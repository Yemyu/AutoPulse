"""DeepSeek ABSA (Aspect-Based Sentiment Analysis) pipeline.

Reads cleaned reviews, calls DeepSeek API to extract per-aspect sentiment,
and writes structured results to data/sentiment/absa/.

Usage:
    # Pilot test (first 5 reviews)
    /opt/miniconda3/envs/nlp-sentiment/bin/python scripts/14_aspect_sentiment_llm.py --limit 5

    # Full run (all filtered reviews, resumable)
    /opt/miniconda3/envs/nlp-sentiment/bin/python scripts/14_aspect_sentiment_llm.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

# Ensure project root is on path so config can be imported.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402

# ---------------------------------------------------------------------------
# ABSA configuration
# ---------------------------------------------------------------------------
ASPECTS: list[str] = settings.ABSA_ASPECTS
ASPECT_LABELS: dict[str, str] = {
    "appearance": "外观",
    "interior": "内饰",
    "space": "空间",
    "power": "动力",
    "control": "操控",
    "comfort": "舒适",
    "fuel_consumption": "油耗",
    "configuration": "配置",
    "intelligence": "智能化",
    "value": "性价比",
}

OUTPUT_DIR = settings.ABSA_OUTPUT_DIR
CHECKPOINT_PATH = OUTPUT_DIR / "absa_checkpoint.json"
RESULT_PATH = OUTPUT_DIR / "absa_results.csv"

SALES_START = "2022-01-01"
SALES_END = "2026-05-31"

BATCH_SIZE = settings.LLM_BATCH_SIZE
MAX_RETRIES = settings.LLM_MAX_RETRIES
REQUEST_TIMEOUT = settings.LLM_REQUEST_TIMEOUT
MAX_TOKENS = settings.LLM_MAX_TOKENS
TEMPERATURE = settings.LLM_TEMPERATURE

# API key validation
if not settings.DEEPSEEK_API_KEY or settings.DEEPSEEK_API_KEY == "xxxxx":
    raise RuntimeError(
        "DEEPSEEK_API_KEY is not set. Please fill it in config/.env before running this script."
    )

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "你是一位汽车评论分析专家。请分析下面这条汽车用户评价，判断用户对以下维度分别表达了什么情感。"
    "如果评论没有提到某个维度，情感为0；提到且正面为1，提到且负面为-1。"
    "输出严格为以下JSON格式，不要任何解释，不要markdown代码块。"
)


def build_user_prompt(content: str) -> str:
    aspects = "\n".join(f"- {cn} ({en})" for en, cn in ASPECT_LABELS.items())
    return f"需要分析的维度：\n{aspects}\n\n评论：\n{content}\n\n输出JSON："


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------
def call_deepseek(content: str) -> dict[str, int]:
    """Call DeepSeek chat completion and return parsed ABSA dict."""
    headers = {
        "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(content)},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }

    last_exception: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                f"{settings.DEEPSEEK_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return parse_absa_json(text)
        except Exception as e:  # noqa: BLE001
            last_exception = e
            wait = 2 ** attempt
            time.sleep(wait)

    # All retries failed
    raise RuntimeError(
        f"DeepSeek API failed after {MAX_RETRIES} retries: {last_exception}"
    ) from last_exception


_JSON_RE = re.compile(r"\{[\s\S]*?\}")


def parse_absa_json(text: str) -> dict[str, int]:
    """Parse JSON response, falling back to regex extraction if needed."""
    text = text.strip()
    if not text:
        return {}

    # Try direct parse first
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Extract first JSON object
        match = _JSON_RE.search(text)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    result: dict[str, int] = {}
    for aspect in ASPECTS:
        val = parsed.get(aspect)
        if isinstance(val, (int, float)):
            result[aspect] = int(val)
        elif isinstance(val, str):
            # Sometimes model returns strings like "1"
            try:
                result[aspect] = int(val)
            except ValueError:
                result[aspect] = 0
        else:
            result[aspect] = 0
    return result


# ---------------------------------------------------------------------------
# Data loading and filtering
# ---------------------------------------------------------------------------
def load_and_filter_reviews(limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(settings.SENTIMENT_DIR / "sentiment_reviews.csv")
    df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")

    # Filter to sales window
    start_dt = pd.Timestamp(SALES_START)
    end_dt = pd.Timestamp(SALES_END)
    df = df[(df["publish_time"] >= start_dt) & (df["publish_time"] <= end_dt)]

    # Drop short / empty content
    df = df[df["content_len"] >= 20]

    # Drop exact duplicate content (keep first)
    df = df.drop_duplicates(subset=["content"], keep="first")

    # Keep only rows with non-empty content
    df = df[df["content"].notna() & (df["content"].astype(str).str.strip() != "")]

    if limit is not None and limit > 0:
        df = df.head(limit)

    print(f"Filtered reviews for ABSA: {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Checkpointing and resumability
# ---------------------------------------------------------------------------
def load_checkpoint() -> dict[str, dict[str, int]]:
    if CHECKPOINT_PATH.exists():
        with CHECKPOINT_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Map review_id -> result dict
        return {str(item["review_id"]): item for item in data.get("results", [])}
    return {}


def save_checkpoint(results: list[dict]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"results": results}
    with CHECKPOINT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------
def process_review(row: pd.Series) -> dict:
    """Process a single review and return a dict with original fields + ABSA."""
    review_id = row["review_id"]
    content = str(row["content"])
    try:
        absa = call_deepseek(content)
        record = {
            "review_id": review_id,
            "series_id": row.get("series_id"),
            "series_name": row.get("series_name"),
            "publish_time": row["publish_time"].strftime("%Y-%m-%d %H:%M:%S"),
            "content": content,
            "success": True,
            "error": "",
        }
        for aspect in ASPECTS:
            record[aspect] = absa.get(aspect, 0)
        return record
    except Exception as e:  # noqa: BLE001
        return {
            "review_id": review_id,
            "series_id": row.get("series_id"),
            "series_name": row.get("series_name"),
            "publish_time": row["publish_time"].strftime("%Y-%m-%d %H:%M:%S"),
            "content": content,
            "success": False,
            "error": str(e),
            **{aspect: 0 for aspect in ASPECTS},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek ABSA pipeline")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N reviews (for pilot testing).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of concurrent API workers (default: 10).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoint and start over.",
    )
    args = parser.parse_args()

    df = load_and_filter_reviews(limit=args.limit)
    if df.empty:
        print("No reviews to process. Exiting.")
        return

    # Load or init checkpoint
    checkpoint = {} if args.no_resume else load_checkpoint()
    if checkpoint:
        print(f"Resuming from checkpoint: {len(checkpoint)} reviews already processed")

    # Filter out already-processed reviews
    df_todo = df[~df["review_id"].astype(str).isin(checkpoint.keys())].copy()
    print(f"Remaining to process: {len(df_todo)} / {len(df)}")

    if df_todo.empty:
        print("All reviews already processed. Saving final CSV.")
        final_df = pd.DataFrame.from_records(list(checkpoint.values()))
        final_df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
        print(f"Saved: {RESULT_PATH}")
        return

    results: list[dict] = list(checkpoint.values())
    total = len(df_todo)
    completed = 0
    failed = 0

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_idx = {
                executor.submit(process_review, row): idx
                for idx, row in df_todo.iterrows()
            }
            for future in as_completed(future_to_idx):
                record = future.result()
                results.append(record)
                if record["success"]:
                    completed += 1
                else:
                    failed += 1

                # Save checkpoint every BATCH_SIZE records
                if len(results) % BATCH_SIZE == 0:
                    save_checkpoint(results)
                    print(
                        f"  Progress: {len(results)}/{total + len(checkpoint)} "
                        f"ok={completed} fail={failed}"
                    )
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving checkpoint...")
    except Exception as e:  # noqa: BLE001
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()
    finally:
        save_checkpoint(results)
        final_df = pd.DataFrame.from_records(results)
        final_df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
        print(f"\nSaved checkpoint: {CHECKPOINT_PATH}")
        print(f"Saved results: {RESULT_PATH}")
        print(f"Total processed: {len(results)} | Success: {completed} | Failed: {failed}")


if __name__ == "__main__":
    main()
