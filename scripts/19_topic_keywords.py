#!/usr/bin/env python3
"""
topic keywords + LDA for top-3 sentiment aspects
针对阶段四 SHAP 最重要的三个维度（舒适/性价比/智能），从评论原文中提取
关键词和 LDA 主题，回答"用户到底在聊什么"。

输出：
  data/processed/stage5/topic_keywords.csv
  data/processed/stage5/lda_topics.csv
  figures/stage5_topic_keywords.png
  figures/stage5_lda_topics.png

Run:
  python scripts/19_topic_keywords.py
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import jieba
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _font_setup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABSA = os.path.join(BASE, "data", "sentiment", "absa", "absa_results.csv")
PROC = os.path.join(BASE, "data", "processed", "stage5")
FIG = os.path.join(BASE, "figures")
os.makedirs(PROC, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

TOP3 = ["comfort", "value", "intelligence"]
ASPECT_LABEL = {"comfort": "舒适性", "value": "性价比", "intelligence": "智能化"}
N_TOPICS = 5
N_WORDS = 10

STOPWORDS = set([
    "的", "了", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说",
    "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "还是", "但是", "就是", "非常",
    "感觉", "有点", "一下", "虽然", "但是", "因为", "所以", "如果", "时候", "不过", "真的", "比较", "还是",
    "觉得", "认为", "已经", "可以", "应该", "可能", "一下", "特别", "其实", "反正", "大概", "确实", "不是",
    "这个", "那个", "这些", "那些", "这么", "那么", "怎么", "什么", "不要", "不能", "不会", "不敢", "不太",
    "不用", "还是", "不是", "没有", "很多", "一点", "一直", "一次", "一样", "一般", "总体", "整体", "大家",
    "我们", "你们", "他们", "它们", "这边", "那边", "这里", "那里", "一下", "一些", "一下",
])


def tokenize(text):
    words = jieba.lcut(str(text))
    return [w.strip() for w in words if len(w.strip()) > 1 and w.strip() not in STOPWORDS
            and not w.strip().isdigit() and not w.strip().replace(".", "").isdigit()]


def tokenize_join(text):
    return " ".join(tokenize(text))


def get_keywords(docs, top_n=20):
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, min_df=3, max_df=0.9)
    tfidf = vectorizer.fit_transform(docs)
    terms = vectorizer.get_feature_names_out()
    scores = np.asarray(tfidf.mean(axis=0)).flatten()
    idx = scores.argsort()[::-1][:top_n]
    return [(terms[i], float(scores[i])) for i in idx]


def get_lda(docs, n_topics=N_TOPICS, n_words=N_WORDS):
    vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None, min_df=3, max_df=0.9, max_features=1000)
    dt = vectorizer.fit_transform(docs)
    if dt.shape[1] < 10:
        return [], []
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=15, learning_method="online")
    lda.fit(dt)
    terms = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top = [terms[i] for i in topic.argsort()[:-n_words - 1:-1]]
        topics.append((topic_idx + 1, top))
    return topics, lda, vectorizer, dt


def main():
    absa = pd.read_csv(ABSA)
    absa["content"] = absa["content"].astype(str)
    print(f"[Stage 5B] loaded {len(absa)} ABSA records")

    keyword_rows = []
    topic_rows = []
    fig_kw, axes_kw = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig_lda, axes_lda = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    for col, aspect in enumerate(TOP3):
        label = ASPECT_LABEL[aspect]
        # keep reviews where this aspect has non-zero sentiment
        sub = absa[absa[aspect].abs() > 0].copy()
        if len(sub) < 50:
            print(f"[WARN] {aspect}: only {len(sub)} records, skip")
            continue
        docs = sub["content"].apply(tokenize_join).tolist()
        docs = [d for d in docs if d.strip()]
        print(f"[{aspect}] {len(docs)} docs for analysis")

        # keywords
        kw = get_keywords(docs, top_n=15)
        for word, score in kw:
            keyword_rows.append({"aspect": label, "word": word, "tfidf_score": score})
        words, scores = zip(*kw[:10]) if kw else ([], [])
        ax = axes_kw[col]
        ax.barh(list(reversed(words)), list(reversed(scores)), color="#4C78A8")
        ax.set_title(f"{label} - top keywords")
        ax.set_xlabel("TF-IDF score")

        # LDA
        topics, lda, vec, dt = get_lda(docs)
        if topics:
            for tid, top_words in topics:
                topic_rows.append({"aspect": label, "topic_id": tid, "top_words": " ".join(top_words)})
            ax = axes_lda[col]
            y_labels = [f"T{t[0]}" for t in topics]
            # use a simple heatmap: topic x top5 words
            top5 = [t[1][:5] for t in topics]
            data = np.zeros((len(topics), 5))
            for i, words in enumerate(top5):
                for j, w in enumerate(words):
                    if w in vec.vocabulary_:
                        data[i, j] = lda.components_[i, vec.vocabulary_[w]]
            im = ax.imshow(data, cmap="Blues", aspect="auto")
            ax.set_xticks(range(5))
            ax.set_xticklabels([f"w{i+1}" for i in range(5)], fontsize=8)
            ax.set_yticks(range(len(topics)))
            ax.set_yticklabels(y_labels, fontsize=8)
            ax.set_title(f"{label} - LDA topics")
            for i in range(len(topics)):
                for j in range(5):
                    ax.text(j, i, top5[i][j], ha="center", va="center", fontsize=7, color="white" if data[i, j] > data.max() / 2 else "black")
        else:
            axes_lda[col].axis("off")
            axes_lda[col].set_title(f"{label} - LDA skipped (too sparse)")

    fig_kw.suptitle("Stage 5 — keywords for top-3 sentiment aspects", fontsize=12)
    fig_kw.savefig(os.path.join(FIG, "stage5_topic_keywords.png"), dpi=130)
    print(f"[Plot] {os.path.join(FIG, 'stage5_topic_keywords.png')}")

    fig_lda.suptitle("Stage 5 — LDA topics for top-3 sentiment aspects", fontsize=12)
    fig_lda.savefig(os.path.join(FIG, "stage5_lda_topics.png"), dpi=130)
    print(f"[Plot] {os.path.join(FIG, 'stage5_lda_topics.png')}")

    pd.DataFrame(keyword_rows).to_csv(os.path.join(PROC, "topic_keywords.csv"), index=False)
    pd.DataFrame(topic_rows).to_csv(os.path.join(PROC, "lda_topics.csv"), index=False)
    print("[Stage 5B] done.")


if __name__ == "__main__":
    main()
