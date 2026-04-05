# -*- coding: utf-8 -*-
"""
topic_model.py — Topic modeling (LDA + BERTopic) and cluster-level analysis.

Covers Section 3.4 (Topic Modeling and Clustering).

Usage:
    python topic_model.py
"""

import os

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from bertopic import BERTopic
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from wordcloud import WordCloud

import config

nltk.download("vader_lexicon", quiet=True)

# Global BERTopic model (populated after training)
bertopic_model = None


# ── LDA ────────────────────────────────────────────────────────────────────

def run_lda(texts: list, n_components: int = config.LDA_N_COMPONENTS) -> None:
    """Fit LDA and print the top words per topic."""
    print("\nRunning LDA …")
    cv = CountVectorizer(max_features=5000, stop_words="english")
    X = cv.fit_transform(texts)
    lda = LatentDirichletAllocation(
        n_components=n_components, random_state=config.RANDOM_STATE,
        learning_method="batch", n_jobs=-1,
    )
    lda.fit(X)
    feat = cv.get_feature_names_out()
    for i, topic in enumerate(lda.components_):
        words = [feat[j] for j in topic.argsort()[:-11:-1]]
        print(f"  LDA Topic {i+1}: {', '.join(words)}")
    print("LDA complete.")


# ── BERTopic ───────────────────────────────────────────────────────────────

def run_bertopic(texts: list) -> BERTopic:
    """Fit BERTopic, save the model, and return it."""
    global bertopic_model
    print("\nRunning BERTopic …")
    sbert = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    embs = sbert.encode(texts, show_progress_bar=True)
    bertopic_model = BERTopic(
        embedding_model=sbert,
        verbose=True,
        min_topic_size=config.BERTOPIC_MIN_TOPIC,
        n_gram_range=(1, 2),
        calculate_probabilities=True,
    )
    bertopic_model.fit_transform(texts, embeddings=embs)
    for tid in bertopic_model.get_topics():
        if tid != -1:
            words = [w for w, _ in bertopic_model.get_topic(tid)]
            print(f"  BERTopic {tid}: {', '.join(words[:8])}")
    bertopic_model.save(config.BERTOPIC_MODEL_PATH)
    print(f"BERTopic model saved → {config.BERTOPIC_MODEL_PATH}")
    return bertopic_model


def load_bertopic() -> BERTopic:
    """Load a previously saved BERTopic model."""
    global bertopic_model
    if os.path.exists(config.BERTOPIC_MODEL_PATH):
        bertopic_model = BERTopic.load(config.BERTOPIC_MODEL_PATH)
        print(f"BERTopic loaded from {config.BERTOPIC_MODEL_PATH}")
    else:
        print(f"BERTopic model not found at {config.BERTOPIC_MODEL_PATH}.")
    return bertopic_model


# ── Clustering ─────────────────────────────────────────────────────────────

def cluster_embeddings(
    embedding_file: str, num_clusters: int = config.KMEANS_N_CLUSTERS,
) -> np.ndarray:
    """K-Means clustering on a pre-computed embedding matrix."""
    if not os.path.exists(embedding_file):
        print(f"Not found: {embedding_file}"); return np.array([])
    X = np.load(embedding_file)
    if X.size == 0: return np.array([])
    km = KMeans(n_clusters=num_clusters, random_state=config.RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)
    print(f"K-Means: {num_clusters} clusters assigned to {len(labels)} samples.")
    return labels


# ── Sentiment ──────────────────────────────────────────────────────────────

def analyze_sentiment(texts: list) -> pd.Series:
    """VADER compound sentiment for each text string."""
    sia = SentimentIntensityAnalyzer()
    return pd.Series([sia.polarity_scores(str(t))["compound"] for t in texts])


# ── Combined enrichment ────────────────────────────────────────────────────

def enrich_dataset(
    preprocessed_csv: str,
    embedding_file: str,
    output_csv: str,
) -> pd.DataFrame:
    """Add cluster labels and sentiment scores to a preprocessed CSV."""
    if not os.path.exists(preprocessed_csv):
        print(f"Not found: {preprocessed_csv}"); return pd.DataFrame()
    df = pd.read_csv(preprocessed_csv)

    cluster_labels = cluster_embeddings(embedding_file)
    if cluster_labels.size > 0:
        df["cluster"] = cluster_labels

    df["sentiment_score"] = analyze_sentiment(df["clean_text"].tolist())
    df.to_csv(output_csv, index=False)
    print(f"Enriched dataset → {output_csv}")
    return df


def generate_wordclouds(df: pd.DataFrame, cluster_col="cluster", text_col="clean_text") -> None:
    """One word cloud per cluster."""
    if df.empty or cluster_col not in df.columns: return
    for cluster in sorted(df[cluster_col].unique()):
        text = " ".join(df[df[cluster_col] == cluster][text_col].dropna().astype(str))
        if not text.strip(): continue
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear"); plt.axis("off")
        plt.title(f"Cluster {cluster}"); plt.show()


# ── Entry point ────────────────────────────────────────────────────────────

def perform_topic_modeling(preprocessed_csv: str = config.TRAIN_PREPROCESSED) -> None:
    """Full topic modeling pipeline: LDA + BERTopic + sentiment."""
    if not os.path.exists(preprocessed_csv):
        print(f"Not found: {preprocessed_csv}"); return
    df = pd.read_csv(preprocessed_csv)
    df["clean_text"] = df["clean_text"].fillna("")

    # Use only rumour texts
    rumour_texts = df[df["label"].astype(str).str.lower() == "fake"]["clean_text"].tolist()
    if not rumour_texts:
        print("No rumour texts found."); return

    run_lda(rumour_texts)
    run_bertopic(rumour_texts)


if __name__ == "__main__":
    perform_topic_modeling()

    enriched = enrich_dataset(
        preprocessed_csv=config.TRAIN_PREPROCESSED,
        embedding_file=config.TRAIN_SBERT_PATH,
        output_csv="train_enriched.csv",
    )
    if not enriched.empty:
        generate_wordclouds(enriched)
