# -*- coding: utf-8 -*-
"""
preprocess.py — Data loading, cleaning, and feature engineering.

Covers Sections 3.2 (Preprocessing) and 3.3 (Feature Engineering).

Usage:
    python preprocess.py
"""

import os
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

import config

# Download NLTK resources once
for _r in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(_r, quiet=True)

# Global TF-IDF vectorizer (fitted on training split)
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=config.TFIDF_MAX_FEATURES)


# ── Preprocessing ──────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Lowercase → tokenize → remove stopwords → lemmatize.
    Section 3.2 of the paper.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    tokens = word_tokenize(text)
    stops = set(stopwords.words("english"))
    tokens = [t for t in tokens if t.isalpha() and t not in stops]
    lem = WordNetLemmatizer()
    return " ".join(lem.lemmatize(t) for t in tokens)


def load_and_split_dataset(
    input_path: str = config.RAW_DATASET_PATH,
    train_path: str = config.TRAIN_CSV,
    val_path: str   = config.VAL_CSV,
    test_path: str  = config.TEST_CSV,
) -> None:
    """Stratified train/val/test split."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset not found at {input_path}")
    df = pd.read_csv(input_path)
    if "label" not in df.columns:
        raise KeyError("Dataset must contain a 'label' column.")

    train_val, test = train_test_split(
        df, test_size=config.DATASET_TEST_SIZE,
        stratify=df["label"], random_state=config.RANDOM_STATE,
    )
    val_frac = config.DATASET_VAL_SIZE / (1 - config.DATASET_TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=val_frac,
        stratify=train_val["label"], random_state=config.RANDOM_STATE,
    )
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"Split → Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")


def apply_preprocessing(
    input_csv: str, output_csv: str,
    text_column: str = config.TEXT_COLUMN,
) -> None:
    """Apply preprocess_text to a CSV and save."""
    if not os.path.exists(input_csv):
        print(f"Not found: {input_csv}"); return
    df = pd.read_csv(input_csv)
    if text_column not in df.columns:
        print(f"Column '{text_column}' missing."); return
    df["clean_text"] = df[text_column].astype(str).apply(preprocess_text)
    df.to_csv(output_csv, index=False)
    print(f"Preprocessed → {output_csv}")


def visualize_splits(
    train_path: str = config.TRAIN_CSV,
    val_path: str   = config.VAL_CSV,
    test_path: str  = config.TEST_CSV,
) -> None:
    for name, path in [("Train", train_path), ("Val", val_path), ("Test", test_path)]:
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        if "label" not in df.columns: continue
        df["label"].value_counts().plot(kind="bar", figsize=(6, 4), title=f"{name} Label Distribution")
        plt.tight_layout(); plt.show()


def visualize_preprocessing(
    input_csv: str, text_column: str = "clean_text", top_n: int = 20
) -> None:
    if not os.path.exists(input_csv): return
    df = pd.read_csv(input_csv)
    if text_column not in df.columns: return
    token_counts = df[text_column].astype(str).apply(lambda x: len(x.split()))
    token_counts.hist(bins=30, figsize=(8, 5))
    plt.title("Token Count Distribution"); plt.xlabel("Tokens"); plt.ylabel("Freq")
    plt.tight_layout(); plt.show()

    all_tokens = " ".join(df[text_column].astype(str)).split()
    freq = Counter(all_tokens).most_common(top_n)
    tokens, counts = zip(*freq)
    plt.figure(figsize=(10, 6)); plt.bar(tokens, counts)
    plt.xticks(rotation=45, ha="right"); plt.title(f"Top {top_n} Tokens")
    plt.tight_layout(); plt.show()


# ── Feature Engineering ────────────────────────────────────────────────────

def compute_tfidf(
    input_csv: str, output_path: str, text_column: str = "clean_text"
) -> None:
    """Fit TF-IDF on corpus and save feature matrix (Section 3.3)."""
    global tfidf_vectorizer
    if not os.path.exists(input_csv): return
    df = pd.read_csv(input_csv)
    if text_column not in df.columns: return
    corpus = df[text_column].astype(str).tolist()
    tfidf_vectorizer = TfidfVectorizer(max_features=config.TFIDF_MAX_FEATURES)
    X = tfidf_vectorizer.fit_transform(corpus)
    np.save(output_path, X.toarray())
    print(f"TF-IDF matrix → {output_path}")


def transform_tfidf(input_csv: str, output_path: str, text_column: str = "clean_text") -> None:
    if not hasattr(tfidf_vectorizer, "vocabulary_"):
        print("Fit TF-IDF first via compute_tfidf()."); return
    if not os.path.exists(input_csv): return
    df = pd.read_csv(input_csv)
    if text_column not in df.columns: return
    X = tfidf_vectorizer.transform(df[text_column].astype(str).tolist())
    np.save(output_path, X.toarray())
    print(f"TF-IDF features → {output_path}")


def load_glove_embeddings(glove_path: str = config.GLOVE_PATH, dim: int = config.GLOVE_DIM):
    if not os.path.exists(glove_path):
        print(f"GloVe file not found at {glove_path}."); return {}
    embeddings = {}
    with open(glove_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            vec = np.array(parts[1:], dtype="float32")
            if len(vec) == dim:
                embeddings[parts[0]] = vec
    print(f"Loaded {len(embeddings)} GloVe vectors.")
    return embeddings


def compute_glove_features(
    input_csv: str, output_path: str, glove_embeddings: dict,
    dim: int = config.GLOVE_DIM, text_column: str = "clean_text",
) -> None:
    if not os.path.exists(input_csv) or not glove_embeddings: return
    df = pd.read_csv(input_csv)
    if text_column not in df.columns: return
    vectors = []
    for doc in df[text_column].astype(str):
        vecs = [glove_embeddings[w] for w in doc.split() if w in glove_embeddings]
        vectors.append(np.mean(vecs, axis=0) if vecs else np.zeros(dim))
    np.save(output_path, np.array(vectors))
    print(f"GloVe features → {output_path}")


def compute_sbert_embeddings(
    input_csv: str, output_path: str,
    model_name: str = config.SBERT_MODEL_NAME,
    text_column: str = "clean_text",
) -> None:
    if not os.path.exists(input_csv): return
    df = pd.read_csv(input_csv)
    if text_column not in df.columns: return
    sbert = SentenceTransformer(model_name)
    embs = sbert.encode(df[text_column].astype(str).tolist(), show_progress_bar=True)
    np.save(output_path, embs)
    print(f"SBERT embeddings → {output_path}")


def plot_embeddings(X, labels=None, method="pca", title="2D Embedding") -> None:
    if not isinstance(X, np.ndarray) or X.size == 0 or X.shape[1] < 2: return
    if method == "pca":
        reducer = PCA(n_components=2, random_state=config.RANDOM_STATE)
    else:
        if X.shape[0] > 2000: X, labels = X[:2000], (labels[:2000] if labels is not None else None)
        reducer = TSNE(n_components=2, perplexity=30, random_state=config.RANDOM_STATE, n_jobs=-1)
    X2 = reducer.fit_transform(X)
    plt.figure(figsize=(8, 6))
    if labels is not None:
        for lbl in np.unique(labels):
            idx = np.array(labels) == lbl
            plt.scatter(X2[idx, 0], X2[idx, 1], label=lbl, alpha=0.6, s=10)
        plt.legend()
    else:
        plt.scatter(X2[:, 0], X2[:, 1], alpha=0.6, s=10)
    plt.title(title); plt.xlabel("C1"); plt.ylabel("C2"); plt.grid(); plt.tight_layout(); plt.show()


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Step 1: Split dataset ===")
    load_and_split_dataset()
    visualize_splits()

    print("\n=== Step 2: Preprocess ===")
    for in_csv, out_csv in [
        (config.TRAIN_CSV, config.TRAIN_PREPROCESSED),
        (config.VAL_CSV,   config.VAL_PREPROCESSED),
        (config.TEST_CSV,  config.TEST_PREPROCESSED),
    ]:
        apply_preprocessing(in_csv, out_csv)
    visualize_preprocessing(config.TRAIN_PREPROCESSED)

    print("\n=== Step 3: TF-IDF ===")
    compute_tfidf(config.TRAIN_PREPROCESSED, config.TRAIN_TFIDF_PATH)
    transform_tfidf(config.VAL_PREPROCESSED, "val_tfidf.npy")
    transform_tfidf(config.TEST_PREPROCESSED, "test_tfidf.npy")

    print("\n=== Step 4: GloVe ===")
    glove = load_glove_embeddings()
    if glove:
        compute_glove_features(config.TRAIN_PREPROCESSED, config.TRAIN_GLOVE_PATH, glove)

    print("\n=== Step 5: SBERT ===")
    compute_sbert_embeddings(config.TRAIN_PREPROCESSED, config.TRAIN_SBERT_PATH)

    print("\n=== Step 6: Visualise embeddings ===")
    if os.path.exists(config.TRAIN_SBERT_PATH) and os.path.exists(config.TRAIN_CSV):
        X = np.load(config.TRAIN_SBERT_PATH)
        lbls = pd.read_csv(config.TRAIN_CSV)["label"].values
        plot_embeddings(X[:1000], lbls[:1000], method="pca", title="PCA of SBERT Embeddings")

    print("\nPreprocessing complete.")
