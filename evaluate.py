# -*- coding: utf-8 -*-
"""
evaluate.py — Batch evaluation on the test split.

Metrics (Table 4 of the paper):
    - Rumour-Anti-Rumour Similarity  (R-A)
    - Anti-Rumour-Context Similarity (A-C)   ← primary grounding indicator
    - Fact Coverage Score            (Cov.)
    - Hallucination Rate             (Hall.)
    - Anti-Rumour Sentiment          (neutral / positive / negative)
    - Source Used for Debunking
    - Number of Relevant Facts Retrieved

Usage:
    python evaluate.py --csv test.csv --sample 10000
"""

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import config
from pipeline import generate_anti_rumor, load_kb, get_embedding

sentiment_analyzer = SentimentIntensityAnalyzer()


# ── Metric functions ─────────────────────────────────────────────────────────

def rumor_anti_rumor_similarity(rumor: str, anti_rumor: str) -> float:
    """Cosine similarity between rumour and anti-rumour SBERT embeddings."""
    if not rumor.strip() or not anti_rumor.strip():
        return 0.0
    e1, e2 = get_embedding(rumor), get_embedding(anti_rumor)
    return float(cosine_similarity(e1.reshape(1,-1), e2.reshape(1,-1))[0][0])


def anti_rumor_context_similarity(anti_rumor: str, facts: List[Dict]) -> float:
    """Cosine similarity between anti-rumour and the concatenated fact context."""
    if not facts or not anti_rumor.strip():
        return 0.0
    ctx = " ".join(f.get("fact","") for f in facts)
    e1, e2 = get_embedding(anti_rumor), get_embedding(ctx)
    return float(cosine_similarity(e1.reshape(1,-1), e2.reshape(1,-1))[0][0])


def fact_coverage_score(anti_rumor: str, facts: List[Dict]) -> float:
    """
    Max cosine similarity between the anti-rumour and each individual fact.
    Measures how well key facts are semantically covered.
    """
    if not facts or not anti_rumor.strip():
        return 0.0
    anti_emb = get_embedding(anti_rumor)
    sims = []
    for fact in facts:
        fact_text = fact.get("fact","")
        if fact_text.strip():
            sims.append(float(cosine_similarity(
                anti_emb.reshape(1,-1), get_embedding(fact_text).reshape(1,-1)
            )[0][0]))
    return max(sims) if sims else 0.0


def is_hallucination(a_c_score: float, threshold: float = config.HALLUCINATION_THRESHOLD) -> bool:
    """Flag as hallucination if Anti-Rumour-Context similarity is below threshold."""
    return a_c_score < threshold


def anti_rumor_sentiment(anti_rumor: str) -> Dict[str, Any]:
    scores = sentiment_analyzer.polarity_scores(anti_rumor)
    compound = scores["compound"]
    label = "Positive" if compound > 0.05 else "Negative" if compound < -0.05 else "Neutral"
    return {"compound": compound, "label": label}


# ── Batch evaluation ─────────────────────────────────────────────────────────

def evaluate(
    csv_path: str = config.TEST_CSV,
    sample_size: int = config.EVAL_SAMPLE_SIZE,
    output_path: str = "outputs/evaluation_results.json",
) -> pd.DataFrame:
    """
    Runs the full pipeline on a random sample of the test set
    and computes all evaluation metrics.

    Args:
        csv_path:    Path to the test CSV (must have a text/tweet column).
        sample_size: Number of instances to evaluate.
        output_path: Where to save the per-instance results JSON.

    Returns:
        DataFrame with per-instance scores and summary statistics.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Test CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    text_col = config.TEXT_COLUMN if config.TEXT_COLUMN in df.columns else "text"
    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not found in {csv_path}.")

    # Sample
    df = df.dropna(subset=[text_col])
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=config.RANDOM_STATE)

    kb = load_kb()
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        rumor = str(row[text_col])
        try:
            anti_rumor, facts, _, _ = generate_anti_rumor(rumor, kb)
        except Exception as e:
            print(f"  Pipeline error: {e}")
            continue

        ra  = rumor_anti_rumor_similarity(rumor, anti_rumor)
        ac  = anti_rumor_context_similarity(anti_rumor, facts)
        cov = fact_coverage_score(anti_rumor, facts)
        hall = is_hallucination(ac)
        sent = anti_rumor_sentiment(anti_rumor)
        src  = facts[0].get("source","N/A") if facts else "N/A"
        n_facts = len(facts)

        results.append({
            "rumor":                rumor,
            "anti_rumor":           anti_rumor,
            "ra_similarity":        round(ra, 4),
            "ac_similarity":        round(ac, 4),
            "fact_coverage":        round(cov, 4),
            "hallucination_flag":   hall,
            "sentiment_compound":   round(sent["compound"], 4),
            "sentiment_label":      sent["label"],
            "source_used":          src,
            "n_facts_retrieved":    n_facts,
        })

    results_df = pd.DataFrame(results)

    # Summary
    print("\n" + "="*55)
    print("  EVALUATION SUMMARY")
    print("="*55)
    print(f"  Rumour–Anti-Rumour Similarity (R-A): {results_df['ra_similarity'].mean():.4f}")
    print(f"  Anti-Rumour–Context Similarity (A-C): {results_df['ac_similarity'].mean():.4f}")
    print(f"  Fact Coverage Score (Cov.):           {results_df['fact_coverage'].mean():.4f}")
    print(f"  Hallucination Rate:                   {results_df['hallucination_flag'].mean()*100:.2f}%")
    print(f"  Avg Facts Retrieved:                  {results_df['n_facts_retrieved'].mean():.2f}")
    print(f"  Source distribution:\n{results_df['source_used'].value_counts().to_string()}")
    print(f"  Sentiment distribution:\n{results_df['sentiment_label'].value_counts().to_string()}")
    print("="*55)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_json(output_path, orient="records", indent=2)
    print(f"\nResults saved → {output_path}")
    return results_df


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the anti-rumour RAG pipeline.")
    parser.add_argument("--csv",    default=config.TEST_CSV,     help="Test CSV path")
    parser.add_argument("--sample", default=config.EVAL_SAMPLE_SIZE, type=int, help="Sample size")
    parser.add_argument("--output", default="outputs/evaluation_results.json", help="Output JSON path")
    args = parser.parse_args()

    evaluate(csv_path=args.csv, sample_size=args.sample, output_path=args.output)
