# -*- coding: utf-8 -*-
"""
visualize.py — Reproduces all evaluation figures from the paper (Figures 2–7).

Requires outputs/evaluation_results.json produced by evaluate.py.

Usage:
    python visualize.py
    python visualize.py --results outputs/evaluation_results.json
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Loader ───────────────────────────────────────────────────────────────────

def load_results(path: str = "outputs/evaluation_results.json") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Results not found at {path}. Run evaluate.py first."
        )
    return pd.read_json(path)


# ── Figure 2: Score distributions ────────────────────────────────────────────

def plot_score_distributions(df: pd.DataFrame) -> None:
    """Histograms of R-A, A-C, and Fact Coverage (Figure 2)."""
    cols   = ["ra_similarity", "ac_similarity", "fact_coverage"]
    titles = [
        "Distribution of Rumour–Anti-Rumour Similarity",
        "Distribution of Anti-Rumour Context Similarity",
        "Distribution of Fact Coverage Score",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col, title in zip(axes, cols, titles):
        ax.hist(df[col].dropna(), bins=20, edgecolor="white", color="steelblue", alpha=0.85)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Score"); ax.set_ylabel("Frequency")
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/fig2_score_distributions.png", dpi=150)
    plt.show()
    print("Figure 2 saved → outputs/fig2_score_distributions.png")


# ── Figure 3a: Fact Coverage vs A-C Similarity ───────────────────────────────

def plot_coverage_vs_context(df: pd.DataFrame) -> None:
    """Scatter of Fact Coverage vs A-C coloured by hallucination flag (Figure 3a)."""
    plt.figure(figsize=(8, 6))
    colors = {False: "C1", True: "C0"}
    for flag, group in df.groupby("hallucination_flag"):
        label = "No Hallucination" if not flag else "Hallucination"
        plt.scatter(group["ac_similarity"], group["fact_coverage"],
                    label=label, color=colors[flag], alpha=0.55, s=50, edgecolors="white", lw=0.4)
    plt.xlabel("Anti-Rumour Context Similarity", fontsize=13)
    plt.ylabel("Fact Coverage Score", fontsize=13)
    plt.title("Fact Coverage vs. Anti-Rumour Context Similarity", fontsize=14)
    plt.legend(title="Hallucination Flag", fontsize=11)
    plt.grid(linestyle="--", alpha=0.5)
    plt.xlim(0, 1.05); plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig("outputs/fig3a_coverage_vs_context.png", dpi=150)
    plt.show()
    print("Figure 3a saved → outputs/fig3a_coverage_vs_context.png")


# ── Figure 3b: Hallucination rate pie ────────────────────────────────────────

def plot_hallucination_pie(df: pd.DataFrame) -> None:
    """Pie chart of hallucinated vs non-hallucinated outputs (Figure 3b)."""
    hall_rate = df["hallucination_flag"].mean() * 100
    sizes = [100 - hall_rate, hall_rate]
    labels = [f"No Hallucination\n{100-hall_rate:.1f}%", f"Hallucination\n{hall_rate:.1f}%"]
    colors = ["#E8534A", "#4CAF50"]
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors, startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    plt.title("Proportion of Hallucinated Anti-Rumours", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/fig3b_hallucination_pie.png", dpi=150)
    plt.show()
    print("Figure 3b saved → outputs/fig3b_hallucination_pie.png")


# ── Figure 4: Key metrics by hallucination flag ───────────────────────────────

def plot_metrics_by_hallucination(df: pd.DataFrame) -> None:
    """Box plots of R-A, A-C, and Coverage split by hallucination flag (Figure 4)."""
    cols   = ["ra_similarity", "ac_similarity", "fact_coverage"]
    titles = [
        "Rumour–Anti-Rumour Similarity",
        "Anti-Rumour Context Similarity",
        "Fact Coverage Score",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, col, title in zip(axes, cols, titles):
        groups = [
            df.loc[~df["hallucination_flag"], col].dropna().values,
            df.loc[ df["hallucination_flag"], col].dropna().values,
        ]
        bplot = ax.boxplot(groups, vert=True, patch_artist=True,
                           tick_labels=["No Hallucination", "Hallucination"])
        for patch, color in zip(bplot["boxes"], ["lightgreen", "salmon"]):
            patch.set_facecolor(color)
        ax.set_title(title, fontsize=13)
        ax.set_ylabel("Score"); ax.set_xlabel("Hallucination Flag")
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("outputs/fig4_metrics_by_hallucination.png", dpi=150)
    plt.show()
    print("Figure 4 saved → outputs/fig4_metrics_by_hallucination.png")


# ── Figure 5: Correlation matrix ─────────────────────────────────────────────

def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Seaborn heatmap of score correlations (Figure 5)."""
    numeric_cols = ["ra_similarity", "ac_similarity", "fact_coverage", "sentiment_compound"]
    rename = {
        "ra_similarity":      "rumor_anti_rumor_similarity",
        "ac_similarity":      "anti_rumor_context_similarity",
        "fact_coverage":      "fact_coverage_score",
        "sentiment_compound": "anti_rumor_sentiment_compound",
    }
    corr = df[numeric_cols].rename(columns=rename).corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, linewidths=0.5, annot_kws={"fontsize": 13})
    plt.title("Correlation Matrix of Anti-Rumour Scores", fontsize=14, pad=15)
    plt.xticks(fontsize=11, rotation=45, ha="right")
    plt.yticks(fontsize=11, rotation=0)
    plt.tight_layout()
    plt.savefig("outputs/fig5_correlation_matrix.png", dpi=150)
    plt.show()
    print("Figure 5 saved → outputs/fig5_correlation_matrix.png")


# ── Figure 6a: Facts retrieved distribution ──────────────────────────────────

def plot_facts_retrieved(df: pd.DataFrame) -> None:
    """Bar chart of number of facts retrieved per rumour (Figure 6a)."""
    counts = df["n_facts_retrieved"].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    plt.bar(counts.index.astype(str), counts.values, color="steelblue", edgecolor="white")
    plt.title("Distribution of Relevant Facts Retrieved Per Rumour", fontsize=13)
    plt.xlabel("Number of Facts Retrieved"); plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("outputs/fig6a_facts_retrieved.png", dpi=150)
    plt.show()
    print("Figure 6a saved → outputs/fig6a_facts_retrieved.png")


# ── Figure 6b: Source distribution pie ───────────────────────────────────────

def plot_source_distribution(df: pd.DataFrame) -> None:
    """Pie chart of debunking source usage (Figure 6b)."""
    counts = df["source_used"].value_counts()
    plt.figure(figsize=(7, 7))
    plt.pie(counts.values, labels=counts.index,
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.2})
    plt.title("Distribution of Fact Sources Used for Debunking", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/fig6b_source_distribution.png", dpi=150)
    plt.show()
    print("Figure 6b saved → outputs/fig6b_source_distribution.png")


# ── Figure 7: Sentiment distribution ─────────────────────────────────────────

def plot_sentiment_distribution(df: pd.DataFrame) -> None:
    """Bar chart of anti-rumour sentiment labels (Figure 7)."""
    counts = df["sentiment_label"].value_counts()
    plt.figure(figsize=(7, 5))
    plt.bar(counts.index, counts.values, color=["#5C85D6", "#E8534A", "#4CAF50"], edgecolor="white")
    plt.title("Anti-Rumour Sentiment Distribution", fontsize=13)
    plt.xlabel("Sentiment"); plt.ylabel("Number of Anti-Rumours")
    plt.tight_layout()
    plt.savefig("outputs/fig7_sentiment_distribution.png", dpi=150)
    plt.show()
    print("Figure 7 saved → outputs/fig7_sentiment_distribution.png")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate all evaluation figures.")
    parser.add_argument("--results", default="outputs/evaluation_results.json",
                        help="Path to evaluation_results.json from evaluate.py")
    args = parser.parse_args()

    df = load_results(args.results)

    plot_score_distributions(df)
    plot_coverage_vs_context(df)
    plot_hallucination_pie(df)
    plot_metrics_by_hallucination(df)
    plot_correlation_matrix(df)
    plot_facts_retrieved(df)
    plot_source_distribution(df)
    plot_sentiment_distribution(df)

    print("\nAll figures saved to outputs/")
