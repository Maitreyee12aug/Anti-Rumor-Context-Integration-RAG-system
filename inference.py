# -*- coding: utf-8 -*-
"""
inference.py — Single-rumour inference with full XAI output.

Usage:
    python inference.py --rumor "5G towers spread COVID-19."
    python inference.py --interactive
"""

import argparse

from pipeline import generate_anti_rumor, load_kb
from evaluate import (
    rumor_anti_rumor_similarity,
    anti_rumor_context_similarity,
    fact_coverage_score,
    is_hallucination,
    anti_rumor_sentiment,
)


def run_inference(rumor: str, kb=None) -> None:
    """
    Runs the full RAG pipeline on a single rumour and prints a
    structured report with scores and XAI explanation.
    """
    if kb is None:
        kb = load_kb()

    anti_rumor, facts, xai_facts, explanation = generate_anti_rumor(rumor, kb)

    ra   = rumor_anti_rumor_similarity(rumor, anti_rumor)
    ac   = anti_rumor_context_similarity(anti_rumor, facts)
    cov  = fact_coverage_score(anti_rumor, facts)
    hall = is_hallucination(ac)
    sent = anti_rumor_sentiment(anti_rumor)
    src  = facts[0].get("source", "N/A") if facts else "N/A"
    url  = facts[0].get("url",    "N/A") if facts else "N/A"

    print("\n" + "=" * 62)
    print("  ANTI-RUMOUR REPORT")
    print("=" * 62)
    print(f"  RUMOUR          : {rumor}")
    print(f"  ANTI-RUMOUR     : {anti_rumor}")
    print(f"  SOURCE USED     : {src}")
    print(f"  SOURCE URL      : {url}")
    print("-" * 62)
    print(f"  R-A Similarity  : {ra:.4f}  (rumour ↔ anti-rumour)")
    print(f"  A-C Similarity  : {ac:.4f}  (anti-rumour ↔ context)")
    print(f"  Fact Coverage   : {cov:.4f}")
    print(f"  Hallucination?  : {'⚠ YES' if hall else '✓ NO'}")
    print(f"  Sentiment       : {sent['label']} (compound: {sent['compound']:.4f})")
    print(f"  Facts Retrieved : {len(facts)}")
    print("-" * 62)
    print("  XAI EXPLANATION:")
    print(f"  {explanation}")
    print("=" * 62)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-sample anti-rumour inference.")
    parser.add_argument("--rumor",       type=str, default=None, help="Rumour text to debunk.")
    parser.add_argument("--interactive", action="store_true",    help="Interactive mode.")
    args = parser.parse_args()

    kb = load_kb()

    if args.rumor:
        run_inference(args.rumor, kb)
    elif args.interactive:
        print("Interactive inference mode. Type 'exit' to quit.\n")
        while True:
            rumor = input("Enter rumour: ").strip()
            if rumor.lower() == "exit":
                break
            if rumor:
                run_inference(rumor, kb)
    else:
        # Default demo
        demo_rumors = [
            "COVID-19 vaccines contain microchips to track people.",
            "5G towers are responsible for spreading COVID-19.",
            "President Zelenskyy fled Ukraine at the start of the war.",
        ]
        for rumor in demo_rumors:
            run_inference(rumor, kb)
