# -*- coding: utf-8 -*-
"""
kb_builder.py — Builds and dynamically refreshes the unified fact knowledge base.

Covers Section 3.5 (Knowledge Base Construction & Management).
KB entry schema matches Table 2 of the paper.

Usage:
    python kb_builder.py               # full build from all sources
    python kb_builder.py --refresh     # add new facts from new_facts.json
"""

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

import config

# ── Globals ────────────────────────────────────────────────────────────────
_model: Optional[SentenceTransformer] = None
_EMBEDDING_DIM: int = 768
kb: List[Dict[str, Any]] = []
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=config.TFIDF_MAX_FEATURES)


def _get_model() -> SentenceTransformer:
    global _model, _EMBEDDING_DIM
    if _model is None:
        _model = SentenceTransformer(config.RAG_EMBEDDING_MODEL)
        _EMBEDDING_DIM = _model.get_sentence_embedding_dimension()
    return _model


def _embed(text: str) -> List[float]:
    if not text.strip():
        return np.zeros(_EMBEDDING_DIM).tolist()
    return _get_model().encode(text).tolist()


def _clean(text: str, limit: int = 300) -> str:
    if not isinstance(text, str): return ""
    text = re.sub(r"\[.*?\]", "", text)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _load_json(path: str):
    if not os.path.exists(path):
        print(f"  Not found: {path}"); return None
    try:
        with open(path) as f: return json.load(f)
    except json.JSONDecodeError as e:
        print(f"  JSON error in {path}: {e}"); return None


# ── Per-source processors ───────────────────────────────────────────────────

def _from_wikipedia(path: str) -> List[Dict]:
    data = _load_json(path)
    if not data: return []
    facts = []
    for topic, sentences in tqdm(data.items(), desc="Wikipedia"):
        for s in sentences:
            c = _clean(s)
            if c:
                facts.append({
                    "fact": c, "source": "Wikipedia",
                    "topic": _clean(topic, 100),
                    "url": f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}",
                    "embedding": _embed(c),
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "confidence_score": 0.80,
                })
    print(f"  Wikipedia: {len(facts)} facts"); return facts


def _from_who(path: str) -> List[Dict]:
    data = _load_json(path)
    if not data: return []
    facts = []
    for title, sentences in tqdm(data.items(), desc="WHO"):
        for s in sentences:
            c = _clean(s)
            if c:
                facts.append({
                    "fact": c, "source": "WHO",
                    "topic": _clean(title, 100),
                    "url": "https://www.who.int/news-room/fact-sheets",
                    "embedding": _embed(c),
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "confidence_score": 0.95,
                })
    print(f"  WHO: {len(facts)} facts"); return facts


def _from_factcheck(path: str) -> List[Dict]:
    data = _load_json(path)
    if not data: return []
    facts = []
    for entry in tqdm(data, desc="FactCheck.org"):
        combined = f"{entry.get('claim_summary','')} {entry.get('verdict','')}".strip()
        c = _clean(combined)
        if c:
            facts.append({
                "fact": c, "source": "FactCheck.org",
                "topic": _clean(entry.get("title", ""), 100),
                "url": entry.get("url", ""),
                "embedding": _embed(c),
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "confidence_score": 0.90,
            })
    print(f"  FactCheck.org: {len(facts)} facts"); return facts


def _from_politifact(path: str) -> List[Dict]:
    data = _load_json(path)
    if not data: return []
    facts = []
    for entry in tqdm(data, desc="PolitiFact"):
        claim   = entry.get("claim",     entry.get("statement", ""))
        ruling  = entry.get("ruling",    entry.get("rating", ""))
        speaker = entry.get("speaker",   "Unknown")
        combined = f"Claim by {speaker}: {claim}. Ruling: {ruling}".strip()
        c = _clean(combined)
        if c:
            facts.append({
                "fact": c, "source": "PolitiFact",
                "topic": _clean(f"PolitiFact: {speaker} – {ruling}", 100),
                "url": entry.get("url", "https://www.politifact.com"),
                "embedding": _embed(c),
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "confidence_score": 0.90,
            })
    print(f"  PolitiFact: {len(facts)} facts"); return facts


# ── Public API ──────────────────────────────────────────────────────────────

def build_initial_unified_kb(
    wikipedia_path: str  = config.WIKIPEDIA_KB_PATH,
    who_path: str        = config.WHO_KB_PATH,
    factcheck_path: str  = config.FACTCHECK_KB_PATH,
    politifact_path: str = config.POLITIFACT_KB_PATH,
    output_path: str     = config.UNIFIED_KB_PATH,
) -> None:
    """
    Merges all source JSONs, deduplicates by fact text, and saves the
    unified knowledge base (Table 2 schema).
    """
    print("Building unified KB …")
    all_facts: List[Dict] = []
    all_facts += _from_wikipedia(wikipedia_path)
    all_facts += _from_who(who_path)
    all_facts += _from_factcheck(factcheck_path)
    all_facts += _from_politifact(politifact_path)

    unique = {f["fact"]: f for f in all_facts}
    final  = list(unique.values())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"Unified KB saved → {output_path}  ({len(final)} unique facts)")


def refresh_knowledge_base(
    new_data_path: Optional[str] = None,
    kb_path: str = config.UNIFIED_KB_PATH,
) -> None:
    """
    Loads the existing KB, ingests new facts from a JSON file,
    re-fits the TF-IDF index, and saves.

    Args:
        new_data_path: JSON list of dicts with keys: fact, source, url, confidence.
        kb_path:       Path to the unified KB JSON.
    """
    global kb, tfidf_vectorizer

    if os.path.exists(kb_path):
        with open(kb_path) as f: kb = json.load(f)
        print(f"Loaded KB: {len(kb)} facts")
    else:
        kb = []; print("No existing KB — starting fresh.")

    if new_data_path and os.path.exists(new_data_path):
        with open(new_data_path) as f: new_data = json.load(f)
        fact_map = {f["fact"]: f for f in kb}
        added = 0
        for entry in new_data:
            c = _clean(entry.get("fact", ""))
            if not c: continue
            new_fact = {
                "fact": c, "source": entry.get("source", "External"),
                "topic": _clean(entry.get("topic", c[:50]), 100),
                "url": entry.get("url", "N/A"),
                "embedding": _embed(c),
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "confidence_score": entry.get("confidence", 0.70),
            }
            if c not in fact_map: added += 1
            fact_map[c] = new_fact
        kb = list(fact_map.values())
        print(f"Ingested: {added} new facts added.")

    corpus = [f["fact"] for f in kb if f.get("fact")]
    if corpus:
        tfidf_vectorizer.fit(corpus)
        print("TF-IDF index re-fitted.")

    with open(kb_path, "w") as f:
        json.dump(kb, f, indent=2)
    print(f"KB saved → {kb_path}  ({len(kb)} facts)")


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or refresh the unified fact KB.")
    parser.add_argument("--refresh", action="store_true",
                        help="Refresh an existing KB with data/new_facts.json")
    args = parser.parse_args()

    os.makedirs(config.DATA_DIR, exist_ok=True)

    if args.refresh:
        refresh_knowledge_base(new_data_path=os.path.join(config.DATA_DIR, "new_facts.json"))
    else:
        build_initial_unified_kb()
