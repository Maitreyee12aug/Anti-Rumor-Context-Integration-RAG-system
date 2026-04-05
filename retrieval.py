# -*- coding: utf-8 -*-
"""
retrieval.py — Hybrid retrieval, semantic re-ranking, and tiered fallback.

Covers Section 3.6 and Table 3 of the paper.

Hybrid score:
    S_hybrid = α · S_semantic + (1 − α) · S_keyword    (paper: α = 0.6)

Tiered Fallback:
    Tier 1 → Internal KB (fastest)
    Tier 2 → Google Fact Check Tools API
    Tier 3 → Live Web Search
"""

import re
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity

import config

# ── Globals (injected by pipeline.py) ──────────────────────────────────────
_kb = []
_tfidf = None
_embedding_model = None
_EMBEDDING_DIM = 768

PREFERRED_DOMAINS = [
    "snopes.com", "factcheck.org", "politifact.com",
    "reuters.com", "apnews.com", "bbc.com", "who.int",
]
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
)


def configure(kb, tfidf_vectorizer, embedding_model) -> None:
    """Inject shared state from the pipeline before any retrieval calls."""
    global _kb, _tfidf, _embedding_model, _EMBEDDING_DIM
    _kb = kb
    _tfidf = tfidf_vectorizer
    _embedding_model = embedding_model
    _EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()


def _tfidf_vec(text: str) -> np.ndarray:
    if _tfidf is None or not hasattr(_tfidf, "vocabulary_") or not text.strip():
        return np.zeros(0)
    return _tfidf.transform([text]).toarray().flatten()


def _embed(text: str) -> np.ndarray:
    if _embedding_model is None or not text.strip():
        return np.zeros(_EMBEDDING_DIM, dtype=np.float32)
    return _embedding_model.encode(text, convert_to_numpy=True)


# ── Tier 1: Internal KB ─────────────────────────────────────────────────────

def retrieve_facts_hybrid(
    query: str,
    query_embedding: np.ndarray,
    top_k: int   = config.TOP_K_INITIAL,
    alpha: float = config.HYBRID_ALPHA,
    min_sem: float = config.MIN_SEMANTIC_SIMILARITY,
    min_kw: float  = config.MIN_KEYWORD_SIMILARITY,
) -> List[Dict[str, Any]]:
    """
    Retrieves facts from the internal KB using hybrid semantic + keyword
    scoring, then re-ranks by semantic cosine similarity.

    Args:
        query:           Raw query string (used for TF-IDF).
        query_embedding: SBERT embedding of the query.
        top_k:           Number of facts to return after re-ranking.
        alpha:           Semantic weight (paper: 0.6).
        min_sem:         Minimum semantic cosine similarity threshold.
        min_kw:          Minimum TF-IDF dot-product threshold.

    Returns:
        Re-ranked list of fact dicts from the internal KB.
    """
    if not _kb:
        return []

    q_tfidf = _tfidf_vec(query)
    scored: List[Tuple[Dict, float]] = []

    for fact in _kb:
        text     = fact.get("fact", "")
        emb_list = fact.get("embedding", [])
        if not emb_list or not text.strip():
            continue
        try:
            fact_emb = np.array(emb_list, dtype=np.float32).reshape(1, -1)
        except ValueError:
            continue

        sem = float(cosine_similarity(query_embedding.reshape(1, -1), fact_emb)[0][0])

        kw = 0.0
        if q_tfidf.size > 0:
            f_tfidf = _tfidf_vec(text)
            if f_tfidf.size > 0:
                kw = float(np.dot(q_tfidf, f_tfidf))

        if sem >= min_sem or kw >= min_kw:
            hybrid = alpha * sem + (1 - alpha) * kw
            scored.append((fact, hybrid))

    scored.sort(key=lambda x: x[1], reverse=True)
    candidates = [f for f, _ in scored[:top_k]]

    # Semantic re-ranking
    re_ranked: List[Tuple[Dict, float]] = []
    for fact in candidates:
        emb_list = fact.get("embedding", [])
        if not emb_list: continue
        try:
            fact_emb = np.array(emb_list, dtype=np.float32).reshape(1, -1)
            score = float(cosine_similarity(query_embedding.reshape(1, -1), fact_emb)[0][0])
            re_ranked.append((fact, score))
        except ValueError:
            continue

    re_ranked.sort(key=lambda x: x[1], reverse=True)
    return [f for f, _ in re_ranked[:top_k]]


# ── Tier 2: Google Fact Check Tools API ─────────────────────────────────────

def fetch_fact_check_claims(
    query: str,
    api_key: str = config.FACT_CHECK_API_KEY,
) -> List[Dict[str, Any]]:
    """
    Queries the Google Fact Check Tools API for verified claims.
    Tier 2 of the tiered fallback (Table 3).

    Args:
        query:   Rumour search string.
        api_key: Google Fact Check API key (set via FACT_CHECK_API_KEY env var).

    Returns:
        List of fact dicts with confidence scores mapped from textual ratings.
    """
    if not api_key:
        print("Warning: FACT_CHECK_API_KEY not set. Skipping Tier 2.")
        return []

    params = {
        "query": query,
        "key": api_key,
        "maxAgeDays": config.FACT_CHECK_MAX_AGE_DAYS,
    }
    try:
        res = requests.get(config.FACT_CHECK_API_URL, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        print(f"Fact Check API error: {e}"); return []

    facts = []
    for claim_item in data.get("claims", []):
        claim_text = claim_item.get("text", "")
        for review in claim_item.get("claimReview", []):
            publisher = review.get("publisher", {}).get("name", "Unknown")
            url = review.get("url", "")
            rating = review.get("textualRating", "").lower()
            # Map rating string → confidence score
            if any(k in rating for k in ["false", "pants on fire", "incorrect", "misleading"]):
                conf = 0.85
            elif any(k in rating for k in ["true", "correct", "accurate"]):
                conf = 0.90
            elif any(k in rating for k in ["mixed", "half", "mostly"]):
                conf = 0.65
            else:
                conf = 0.50

            fact_text = f"{claim_text} [{publisher} rating: {review.get('textualRating','N/A')}]"
            facts.append({
                "fact": fact_text[:300], "source": publisher,
                "url": url, "confidence_score": conf, "embedding": [],
            })
    return facts


# ── Tier 3: Live Web Search ──────────────────────────────────────────────────

def perform_web_search(
    query: str,
    num_results: int = config.LIVE_SEARCH_MAX_URLS,
) -> List[Dict[str, Any]]:
    """
    Searches the web for fact-checking content (Tier 3 fallback).
    Prioritises known fact-checking domains.
    """
    headers = {"User-Agent": USER_AGENT}
    search_url = (
        f"https://www.google.com/search"
        f"?q={requests.utils.quote(query + ' fact check')}&num={num_results * 3}"
    )
    results = []
    try:
        res = requests.get(search_url, headers=headers, timeout=10, verify=False)
        soup = BeautifulSoup(res.text, "html.parser")
        for g in soup.select("div.g")[: num_results * 3]:
            a_tag   = g.find("a", href=True)
            snippet = g.find("span") or g.find("div", class_="VwiC3b")
            if not (a_tag and snippet): continue
            href = a_tag["href"]
            domain = re.search(r"https?://([^/]+)", href)
            domain_str = domain.group(1) if domain else ""
            preferred = any(d in domain_str for d in PREFERRED_DOMAINS)
            results.append({
                "fact": snippet.get_text(strip=True)[:300],
                "source": domain_str, "url": href,
                "confidence_score": 0.75 if preferred else 0.55,
                "embedding": [],
            })
        results.sort(key=lambda x: x["confidence_score"], reverse=True)
    except Exception as e:
        print(f"Web search error: {e}")
    return results[:num_results]


def extract_facts_from_web_content(
    url: str,
    query_embedding: np.ndarray,
    num_facts: int = config.LIVE_SEARCH_FACTS_PER_URL,
) -> List[Dict[str, Any]]:
    """
    Fetches a page and returns the top-N sentences by cosine similarity
    to the query embedding.
    """
    headers = {"User-Agent": USER_AGENT}
    facts = []
    try:
        res = requests.get(url, headers=headers, timeout=10, verify=False)
        soup = BeautifulSoup(res.text, "html.parser")
        sentences = []
        for p in soup.find_all("p"):
            for sent in re.split(r"(?<=[.!?]) +", p.get_text(strip=True)):
                if len(sent) >= 60:
                    sentences.append(sent)

        ranked = []
        for sent in sentences:
            emb = _embed(sent)
            sim = float(cosine_similarity(query_embedding.reshape(1,-1), emb.reshape(1,-1))[0][0])
            ranked.append((sent, sim))
        ranked.sort(key=lambda x: x[1], reverse=True)

        domain = re.search(r"https?://([^/]+)", url)
        domain_str = domain.group(1) if domain else url
        for sent, _ in ranked[:num_facts]:
            facts.append({
                "fact": sent[:300], "source": domain_str,
                "url": url, "confidence_score": 0.60, "embedding": [],
            })
    except Exception as e:
        print(f"Content extraction failed ({url}): {e}")
    return facts
