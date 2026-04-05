# -*- coding: utf-8 -*-
"""
pipeline.py — Full multi-tier RAG pipeline for anti-rumour generation.

Covers Sections 3.7 (Augmentation & Generation) and 3.8 (XAI).

Tiered fallback order (Table 3):
    Tier 1 → Internal KB
    Tier 2 → Google Fact Check Tools API
    Tier 3 → Live Web Search

Usage:
    python pipeline.py --rumor "COVID vaccines contain microchips."
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import pipeline as hf_pipeline

import config
import retrieval

# Local TF-IDF vectorizer — fitted on the KB corpus inside load_kb()
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=config.TFIDF_MAX_FEATURES)

# ── Model initialisation ────────────────────────────────────────────────────

print("Loading models …")
embedding_model = SentenceTransformer(config.RAG_EMBEDDING_MODEL)
EMBEDDING_DIM   = embedding_model.get_sentence_embedding_dimension()

tokenizer  = T5Tokenizer.from_pretrained(config.GENERATION_MODEL)
gen_model  = T5ForConditionalGeneration.from_pretrained(config.GENERATION_MODEL)
summarizer = hf_pipeline(
    "summarization", model=config.SUMMARIZER_MODEL,
    tokenizer=config.SUMMARIZER_MODEL,
    device=0 if torch.cuda.is_available() else -1,
)
sentiment_analyzer = SentimentIntensityAnalyzer()
print("Models loaded.")


# ── Load KB and configure retrieval ─────────────────────────────────────────

def load_kb(kb_path: str = config.UNIFIED_KB_PATH) -> List[Dict]:
    if not os.path.exists(kb_path):
        print(f"KB not found at {kb_path}. Run kb_builder.py first.")
        return []
    with open(kb_path) as f:
        kb = json.load(f)
    print(f"KB loaded: {len(kb)} facts")

    # Re-fit TF-IDF on KB corpus
    corpus = [f["fact"] for f in kb if f.get("fact")]
    if corpus:
        tfidf_vectorizer.fit(corpus)
    retrieval.configure(kb, tfidf_vectorizer, embedding_model)
    return kb


# ── Helper functions ─────────────────────────────────────────────────────────

def _get_emb(text: str) -> np.ndarray:
    if not text.strip():
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    return embedding_model.encode(text, convert_to_numpy=True)


# Public alias used by evaluate.py and inference.py
get_embedding = _get_emb


def _sanitize(text: str) -> str:
    """Heuristically removes emotionally charged phrases from the query."""
    for phrase in ["genius", "apparently", "of course", "unlike", "ridiculous"]:
        text = text.replace(phrase, "")
    return text.strip()


def _summarize_facts(facts: List[Dict], max_length: int = config.SUMMARIZER_MAX_LENGTH) -> str:
    """Abstractive summarisation of retrieved facts (Section 3.7)."""
    if not facts:
        return ""
    full_text = " ".join(f.get("fact", "") for f in facts if f.get("fact"))
    if not full_text.strip():
        return ""
    try:
        if len(full_text) > 1000:
            full_text = full_text[:1000] + "…"
        summary = summarizer(full_text, max_length=max_length, min_length=20, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        print(f"Summarisation warning: {e}")
        return full_text


def _build_prompt(
    query: str,
    facts: List[Dict],
    sentiment_label: str,
    source_type: str = "internal_kb",
) -> str:
    """Constructs the constrained RAG prompt with negative constraints (Section 3.7)."""
    if not facts:
        return (
            f"Rumor: {query}\n\n"
            "No relevant facts found.\n"
            "Anti-Rumor Response: Insufficient factual information to debunk this claim."
        )

    summarized = _summarize_facts(facts)
    if not summarized.strip():
        summarized = " ".join(f"[{f['source']}] {f['fact']}" for f in facts)

    tone = (
        "Maintain a strictly neutral, objective, and factual tone."
        if sentiment_label in ("Positive", "Negative")
        else "Maintain an objective and factual tone."
    )

    source_note = ""
    if source_type == "fact_check_api":
        source_note = "Facts retrieved from Google Fact Check Tools API. "
    elif source_type == "live_search":
        source_note = "Facts retrieved via live web search (not pre-vetted). "

    return (
        "You are an expert fact-checker. Provide a concise, factual debunking "
        "of the following rumour based *strictly* on the provided facts. "
        "Do NOT invent or infer any details not present in the facts. "
        f"{source_note}"
        f"{tone} "
        "If facts are insufficient, state: "
        "'Insufficient factual information to debunk this claim.' "
        "Respond with only the debunking statement.\n\n"
        f"Rumor: {query}\n\n"
        f"Summarized Facts:\n{summarized}\n\n"
        "Anti-rumor:"
    )


def _post_process(response: str, query: str, facts: List[Dict]) -> str:
    """Strips prompt artefacts from the T5 output."""
    for pattern in [
        r".*Anti-rumor:\s*", r".*Anti-Rumor Response:\s*",
        r".*Debunking:\s*",  r".*Response:\s*",
    ]:
        response = re.sub(pattern, "", response, flags=re.I).strip()

    for s in [
        "Based strictly on the following facts",
        "Your response must be ONLY the debunking statement",
        "DO NOT include the rumor text",
        "Maintain a strictly neutral",
        "Maintain an objective",
        "Rumor:", "Facts:", "Summarized Facts:", "Anti-rumor:",
        query,
    ]:
        response = response.replace(s, "").strip()

    for fact in facts:
        if "source" in fact:
            response = response.replace(f"[{fact['source']}]", "").strip()

    response = response.strip('\'".,;:-_ ')
    return " ".join(response.split()) or "Unable to generate a concise debunking."


# ── XAI component ───────────────────────────────────────────────────────────

def build_xai_explanation(
    response: str,
    facts: List[Dict],
    source_type: str,
) -> Tuple[List[Dict], str]:
    """
    Identifies the facts most semantically similar to the generated response
    and returns an auditable evidence trail (Section 3.8).
    """
    if not facts or not response.strip():
        return [], ""

    resp_emb = _get_emb(response)
    sims = []
    for fact in facts:
        fact_emb = _get_emb(fact["fact"])
        sim = float(cosine_similarity(resp_emb.reshape(1,-1), fact_emb.reshape(1,-1))[0][0])
        sims.append((fact, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    contributing = [f for f, s in sims if s > config.XAI_SIMILARITY_THRESHOLD][: config.XAI_MAX_CONTRIBUTING]

    if contributing:
        explanation = (
            "The debunking is primarily based on the following key facts:\n"
            + "\n".join(
                f"  - [{f['source']}] \"{f['fact']}\"  (URL: {f['url']})"
                for f in contributing
            )
            + "\nThese facts directly contradict or contextualise the rumour."
        )
    else:
        explanation = "The debunking synthesises the provided facts; no single fact dominates."

    if source_type == "fact_check_api":
        explanation += "\n\n[Source: Google Fact Check Tools API]"
    elif source_type == "live_search":
        explanation += "\n\n[Source: Live web search — results not pre-vetted by the internal KB]"

    return contributing, explanation


# ── Main generation function ─────────────────────────────────────────────────

def generate_anti_rumor(
    query: str,
    kb: List[Dict] = None,
) -> Tuple[str, List[Dict], List[Dict], str]:
    """
    Full multi-tier RAG pipeline: retrieves facts → summarises → generates
    anti-rumour → produces XAI explanation.

    Args:
        query: Raw rumour text.
        kb:    Knowledge base (loaded via load_kb(); pass None to use global).

    Returns:
        Tuple of:
          - generated_anti_rumor  (str)
          - relevant_facts        (List[Dict])
          - xai_contributing_facts (List[Dict])
          - xai_reasoning         (str)
    """
    query = _sanitize(query)
    if not query:
        return "Empty query.", [], [], ""

    # Sentiment of original rumour
    scores = sentiment_analyzer.polarity_scores(query)
    compound = scores["compound"]
    sentiment = "Positive" if compound > 0.05 else "Negative" if compound < -0.05 else "Neutral"
    print(f"Rumour sentiment: {sentiment} ({compound:.4f})")

    query_emb = _get_emb(query)
    relevant_facts: List[Dict] = []
    source_used = "internal_kb"

    # ── Tier 1: Internal KB ──
    print("Tier 1: searching internal KB …")
    candidates = retrieval.retrieve_facts_hybrid(query, query_emb)
    for cand in candidates:
        emb = _get_emb(cand["fact"])
        sim = float(cosine_similarity(query_emb.reshape(1,-1), emb.reshape(1,-1))[0][0])
        if sim >= config.CONTEXT_RELEVANCE_THRESHOLD:
            relevant_facts.append(cand)
        if len(relevant_facts) >= config.FACTS_FOR_GENERATION:
            break

    if relevant_facts:
        print(f"  Tier 1: {len(relevant_facts)} relevant facts found.")

    # ── Tier 2: Fact Check API ──
    if len(relevant_facts) < config.FACTS_FOR_GENERATION:
        print("Tier 2: querying Google Fact Check API …")
        api_facts = retrieval.fetch_fact_check_claims(query)
        for claim in api_facts:
            if claim.get("confidence_score", 0) < config.MIN_CONFIDENCE_EXTERNAL:
                continue
            emb = _get_emb(claim["fact"])
            claim["embedding"] = emb.tolist()
            sim = float(cosine_similarity(query_emb.reshape(1,-1), emb.reshape(1,-1))[0][0])
            if sim >= config.CONTEXT_RELEVANCE_THRESHOLD:
                relevant_facts.append(claim)
        if len(relevant_facts) > config.FACTS_FOR_GENERATION:
            relevant_facts = relevant_facts[:config.FACTS_FOR_GENERATION]
            source_used = "fact_check_api"
            print(f"  Tier 2: augmented to {len(relevant_facts)} facts.")

    # ── Tier 3: Live Web Search ──
    if len(relevant_facts) < config.FACTS_FOR_GENERATION:
        print("Tier 3: live web search …")
        snippets = retrieval.perform_web_search(query)
        for snip in snippets:
            emb = _get_emb(snip["fact"])
            snip["embedding"] = emb.tolist()
            sim = float(cosine_similarity(query_emb.reshape(1,-1), emb.reshape(1,-1))[0][0])
            if sim >= config.CONTEXT_RELEVANCE_THRESHOLD:
                relevant_facts.append(snip)
            # Also extract page-level facts
            page_facts = retrieval.extract_facts_from_web_content(snip["url"], query_emb)
            for pf in page_facts:
                emb = _get_emb(pf["fact"])
                pf["embedding"] = emb.tolist()
                sim2 = float(cosine_similarity(query_emb.reshape(1,-1), emb.reshape(1,-1))[0][0])
                if sim2 >= config.CONTEXT_RELEVANCE_THRESHOLD:
                    relevant_facts.append(pf)
        relevant_facts = relevant_facts[:config.FACTS_FOR_GENERATION]
        source_used = "live_search"
        print(f"  Tier 3: augmented to {len(relevant_facts)} facts.")

    # ── Context relevance check ──
    if not relevant_facts:
        return "Insufficient factual information to debunk this rumour.", [], [], ""
    ctx_text = " ".join(f.get("fact", "") for f in relevant_facts)
    ctx_emb  = _get_emb(ctx_text)
    ctx_sim  = float(cosine_similarity(query_emb.reshape(1,-1), ctx_emb.reshape(1,-1))[0][0])
    if ctx_sim < config.CONTEXT_RELEVANCE_THRESHOLD:
        return "Insufficient factual information to provide a relevant debunking.", relevant_facts, [], ""

    # ── T5 Generation ──
    prompt = _build_prompt(query, relevant_facts, sentiment, source_used)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = gen_model.generate(
            inputs.input_ids,
            max_length=config.MAX_GEN_LENGTH,
            num_beams=config.NUM_BEAMS,
            early_stopping=True,
        )
    raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = _post_process(raw_response, query, relevant_facts)

    # ── XAI ──
    contributing, explanation = build_xai_explanation(response, relevant_facts, source_used)

    return response, relevant_facts, contributing, explanation


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an anti-rumour for a given claim.")
    parser.add_argument("--rumor", type=str, default=None, help="Rumour text to debunk.")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode.")
    args = parser.parse_args()

    kb = load_kb()

    def _run(rumor_text: str):
        anti_rumor, facts, xai_facts, explanation = generate_anti_rumor(rumor_text, kb)
        print(f"\n{'='*60}")
        print(f"  RUMOUR      : {rumor_text}")
        print(f"  ANTI-RUMOUR : {anti_rumor}")
        if facts:
            print(f"  SOURCE USED : {facts[0].get('source','N/A')}  ({facts[0].get('url','N/A')})")
        print(f"  XAI         : {explanation[:200]}…" if len(explanation) > 200 else f"  XAI         : {explanation}")
        print("="*60)

    if args.rumor:
        _run(args.rumor)
    elif args.interactive:
        print("Interactive mode. Type 'exit' to quit.\n")
        while True:
            rumor = input("Enter a rumour: ").strip()
            if rumor.lower() == "exit":
                break
            _run(rumor)
    else:
        parser.print_help()
