# -*- coding: utf-8 -*-
"""
config.py — Centralised hyperparameters and path settings.
All tuneable values live here; edit this file before running any script.

⚠️  API KEY: Set FACT_CHECK_API_KEY via environment variable.
    Never hard-code your key in this file.
    export FACT_CHECK_API_KEY="your_key_here"
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR              = "data"
OUTPUTS_DIR           = "outputs"
CHECKPOINTS_DIR       = "checkpoints"

# Raw scraped KB JSONs (produced by kb_scrapers.py)
WIKIPEDIA_KB_PATH     = os.path.join(DATA_DIR, "auto_fact_base.json")
WHO_KB_PATH           = os.path.join(DATA_DIR, "who_fact_base.json")
FACTCHECK_KB_PATH     = os.path.join(DATA_DIR, "factcheck_kb.json")
POLITIFACT_KB_PATH    = os.path.join(DATA_DIR, "politifact_data.json")

# Unified KB (produced by kb_builder.py)
UNIFIED_KB_PATH       = os.path.join(DATA_DIR, "unified_fact_kb.json")

# Dataset CSV paths (CONSTRAINT 2021 / ISOT)
RAW_DATASET_PATH      = os.path.join(DATA_DIR, "Constraint_Train.csv")
TRAIN_CSV             = "train.csv"
VAL_CSV               = "val.csv"
TEST_CSV              = "test.csv"
TRAIN_PREPROCESSED    = "train_preprocessed.csv"
VAL_PREPROCESSED      = "val_preprocessed.csv"
TEST_PREPROCESSED     = "test_preprocessed.csv"

# Embedding artefacts
TRAIN_TFIDF_PATH      = "train_tfidf.npy"
TRAIN_GLOVE_PATH      = "train_glove.npy"
TRAIN_SBERT_PATH      = "train_sbert.npy"
GLOVE_PATH            = "glove.6B.100d.txt"      # download separately

# BERTopic model artefact
BERTOPIC_MODEL_PATH   = "bertopic_model"

# ── Preprocessing ──────────────────────────────────────────────────────────
TEXT_COLUMN           = "tweet"       # raw text column in dataset CSV
DATASET_TEST_SIZE     = 0.20
DATASET_VAL_SIZE      = 0.10
RANDOM_STATE          = 42

# ── Feature Engineering ────────────────────────────────────────────────────
TFIDF_MAX_FEATURES    = 5000
GLOVE_DIM             = 100
SBERT_MODEL_NAME      = "all-MiniLM-L6-v2"     # lightweight; swap for all-mpnet-base-v2 if GPU

# ── Topic Modeling ─────────────────────────────────────────────────────────
LDA_N_COMPONENTS      = 5
BERTOPIC_MIN_TOPIC    = 10
KMEANS_N_CLUSTERS     = 10

# ── Knowledge Base ─────────────────────────────────────────────────────────
WIKIPEDIA_TOPICS = [
    "COVID-19", "Climate change", "Mail-in voting", "Inflation",
    "Natural disaster", "Misinformation", "Russia-Ukraine war",
    "Syrian civil war", "Barack Obama",
]
WIKIPEDIA_SENTENCES_PER_TOPIC = 3
WHO_SCRAPE_LIMIT              = 100
FACTCHECK_SCRAPE_PAGES        = 100
POLITIFACT_SCRAPE_PAGES       = 100

# ── Retrieval ──────────────────────────────────────────────────────────────
# Embedding model for RAG (heavier than SBERT_MODEL_NAME; used at inference)
RAG_EMBEDDING_MODEL       = "all-mpnet-base-v2"

HYBRID_ALPHA              = 0.6    # weight for semantic similarity vs keyword
TOP_K_INITIAL             = 20     # initial candidate pool size before re-ranking
MIN_SEMANTIC_SIMILARITY   = 0.40
MIN_KEYWORD_SIMILARITY    = 0.05
CONTEXT_RELEVANCE_THRESHOLD = 0.65  # Table 3 — minimum score to pass Tier 1 filter
MIN_CONFIDENCE_EXTERNAL   = 0.60   # minimum confidence for Tier 2/3 facts

# Google Fact Check Tools API
FACT_CHECK_API_KEY        = os.environ.get("FACT_CHECK_API_KEY", "")
FACT_CHECK_API_URL        = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
FACT_CHECK_MAX_AGE_DAYS   = 3650

# ── Augmentation & Generation ──────────────────────────────────────────────
SUMMARIZER_MODEL          = "t5-small"
GENERATION_MODEL          = "t5-small"
SUMMARIZER_MAX_LENGTH     = 150
MAX_GEN_LENGTH            = 50     # T5 max_new_tokens (paper: 50)
NUM_BEAMS                 = 5

# ── XAI ───────────────────────────────────────────────────────────────────
XAI_SIMILARITY_THRESHOLD  = 0.60   # min cosine sim to count as contributing fact
XAI_MAX_CONTRIBUTING      = 3

# ── Evaluation ─────────────────────────────────────────────────────────────
EVAL_SAMPLE_SIZE          = 10_000  # number of test instances (paper: 10K)
HALLUCINATION_THRESHOLD   = 0.50    # A-C similarity below this → flagged
FACTS_FOR_GENERATION      = 3       # facts passed to the generator

# ── Live Web Search (Tier 3) ────────────────────────────────────────────────
LIVE_SEARCH_MAX_URLS        = 2
LIVE_SEARCH_FACTS_PER_URL   = 2
