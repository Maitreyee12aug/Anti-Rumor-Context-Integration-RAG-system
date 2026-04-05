Anti-Rumor Context Integration: A Novel RAG System for Automated Factual Correction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/Paper-Accepted-green)](.)

---

> ⚠️ Citation Notice: This repository is the official implementation of "Anti-Rumor Context Integration: A Novel RAG System for Automated Factual Correction," accepted in the 7th International Conference on Advances in Distributed Computing and Machine Learning.
> (Maitreyee Ganguly, Paramita Dey, Soumik Pal — Government College of Engineering and Ceramic Technology, Kolkata).
> If you use this code or build upon this work, please cite our paper using the BibTeX entry at the bottom of this README.

---

## Overview

We propose a multi-tier Retrieval-Augmented Generation (RAG) system that goes beyond binary rumour classification to generate fact-grounded, explainable counter-narratives automatically.

Key contributions:

- Tiered Retrieval — Internal KB → Google Fact Check API → Live Web Search (trust–efficiency hierarchy)
- Hybrid Semantic + Keyword Scoring** — α·S_semantic + (1−α)·S_keyword with semantic re-ranking
- Constrained T5 Generation — Negative-prompt constraints to suppress hallucinations
- XAI Module — Auditable evidence trail with source attribution per generated response
- Zero-Shot Generalisation — Evaluated across COVID-19, ISOT, and Russia–Ukraine datasets with no task-specific fine-tuning

### Results (10K test sample)

| Metric | Score |
|---|---|
| Rumour–Anti-Rumour Similarity (R-A) | 0.71 |
| Anti-Rumour–Context Similarity (A-C) | 0.73 |
| Fact Coverage Score (Cov.) | 0.72 |
| Hallucination Rate | 10.05% |

---

## Architecture

```
Raw Rumour Text
       │
       ▼
┌─────────────────────────────────────┐
│  DATA PREPROCESSING                 │
│  Tokenisation → Stopword Removal    │
│  → Lemmatisation                    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  FEATURE ENGINEERING                │
│  TF-IDF  │  GloVe  │  SBERT        │
│  Topic Modelling (LDA + BERTopic)   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐      ┌──────────────────────┐
│  RETRIEVAL COMPONENT                │      │  Knowledge Base      │
│                                     │◄─────│  Wikipedia / WHO     │
│  Query Embedding (SBERT)            │      │  FactCheck.org       │
│  Hybrid Scoring (α=0.6)             │      │  PolitiFact          │
│  Semantic Re-ranking                │      └──────────────────────┘
│                                     │
│  Tiered Fallback:                   │      ┌──────────────────────┐
│    Tier 1 → Internal KB             │      │  Google Fact Check   │
│    Tier 2 → Fact Check API  ────────┼─────►│  Tools API           │
│    Tier 3 → Live Web Search ────────┼─────►│  Live Web            │
└──────────────┬──────────────────────┘      └──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  AUGMENTATION & GENERATION          │
│  Contextual Summarisation (T5)      │
│  Dynamic Prompt Construction        │
│  T5-small Generation (5-beam)       │
│  Post-processing                    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  XAI COMPONENT                      │
│  Contributing Facts (cosine sim)    │
│  Source Attribution + URL           │
│  Reasoning Explanation              │
└─────────────────────────────────────┘
               │
               ▼
     Explainable Anti-Rumour Output
```

---

## Repository Structure

```
anti-rumor-rag/
│
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
│
├── config.py            ← All hyperparameters and path settings
│
├── preprocess.py        ← Section 3.2–3.3: Cleaning, TF-IDF, GloVe, SBERT
├── topic_model.py       ← Section 3.4: LDA, BERTopic, K-Means, sentiment
├── kb_scrapers.py       ← Section 3.5: Wikipedia / WHO / FactCheck / PolitiFact crawlers
├── kb_builder.py        ← Section 3.5: Unified KB construction & dynamic refresh
├── retrieval.py         ← Section 3.6: Hybrid retrieval + tiered fallback
├── pipeline.py          ← Section 3.7–3.8: T5 generation + XAI (main RAG pipeline)
├── evaluate.py          ← Section 4: Batch evaluation & all metrics (Table 4)
├── visualize.py         ← Section 5: All paper figures (Figures 2–7)
├── inference.py         ← Single-sample inference with full XAI report
│
└── data/
    └── sample/          ← Sample rumours for quick testing
```

---

## Requirements

### System Requirements

- Python >= 3.8
- CUDA >= 11.3 recommended; CPU inference also supported
- ~4 GB RAM minimum; 8 GB recommended for KB construction

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Maitreyee12aug/anti-rumor-rag.git
cd anti-rumor-rag

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy language model
python -m spacy download en_core_web_sm

# 5. (Optional) Download GloVe vectors
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d .
```

### Core Dependencies

| Package | Version | Purpose |
|---|---|---|
| PyTorch | >= 1.12.0 | Deep learning framework |
| transformers | >= 4.20.0 | T5 generation + tokenisation |
| sentence-transformers | >= 2.2.0 | SBERT embeddings (all-mpnet-base-v2) |
| bertopic | >= 0.15.0 | BERTopic topic modelling |
| scikit-learn | >= 1.0.0 | TF-IDF, KMeans, cosine similarity |
| nltk | >= 3.7 | Preprocessing, VADER sentiment |
| requests / beautifulsoup4 | latest | KB web scraping |
| wikipedia | 1.4+ | Wikipedia KB construction |

---

## Datasets

Three publicly available datasets were used (zero-shot — no fine-tuning):

| Dataset | Domain | Size |
|---|---|---|
| [CONSTRAINT 2021](https://constraint-shared-task-2021.github.io/) | COVID-19 fake news | ~10K |
| [ISOT](https://www.uvic.ca/ecs/ece/isot/datasets/) | General fake news | ~45K |
| [Russia–Ukraine War](https://huggingface.co/) | Conflict misinformation | ~65K |

Place your dataset CSV at `data/Constraint_Train.csv` (or update `RAW_DATASET_PATH` in `config.py`).
The CSV must contain at a minimum a text column (default: `tweet`) and a `label` column.

---

## Usage

### ⚠️ API Key Setup

The Google Fact Check Tools API key must be set as an **environment variable**. Never hard-code it.

```bash
export FACT_CHECK_API_KEY="your_google_api_key_here"
```

Get a free key at: https://developers.google.com/fact-check/tools/api/reference/rest

---

### Step 1 — Preprocess & Feature Engineering

```bash
python preprocess.py
```

Produces: `train.csv`, `val.csv`, `test.csv`, `train_preprocessed.csv`, `train_tfidf.npy`, `train_sbert.npy`

---

### Step 2 — Topic Modeling & Clustering

```bash
python topic_model.py
```

Produces: `bertopic_model/`, `train_enriched.csv`, cluster word clouds

---

### Step 3 — Build the Knowledge Base

```bash
# 3a. Scrape all KB sources (Wikipedia, WHO, FactCheck, PolitiFact)
python kb_scrapers.py

# 3b. Merge sources and generate SBERT embeddings
python kb_builder.py

# 3c. (Optional) Refresh KB with new facts
python kb_builder.py --refresh
```

Produces: `data/unified_fact_kb.json`

---

### Step 4 — Single Rumour Inference

```bash
# Single rumour
python inference.py --rumor "COVID-19 vaccines contain microchips."

# Interactive mode
python inference.py --interactive

# Quick demo (3 built-in examples)
python inference.py
```

Example output:
```
══════════════════════════════════════════════════════════════
  ANTI-RUMOUR REPORT
══════════════════════════════════════════════════════════════
  RUMOUR          : COVID-19 vaccines contain microchips.
  ANTI-RUMOUR     : COVID-19 vaccines do not contain microchips.
                    They are composed of biological components
                    designed to build immune response, a claim
                    widely debunked by health authorities.
  SOURCE USED     : WHO
  SOURCE URL      : https://www.who.int/news-room/fact-sheets
──────────────────────────────────────────────────────────────
  R-A Similarity  : 0.8214
  A-C Similarity  : 0.8791
  Fact Coverage   : 0.8623
  Hallucination?  : ✓ NO
  Sentiment       : Neutral (compound: 0.0000)
  Facts Retrieved : 3
──────────────────────────────────────────────────────────────
  XAI EXPLANATION:
  The debunking is primarily based on the following key facts:
    - [WHO] "COVID-19 vaccines do not contain any microchips..."
══════════════════════════════════════════════════════════════
```

---

### Step 5 — Batch Evaluation

```bash
python evaluate.py --csv test.csv --sample 10000
```

Produces: `outputs/evaluation_results.json` with per-instance scores and a summary.

---

### Step 6 — Reproduce Paper Figures

```bash
python visualize.py
```

Generates all figures from Section 5 to `outputs/`:

| File | Figure |
|---|---|
| `fig2_score_distributions.png` | Figure 2: Score histograms (R-A, A-C, Coverage) |
| `fig3a_coverage_vs_context.png` | Figure 3a: Coverage vs. context similarity scatter |
| `fig3b_hallucination_pie.png` | Figure 3b: Hallucination rate pie chart |
| `fig4_metrics_by_hallucination.png` | Figure 4: Box plots by hallucination flag |
| `fig5_correlation_matrix.png` | Figure 5: Metric correlation heatmap |
| `fig6a_facts_retrieved.png` | Figure 6a: Facts retrieved per rumour |
| `fig6b_source_distribution.png` | Figure 6b: Source usage distribution |
| `fig7_sentiment_distribution.png` | Figure 7: Anti-rumour sentiment distribution |

---

## Hyperparameters

All hyperparameters are centralised in `config.py`.

| Parameter | Value | Description |
|---|---|---|
| `RAG_EMBEDDING_MODEL` | `all-mpnet-base-v2` | SBERT model for retrieval embeddings |
| `HYBRID_ALPHA` | `0.6` | Semantic vs. keyword weight (α) |
| `TOP_K_INITIAL` | `20` | Candidate pool size before re-ranking |
| `CONTEXT_RELEVANCE_THRESHOLD` | `0.65` | Min A-C score to pass Tier 1 filter |
| `MIN_CONFIDENCE_EXTERNAL` | `0.60` | Min confidence for Tier 2/3 facts |
| `GENERATION_MODEL` | `t5-small` | HuggingFace generation model |
| `MAX_GEN_LENGTH` | `50` | T5 max new tokens |
| `NUM_BEAMS` | `5` | Beam search width |
| `FACTS_FOR_GENERATION` | `3` | Facts passed to the T5 prompt |
| `XAI_SIMILARITY_THRESHOLD` | `0.60` | Min cosine sim to count as contributing |
| `HALLUCINATION_THRESHOLD` | `0.50` | A-C score below this → hallucination flag |

---

## Baseline Comparison

| System | R-A | A-C | Coverage | Hall. Rate |
|---|---|---|---|---|
| Single-Tier RAG (baseline) | 0.57 | 0.60 | 0.52 | 25.0% |
| RARG (2024) | 0.63 | 0.66 | 0.59 | 18–22% |
| Speculative RAG (2024) | 0.65 | 0.68 | 0.63 | 19.0% |
| MADAM-RAG (2025) | 0.68 | 0.71 | 0.67 | 15.0% |
| RAGChecker (2024) | — | — | 0.64 | 14.0% |
| **Proposed System** | **0.71** | **0.73** | **0.72** | **10.05%** |

---

## Citation

If you use this code in your research, please cite our paper.

---

## Contact

For questions or issues, please open a GitHub Issue or contact:

- **Maitreyee Ganguly** — maitreyee12aug@gmail.com
- **Paramita Dey** — dey.paramita77@gmail.com
- **Soumik Pal** — soumik.kms@gmail.com

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Google Fact Check Tools API](https://developers.google.com/fact-check/tools/api/reference/rest)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [Sentence-Transformers](https://www.sbert.net/)
- [BERTopic](https://github.com/MaartenGr/BERTopic)
- [FactCheck.org](https://www.factcheck.org) · [PolitiFact](https://www.politifact.com) · [WHO](https://www.who.int)
```
