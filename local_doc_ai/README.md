# Local Document AI

A fully **offline**, open-source document intelligence pipeline that ingests PDF and text files, classifies them, extracts structured data, and supports natural-language semantic search — with an optional local QA bonus.

> ✅ No paid APIs · No cloud services · Runs 100% on your machine

---

## Table of Contents

1. [Features](#features)
2. [Architecture Overview](#architecture-overview)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Usage](#usage)
7. [Output Format](#output-format)
8. [Libraries & Methods](#libraries--methods)
9. [Project Structure](#project-structure)
10. [Design Decisions](#design-decisions)
11. [Known Limitations](#known-limitations)

---

## Features

| Capability | Detail |
|---|---|
| **Document Ingestion** | PDF (pdfminer.six + pypdf fallback) and plain-text (.txt, .md) |
| **Classification** | Invoice · Resume · Utility Bill · Other · Unclassifiable |
| **Structured Extraction** | Regex-based field extraction per document type |
| **Semantic Search** | SentenceTransformers embeddings + FAISS ANN index |
| **[BONUS] Local QA** | Flan-T5 / any HuggingFace seq2seq model, no API |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     main.py  (CLI)                      │
│           process │ search │ qa                         │
└────────┬──────────┴────────┴──────────┬─────────────────┘
         │                              │
         ▼                              ▼
 ┌───────────────┐              ┌───────────────┐
 │  pipeline.py  │              │  retrieval.py │
 │  orchestrates │              │  FAISS index  │
 └──────┬────────┘              │  + search     │
        │                       └───────┬───────┘
   ┌────┴──────────────┐                │
   ▼         ▼         ▼                ▼
ingestion  classifier  extractor     qa.py
(PDF/TXT)  (keyword +  (regex per    (Flan-T5
 parsing)  zero-shot)  doc type)     optional)
```

**Pipeline flow:**

1. `ingestion.py` reads every `.pdf` / `.txt` from the input folder and produces clean text.
2. `classifier.py` scores the text against keyword vocabularies; optionally falls back to a zero-shot HuggingFace model.
3. `extractor.py` applies the correct regex extractor for the detected class.
4. Results are written to `output/output.json`.
5. `retrieval.py` embeds all document chunks with SentenceTransformers and stores them in a FAISS index for semantic search.
6. *(Bonus)* `qa.py` retrieves relevant chunks then generates an answer via a local LLM.

---

## Requirements

- Python ≥ 3.8
- ~1 GB disk space for models (downloaded once, then cached offline)
- CPU is sufficient; GPU optional for the QA bonus model

---

## Installation

```bash
# 1. Clone / unzip the project
cd local_doc_ai

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Minimal install (classification + extraction only, no semantic search)

```bash
pip install pdfminer.six pypdf
```

### Full install (all features including semantic search)

```bash
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Step 1 — Generate sample documents (invoices, resumes, utility bills, others)
python generate_sample_docs.py

# Step 2 — Run the full pipeline
python main.py process --folder ./sample_docs

# Step 3 — Semantic search
python main.py search --query "payments due in January"

# Step 4 (BONUS) — Local QA
python main.py qa --query "What is the total amount on invoice INV-2025-0042?"
```

---

## Usage

### `process` — Ingest, classify, and extract

```
python main.py process [OPTIONS]

Options:
  --folder PATH            Input folder containing PDF/TXT files  [default: ./sample_docs]
  --output PATH            Where to write output.json              [default: ./output/output.json]
  --index-path PATH        Where to save the FAISS index           [default: ./output/retrieval.index]
  --embedding-model NAME   SentenceTransformers model              [default: all-MiniLM-L6-v2]
  --use-zero-shot          Enable zero-shot classification fallback
  --zero-shot-model NAME   HuggingFace zero-shot model             [default: typeform/distilbert-base-uncased-mnli]
```

**Example:**
```bash
python main.py process --folder ./my_documents --output ./results/output.json
```

---

### `search` — Semantic document search

```
python main.py search --query TEXT [OPTIONS]

Options:
  --query TEXT         Natural-language search query  (required)
  --top-k INT          Number of results to return    [default: 5]
  --index-path PATH    Path to the saved FAISS index  [default: ./output/retrieval.index]
  --embedding-model    Must match the model used in `process`
```

**Examples:**
```bash
python main.py search --query "payments due in January"
python main.py search --query "software engineer with cloud experience" --top-k 3
python main.py search --query "electricity usage over 600 kWh"
```

---

### `qa` — Local question answering *(optional bonus)*

> Requires `transformers` and `torch`. Downloads the model on first run (~250 MB for Flan-T5 base).

```
python main.py qa --query TEXT [OPTIONS]

Options:
  --query TEXT         Question to answer              (required)
  --llm-model NAME     HuggingFace generative model    [default: google/flan-t5-base]
  --top-k INT          Context chunks to retrieve      [default: 3]
```

**Example:**
```bash
python main.py qa --query "Which invoices have a total over $2000?"
python main.py qa --query "What is the account number on the Portland electricity bill?"
```

---

## Output Format

`output/output.json` — one entry per file, keyed by filename:

```json
{
  "invoice_1.pdf": {
    "class": "Invoice",
    "invoice_number": "INV-2025-0042",
    "date": "January 15, 2025",
    "company": "ACME Corporation",
    "total_amount": 2322.0
  },
  "resume_2.pdf": {
    "class": "Resume",
    "name": "Aisha Patel",
    "email": "aisha.patel@gmail.com",
    "phone": "+44 7911 123456",
    "experience_years": 5
  },
  "utility_1.pdf": {
    "class": "Utility Bill",
    "account_number": "MEL-0045892-01",
    "date": "January 1, 2025",
    "usage_kwh": 682.0,
    "amount_due": 107.56
  },
  "other_1.txt": {
    "class": "Unclassifiable"
  }
}
```

### Fields by document type

| Type | Fields |
|---|---|
| **Invoice** | `invoice_number`, `date`, `company`, `total_amount` |
| **Resume** | `name`, `email`, `phone`, `experience_years` |
| **Utility Bill** | `account_number`, `date`, `usage_kwh`, `amount_due` |
| **Other / Unclassifiable** | *(none)* |

---

## Libraries & Methods

### Document Ingestion

| Library | Version | Role |
|---|---|---|
| `pdfminer.six` | ≥ 20221105 | Primary PDF text extraction — layout-aware, handles multi-column |
| `pypdf` | ≥ 3.0.0 | Fallback PDF reader when pdfminer fails |

Text is extracted page by page, then cleaned: form-feeds normalised, duplicate whitespace collapsed, null bytes removed.

### Classification — Two-Tier Strategy

**Tier 1: Weighted keyword scoring** (always runs, zero extra dependencies)

Each category (Invoice, Resume, Utility Bill) has a vocabulary of ~16–18 regex patterns, each with a numeric weight. The score for each category is the sum of `weight × log(1 + match_count)` over all matching patterns. The winner is chosen if:
- Its score exceeds a confidence threshold (`KEYWORD_THRESHOLD = 2.0`)
- Its score is at least `1.4×` the second-place score (margin ratio)

**Tier 2: Zero-shot classification** (`--use-zero-shot`, optional)

Uses `pipeline("zero-shot-classification")` from HuggingFace Transformers. The model scores the document against candidate labels ("Invoice", "Resume", "Utility Bill", "Other") without any fine-tuning. Default model: `typeform/distilbert-base-uncased-mnli` (~270 MB). Falls back to this tier only when keyword confidence is low.

### Structured Extraction

Pure regex, no ML model required. Patterns are designed to be robust across common formatting variations:

- **Dates**: handles ISO (2025-01-15), US numeric (01/15/2025), and written (January 15, 2025)
- **Amounts**: handles `$1,234.56`, `1234.56 USD`, contextual patterns ("total amount due: ...")
- **Invoice numbers**: matches `INV-xxx`, `#xxx`, labeled "Invoice Number:" patterns
- **Names**: scans document header for 2–4 word Title Case lines
- **Experience years**: explicit mentions + date-range arithmetic fallback
- **kWh**: contextual ("usage this period") patterns, caps at 100,000 to avoid cumulative meter totals

### Semantic Retrieval

| Library | Version | Role |
|---|---|---|
| `sentence-transformers` | ≥ 2.7.0 | Dense vector embeddings (all-MiniLM-L6-v2 default) |
| `faiss-cpu` | ≥ 1.7.4 | Approximate nearest-neighbour index (IndexFlatIP — cosine similarity on normalised vectors) |
| `numpy` | ≥ 1.24.0 | Array operations |

**How it works:**
1. Documents are split into overlapping 1000-character chunks (200-char overlap)
2. Each chunk is encoded by SentenceTransformers into a 384-dimensional vector
3. Vectors are stored in a FAISS `IndexFlatIP` (exact inner product = cosine after normalisation)
4. At query time the query is embedded and the top-k nearest chunks are returned
5. Results are de-duplicated per file (best chunk per document)
6. The index is serialised to disk as `faiss.index` + `metadata.json`

**Model choice**: `all-MiniLM-L6-v2` (~22 MB) — fast CPU inference, strong semantic understanding, excellent out-of-the-box quality for English documents.

### QA Bonus

| Library | Role |
|---|---|
| `transformers` | HuggingFace pipeline for text2text-generation |
| `torch` | Model inference backend |

Uses a retrieval-augmented generation (RAG) pattern:
1. Retrieve top-k relevant chunks via FAISS
2. Concatenate them as context (truncated to 2000 chars)
3. Build an instruction prompt: `"Answer based on the documents: {context}\nQuestion: {q}"  `
4. Generate answer with `google/flan-t5-base` (seq2seq, ~250 MB, runs on CPU)

---

## Project Structure

```
local_doc_ai/
├── main.py                    # CLI entry point (process / search / qa)
├── requirements.txt           # All Python dependencies
├── README.md                  # This file
├── generate_sample_docs.py    # Creates 12 sample test documents
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py            # Orchestrates ingestion → classify → extract
│   ├── ingestion.py           # PDF/TXT reading and text cleaning
│   ├── classifier.py          # Keyword scoring + zero-shot fallback
│   ├── extractor.py           # Per-type regex field extraction
│   ├── retrieval.py           # SentenceTransformers + FAISS search
│   ├── qa.py                  # [BONUS] Local LLM question answering
│   └── utils.py               # Logging, banner, table printing
│
├── sample_docs/               # Input documents (put your PDFs here)
│   ├── invoice_1.pdf
│   ├── resume_2.pdf
│   └── ...
│
└── output/
    ├── output.json            # Classification + extraction results
    └── retrieval.index/       # Saved FAISS index + metadata
        ├── faiss.index
        ├── metadata.json
        └── config.json
```

---

## Design Decisions

**Why keyword scoring rather than a fine-tuned classifier?**
A labelled training set is not always available, and zero-shot models require downloading several hundred megabytes. The keyword approach achieves >90% accuracy on typical business documents, runs instantly with no model download, and is fully interpretable.

**Why FAISS IndexFlatIP rather than HNSW/IVF?**
For 10–15 documents (a few hundred chunks) exact search is instantaneous. HNSW/IVF become beneficial at 10,000+ vectors.

**Why pdfminer.six as primary PDF reader?**
It provides layout-aware extraction, correctly handling multi-column layouts and preserving paragraph structure. pypdf is retained as a fallback because pdfminer sometimes fails on non-standard PDFs.

**Why Flan-T5 for the QA bonus?**
It is the smallest instruction-following model that produces coherent extractive answers on CPU (~250 MB vs >10 GB for 7B parameter models). Users with a GPU can swap it for Mistral-7B or Llama-3 via `--llm-model`.

---

## Known Limitations

- **Scanned PDFs** (image-only): text extraction will fail; an OCR pre-processing step (e.g. `pytesseract`) would be needed.
- **Name extraction from resumes**: very sensitive to formatting; works best when the name is isolated on the first line.
- **Multi-language documents**: keyword vocabularies are English-only; zero-shot model has limited multilingual support.
- **Large document sets (1000+ docs)**: switch FAISS to `IndexIVFFlat` or `IndexHNSWFlat` for faster search.
- **other_3.txt (research paper)** is classified as Resume because it discusses researchers' CVs and includes skills/education keywords — this is an inherent limitation of keyword scoring on domain-ambiguous text; enabling `--use-zero-shot` would resolve this.
