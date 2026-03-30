---
title: Florida County RAG
emoji: 🐊
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.30.0
app_file: app.py
pinned: false
---

# Florida County RAG

Citation-aware **Retrieval-Augmented Generation (RAG)** over Florida county comprehensive plans. PDFs are chunked and embedded into **Chroma**; retrieval feeds an **LLM** for answers with **page-level citations**. Optional pipelines classify five environmental policy topics and score predictions against a labeled dataset.

**Research question:** Do Florida county comprehensive plans include enforceable policies on wildlife corridors, wildlife crossings, wildlife surveys, land acquisition, and open space planning?

---

## Tech stack

| Layer | Choice |
|--------|--------|
| Language | Python 3 |
| Vector DB | Chroma (persistent, local `chroma_db/`) |
| Embeddings | `nomic-embed-text-v1.5` (via configurable HTTP API) |
| LLM | `llama-3.1-8b-instruct` default (configurable; UF AI Gateway–compatible) |
| UI | Streamlit (`app.py`) |
| Eval | Per-topic accuracy / precision / recall / F1 vs `data/manifests/county_labels.csv` |

---

## Repository layout

```
florida-county-rag/
├── app.py                    # Streamlit web UI (RAG Q&A, search, topic scan)
├── requirements.txt
├── .env.example              # Template for API keys (copy → .env)
├── scripts/
│   └── run_pipeline.py       # Optional: chunk → index → classify → evaluate
├── src/
│   ├── config.py             # Paths, models, topic definitions
│   ├── ingestion/            # PDF → pages → chunks
│   ├── indexing/             # Embeddings → Chroma
│   ├── rag/                  # Retriever, answer engine, CLI
│   ├── retrieval/            # query_chroma.py (semantic smoke tests)
│   ├── classification/     # Rule + RAG classifiers, threshold helper, postprocess
│   ├── evaluation/         # metrics.py, label comparison
│   └── utils/                # Embeddings, LLM client, county normalization
├── data/
│   ├── raw_pdfs/             # County PDFs (not in git — see below)
│   ├── processed/            # pages.jsonl, chunks.jsonl, CSVs (jsonl gitignored)
│   └── manifests/
│       └── county_labels.csv # Ground-truth labels for evaluation
├── chroma_db/                # Built index (not in git — regenerate locally)
└── assets/                   # Optional static assets for UI
```

---

## Quick start (for reviewers)

1. **Clone** the repository.

2. **Create a virtual environment** and install:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with valid `EMBEDDINGS_*` and `LLM_*` values. The project targets a **UF-compatible OpenAI-style** `/v1/embeddings` and `/v1/chat/completions` gateway; adjust URLs if your provider differs.

4. **Data & index** (required for full functionality)
   - Place county PDFs under `data/raw_pdfs/` (expected layout is per-county folders; see `src/ingestion/extract_pages.py` / `config` for conventions).
   - Run the pipeline to produce `data/processed/*.jsonl` and `chroma_db/` (see **Pipeline** below).  
   - *Without* `chroma_db/` and chunks, the Streamlit app and retriever will fail at runtime.

5. **Run the UI**
   ```bash
   streamlit run app.py
   ```

6. **CLI RAG** (optional)
   ```bash
   python -m src.rag.cli "Your question here?" --county "Alachua County"
   ```

---

## Pipeline (ingestion → index)

Run in order after PDFs are in place:

| Step | Command | Output |
|------|---------|--------|
| Extract pages | `python src/ingestion/extract_pages.py` | `data/processed/pages.jsonl` |
| Chunk | `python src/ingestion/smart_chunker.py` (sentence-aware; or `chunk_pages.py`) | `data/processed/chunks.jsonl` |
| Embed + Chroma | `python src/indexing/build_chroma.py` | `chroma_db/` |

**One-shot orchestration** (chunk → index → optional classifiers → eval):

```bash
python scripts/run_pipeline.py --all
# Or steps: --chunk --index --classify-baseline --classify --evaluate
```

Semantic search smoke test:

```bash
python src/retrieval/query_chroma.py "wildlife corridor" --k 5
```

---

## Classification & evaluation

| Purpose | Script | Notes |
|---------|--------|--------|
| Keyword baseline | `python src/classification/rule_classifier.py` | Writes `data/processed/topic_evidence_by_county.csv` |
| RAG + LLM classifier | `python src/classification/rag_classifier.py` | Uses retrieval + LLM; configurable `--top-k-chunks` |
| Metrics vs labels | `python src/evaluation/metrics.py data/processed/topic_evidence_by_county.csv` | Optional: `--labels`, `--output report.json` |
| RAG predictions | `python src/evaluation/metrics.py data/processed/rag_topic_evidence_by_county.csv` | Same, if that CSV exists |

Ground-truth format: `data/manifests/county_labels.csv` (per-topic Yes/No by county).

---

## Configuration highlights

- **Models and topics** live in `src/config.py` (`TOPICS`, paths, `CHROMA_DIR`, etc.).
- **Retries** exist on embedding/indexing and LLM calls for transient API errors (see `src/utils/llm.py`, `src/indexing/build_chroma.py`).

---

## What is *not* in this repository

Per `.gitignore`:

- **`.env`** — API keys must never be committed.
- **`chroma_db/`** — Rebuilt locally after ingestion.
- **`data/raw_pdfs/`** — Large PDF corpus.
- **`data/processed/*.jsonl`** — Large generated text artifacts.

**`data/manifests/county_labels.csv`** is intended to stay versioned for reproducible evaluation. Regenerated CSVs under `data/processed/` may be present locally for experiments; add only small samples if you need them in git.

---

## Deployment note

**Streamlit** apps are commonly deployed to **Streamlit Community Cloud** (GitHub + Secrets for env vars). **Vercel** is not a good fit for a long-running Streamlit server. See platform docs for size limits on repos and artifacts (`chroma_db` may need external storage or a build step for cloud).

---

## License

Add a `LICENSE` file appropriate to your institution and data use; this repo does not include one by default.

---

## Contact / context

Internal / academic use: Florida county comprehensive plans, environmental policy retrieval, and optional classification benchmarks against hand-labeled counties.
