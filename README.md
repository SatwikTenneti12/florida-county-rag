---
title: Florida County RAG
emoji: 🐊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Florida Policy Navigator

Florida Policy Navigator is a citation-aware agentic RAG web application for exploring Florida county comprehensive plans. It lets users ask policy questions, run semantic search, and review AI-assisted topic analysis across county planning documents.

The current application uses a FastAPI backend, a React/Vite frontend, ChromaDB for local vector search, and UF AI Gateway-compatible embedding and chat-completion APIs.

## Current Features

- User signup and login with JWT authentication.
- Optional Google reCAPTCHA verification for production.
- County-aware RAG answers with cited plan excerpts.
- Semantic search over county comprehensive plan chunks.
- Topic analysis for five conservation policy themes.
- Input guardrails for unsafe or irrelevant requests.
- Email verification support through SMTP configuration.
- Evaluation scripts for LLM-as-judge validation and AI-human policy label agreement.

## Architecture

| Layer | Current implementation |
| --- | --- |
| Backend | FastAPI in `api.py` |
| Frontend | React + Vite in `frontend/` |
| Auth storage | Local SQLite database through `src/auth/database.py` |
| PDF extraction | PyMuPDF / `fitz`, using page-level text extraction |
| Chunking | Sentence-aware chunks from `src/ingestion/smart_chunker.py` |
| Chunk size | 4000 characters |
| Chunk overlap | 800 characters |
| Vector database | ChromaDB persistent local store in `chroma_db/` |
| Chroma collection | `county_chunks` |
| Embedding model | `nomic-embed-text-v1.5` |
| Embedding dimensions | 768 |
| LLM | `llama-3.1-8b-instruct` by default |
| Default retrieval | Top-k 8 for Ask/Search; top-k 3 for Topic Analysis |
| Answer temperature | 0.1 for answer generation; 0.0 for evaluation/scoring |

## Repository Layout

```text
florida-county-rag/
├── api.py                         # FastAPI backend and API routes
├── Dockerfile                     # Backend deployment container
├── requirements.txt               # Python dependencies
├── .env.example                   # Backend environment template
├── frontend/                      # React/Vite web app
│   ├── .env.example               # Frontend environment template
│   ├── package.json
│   └── src/
├── src/
│   ├── auth/                      # User database helpers
│   ├── classification/            # Topic scoring and guardrails
│   ├── evaluation/                # Validation and agreement scripts
│   ├── indexing/                  # Chroma index builder
│   ├── ingestion/                 # PDF extraction and chunking
│   ├── rag/                       # Retriever and answer engine
│   ├── retrieval/                 # Search smoke-test scripts
│   └── utils/                     # LLM, embedding, email, county helpers
├── data/
│   ├── benchmarks/                # Benchmark question files
│   ├── manifests/                 # Human label reference data
│   └── processed/                 # Small reproducible outputs in git
└── chroma_db/                     # Local vector DB, not committed
```

## What Is In GitHub

The GitHub repository contains the current application code, frontend, backend, setup templates, evaluation scripts, and small research/evaluation outputs.

The following are intentionally not committed:

- `.env` and `frontend/.env`: contain API keys and local settings.
- `data/auth.db`: local user database.
- `frontend/node_modules/`: frontend dependency install folder.
- `frontend/dist/`: generated frontend build.
- `chroma_db/`: generated vector database.
- `data/raw_pdfs/`: large source PDF corpus.
- Large generated JSONL intermediates unless specifically unignored.

The complete private handoff zip includes these local artifacts. Use the zip for full transfer to the project owner. Use GitHub for clean source control and deployment setup.

## Why ChromaDB And Raw PDFs Are Not In GitHub

`chroma_db/` and `data/raw_pdfs/` are large binary/data artifacts. They make normal Git history heavy, slow to clone, and difficult to maintain. The better handoff pattern is:

1. Keep code and small reproducible outputs in GitHub.
2. Share the full data/index bundle privately through the complete handoff zip.
3. Rebuild `chroma_db/` from `data/raw_pdfs/` when needed.

If the hosting team wants every artifact in GitHub, use Git LFS or a GitHub Release asset instead of normal git tracking.

## Backend Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with real values:

```text
EMBEDDINGS_API_KEY=
EMBEDDINGS_BASE_URL=https://api.ai.it.ufl.edu/v1
EMBEDDINGS_MODEL=nomic-embed-text-v1.5

LLM_API_KEY=
LLM_BASE_URL=https://api.ai.it.ufl.edu/v1
LLM_MODEL=llama-3.1-8b-instruct

JWT_SECRET=
CORS_ORIGINS=http://localhost:5173

RECAPTCHA_SECRET_KEY=
DISABLE_RECAPTCHA=false

SMTP_HOST=
SMTP_PORT=587
SMTP_USERNAME=
SMTP_PASSWORD=
SMTP_FROM_EMAIL=
SMTP_FROM_NAME=Florida Policy Navigator
SMTP_USE_TLS=true
```

For local testing without CAPTCHA, set:

```text
DISABLE_RECAPTCHA=true
```

Start the backend:

```bash
python api.py
```

Default backend URL:

```text
http://127.0.0.1:8000
```

## Frontend Setup

```bash
cd frontend
npm install
cp .env.example .env
```

Fill `frontend/.env`:

```text
VITE_API_BASE_URL=http://localhost:8000
VITE_RECAPTCHA_SITE_KEY=
VITE_DISABLE_RECAPTCHA=false
```

For local testing without CAPTCHA, set:

```text
VITE_DISABLE_RECAPTCHA=true
```

Start the frontend:

```bash
npm run dev
```

Default frontend URL:

```text
http://127.0.0.1:5173
```

## Data And Index Restore

For a full working local copy, restore these folders/files from the private handoff zip:

```text
data/raw_pdfs/
data/processed/pages.jsonl
data/processed/chunks.jsonl
chroma_db/
.env
frontend/.env
```

If the Chroma index is missing, rebuild it:

```bash
python src/ingestion/extract_pages.py
python src/ingestion/smart_chunker.py
python src/indexing/build_chroma.py
```

The app needs `data/processed/chunks.jsonl` and `chroma_db/` for semantic retrieval. Without them, search and RAG answers cannot work fully.

## Main API Workflows

The FastAPI backend exposes routes for:

- Authentication: signup, login, and current-user checks.
- County list retrieval for signup/search filtering.
- RAG question answering.
- Semantic search.
- Topic analysis.
- Feedback capture.

The frontend calls the backend through `frontend/src/api.js`, using `VITE_API_BASE_URL`.

## Evaluation

The project includes two current validation tracks:

1. LLM-as-judge RAG validation over a 50-question benchmark.
   - Output: `data/processed/validation_results.json`
   - Metrics: context relevance, answer relevance, and groundedness.

2. AI-human county-topic agreement over 335 county-topic comparisons.
   - Output: `data/processed/ai_human_policy_agreement.json`
   - Human labels source: `data/County Comprehensive Plans.xlsx`
   - Generated labels: `data/processed/ai_policy_scores_by_county.csv`

Useful scripts:

```bash
python src/evaluation/generate_benchmark.py
python src/evaluation/llm_evaluator.py
python src/evaluation/compare_ai_human_policy_labels.py
python src/classification/rubric_scorer.py
```

## Local Checks

Backend syntax check:

```bash
python -m py_compile api.py src/rag/answer_engine.py src/rag/retriever.py src/utils/embeddings.py src/utils/llm.py
```

Frontend checks:

```bash
cd frontend
npm run lint
npm run build
```

## Deployment Notes

The frontend can be hosted on a static frontend host such as Vercel, Netlify, or another web server that can serve the Vite build.

The backend must be hosted as a Python service because it performs authentication, retrieval, LLM calls, and ChromaDB access. The backend host needs:

- Python dependencies from `requirements.txt`.
- A populated `.env`.
- Access to `data/processed/chunks.jsonl`.
- Access to `chroma_db/`, or enough permissions/API access to rebuild it.
- SMTP settings if email verification is enabled.
- reCAPTCHA settings if CAPTCHA is enabled.

## Important Security Notes

Never commit `.env`, API keys, SMTP passwords, CAPTCHA secrets, JWT secrets, or local auth databases to GitHub.

For production:

- Set a strong `JWT_SECRET`.
- Set `DISABLE_RECAPTCHA=false`.
- Configure real `RECAPTCHA_SECRET_KEY` and `VITE_RECAPTCHA_SITE_KEY`.
- Use a production database instead of a local SQLite file if multiple users or persistence guarantees are required.
- Restrict `CORS_ORIGINS` to the production frontend domain.

## Project Context

This project supports research and demonstration work around Florida county comprehensive plans, especially retrieval and analysis of policies related to wildlife corridors, wildlife crossings, wildlife surveys, land acquisition, and open space planning.
