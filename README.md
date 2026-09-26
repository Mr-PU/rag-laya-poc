# RAG Showdown: OpenAI-only vs Laya-reranked RAG (POC)

A small proof-of-concept comparing two RAG pipelines over the same sample corpus and the same
OpenAI generation step, isolating **one variable**: whether retrieved chunks are reranked by
[Laya](https://github.com/NandhaKishorM/laya) — a fast, local, non-autoregressive decision model —
before being sent to the LLM.

Laya doesn't generate text answers; it makes fast typed yes/no ("noul") decisions with calibrated
probabilities. Here it's used to judge, for every candidate chunk, "does this passage directly
answer the question?" — filtering and reordering what the LLM sees. Both pipelines still use
OpenAI for the final answer, so the comparison is fair: same retrieval, same generator, only the
reranking step differs.

## Architecture

```
                 ┌─────────────┐
   query ───────▶│   backend   │──(embed + vector search, shared)──▶ candidate chunks
                 │  (FastAPI)  │
                 └──────┬──────┘
                         │
          ┌──────────────┴───────────────┐
          ▼                               ▼
   Path A: naive RAG              Path B: Laya-reranked RAG
   top-K chunks, as retrieved     candidates scored concurrently by Laya
          │                       (local Docker container, noul relevance)
          │                               │
          ▼                               ▼
   OpenAI chat completion          OpenAI chat completion
          │                               │
          ▼                               ▼
      answer A                        answer B
```

Three containers:
- **laya** — runs `laya-serve` (the official Laya HTTP server) locally on CPU, fully offline after
  the first model download.
- **backend** — FastAPI app: loads the sample corpus into an in-memory vector store (OpenAI
  embeddings + cosine similarity — no external vector DB needed for a POC this size), runs both
  pipelines, and exposes `/api/query` and `/api/evaluate`.
- **frontend** — a single static page (nginx) with a side-by-side comparison view, a batch-eval
  dashboard, and a corpus browser.

## Requirements

- Docker and Docker Compose
- An OpenAI API key (used for embeddings + generation on both paths)
- ~2 GB free disk for the Laya model download on first run (cached in a Docker volume afterward)

## Running it

```bash
cp .env.example .env
# edit .env and paste your OPENAI_API_KEY

docker compose up --build
```

First startup downloads the Laya English checkpoint (a few hundred MB) — this only happens once,
it's cached in the `laya-model-cache` volume. Give the `laya` container a minute or two on first run.

Once it's up:
- Frontend: **http://localhost:8092**
- Backend API: **http://localhost:8090** (docs at `/docs`)
- Laya service directly: **http://localhost:8091** (`POST /v1/systemone`)

If you're running this on a remote machine or a different port mapping, update the "Backend URL"
field in the top-right of the frontend page.

## Using it

**Compare tab** — ask a question about the sample corpus (a fictional company's HR, expense,
security, onboarding and product docs). You'll see both answers side by side, a timing breakdown
per stage (embed / retrieve / rerank / generate), and — on the Laya side — every candidate chunk
with its calibrated relevance score and which ones were kept vs. filtered out.

**Batch Eval tab** — runs a fixed set of 10 questions (each with a known source document and an
expected keyword) through both pipelines and reports:
- **Retrieval accuracy** — did the expected document make it into the context sent to the LLM?
- **Answer keyword accuracy** — does the final answer contain the expected fact? (a crude but
  transparent proxy, not a substitute for human or LLM-graded eval)
- **Average latency** per pipeline

**Corpus tab** — lists the indexed sample documents and chunk counts.

## Tuning

Environment variables (set in `.env`, read by the backend):

| Variable | Default | Meaning |
|---|---|---|
| `TOP_K_RETRIEVE` | 6 | candidates pulled from the vector store before reranking |
| `TOP_N_CONTEXT` | 3 | chunks actually sent to the LLM after reranking |
| `LAYA_RELEVANCE_THRESHOLD` | 0.35 | Laya `noul` probability below which a chunk is discarded |
| `OPENAI_CHAT_MODEL` | gpt-4o-mini | generation model |
| `OPENAI_EMBED_MODEL` | text-embedding-3-small | embedding model |

## Using your own documents

Drop `.txt` files into `backend/data/docs/` (one paragraph per retrieval chunk, first paragraph
treated as the title) and rebuild the backend: `docker compose up --build backend`. If you replace
the corpus, also update `backend/app/eval_data.py` with matching questions/expected docs/keywords
for the eval tab to stay meaningful.

## Notes on what this does and doesn't measure

- Retrieval accuracy and keyword accuracy here are proxies computed against a small hand-written
  eval set — good for illustrating the effect of reranking on this corpus, not a general benchmark.
- Both pipelines call OpenAI for generation, so OpenAI API latency/cost is common to both; the
  difference you'll see in the eval dashboard is attributable to the reranking step.
- Laya's reranking calls run concurrently (one HTTP call per candidate chunk, fired in parallel),
  which is part of the point — see the "rerank" segment of the timing breakdown.
