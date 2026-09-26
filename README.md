# RAG Showdown: OpenAI-only vs Laya-reranked RAG (POC)

A small proof-of-concept comparing two RAG pipelines over the same sample corpus and the same
OpenAI generation step, isolating **one variable**: whether retrieved chunks are reranked by
[Laya](https://github.com/NandhaKishorM/laya) — a fast, local, non-autoregressive decision model
before being sent to the LLM.

Laya doesn't generate text answers; it makes fast typed yes/no ("noul") decisions with calibrated
probabilities. Here it's used to judge, for every candidate chunk, "does this passage directly
answer the question?" filtering and reordering what the LLM sees. Both pipelines still use
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
- **laya** : runs `laya-serve` (the official Laya HTTP server) locally on CPU, fully offline after
  the first model download.
- **backend** : FastAPI app: loads the sample corpus into an in-memory vector store (OpenAI
  embeddings + cosine similarity, no external vector DB needed for a POC this size), runs both
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

First startup downloads the Laya English checkpoint (a few hundred MB), this only happens once,
it's cached in the `laya-model-cache` volume. Give the `laya` container a minute or two on first run.

Once it's up:
- Frontend: **http://localhost:8092**
- Backend API: **http://localhost:8090** (docs at `/docs`)
- Laya service directly: **http://localhost:8091** (`POST /v1/systemone`)

If you're running this on a remote machine or a different port mapping, update the "Backend URL"
field in the top-right of the frontend page.

## Using it

**Compare tab**, ask a question about the sample corpus (a fictional company's HR, expense,
security, onboarding and product docs). You'll see both answers side by side, a timing breakdown
per stage (embed / retrieve / rerank / generate), and, on the Laya side, every candidate chunk
with its calibrated relevance score and which ones were kept vs. filtered out.

**Batch Eval tab** runs a fixed set of 10 questions (each with a known source document and an
expected keyword) through both pipelines and reports:
- **Retrieval accuracy** did the expected document make it into the context sent to the LLM?
- **Answer keyword accuracy** does the final answer contain the expected fact? (a crude but
  transparent proxy, not a substitute for human or LLM-graded eval)
- **Average latency** per pipeline

**Corpus tab** lists the indexed sample documents and chunk counts.

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
  eval set, good for illustrating the effect of reranking on this corpus, not a general benchmark.
- Both pipelines call OpenAI for generation, so OpenAI API latency/cost is common to both; the
  difference you'll see in the eval dashboard is attributable to the reranking step.
- Laya's reranking calls run concurrently (one HTTP call per candidate chunk, fired in parallel),
  which is part of the point, see the "rerank" segment of the timing breakdown.

## Understanding the Results

Both pipelines gave the same correct answer here, and that's actually a meaningful (if slightly boring) result, let me walk through why.

**Why embed/retrieve are identical on both cards:** by design, the backend computes the query embedding and does vector search once and shares those candidate chunks with both paths (see pipeline.py's `_retrieve_candidates`). That isolates the one variable being tested, reranking, so 1244.7ms / 0.41ms aren't two separate calls, they're the same numbers displayed twice.

**Why rerank only costs 10.7ms:** Laya is running locally in its own container, and all candidate chunks get scored concurrently (one HTTP call per chunk, fired in parallel via asyncio.gather) rather than one after another. That's the concurrency story from the Laya README playing out, a handful of parallel local forward passes adds almost nothing to the total.

**Why generate differs (1686ms vs 2259ms) even though both send similar context:** that's just normal call-to-call latency variance from OpenAI's API, not something reranking caused. Two back-to-back API calls rarely take identical time.

**Why the answers are identical:** for this specific question ("how many days of paid annual leave"), the correct chunk is also the single closest match by raw cosine similarity, so naive top-K already puts it in the context Path A sends to the LLM. Laya reordering the same candidates doesn't change anything when the top-similarity chunk was already the right one. This is expected and honest: reranking's value shows up specifically when semantic similarity picks a near-miss over the actually-relevant passage, e.g. a query that shares vocabulary with the wrong document, or where two chunks are topically close but only one directly answers the question.

To actually see divergence, try:

1. A vaguer/cross-topic question where multiple docs share keywords (e.g. "how long do I have to submit something after an expense" vs. "how long after starting do I need to do something", both involve short time windows, different docs)
2. Or just run the Batch Eval tab, the 10-question set is built to include a couple of these harder cases, so the retrieval-accuracy bar chart should show a real gap there rather than a tie.
