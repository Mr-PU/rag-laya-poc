import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import config
from .eval_data import EVAL_QUESTIONS
from .pipeline import run_compare, run_evaluation
from .vectorstore import store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-laya-poc")

app = FastAPI(title="RAG: OpenAI-only vs Laya-reranked (POC)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


@app.on_event("startup")
async def startup() -> None:
    logger.info("Building in-memory vector store from %s ...", config.DOCS_DIR)
    await store.build()
    logger.info("Vector store ready with %d chunks.", len(store.chunks))


@app.get("/api/health")
async def health():
    return {
        "status": "ok" if store.is_ready() else "corpus_not_loaded",
        "chunks_indexed": len(store.chunks),
        "laya_url": config.LAYA_URL,
        "chat_model": config.OPENAI_CHAT_MODEL,
        "embed_model": config.OPENAI_EMBED_MODEL,
    }


@app.get("/api/corpus")
async def corpus():
    docs: dict[str, dict] = {}
    for c in store.chunks:
        d = docs.setdefault(c.doc_id, {"doc_id": c.doc_id, "title": c.doc_title, "chunks": 0})
        d["chunks"] += 1
    return {"documents": list(docs.values()), "total_chunks": len(store.chunks)}


@app.get("/api/eval-questions")
async def eval_questions():
    return {"questions": EVAL_QUESTIONS}


@app.post("/api/query")
async def query(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    if not store.is_ready():
        raise HTTPException(status_code=503, detail="corpus not indexed yet")
    try:
        return await run_compare(req.query.strip())
    except RuntimeError as e:
        # e.g. missing OPENAI_API_KEY
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/evaluate")
async def evaluate():
    if not store.is_ready():
        raise HTTPException(status_code=503, detail="corpus not indexed yet")
    try:
        return await run_evaluation(EVAL_QUESTIONS)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
