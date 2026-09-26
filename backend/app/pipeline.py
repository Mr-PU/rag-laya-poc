"""Orchestrates both RAG paths for a single query:

  Path A (naive)         : vector-similarity top-K -> top-N straight to the LLM
  Path B (laya_reranked)  : same top-K candidates -> Laya reranks/filters -> top-N to the LLM

Retrieval is computed once and shared between both paths so the only variable
being compared is the reranking step itself.
"""
import time

from . import config
from .corpus import Chunk
from .openai_client import embed_query, generate_answer
from .reranker import rerank
from .vectorstore import store


def _chunk_dict(c: Chunk, extra: dict | None = None) -> dict:
    d = {"id": c.id, "doc_id": c.doc_id, "doc_title": c.doc_title, "text": c.text}
    if extra:
        d.update(extra)
    return d


async def _retrieve_candidates(query: str) -> tuple[list[Chunk], list[float], float, float]:
    t0 = time.perf_counter()
    qvec = await embed_query(query)
    embed_ms = round((time.perf_counter() - t0) * 1000, 1)

    chunks, sims, retrieve_ms = store.search(qvec, config.TOP_K_RETRIEVE)
    return chunks, sims, embed_ms, retrieve_ms


async def run_compare(query: str) -> dict:
    t_total0 = time.perf_counter()

    candidates, sims, embed_ms, retrieve_ms = await _retrieve_candidates(query)

    # ---- Path A: naive (vector similarity order only) ----
    naive_context_chunks = candidates[: config.TOP_N_CONTEXT]
    naive_context_texts = [c.text for c in naive_context_chunks]

    t0 = time.perf_counter()
    naive_answer, naive_usage = await generate_answer(query, naive_context_texts)
    naive_gen_ms = round((time.perf_counter() - t0) * 1000, 1)

    naive_total_ms = round(embed_ms + retrieve_ms + naive_gen_ms, 1)

    # ---- Path B: Laya-reranked ----
    reranked, laya_wall_ms = await rerank(query, candidates)
    above_threshold = [r for r in reranked if r["relevance"] >= config.LAYA_RELEVANCE_THRESHOLD]
    chosen = above_threshold if above_threshold else reranked  # never send empty context
    laya_context_items = chosen[: config.TOP_N_CONTEXT]
    laya_context_texts = [item["chunk"].text for item in laya_context_items]

    t0 = time.perf_counter()
    laya_answer, laya_usage = await generate_answer(query, laya_context_texts)
    laya_gen_ms = round((time.perf_counter() - t0) * 1000, 1)

    laya_total_ms = round(embed_ms + retrieve_ms + laya_wall_ms + laya_gen_ms, 1)

    total_ms = round((time.perf_counter() - t_total0) * 1000, 1)

    return {
        "query": query,
        "shared": {
            "embed_ms": embed_ms,
            "retrieve_ms": retrieve_ms,
            "candidates_considered": len(candidates),
        },
        "naive": {
            "label": "Naive RAG (vector similarity only)",
            "answer": naive_answer,
            "context_used": [
                _chunk_dict(c, {"similarity": round(s, 3)})
                for c, s in zip(naive_context_chunks, sims[: config.TOP_N_CONTEXT])
            ],
            "timing_ms": {
                "embed": embed_ms,
                "retrieve": retrieve_ms,
                "generate": naive_gen_ms,
                "total": naive_total_ms,
            },
            "usage": naive_usage,
        },
        "laya_reranked": {
            "label": "Laya-reranked RAG (local, dockerized)",
            "answer": laya_answer,
            "context_used": [
                _chunk_dict(item["chunk"], {"laya_relevance": round(item["relevance"], 3)})
                for item in laya_context_items
            ],
            "all_candidates_scored": [
                {
                    "doc_id": item["chunk"].doc_id,
                    "text_preview": item["chunk"].text[:120],
                    "laya_relevance": round(item["relevance"], 3),
                    "call_ms": item["call_ms"],
                    "kept": item in laya_context_items,
                    "error": item["error"],
                }
                for item in reranked
            ],
            "timing_ms": {
                "embed": embed_ms,
                "retrieve": retrieve_ms,
                "rerank": laya_wall_ms,
                "generate": laya_gen_ms,
                "total": laya_total_ms,
            },
            "usage": laya_usage,
        },
        "total_wall_ms": total_ms,
    }


def _keyword_hit(answer: str, keywords: list[str]) -> bool:
    lower = answer.lower()
    return any(kw.lower() in lower for kw in keywords)


def _retrieval_hit(context_used: list[dict], expected_doc_id: str) -> bool:
    return any(c["doc_id"] == expected_doc_id for c in context_used)


async def run_evaluation(questions: list[dict]) -> dict:
    per_question = []
    for item in questions:
        result = await run_compare(item["question"])

        naive_hit = _retrieval_hit(result["naive"]["context_used"], item["expected_doc_id"])
        laya_hit = _retrieval_hit(result["laya_reranked"]["context_used"], item["expected_doc_id"])
        naive_kw = _keyword_hit(result["naive"]["answer"], item["expected_keywords"])
        laya_kw = _keyword_hit(result["laya_reranked"]["answer"], item["expected_keywords"])

        per_question.append(
            {
                "question": item["question"],
                "expected_doc_id": item["expected_doc_id"],
                "naive": {
                    "answer": result["naive"]["answer"],
                    "retrieval_hit": naive_hit,
                    "keyword_hit": naive_kw,
                    "total_ms": result["naive"]["timing_ms"]["total"],
                },
                "laya_reranked": {
                    "answer": result["laya_reranked"]["answer"],
                    "retrieval_hit": laya_hit,
                    "keyword_hit": laya_kw,
                    "total_ms": result["laya_reranked"]["timing_ms"]["total"],
                },
            }
        )

    n = len(per_question) or 1

    def _agg(path: str) -> dict:
        rh = sum(1 for q in per_question if q[path]["retrieval_hit"]) / n
        kh = sum(1 for q in per_question if q[path]["keyword_hit"]) / n
        avg_ms = sum(q[path]["total_ms"] for q in per_question) / n
        return {
            "retrieval_accuracy": round(rh, 3),
            "answer_keyword_accuracy": round(kh, 3),
            "avg_latency_ms": round(avg_ms, 1),
        }

    return {
        "per_question": per_question,
        "summary": {
            "naive": _agg("naive"),
            "laya_reranked": _agg("laya_reranked"),
            "n_questions": len(per_question),
        },
    }
