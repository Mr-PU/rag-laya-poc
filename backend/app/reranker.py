"""Reranks candidate chunks using Laya (local, dockerized) as a typed `noul`
(yes/no with calibrated probability) relevance judge, fired concurrently
across all candidates rather than serially."""
import asyncio
import time

import httpx

from . import config
from .corpus import Chunk


async def _score_one(client: httpx.AsyncClient, query: str, chunk: Chunk) -> dict:
    payload = {
        "state": {"query": query, "passage": chunk.text},
        "questions": {
            "relevant": {
                "type": "noul",
                "instructions": (
                    f'Does this passage directly and specifically answer the question: "{query}"? '
                    "Answer true only if the passage contains the actual information needed, "
                    "not just overlapping topic or keywords."
                ),
            }
        },
    }
    t0 = time.perf_counter()
    try:
        resp = await client.post(f"{config.LAYA_URL}/v1/systemone", json=payload)
        resp.raise_for_status()
        data = resp.json()
        prob = float(data["answers"]["relevant"]["noul"])
        error = None
    except Exception as e:  # noqa: BLE001 - surface any failure into the eval UI instead of crashing
        prob = 0.0
        error = str(e)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {"chunk": chunk, "relevance": prob, "call_ms": round(elapsed_ms, 1), "error": error}


async def rerank(query: str, candidates: list[Chunk]) -> tuple[list[dict], float]:
    """Scores every candidate concurrently against Laya, returns them sorted by
    relevance (desc). Returns (scored_results, wall_clock_ms) — wall clock reflects
    the concurrency win: N calls in parallel, not N calls back-to-back."""
    if not candidates:
        return [], 0.0

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=config.LAYA_TIMEOUT_S) as client:
        results = await asyncio.gather(
            *[_score_one(client, query, c) for c in candidates]
        )
    wall_ms = (time.perf_counter() - t0) * 1000

    results_sorted = sorted(results, key=lambda r: r["relevance"], reverse=True)
    return results_sorted, round(wall_ms, 1)
