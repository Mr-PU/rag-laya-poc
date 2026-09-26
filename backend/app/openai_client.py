"""Thin wrappers around the OpenAI API for embeddings and chat generation."""
import time

import numpy as np
from openai import AsyncOpenAI

from . import config

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file or environment."
            )
        _client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    return _client


async def embed_texts(texts: list[str]) -> np.ndarray:
    client = get_client()
    resp = await client.embeddings.create(model=config.OPENAI_EMBED_MODEL, input=texts)
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype=np.float32)


async def embed_query(query: str) -> np.ndarray:
    vecs = await embed_texts([query])
    return vecs[0]


async def generate_answer(query: str, context_chunks: list[str]) -> tuple[str, dict]:
    """Generate a final answer from OpenAI given retrieved context. Returns (answer, usage)."""
    client = get_client()

    if context_chunks:
        context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))
    else:
        context_block = "(no context retrieved)"

    system_prompt = (
        "You are a helpful internal assistant answering questions using only the provided "
        "context excerpts. Cite the excerpt number(s) you used, like [1]. If the context does "
        "not contain the answer, say so clearly instead of guessing."
    )
    user_prompt = f"Context:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"

    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model=config.OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    answer = resp.choices[0].message.content or ""
    usage = {
        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
        "total_tokens": resp.usage.total_tokens if resp.usage else None,
        "generation_ms": round(elapsed_ms, 1),
    }
    return answer, usage
