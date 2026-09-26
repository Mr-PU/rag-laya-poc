"""A minimal in-memory vector store. No external DB needed for this POC:
embeddings are computed once at startup and kept in a numpy matrix."""
import time

import numpy as np

from . import config
from .corpus import Chunk, load_corpus
from .openai_client import embed_texts


class VectorStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None  # (n_chunks, dim), L2-normalized

    async def build(self) -> None:
        self.chunks = load_corpus()
        if not self.chunks:
            self.matrix = np.zeros((0, 0), dtype=np.float32)
            return
        texts = [c.text for c in self.chunks]
        vecs = await embed_texts(texts)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        self.matrix = vecs / norms

    def is_ready(self) -> bool:
        return self.matrix is not None and self.matrix.shape[0] > 0

    def search(self, query_vec: np.ndarray, top_k: int) -> tuple[list[Chunk], list[float], float]:
        """Returns (chunks, similarity_scores, elapsed_ms), highest similarity first."""
        t0 = time.perf_counter()
        if not self.is_ready():
            return [], [], 0.0

        q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        sims = self.matrix @ q  # cosine similarity since both are L2-normalized
        top_k = min(top_k, len(self.chunks))
        idx = np.argsort(-sims)[:top_k]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        result_chunks = [self.chunks[i] for i in idx]
        result_scores = [float(sims[i]) for i in idx]
        return result_chunks, result_scores, round(elapsed_ms, 2)


store = VectorStore()
