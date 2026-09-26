"""Loads sample documents from disk and splits them into retrieval chunks."""
import os
import re
import uuid
from dataclasses import dataclass, field

from . import config


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_title: str
    text: str


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts


def load_corpus() -> list[Chunk]:
    chunks: list[Chunk] = []
    if not os.path.isdir(config.DOCS_DIR):
        return chunks

    for fname in sorted(os.listdir(config.DOCS_DIR)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(config.DOCS_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        paragraphs = _split_paragraphs(raw)
        if not paragraphs:
            continue

        title = paragraphs[0].split("—")[-1].strip() if "—" in paragraphs[0] else paragraphs[0]
        doc_id = fname.replace(".txt", "")

        # Skip the title paragraph itself, chunk the rest by paragraph.
        body_paragraphs = paragraphs[1:] if len(paragraphs) > 1 else paragraphs
        for para in body_paragraphs:
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4())[:8],
                    doc_id=doc_id,
                    doc_title=title,
                    text=para,
                )
            )
    return chunks
