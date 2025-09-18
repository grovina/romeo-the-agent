import dataclasses
import functools
import json
import re
from typing import Any

import numpy as np

from ..core import config, embedder
from . import base


@dataclasses.dataclass
class RagItem:
    id: int
    text: str
    source: str
    vector: list[float]


def _split_into_chunks(text: str) -> list[str]:
    """
    Extremely simple chunking: split on 1+ blank lines; trim whitespace; drop
    tiny pieces.
    """
    raw_parts = re.split(r"\n\s*\n+", text)
    parts = [p.strip() for p in raw_parts]
    return [p for p in parts if len(p) > 20]


def _build_index() -> None:
    """Build an in-memory list of RagItem objects from local docs."""
    items: list[RagItem] = []

    for path in config.DOCS_DIR.glob("*.txt"):
        text = path.read_text(encoding="utf-8")

        chunks = _split_into_chunks(text)
        vectors = embedder.embed(chunks)

        for chunk, vector in zip(chunks, vectors):
            item = RagItem(
                id=len(items),
                text=chunk,
                source=path.name,
                vector=vector,
            )
            items.append(item)

    config.RAG_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.RAG_INDEX_PATH.write_text(
        json.dumps([dataclasses.asdict(item) for item in items], indent=2)
    )


def _load_index() -> list[RagItem]:
    data = json.loads(config.RAG_INDEX_PATH.read_text(encoding="utf-8"))
    results: list[RagItem] = []
    for item in data:
        result = RagItem(
            id=int(item["id"]),
            text=str(item["text"]),
            source=str(item["source"]),
            vector=[float(x) for x in item["vector"]],
        )
        results.append(result)
    return results


@functools.cache
def _get_index() -> list[RagItem]:
    if not config.RAG_INDEX_PATH.exists():
        print("No RAG index found. Building it...")
        _build_index()

    return _load_index()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def rag_search(query: str, top_k: int = config.RAG_TOP_K) -> dict[str, Any]:
    """
    Search the local course docs and return the most relevant snippets.

    We return a compact dict so the LLM can easily use it to write the answer:
    {
      "snippets": [
        {"text": "...", "source": "syllabus.md"},
        ...
      ]
    }
    """
    index = _get_index()

    [query_vector] = embedder.embed([query])

    scored: list[tuple[float, RagItem]] = []
    for chunk in index:
        score = _cosine_similarity(query_vector, chunk.vector)
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)

    return {
        "snippets": [
            {"text": chunk.text, "source": chunk.source, "score": round(score, 3)}
            for score, chunk in scored[:top_k]
        ]
    }


class RagTool(base.Tool):
    name = "rag"

    def schema(self):  #
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Search local docs and get relevant snippets",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for"}
                    },
                    "required": ["query"],
                },
            },
        }

    def run(self, kwargs):
        query = kwargs["query"]
        print(f"[RAG] Searching for `{query}`...")
        result = rag_search(query)
        print(f"[RAG] Result:\n{json.dumps(result, indent=2)}")
        return result
