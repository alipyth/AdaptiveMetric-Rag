from __future__ import annotations

import math
import os
import httpx

from . import database
from .models import AppSettings
from .retrieval import DIMENSION, embed as local_embed, query_variants


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [float(value) / norm for value in vector]


def embedding_info(settings: AppSettings) -> dict:
    if settings.embedding_provider == "ollama":
        return {
            "provider": "ollama",
            "model": settings.embedding_model,
            "display_name": settings.embedding_model,
            "dimensions": None,
            "requires_api_key": False,
            "base_url": settings.embedding_base_url or os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
        }
    return {
        "provider": "local",
        "model": "multilingual-feature-hashing-v1",
        "display_name": "هش ویژگی چندزبانه محلی",
        "dimensions": DIMENSION,
        "requires_api_key": False,
        "base_url": "",
    }


async def create_embeddings(settings: AppSettings, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if settings.embedding_provider == "local":
        return [local_embed(text) for text in texts]

    base_url = (settings.embedding_base_url or os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")).rstrip("/")
    vectors: list[list[float]] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=10)) as client:
        for start in range(0, len(texts), 32):
            batch = texts[start:start + 32]
            response = await client.post(f"{base_url}/api/embed", json={"model": settings.embedding_model, "input": batch})
            if response.status_code == 404:
                # Compatibility with older Ollama releases.
                for text in batch:
                    legacy = await client.post(f"{base_url}/api/embeddings", json={"model": settings.embedding_model, "prompt": text})
                    legacy.raise_for_status()
                    vectors.append(_normalize(legacy.json()["embedding"]))
                continue
            response.raise_for_status()
            payload = response.json()
            batch_vectors = payload.get("embeddings") or ([payload["embedding"]] if payload.get("embedding") else [])
            if len(batch_vectors) != len(batch):
                raise ValueError("Ollama returned an unexpected number of embeddings")
            vectors.extend(_normalize(vector) for vector in batch_vectors)
    return vectors


async def create_query_embedding(settings: AppSettings, query: str) -> list[float]:
    """Blend the original query with multilingual keyword-expanded variants."""
    variants = query_variants(query)
    vectors = await create_embeddings(settings, variants)
    if len(vectors) == 1:
        return vectors[0]
    weights = [.55] + [(.45 / (len(vectors) - 1))] * (len(vectors) - 1)
    blended = [sum(weight * vector[index] for weight, vector in zip(weights, vectors)) for index in range(len(vectors[0]))]
    return _normalize(blended)


def document_embedding_text(name: str, section: str | None, content: str) -> str:
    parts = [f"Document title: {name}"]
    if section:
        parts.append(f"Section: {section}")
    parts.append(f"Content: {content}")
    return "\n".join(parts)


async def reindex_all(settings: AppSettings) -> dict:
    chunks = database.rows(
        "SELECT c.id,c.content,c.section,d.name document_name FROM chunks c "
        "JOIN documents d ON d.id=c.document_id ORDER BY c.document_id,c.position"
    )
    if not chunks:
        probe = await create_embeddings(settings, ["embedding readiness check"])
        return {"chunks": 0, "dimensions": len(probe[0]) if probe else None}
    vectors = await create_embeddings(
        settings,
        [document_embedding_text(chunk["document_name"], chunk.get("section"), chunk["content"]) for chunk in chunks],
    )
    with database.connect() as db:
        db.executemany(
            "UPDATE chunks SET embedding=? WHERE id=?",
            [(database.json_value(vector), chunk["id"]) for chunk, vector in zip(chunks, vectors)],
        )
    return {"chunks": len(chunks), "dimensions": len(vectors[0]) if vectors else None}
