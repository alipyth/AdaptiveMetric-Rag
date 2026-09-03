from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


ProviderName = Literal["local", "ollama", "openai", "gemini"]
EmbeddingProviderName = Literal["local", "ollama"]


class AppSettings(BaseModel):
    provider: ProviderName = "local"
    model: str = "local-extractive"
    base_url: str = ""
    api_key: str = ""
    temperature: float = Field(0.15, ge=0, le=2)
    max_tokens: int = Field(900, ge=128, le=8192)
    candidate_count: int = Field(100, ge=10, le=500)
    context_count: int = Field(5, ge=1, le=15)
    chunk_size: int = Field(900, ge=250, le=3000)
    chunk_overlap: int = Field(140, ge=0, le=800)
    enable_early_exit: bool = True
    confidence_threshold: float = Field(0.58, ge=0, le=1)
    enable_query_expansion: bool = True
    system_prompt: str = "Answer only from the supplied sources. Cite claims using [1], [2], etc. If the sources are insufficient, say so clearly."
    embedding_provider: EmbeddingProviderName = "local"
    embedding_model: str = "multilingual-feature-hashing-v1"
    embedding_base_url: str = ""

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, value: int, info: Any) -> int:
        size = info.data.get("chunk_size", 900)
        if value >= size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value


class SettingsView(AppSettings):
    api_key: str = ""
    has_api_key: bool = False


class ChatRequest(BaseModel):
    message: str = Field(min_length=2, max_length=8000)
    conversation_id: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    id: int
    document_id: str
    document_name: str
    chunk_id: str
    page: int | None = None
    section: str | None = None
    excerpt: str
    highlight: str = ""
    score: float


class QueryAnalysis(BaseModel):
    intent: str
    language: str
    weights: dict[str, float]
    entities: list[str]
    numbers: list[str]
    temporal_terms: list[str]
    keywords: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    citations: list[Citation]
    analysis: QueryAnalysis
    confidence: float
    latency_ms: int
    provider: str
    early_exit: bool = False
    evidence_found: bool = True
