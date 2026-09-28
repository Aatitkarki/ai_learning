from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

StoreName = Literal["sqlite", "pgvector", "chroma", "qdrant", "faiss"]
Method = Literal["bm25", "vector", "hybrid", "mmr"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ChunkConfig(StrictModel):
    strategy: Literal["fixed", "sentence", "markdown", "langchain"] = "markdown"
    size: int = Field(default=900, ge=100, le=4000)
    overlap: int = Field(default=120, ge=0, le=1500)

    @model_validator(mode="after")
    def valid_overlap(self):
        if self.overlap >= self.size:
            raise ValueError("Overlap must be smaller than chunk size.")
        return self


class IndexConfig(StrictModel):
    name: str = Field(min_length=1, max_length=80)
    document_ids: list[str] = Field(min_length=1, max_length=100)
    store: StoreName = "sqlite"
    embedding: Literal["hash", "ollama", "openai_compatible"] = "hash"
    embedding_model: str = Field(default="", max_length=150)
    hash_dimensions: int = Field(default=384, ge=64, le=2048)
    chunking: ChunkConfig = Field(default_factory=ChunkConfig)
    vector_index: Literal["native", "exact", "hnsw"] = "native"

    @model_validator(mode="after")
    def validate_config(self):
        if len(set(self.document_ids)) != len(self.document_ids):
            raise ValueError("Document IDs must be unique.")
        if self.vector_index != "native" and self.store not in {"pgvector", "faiss"}:
            raise ValueError("Explicit index selection is available for pgvector and FAISS.")
        return self


class SearchConfig(StrictModel):
    index_id: str
    method: Method = "hybrid"
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int = Field(default=30, ge=1, le=200)
    vector_weight: float = Field(default=0.5, ge=0, le=1)
    mmr_lambda: float = Field(default=0.65, ge=0, le=1)
    min_vector_score: float = Field(default=0.15, ge=0, le=1)
    min_bm25_score: float = Field(default=0.1, ge=0)
    rerank: Literal["none", "lexical"] = "none"
    expand_query: bool = False
    document_ids: list[str] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def candidates_cover_results(self):
        if self.candidate_k < self.top_k:
            raise ValueError("Candidate count must be at least top k.")
        return self


class SearchRequest(StrictModel):
    question: str = Field(min_length=1, max_length=2000)
    config: SearchConfig


class CompareRequest(StrictModel):
    question: str = Field(min_length=1, max_length=2000)
    configs: list[SearchConfig] = Field(min_length=2, max_length=4)


class ChatRequest(SearchRequest):
    mode: Literal["evidence", "ollama", "openai_compatible"] = "evidence"
    conversation_id: str | None = None
    use_followup_context: bool = True
    temperature: float = Field(default=0, ge=0, le=1)
    max_tokens: int = Field(default=1000, ge=64, le=4096)
    context_chars: int = Field(default=14000, ge=1000, le=32000)


class EvaluationRequest(StrictModel):
    configs: list[SearchConfig] = Field(min_length=1, max_length=8)
    split: Literal["dev", "holdout", "all"] = "dev"


class ReviewRequest(StrictModel):
    rating: Literal["supported", "unsupported", "partial", "not_reviewed"]
    notes: str = Field(default="", max_length=4000)


class Page(BaseModel):
    number: int
    text: str


class Chunk(BaseModel):
    id: str
    document_id: str
    filename: str
    page: int
    ordinal: int
    start: int
    end: int
    text: str
    heading: str = ""


class Hit(BaseModel):
    chunk: Chunk
    score: float
    scores: dict[str, float] = Field(default_factory=dict)


class SearchResult(BaseModel):
    query: str
    expanded_query: str
    config: SearchConfig
    hits: list[Hit]
    timings_ms: dict[str, float]
    warnings: list[str]
    corpus_fingerprint: str
