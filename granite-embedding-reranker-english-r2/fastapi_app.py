import os
from functools import lru_cache
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from infer import DEFAULT_MODEL_PATH, RerankInferencer


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


class RankRequest(BaseModel):
    query: str = Field(..., min_length=1)
    documents: List[str] = Field(..., min_length=1)


class RankResult(BaseModel):
    document: str
    score: float


class RankResponse(BaseModel):
    results: List[RankResult]


@lru_cache(maxsize=1)
def get_inferencer() -> RerankInferencer:
    return RerankInferencer(os.getenv("MODEL_PATH") or os.getenv("MODEL_ID") or DEFAULT_MODEL_PATH, os.getenv("DEVICE", "auto"), int(os.getenv("BATCH_SIZE", "8")))


app = FastAPI(title="granite-embedding-reranker-english-r2 API", version="1.0.0", description="FastAPI service for query-document reranking.")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    inferencer = get_inferencer()
    return HealthResponse(status="ok", model=inferencer.model_name_or_path, device=os.getenv("DEVICE", "auto"))


@app.post("/rank", response_model=RankResponse)
def rank(request: RankRequest) -> RankResponse:
    return RankResponse(results=get_inferencer().rank(request.query, request.documents))
