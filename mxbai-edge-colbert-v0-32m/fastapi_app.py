import os
from functools import lru_cache
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from infer import ColBERTInferencer, DEFAULT_MODEL_PATH


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
def get_inferencer() -> ColBERTInferencer:
    return ColBERTInferencer(
        model_name_or_path=os.getenv("MODEL_PATH") or os.getenv("MODEL_ID") or DEFAULT_MODEL_PATH,
        device=os.getenv("DEVICE", "auto"),
        batch_size=int(os.getenv("BATCH_SIZE", "8")),
    )


app = FastAPI(
    title="mxbai-edge-colbert-v0-32m API",
    version="1.0.0",
    description="FastAPI service for ColBERT-style query-document ranking.",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    inferencer = get_inferencer()
    return HealthResponse(status="ok", model=inferencer.model_name_or_path, device=os.getenv("DEVICE", "auto"))


@app.post("/rank", response_model=RankResponse)
def rank(request: RankRequest) -> RankResponse:
    inferencer = get_inferencer()
    results = inferencer.score(request.query, request.documents)
    return RankResponse(results=results)
