import os
from functools import lru_cache
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from infer import DEFAULT_MODEL_PATH, EmbeddingInferencer


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


class EncodeRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)
    normalize: bool = True


class EncodeResponse(BaseModel):
    embeddings: List[List[float]]


@lru_cache(maxsize=1)
def get_inferencer() -> EmbeddingInferencer:
    return EmbeddingInferencer(
        os.getenv("MODEL_PATH") or os.getenv("MODEL_ID") or DEFAULT_MODEL_PATH,
        os.getenv("DEVICE", "auto"),
        int(os.getenv("BATCH_SIZE", "8")),
    )


app = FastAPI(title="SecureBERT2_0-biencoder API", version="1.0.0", description="FastAPI service for text embeddings.")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    inferencer = get_inferencer()
    return HealthResponse(status="ok", model=inferencer.model_name_or_path, device=os.getenv("DEVICE", "auto"))


@app.post("/encode", response_model=EncodeResponse)
def encode(request: EncodeRequest) -> EncodeResponse:
    return EncodeResponse(embeddings=get_inferencer().encode(request.texts, request.normalize))
