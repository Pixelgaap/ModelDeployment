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


class InferenceRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)
    normalize: bool = True


class InferenceResponse(BaseModel):
    embeddings: List[List[float]]


@lru_cache(maxsize=1)
def get_inferencer() -> EmbeddingInferencer:
    return EmbeddingInferencer(os.getenv("MODEL_PATH") or os.getenv("MODEL_ID") or DEFAULT_MODEL_PATH, os.getenv("DEVICE", "auto"), int(os.getenv("BATCH_SIZE", "8")))


app = FastAPI(title='lt-wikidata-comp-en API', version="1.0.0", description="FastAPI service for text embeddings.")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    inferencer = get_inferencer()
    return HealthResponse(status="ok", model=inferencer.model_name_or_path, device=os.getenv("DEVICE", "auto"))


@app.post("/predict", response_model=InferenceResponse)
def encode(request: InferenceRequest) -> InferenceResponse:
    return InferenceResponse(embeddings=get_inferencer().encode(request.texts, request.normalize))

if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the FastAPI model service.")
    parser.add_argument("--host", default="0.0.0.0", help="Service host.")
    parser.add_argument("--port", type=int, default=8080, help="Service port.")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload.")
    args = parser.parse_args()
    uvicorn.run("fastapi_app:app", host=args.host, port=args.port, reload=args.reload)

