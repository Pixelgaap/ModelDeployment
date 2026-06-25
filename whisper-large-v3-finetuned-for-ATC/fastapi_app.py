import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from infer import DEFAULT_MODEL_PATH, PipelineInferencer


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


class InferenceRequest(BaseModel):
    input: str = Field(..., min_length=1)


class InferenceResponse(BaseModel):
    result: Any


@lru_cache(maxsize=1)
def get_inferencer() -> PipelineInferencer:
    return PipelineInferencer(os.getenv("MODEL_PATH") or os.getenv("MODEL_ID") or DEFAULT_MODEL_PATH, os.getenv("DEVICE", "auto"), int(os.getenv("BATCH_SIZE", "8")))


app = FastAPI(title="whisper-large-v3-finetuned-for-ATC API", version="1.0.0", description="FastAPI service for automatic speech recognition.")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    inferencer = get_inferencer()
    return HealthResponse(status="ok", model=inferencer.model_name_or_path, device=os.getenv("DEVICE", "auto"))


@app.post("/transcribe", response_model=InferenceResponse)
def infer(request: InferenceRequest) -> InferenceResponse:
    return InferenceResponse(result=get_inferencer().run(request.input))
