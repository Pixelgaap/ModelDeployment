import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from infer import DEFAULT_MODEL_PATH, PipelineInferencer


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


class InferenceRequest(BaseModel):
    text: str = Field(..., min_length=1)


class InferenceResponse(BaseModel):
    result: Any


@lru_cache(maxsize=1)
def get_inferencer() -> PipelineInferencer:
    return PipelineInferencer(os.getenv("MODEL_PATH") or os.getenv("MODEL_ID") or DEFAULT_MODEL_PATH, os.getenv("DEVICE", "auto"), int(os.getenv("BATCH_SIZE", "8")))


app = FastAPI(title='OpenMed-NER-OrganismDetect-ModernClinical-395M API', version="1.0.0", description='FastAPI service for token-classification.')


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    inferencer = get_inferencer()
    return HealthResponse(status="ok", model=inferencer.model_name_or_path, device=os.getenv("DEVICE", "auto"))


@app.post("/predict", response_model=InferenceResponse)
def infer(request: InferenceRequest) -> InferenceResponse:
    return InferenceResponse(result=get_inferencer().run(**request.model_dump(exclude_none=True)))

if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the FastAPI model service.")
    parser.add_argument("--host", default="0.0.0.0", help="Service host.")
    parser.add_argument("--port", type=int, default=8080, help="Service port.")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload.")
    args = parser.parse_args()
    uvicorn.run("fastapi_app:app", host=args.host, port=args.port, reload=args.reload)

