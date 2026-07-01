import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from infer import DEFAULT_MODEL_PATH, PipelineInferencer


def _decode_data_url(value: str) -> bytes:
    import base64

    payload = value.split(",", 1)[1] if "," in value and value.lstrip().startswith("data:") else value
    return base64.b64decode(payload)


def _download_url(url: str) -> bytes:
    import urllib.request

    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def _suffix_from_remote_source(base64_data: str | None, url: str | None, default_suffix: str) -> str:
    import mimetypes
    import os
    import urllib.parse

    if base64_data and base64_data.lstrip().startswith("data:"):
        media_type = base64_data.split(";", 1)[0].split(":", 1)[1]
        return mimetypes.guess_extension(media_type) or default_suffix
    if url:
        suffix = os.path.splitext(urllib.parse.urlparse(url).path)[1]
        if suffix:
            return suffix
    return default_suffix


def _materialize_remote_file(path: str | None, base64_data: str | None, url: str | None, suffix: str) -> tuple[str, bool]:
    import tempfile

    provided = [value for value in (path, base64_data, url) if value]
    if len(provided) != 1:
        raise ValueError("Exactly one of path, base64, or url must be provided for file input.")
    if path:
        return path, False
    data = _decode_data_url(base64_data) if base64_data else _download_url(url)
    tmp_suffix = _suffix_from_remote_source(base64_data, url, suffix)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=tmp_suffix)
    try:
        tmp.write(data)
        return tmp.name, True
    finally:
        tmp.close()


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


class InferenceRequest(BaseModel):
    audio_path: Optional[str] = Field(None, min_length=1, description="Server-local audio path. Use audio_base64 or audio_url for remote API calls.")
    audio_base64: Optional[str] = Field(None, min_length=1, description="Base64 audio bytes. Data URLs are accepted.")
    audio_url: Optional[str] = Field(None, min_length=1, description="HTTP/HTTPS audio URL reachable from the server.")


class InferenceResponse(BaseModel):
    result: Any


@lru_cache(maxsize=1)
def get_inferencer() -> PipelineInferencer:
    return PipelineInferencer(os.getenv("MODEL_PATH") or os.getenv("MODEL_ID") or DEFAULT_MODEL_PATH, os.getenv("DEVICE", "auto"), int(os.getenv("BATCH_SIZE", "8")))


app = FastAPI(title='hubert-large-speech-emotion-recognition-russian-dusha-finetuned API', version="1.0.0", description='FastAPI service for audio-classification.')


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    inferencer = get_inferencer()
    return HealthResponse(status="ok", model=inferencer.model_name_or_path, device=os.getenv("DEVICE", "auto"))


@app.post("/predict", response_model=InferenceResponse)
def infer(request: InferenceRequest) -> InferenceResponse:
    payload = request.model_dump(exclude_none=True)
    temp_paths: List[str] = []
    try:
        if any(key in payload for key in ("audio_path", "audio_base64", "audio_url")):
            audio_path, should_cleanup = _materialize_remote_file(
                payload.pop("audio_path", None),
                payload.pop("audio_base64", None),
                payload.pop("audio_url", None),
                ".audio",
            )
            payload["audio_path"] = audio_path
            if should_cleanup:
                temp_paths.append(audio_path)
        return InferenceResponse(result=get_inferencer().run(**payload))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        for temp_path in temp_paths:
            try:
                os.remove(temp_path)
            except OSError:
                pass

if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the FastAPI model service.")
    parser.add_argument("--host", default="0.0.0.0", help="Service host.")
    parser.add_argument("--port", type=int, default=8080, help="Service port.")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload.")
    args = parser.parse_args()
    uvicorn.run("fastapi_app:app", host=args.host, port=args.port, reload=args.reload)

