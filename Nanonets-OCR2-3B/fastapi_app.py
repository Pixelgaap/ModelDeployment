import argparse
import base64
import io
import os
import urllib.request
from typing import Optional

from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

app = FastAPI(title="Nanonets-OCR2-3B 模型服务", version="1.0.0", description="基于Nanonets-OCR2-3B模型的OCR文本识别服务")

pipe = None
DEFAULT_MODEL_PATH = os.path.dirname(os.path.abspath(__file__))


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    gpu_available: bool


class OCRResponse(BaseModel):
    success: bool
    text: str
    model: str


class OCRRequest(BaseModel):
    image_path: Optional[str] = Field(
        None,
        min_length=1,
        description="Server-local image path. Use image_base64 or image_url for remote API calls.",
    )
    image_base64: Optional[str] = Field(
        None,
        min_length=1,
        description="Base64 image bytes. Data URLs are accepted.",
    )
    image_url: Optional[str] = Field(
        None,
        min_length=1,
        description="HTTP/HTTPS image URL reachable from the server.",
    )
    max_new_tokens: int = Field(512, ge=1, le=2048)


def _decode_data_url(value: str) -> bytes:
    payload = value.split(",", 1)[1] if "," in value and value.lstrip().startswith("data:") else value
    return base64.b64decode(payload)


def _load_image(request: OCRRequest) -> Image.Image:
    provided = [value for value in (request.image_path, request.image_base64, request.image_url) if value]
    if len(provided) != 1:
        raise ValueError("Exactly one of image_path, image_base64, or image_url must be provided.")
    if request.image_path:
        return Image.open(request.image_path).convert("RGB")
    if request.image_base64:
        return Image.open(io.BytesIO(_decode_data_url(request.image_base64))).convert("RGB")
    with urllib.request.urlopen(request.image_url, timeout=30) as response:
        return Image.open(io.BytesIO(response.read())).convert("RGB")


@app.on_event("startup")
def load_model():
    global pipe
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH), help="本地模型路径")
    parser.add_argument("--device", type=int, default=0, help="使用的GPU设备索引，-1表示CPU")
    args, _ = parser.parse_known_args()

    try:
        pipe = pipeline(task="image-to-text", model=args.model, device=args.device)
        print(f"模型加载成功: {args.model}")
        print(f"使用设备: {'GPU' if args.device >= 0 else 'CPU'}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise e


@app.get("/health", response_model=HealthResponse, tags=["健康检查"])
async def health_check():
    if pipe is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    import torch
    gpu_available = torch.cuda.is_available()
    device_info = pipe.device if hasattr(pipe, 'device') else "Unknown"

    return HealthResponse(
        status="healthy" if pipe is not None else "unhealthy",
        model="nanonets/Nanonets-OCR2-3B",
        device=str(device_info),
        gpu_available=gpu_available
    )


@app.post("/predict", response_model=OCRResponse, tags=["OCR识别"])
async def ocr_inference(request: OCRRequest):
    if pipe is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        image_pil = _load_image(request)
        result = pipe(image_pil, max_new_tokens=request.max_new_tokens)

        if isinstance(result, list) and len(result) > 0:
            text = result[0].get("generated_text", "") if isinstance(result[0], dict) else str(result[0])
        else:
            text = str(result)

        return OCRResponse(
            success=True,
            text=text,
            model="nanonets/Nanonets-OCR2-3B"
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH), help="本地模型路径")
    parser.add_argument("--device", type=int, default=0, help="使用的GPU设备索引，-1表示CPU")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务监听地址")
    parser.add_argument("--port", type=int, default=8080, help="服务监听端口")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
