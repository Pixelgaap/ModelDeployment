import argparse
import io
import os
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel
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
async def ocr_inference(
    image: UploadFile = File(..., description="待识别的图像文件"),
    max_new_tokens: int = Query(512, description="生成文本的最大长度")
):
    if pipe is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        image_bytes = await image.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        result = pipe(image_pil, max_new_tokens=max_new_tokens)

        if isinstance(result, list) and len(result) > 0:
            text = result[0].get("generated_text", "") if isinstance(result[0], dict) else str(result[0])
        else:
            text = str(result)

        return OCRResponse(
            success=True,
            text=text,
            model="nanonets/Nanonets-OCR2-3B"
        )
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
