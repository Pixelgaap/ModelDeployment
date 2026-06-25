import argparse
import json
import os
from typing import Any, Optional

import torch
from transformers import pipeline

DEFAULT_MODEL_ID = 'anferico/bert-for-patents'
DEFAULT_MODEL_PATH = os.path.dirname(os.path.abspath(__file__))
TASK = 'fill-mask'


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if value.__class__.__name__ == "Image":
        return {"type": "PIL.Image", "size": list(value.size), "mode": value.mode}
    return str(value)


class PipelineInferencer:
    def __init__(self, model_name_or_path: Optional[str] = None, device: Optional[str] = None, batch_size: int = 8) -> None:
        self.model_name_or_path = model_name_or_path or os.getenv("MODEL_PATH") or os.getenv("MODEL_ID") or DEFAULT_MODEL_PATH
        self.batch_size = batch_size
        self.device = self._resolve_device(device)
        self.pipe = pipeline(TASK, model=self.model_name_or_path, device=self.device, trust_remote_code=True)

    @staticmethod
    def _resolve_device(device: Optional[str]) -> int:
        requested = device or os.getenv("DEVICE", "auto")
        if requested == "auto":
            return 0 if torch.cuda.is_available() else -1
        if requested in {"cpu", "-1"}:
            return -1
        if requested.startswith("cuda"):
            return int(requested.split(":", 1)[1]) if ":" in requested else 0
        return int(requested)

    def run(self, **payload: Any) -> Any:
        if TASK == "text-classification":
            return jsonable(self.pipe(payload["texts"], batch_size=self.batch_size, truncation=True))
        if TASK == "token-classification":
            return jsonable(self.pipe(payload["text"], aggregation_strategy="simple"))
        if TASK == "fill-mask":
            return jsonable(self.pipe(payload["text"], top_k=payload.get("top_k", 5)))
        if TASK == "summarization":
            return jsonable(self.pipe(payload["text"], max_length=payload.get("max_length"), min_length=payload.get("min_length"), truncation=True))
        if TASK == "translation":
            return jsonable(self.pipe(payload["text"], max_length=payload.get("max_length")))
        if TASK == "zero-shot-classification":
            return jsonable(self.pipe(payload["text"], candidate_labels=payload["candidate_labels"]))
        if TASK == "question-answering":
            return jsonable(self.pipe(question=payload["question"], context=payload["context"]))
        if TASK == "table-question-answering":
            return jsonable(self.pipe(table=payload["table"], query=payload["query"]))
        if TASK in {"document-question-answering", "visual-question-answering"}:
            return jsonable(self.pipe(image=payload["image_path"], question=payload["question"]))
        if TASK == "multiple-choice":
            return jsonable(self.pipe({"question": payload["question"], "choices": payload["choices"]}))
        if TASK in {"image-classification", "image-to-text", "object-detection", "image-segmentation", "mask-generation", "depth-estimation", "image-feature-extraction"}:
            return jsonable(self.pipe(payload["image_path"]))
        if TASK in {"zero-shot-image-classification", "zero-shot-object-detection"}:
            return jsonable(self.pipe(payload["image_path"], candidate_labels=payload["candidate_labels"]))
        if TASK in {"automatic-speech-recognition", "audio-classification", "voice-activity-detection"}:
            return jsonable(self.pipe(payload["audio_path"]))
        return jsonable(self.pipe(payload.get("input")))


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{TASK} inference.")
    parser.add_argument("--model", default=None, help="Local model path. Defaults to the directory containing this file.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--input", required=True, help="Input text or local file path, depending on the task.")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    inferencer = PipelineInferencer(args.model, args.device, args.batch_size)
    key = "audio_path" if TASK.startswith("audio") or TASK == "automatic-speech-recognition" else "image_path" if TASK.startswith("image") or TASK in {"object-detection", "depth-estimation", "mask-generation"} else "text"
    print(json.dumps({"result": inferencer.run(**{key: args.input, "input": args.input, "texts": [args.input]})}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
