import argparse
import json
import os
from typing import Any, Dict, List, Optional

import torch
from transformers import pipeline

DEFAULT_MODEL_ID = 'macpaw-research/blip-icon-captioning'
DEFAULT_MODEL_PATH = os.path.dirname(os.path.abspath(__file__))


class PipelineInferencer:
    def __init__(self, model_name_or_path: Optional[str] = None, device: Optional[str] = None, batch_size: int = 8) -> None:
        self.model_name_or_path = model_name_or_path or os.getenv("MODEL_PATH") or os.getenv("MODEL_ID") or DEFAULT_MODEL_PATH
        self.batch_size = batch_size
        self.device = self._resolve_device(device)
        self.pipe = pipeline('image-to-text', model=self.model_name_or_path, tokenizer=self.model_name_or_path, device=self.device, trust_remote_code=True)

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

    def run(self, value: Any, **kwargs: Any) -> Any:
        return self.pipe(value, **kwargs)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="blip-icon-captioning inference.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--input", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    inferencer = PipelineInferencer(args.model, args.device, args.batch_size)
    print(json.dumps({"result": inferencer.run(args.input)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
