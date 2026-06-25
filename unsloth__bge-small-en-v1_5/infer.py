import argparse
import json
import os
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import pipeline

DEFAULT_MODEL_ID = 'unsloth/bge-small-en-v1.5'
DEFAULT_MODEL_PATH = os.path.dirname(os.path.abspath(__file__))


class EmbeddingInferencer:
    def __init__(self, model_name_or_path: Optional[str] = None, device: Optional[str] = None, batch_size: int = 8) -> None:
        self.model_name_or_path = model_name_or_path or os.getenv("MODEL_PATH") or os.getenv("MODEL_ID") or DEFAULT_MODEL_PATH
        self.batch_size = batch_size
        self.device = self._resolve_device(device)
        self.extractor = pipeline("feature-extraction", model=self.model_name_or_path, tokenizer=self.model_name_or_path, device=self.device, trust_remote_code=True)

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

    def encode(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        outputs = self.extractor(texts, padding=True, truncation=True, batch_size=self.batch_size, return_tensors=True)
        hidden = torch.as_tensor(outputs)
        tokenized = self.extractor.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        mask = tokenized["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        if normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled.detach().cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding inference.")
    parser.add_argument("--model", default=None, help="Local model path. Defaults to the directory containing this file.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--texts", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps({"embeddings": EmbeddingInferencer(args.model, args.device, args.batch_size).encode(args.texts)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
