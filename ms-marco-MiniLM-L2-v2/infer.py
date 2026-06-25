import argparse
import json
import os
from typing import Any, Dict, List, Optional

import torch
from transformers import pipeline

DEFAULT_MODEL_ID = 'cross-encoder/ms-marco-MiniLM-L2-v2'
DEFAULT_MODEL_PATH = os.path.dirname(os.path.abspath(__file__))


class RerankInferencer:
    def __init__(self, model_name_or_path: Optional[str] = None, device: Optional[str] = None, batch_size: int = 8) -> None:
        self.model_name_or_path = model_name_or_path or os.getenv("MODEL_PATH") or os.getenv("MODEL_ID") or DEFAULT_MODEL_PATH
        self.batch_size = batch_size
        self.device = self._resolve_device(device)
        self.pipe = pipeline("text-classification", model=self.model_name_or_path, device=self.device, trust_remote_code=True, top_k=None)

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

    @staticmethod
    def _score(item: Any) -> float:
        labels = item if isinstance(item, list) else [item]
        preferred = [x for x in labels if str(x.get("label", "")).upper() in {"LABEL_1", "POSITIVE", "RELEVANT"}]
        return float((preferred or labels)[0].get("score", 0.0)) if labels else 0.0

    def rank(self, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        outputs = self.pipe([{"text": query, "text_pair": d} for d in documents], batch_size=self.batch_size, truncation=True)
        results = [{"document": d, "score": self._score(o)} for d, o in zip(documents, outputs)]
        results.sort(key=lambda item: item["score"], reverse=True)
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank inference.")
    parser.add_argument("--model", default=None, help="Local model path. Defaults to the directory containing this file.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--query", required=True)
    parser.add_argument("--documents", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps({"results": RerankInferencer(args.model, args.device, args.batch_size).rank(args.query, args.documents)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
