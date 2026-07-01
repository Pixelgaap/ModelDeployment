import argparse
import json
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import pipeline

DEFAULT_MODEL_ID = 'jinaai/jina-code-embeddings-1.5b'
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
        if not texts:
            return []
        outputs = self.extractor(texts, padding=True, truncation=True, batch_size=self.batch_size, return_tensors=True)
        hidden = torch.as_tensor(outputs)
        tokenized = self.extractor.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        mask = tokenized["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        if normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled.detach().cpu().tolist()

    def similarity(self, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        vectors = torch.tensor(self.encode([query] + documents, normalize=True))
        query_vector = vectors[0]
        results = [{"document": doc, "score": float(torch.dot(query_vector, vec).item())} for doc, vec in zip(documents, vectors[1:])]
        results.sort(key=lambda item: item["score"], reverse=True)
        return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embedding inference and similarity ranking.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--texts", nargs="+")
    parser.add_argument("--query")
    parser.add_argument("--documents", nargs="+")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    inferencer = EmbeddingInferencer(args.model, args.device, args.batch_size)
    payload = {"results": inferencer.similarity(args.query, args.documents)} if args.query and args.documents else {"embeddings": inferencer.encode(args.texts or ["hello world"])}
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
