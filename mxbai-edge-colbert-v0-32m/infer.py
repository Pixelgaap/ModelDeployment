import argparse
import json
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import pipeline


DEFAULT_MODEL_ID = "mixedbread-ai/mxbai-edge-colbert-v0-32m"
DEFAULT_MODEL_PATH = os.path.dirname(os.path.abspath(__file__))


class ColBERTInferencer:
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 8,
    ) -> None:
        self.model_name_or_path = (
            model_name_or_path
            or os.getenv("MODEL_PATH")
            or os.getenv("MODEL_ID")
            or DEFAULT_MODEL_PATH
        )
        self.batch_size = batch_size
        self.device = self._resolve_device(device)
        self.extractor = pipeline(
            task="feature-extraction",
            model=self.model_name_or_path,
            tokenizer=self.model_name_or_path,
            device=self.device,
            trust_remote_code=True,
        )
        tokenizer = self.extractor.tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id

    @staticmethod
    def _resolve_device(device: Optional[str]) -> int:
        requested = device or os.getenv("DEVICE", "auto")
        if requested == "auto":
            return 0 if torch.cuda.is_available() else -1
        if requested in {"cpu", "-1"}:
            return -1
        if requested.startswith("cuda"):
            if ":" in requested:
                return int(requested.split(":", 1)[1])
            return 0
        return int(requested)

    def encode(self, texts: List[str], normalize: bool = True) -> List[List[List[float]]]:
        if not texts:
            return []
        outputs = self.extractor(
            texts,
            padding=True,
            truncation=True,
            batch_size=self.batch_size,
            return_tensors=True,
        )
        embeddings = torch.as_tensor(outputs)
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings.detach().cpu().tolist()

    def score(self, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        if not documents:
            return []

        texts = [query] + documents
        tokenized = self.extractor.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        embeddings = torch.tensor(self.encode(texts, normalize=True))
        masks = self._content_masks(tokenized["input_ids"])

        query_embeddings = embeddings[0][masks[0]]
        results: List[Dict[str, Any]] = []
        for index, document in enumerate(documents, start=1):
            document_embeddings = embeddings[index][masks[index]]
            if query_embeddings.numel() == 0 or document_embeddings.numel() == 0:
                score = 0.0
            else:
                similarities = query_embeddings @ document_embeddings.T
                score = similarities.max(dim=1).values.sum().item()
            results.append({"document": document, "score": score})

        results.sort(key=lambda item: item["score"], reverse=True)
        return results

    def _content_masks(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        for token_id in (self.pad_token_id, self.cls_token_id, self.sep_token_id):
            if token_id is not None:
                mask &= input_ids != token_id
        return mask


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ColBERT-style retrieval inference.")
    parser.add_argument("--model", default=None, help="Local model path. Defaults to the directory containing this file.")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:0, or a pipeline device id.")
    parser.add_argument("--query", required=True, help="Query text.")
    parser.add_argument("--documents", nargs="+", required=True, help="Candidate documents.")
    parser.add_argument("--batch-size", type=int, default=8, help="Feature extraction batch size.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    inferencer = ColBERTInferencer(
        model_name_or_path=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )
    results = inferencer.score(args.query, args.documents)
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
