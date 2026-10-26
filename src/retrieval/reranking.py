"""CPU cross-encoder reranking. Model files must be provisioned explicitly."""
import math
import os
import threading
from pathlib import Path


class RerankerUnavailable(RuntimeError):
    pass


class LocalReranker:
    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        path = Path(os.environ.get("RERANK_MODEL_PATH", "/app/assets/models/reranker"))
        if not path.is_dir():
            raise RerankerUnavailable("Local reranker is not provisioned. See docs/retrieval.md.")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            model = AutoModelForSequenceClassification.from_pretrained(
                path, local_files_only=True, trust_remote_code=False, use_safetensors=True
            ).to("cpu").eval()
        except Exception as exc:
            raise RerankerUnavailable("Cannot load the local reranker; check optional dependencies and model files.") from exc
        self._tokenizer, self._model = tokenizer, model

    def _remote(self, path, payload=None):
        import json
        from urllib.request import Request, urlopen
        base = os.environ.get("RERANK_SERVICE_URL", "").rstrip("/")
        request = Request(base + path, data=None if payload is None else json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=120) as response:
                return json.load(response)
        except Exception as exc:
            raise RerankerUnavailable("Reranking service unavailable. See docs/retrieval.md.") from exc

    def ready(self):
        if os.environ.get("RERANK_SERVICE_URL"):
            self._remote("/ready")
            return
        with self._lock:
            self._load()

    def score(self, query, documents):
        if os.environ.get("RERANK_SERVICE_URL"):
            return self._remote("/score", {"query": query, "texts": [doc.text for doc in documents]})["scores"]
        # Serialize model loading/inference so concurrent requests do not duplicate it.
        with self._lock:
            self._load()
            import torch
            scores = []
            with torch.inference_mode():
                for start in range(0, len(documents), 8):
                    batch = documents[start:start + 8]
                    encoded = self._tokenizer(
                        [query] * len(batch), [doc.text for doc in batch],
                        padding=True, truncation="only_second", max_length=512,
                        return_tensors="pt",
                    )
                    scores.extend(self._model(**encoded).logits.reshape(-1).tolist())
            return scores


def rank_documents(query, documents, limit, scorer):
    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")
    documents = list(documents)
    if not documents:
        return []
    scores = scorer(query, documents)
    if len(scores) != len(documents) or any(not math.isfinite(float(s)) for s in scores):
        raise ValueError("Reranker returned invalid scores")
    ranked = sorted(zip(documents, scores), key=lambda pair: -float(pair[1]))
    return [(doc, float(score)) for doc, score in ranked[:limit]]


reranker = LocalReranker()
