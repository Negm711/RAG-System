"""Explicit one-time download. Inference never downloads models."""
import argparse
from pathlib import Path

MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=MODEL, revision=REVISION, local_dir=args.output,
                      allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                                      "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "README.md"])
    print("Reranker saved:", args.output)
    print("Revision:", REVISION)
