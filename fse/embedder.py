"""Local dense embeddings (BAAI/bge-small-en-v1.5, MIT). Replaces the Gemini free tier; see AMENDMENTS.md A1."""
import os
from functools import lru_cache

from fse.ingest import ROOT

MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # bge's documented query instruction


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(os.cpu_count())
    return SentenceTransformer(MODEL, cache_folder=str(ROOT / "models"), device="cpu")


def embed_passages(texts):
    return _model().encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)


def embed_queries(texts):
    return _model().encode([QUERY_PREFIX + t for t in texts], batch_size=64,
                           normalize_embeddings=True, convert_to_numpy=True)
