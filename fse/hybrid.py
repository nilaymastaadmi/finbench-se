"""Hybrid retrieval: dense (bge-small) and BM25 rankings fused by reciprocal rank fusion (AMENDMENTS.md A3).

Dense embeddings miss numeric tables (a cash-flow statement is mostly numbers); BM25 matches the exact
line-item words. RRF (Cormack et al. 2009) merges the two rankings without tuning score scales.
"""
import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from fse.retrieve import _index

POOL = 50       # candidates taken from each ranking before fusion
RRF_K = 60      # the constant from the RRF paper


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


@lru_cache(maxsize=None)
def _bm25(doc):
    _, meta = _index(doc)
    return BM25Okapi([tokens(c["text"]) for c in meta])


def rrf(rankings, k=RRF_K):
    """Fuse ranked id lists; an id's score is the sum of 1/(k + rank) over lists that contain it."""
    score = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking, 1):
            score[i] = score.get(i, 0.0) + 1.0 / (k + rank)
    return sorted(score, key=lambda i: -score[i])


def search(doc, question, query_vec, k=5):
    index, meta = _index(doc)
    q = np.asarray([query_vec], dtype=np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    dense_scores, dense_ids = index.search(q, min(POOL, index.ntotal))
    dense = [int(i) for i in dense_ids[0] if i >= 0]
    bm = _bm25(doc).get_scores(tokens(question))
    lexical = [int(i) for i in np.argsort(-bm)[:POOL]]
    dense_score = dict(zip(dense, dense_scores[0].tolist()))
    hits = [dict(meta[i], score=float(dense_score.get(i, 0.0)), bm25=float(bm[i]))
            for i in rrf([dense, lexical])[:k]]
    return hits, float(dense_scores[0][0])            # dense top-1 cosine: the trivial-baseline signal
