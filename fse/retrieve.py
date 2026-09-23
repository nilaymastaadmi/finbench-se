"""Top-k retrieval restricted to the question's own filing (FinanceBench "single vector store")."""
import json
from functools import lru_cache

import faiss
import numpy as np

from fse.build_index import IDX
from fse.embedder import embed_queries

TOP_K = 5


@lru_cache(maxsize=None)
def _index(doc):
    vecs = np.load(IDX / f"{doc}.npz")["vecs"]
    index = faiss.IndexFlatIP(vecs.shape[1])            # vectors are unit-norm, so IP = cosine
    index.add(vecs)
    meta = [json.loads(line) for line in (IDX / f"{doc}.jsonl").open(encoding="utf-8")]
    assert len(meta) == index.ntotal, (doc, len(meta), index.ntotal)
    return index, meta


def search(doc, query_vec, k=TOP_K):
    index, meta = _index(doc)
    q = np.asarray([query_vec], dtype=np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    scores, ids = index.search(q, k)
    return [dict(meta[i], score=float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]


def embed_questions(questions):
    return embed_queries([q["question"] for q in questions])


def evidence_hit(hits, question):
    """True if any retrieved chunk comes from a gold evidence page of this question."""
    gold = {(e["doc_name"], e["evidence_page_num"]) for e in question["evidence"]}
    return any((h["doc"], h["page"]) in gold for h in hits)
