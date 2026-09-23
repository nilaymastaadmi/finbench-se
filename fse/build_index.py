"""Embed every chunk of the 84 referenced filings once; one .npz + .jsonl per filing.

Per-filing indexes match FinanceBench's "single vector store" setting: retrieval is
restricted to the filing the question is about. Safe to re-run: finished filings are skipped.
"""
import json
import sys
import time

import numpy as np

from fse.gemini import embed
from fse.ingest import ROOT, chunks_for, load_questions

IDX = ROOT / "data" / "index"


def build(doc):
    npz, meta = IDX / f"{doc}.npz", IDX / f"{doc}.jsonl"
    if npz.exists() and meta.exists():
        return "cached"
    chunks = chunks_for(doc)
    vecs = np.asarray(embed([c["text"] for c in chunks], "RETRIEVAL_DOCUMENT"), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    with meta.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    np.savez_compressed(npz, vecs=vecs)
    return f"{len(chunks)} chunks"


if __name__ == "__main__":
    IDX.mkdir(parents=True, exist_ok=True)
    docs = sorted({q["doc_name"] for q in load_questions()})
    t0 = time.time()
    for i, d in enumerate(docs, 1):
        print(f"[{i}/{len(docs)}] {d}: {build(d)} ({time.time() - t0:.0f}s)", flush=True)
    sys.exit(0)
