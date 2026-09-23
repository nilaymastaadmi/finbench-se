"""Extract page text from the FinanceBench filings and cut it into page-tagged chunks.

Each chunk keeps its 0-based page index so retrieval can be scored against the
dataset's gold evidence page, separating retrieval misses from generation errors.
"""
import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdfs"
CHUNK_CHARS = 1200
OVERLAP = 200


def chunk_page(text, size=CHUNK_CHARS, overlap=OVERLAP):
    """Split one page into overlapping windows. Short pages stay whole."""
    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= size:
        return [text]
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step)]


def pages(doc_name):
    with pymupdf.open(PDF_DIR / f"{doc_name}.pdf") as doc:
        return [p.get_text() for p in doc]


def chunks_for(doc_name):
    out = []
    for pno, text in enumerate(pages(doc_name)):
        for j, c in enumerate(chunk_page(text)):
            out.append({"doc": doc_name, "page": pno, "id": f"{doc_name}:{pno}:{j}", "text": c})
    return out


def load_questions():
    path = ROOT / "data" / "financebench_open_source.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8")]
