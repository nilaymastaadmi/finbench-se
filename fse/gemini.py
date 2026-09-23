"""Minimal Gemini REST client: embeddings and text generation, with retry on 429/5xx.

Only public FinanceBench text is ever sent (the free tier may use inputs for training).
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = "https://generativelanguage.googleapis.com/v1beta/models"
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
GEN_MODEL = "gemini-3-flash-preview"


def _post(url, body, tries=10):
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]}
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                wait = min(60, 4 * 2 ** attempt)
                print(f"[gemini] HTTP {e.code}, retry {attempt + 1} in {wait}s", file=sys.stderr, flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"Gemini HTTP {e.code}: {e.read()[:300]!r}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < tries - 1:
                print(f"[gemini] {type(e).__name__}, retry {attempt + 1}", file=sys.stderr, flush=True)
                time.sleep(4 * 2 ** attempt)
                continue
            raise


EMBED_TEXTS_PER_MIN = 850        # free tier counts every text as a request: 1,000 per minute (429 seen 2026-09-23)
_last_batch = [0.0]


def embed(texts, task):
    """task: RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY. Returns one 768-d vector per text."""
    out = []
    for i in range(0, len(texts), 100):
        gap = 60 * min(100, len(texts) - i) / EMBED_TEXTS_PER_MIN - (time.time() - _last_batch[0])
        if gap > 0:
            time.sleep(gap)
        _last_batch[0] = time.time()
        reqs = [{"model": f"models/{EMBED_MODEL}", "content": {"parts": [{"text": t[:8000]}]},
                 "taskType": task, "outputDimensionality": EMBED_DIM} for t in texts[i:i + 100]]
        r = _post(f"{BASE}/{EMBED_MODEL}:batchEmbedContents", {"requests": reqs})
        out.extend(e["values"] for e in r["embeddings"])
    assert len(out) == len(texts), (len(out), len(texts))
    return out


def generate(prompt, temperature=0.0, max_tokens=1024):
    r = _post(f"{BASE}/{GEN_MODEL}:generateContent",
              {"contents": [{"parts": [{"text": prompt}]}],
               "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}})
    cand = r.get("candidates") or []
    parts = cand[0].get("content", {}).get("parts", []) if cand else []
    return "".join(p.get("text", "") for p in parts).strip()
