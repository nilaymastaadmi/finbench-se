"""Retrieve, then generate 1 greedy answer + K sampled answers per question with a local open model.

Resume-safe: every finished question is appended to results/generations.jsonl and skipped on re-run.
Settings are fixed by PREREGISTRATION.md; do not change them without an AMENDMENTS.md entry.
"""
import json
import time
import urllib.request

from fse.ingest import ROOT, load_questions
from fse.build_index import IDX
from fse.retrieve import TOP_K, embed_questions, evidence_hit, search

MODEL = "qwen2.5:7b-instruct"
K_SAMPLES = 5
SAMPLE_TEMPERATURE = 0.5
OPTIONS = {"num_ctx": 4096, "num_predict": 96, "num_thread": 16}
OUT = ROOT / "results" / "generations.jsonl"

PROMPT = """You are a financial analyst answering a question about one company filing.
Use only the excerpts below. If they do not contain the information needed, reply exactly:
"The excerpts do not contain this information."
Answer in at most two sentences and state the key figure with its unit. Do not show working.
Then end with one final line of the form "Confidence: N", where N (0 to 100) is your probability
that your answer is correct.

Excerpts from {doc}:
{excerpts}

Question: {question}"""


def build_prompt(q, hits):
    excerpts = "\n\n".join(f"[{i + 1}] (page {h['page'] + 1}) {h['text']}" for i, h in enumerate(hits))
    return PROMPT.format(doc=q["doc_name"], excerpts=excerpts, question=q["question"])


def ollama(prompt, temperature, seed):
    body = {"model": MODEL, "prompt": prompt, "stream": False,
            "options": dict(OPTIONS, temperature=temperature, seed=seed)}
    for attempt in range(3):                       # transport errors only, per the pre-registration
        try:
            req = urllib.request.Request("http://localhost:11434/api/generate", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.loads(r.read())["response"].strip()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(10)


def main(limit=None):
    OUT.parent.mkdir(exist_ok=True)
    done = {json.loads(line)["id"] for line in OUT.open(encoding="utf-8")} if OUT.exists() else set()
    questions = sorted(load_questions(), key=lambda q: (q["doc_name"], q["financebench_id"]))[:limit]
    todo = [q for q in questions if q["financebench_id"] not in done]
    qvecs = embed_questions(todo) if todo else []
    t0 = time.time()
    for n, (q, qv) in enumerate(zip(todo, qvecs), 1):
        t = time.time()
        while not (IDX / f"{q['doc_name']}.npz").exists():   # the index builds alongside, filing by filing
            time.sleep(15)
        hits = search(q["doc_name"], qv, TOP_K)
        prompt = build_prompt(q, hits)
        greedy = ollama(prompt, 0.0, 0)
        samples = [ollama(prompt, SAMPLE_TEMPERATURE, s) for s in range(1, K_SAMPLES + 1)]
        rec = {"id": q["financebench_id"], "doc": q["doc_name"], "question": q["question"],
               "gold": q["answer"], "justification": q.get("justification"),
               "question_type": q["question_type"], "question_reasoning": q.get("question_reasoning"),
               "hits": [{"id": h["id"], "page": h["page"], "score": h["score"]} for h in hits],
               "evidence_hit": evidence_hit(hits, q), "context": [h["text"] for h in hits],
               "greedy": greedy, "samples": samples, "seconds": round(time.time() - t, 1)}
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{n}/{len(todo)}] {rec['id']} hit={rec['evidence_hit']} {rec['seconds']}s "
              f"(elapsed {time.time() - t0:.0f}s) greedy={greedy[:70]!r}", flush=True)


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
