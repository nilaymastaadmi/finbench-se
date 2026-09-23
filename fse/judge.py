"""Grade each greedy answer (FinanceBench's 3 labels) and cluster each question's samples by meaning.

The judge (Gemini) is a different model family from the generator (Qwen). Grading never sees the
samples; clustering never sees the gold answer. Resume-safe via results/judgements.jsonl.
"""
import json
import re
import time

from fse.gemini import generate
from fse.ingest import ROOT

GEN = ROOT / "results" / "generations.jsonl"
OUT = ROOT / "results" / "judgements.jsonl"
LABELS = ("correct", "incorrect", "failed")
CONF_LINE = re.compile(r"\s*\bconfidence\s*[:=]\s*(\d{1,3})\s*%?\s*\.?\s*$", re.I)   # last thing in the text, own line or not

GRADE = """You grade answers to questions about company financial filings, using FinanceBench's rubric.
Labels:
- correct: the answer gives the same fact or value as the gold answer. Minor rounding, unit formatting
  (e.g. $1,577 million vs 1577.00) or extra correct detail is fine.
- incorrect: the answer gives a different value or fact, a wrong calculation, or contradicts the gold answer.
- failed: the answer declines, or says the information is not available, without committing to an answer.
Question: {question}
Gold answer: {gold}
Gold justification: {justification}
Answer to grade: {answer}
Reply with JSON only: {{"label": "correct" | "incorrect" | "failed", "reason": "<one sentence>"}}"""

CLUSTER = """Group these {n} answers to the same question by meaning.
Two answers belong to the same group only if each one entails the other: they commit to the same final
value or fact. Ignore wording, formatting, and rounding at the precision stated. An answer that says the
information is unavailable forms its own group with other such answers.
Question: {question}
{answers}
Reply with JSON only: {{"groups": [<group number for answer 1>, <for answer 2>, ...]}} using small integers starting at 1."""


def strip_confidence(text):
    """Return (answer without the confidence line, stated confidence or None)."""
    m = CONF_LINE.search(text)
    conf = int(m.group(1)) if m and int(m.group(1)) <= 100 else None
    return CONF_LINE.sub("", text).strip(), conf


def _json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in judge reply: {text[:200]!r}")
    return json.loads(m.group(0))


def grade(rec):
    answer, _ = strip_confidence(rec["greedy"])
    reply = _json(generate(GRADE.format(question=rec["question"], gold=rec["gold"],
                                        justification=rec.get("justification") or "", answer=answer)))
    assert reply["label"] in LABELS, reply
    return reply


def cluster(rec):
    samples = [strip_confidence(s)[0] for s in rec["samples"]]
    listing = "\n".join(f"Answer {i + 1}: {s}" for i, s in enumerate(samples))
    groups = _json(generate(CLUSTER.format(n=len(samples), question=rec["question"], answers=listing)))["groups"]
    assert len(groups) == len(samples) and all(isinstance(g, int) for g in groups), groups
    return groups


def main():
    done = {json.loads(line)["id"] for line in OUT.open(encoding="utf-8")} if OUT.exists() else set()
    recs = [json.loads(line) for line in GEN.open(encoding="utf-8")]
    for n, rec in enumerate(r for r in recs if r["id"] not in done):
        t = time.time()
        out = {"id": rec["id"], "grade": grade(rec), "clusters": cluster(rec)}
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")
        print(f"[{n + 1}] {rec['id']} {out['grade']['label']} clusters={out['clusters']} {time.time() - t:.1f}s",
              flush=True)


if __name__ == "__main__":
    main()
