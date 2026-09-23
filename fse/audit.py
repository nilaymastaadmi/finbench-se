"""Blind audit of the LLM judge (pre-registered: 30 answers, stratified by question type, seed 7).

    python -m fse.audit sheet    # writes results/audit_sheet.csv WITHOUT judge labels; fill the `label` column
    python -m fse.audit score    # compares your labels with the judge's, writes results/AUDIT.md
"""
import csv
import json
import random
import sys

from fse.ingest import ROOT
from fse.judge import strip_confidence

RES = ROOT / "results"
SHEET = RES / "audit_sheet.csv"
LABELS = ("correct", "incorrect", "failed")


def sheet():
    gen = [json.loads(line) for line in (RES / "generations.jsonl").open(encoding="utf-8")]
    rng = random.Random(7)
    picks = []
    for qtype in sorted({g["question_type"] for g in gen}):
        pool = sorted((g for g in gen if g["question_type"] == qtype), key=lambda g: g["id"])
        picks += rng.sample(pool, 10)
    assert len(picks) == 30
    with SHEET.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "gold_answer", "model_answer", "label"])
        for g in picks:
            w.writerow([g["id"], g["question"], g["gold"], strip_confidence(g["greedy"])[0], ""])
    print(f"wrote {SHEET} (30 rows). Label each: correct / incorrect / failed. Do not open judgements.jsonl first.")


def score():
    mine = {r["id"]: r["label"].strip().lower() for r in csv.DictReader(SHEET.open(encoding="utf-8"))}
    assert all(v in LABELS for v in mine.values()), sorted(set(mine.values()))
    judge = {j["id"]: j["grade"]["label"] for j in map(json.loads, (RES / "judgements.jsonl").open(encoding="utf-8"))}
    pairs = [(mine[i], judge[i]) for i in mine]
    agree = sum(a == b for a, b in pairs)
    table = {(a, b): sum(1 for p in pairs if p == (a, b)) for a in LABELS for b in LABELS}
    lines = [f"# Judge audit: {agree} of {len(pairs)} agree ({agree / len(pairs):.0%}); pre-registered bar 80%", "",
             "| manual \\ judge | " + " | ".join(LABELS) + " |", "|---|---|---|---|"]
    lines += [f"| {a} | " + " | ".join(str(table[(a, b)]) for b in LABELS) + " |" for a in LABELS]
    lines += ["", "Disagreements:"] + [f"- {i}: manual {mine[i]}, judge {judge[i]}" for i in mine if mine[i] != judge[i]]
    (RES / "AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"sheet": sheet, "score": score}[sys.argv[1]]()
