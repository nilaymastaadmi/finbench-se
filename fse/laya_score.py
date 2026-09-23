"""Exploratory arm: a local 'typed decision' model (Laya) scores whether each greedy answer is
supported by its retrieved context. Runs fully offline against the Laya daemon on 127.0.0.1:8731.
"""
import json
import urllib.request

from fse.ingest import ROOT
from fse.judge import strip_confidence

GEN = ROOT / "results" / "generations.jsonl"
OUT = ROOT / "results" / "laya.jsonl"
QUESTION = {"supported": {"type": "noul", "instructions":
            "Is every factual claim and figure in the ANSWER stated in, or directly computable from, the CONTEXT?"}}


def ask(state):
    req = urllib.request.Request("http://127.0.0.1:8731/ask",
                                 data=json.dumps({"state": state, "questions": QUESTION}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["answers"]["supported"]["noul"]


def main():
    done = {json.loads(line)["id"] for line in OUT.open(encoding="utf-8")} if OUT.exists() else set()
    for rec in map(json.loads, GEN.open(encoding="utf-8")):
        if rec["id"] in done:
            continue
        answer = strip_confidence(rec["greedy"])[0]
        state = "CONTEXT:\n" + "\n\n".join(rec["context"])[:3500] + "\n\nANSWER: " + answer
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": rec["id"], "supported": float(ask(state))}) + "\n")


if __name__ == "__main__":
    main()
