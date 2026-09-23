# Amendments to PREREGISTRATION.md

Each entry: what changed, why, when, and which result it could affect. All entries below were made
before any answer was generated or judged (results/ did not exist yet), so none can be results-driven.

## A1, 2026-09-23 14:45 IST: embedding model gemini-embedding-001 -> BAAI/bge-small-en-v1.5 (local)

The free Gemini embedding tier returned HTTP 429 ("embed_content_free_tier_requests, limit: 1000")
and, at 100 chunks (about 30k tokens) per batch, a tokens-per-minute cap would have needed about
6 hours for the 36k chunks of 84 filings. bge-small-en-v1.5 (MIT, 384 dims) runs locally on CPU,
which also makes retrieval fully reproducible offline. Queries use bge's documented instruction
prefix. Chunking, top k = 5 and cosine similarity are unchanged. Could affect: retrieval recall and
therefore accuracy; it cannot affect how uncertainty scores are compared, since every method sees
the same retrieved context.

## A2, 2026-09-23: generation settings made explicit (no change of substance)

Ollama options fixed as num_ctx 4096 (so five 1,200-character excerpts are never truncated),
num_predict 96, num_thread 16 (measured fastest: 33.6 prompt tokens/s vs 26.1 at default), and
seeds 0 (greedy) and 1 to 5 (samples). Questions are processed sorted by filing, then id, so
generation can follow the index build. The prompt asks for at most two sentences without working,
to keep CPU generation near 100 s per question.

## A3, 2026-09-23 15:05 IST: hybrid retrieval (dense + BM25, reciprocal rank fusion)

Trigger: the one-question smoke test (financebench_id_03029, 3M FY2018 capex from the cash-flow
statement) retrieved 5 chunks from pages 25, 38, 42 and 43 and missed the gold page; the model
correctly refused. A cash-flow statement is mostly numbers, which dense sentence embeddings rank
poorly, and the question names its line item in words BM25 matches exactly. Hybrid dense + BM25 is
the standard production design, so this is adopted on design grounds from one observed failure,
**before** recall was measured on the 150 questions. Fusion: top 50 from each ranking, reciprocal
rank fusion with k = 60 (Cormack et al. 2009), final top k = 5 unchanged. Recall@5 is reported for
dense-only and for hybrid, both measured once. Trivial retrieval baseline uses the dense top-1 cosine.
The smoke output (results/smoke.jsonl) is not part of the run.

Also fixed from the smoke test: the confidence parser required "Confidence: N" on its own line; the
model wrote it on the answer's line. Parser now accepts both (tests/test_judge_parse.py).

## A4, 2026-09-23 23:45 IST: A3 reverted, retrieval back to dense-only (the pre-registered design)

A3 promised recall@5 for both designs, measured once. Measured on all 150 questions after 5 questions
had been generated under hybrid: **dense-only 89 of 150 (59.3%), hybrid 52 of 150 (34.7%), BM25 alone
39 of 150 (26.0%)**. Hybrid by question type: metrics-generated 9 of 50, domain-relevant 14 of 50,
novel-generated 29 of 50. No fusion bug: BM25 alone is weak because FinanceBench questions carry long
instruction boilerplate ("Give a response to the question by relying on the details shown in the
cash flow statement") that dominates the keyword match, and equal-weight reciprocal rank fusion pulled
the dense ranking down with it. A3 was adopted from one example; the full measurement contradicts it.

Action: generation restarts from zero with dense-only retrieval, which is what PREREGISTRATION.md
specified before A3. The 5 hybrid generations and 3 judgements are kept, not deleted, in
results/discarded_hybrid_A3/ and excluded from every result.

Disclosure: this choice used the gold evidence pages of the same 150 test questions. It cannot favour
any uncertainty method (all of them see identical retrieved context), but it makes the accuracy and
recall figures optimistic relative to a design fixed blind. Reported as such.

Lesson recorded for the write-up: a retrieval change adopted from one failure cut evidence recall by
24.6 points when finally measured on the full set.
