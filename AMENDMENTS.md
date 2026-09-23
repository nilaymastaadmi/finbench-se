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
