# Pre-registration: can a RAG system tell when it is about to be wrong?

Written 2026-09-23, committed before any generation, judging or analysis code exists
(`git log --diff-filter=A -- PREREGISTRATION.md fse/generate.py fse/judge.py fse/analyse.py` proves the order).
Author: Nilay Toshniwal. Built with AI assistance (Claude Code); disclosed in the README.

## Question

A retrieval-augmented generation (RAG) pipeline answers questions about company filings. When it
answers wrongly it usually does so fluently. If it could flag its own likely-wrong answers, it could
abstain and hand the question to a person instead. Does **discrete semantic entropy**
(Kuhn, Gal, Farquhar, ICLR 2023, arXiv 2302.09664; Farquhar, Kossen, Kuhn, Gal, Nature 2024)
detect wrong answers in a RAG setting, and does it beat cheap alternatives?

Semantic entropy was validated on short-form QA (TriviaQA AUROC about 0.83, CoQA about 0.77 for a
30B OPT model, Kuhn 2023 Table 2). No published result places it inside a RAG pipeline over long,
numeric financial documents, which is why this is a test and not a demonstration.

## Data (fixed)

- FinanceBench open-source sample (Islam et al. 2023, arXiv 2311.11944): **150 questions** over
  **84 filings** (10-K, 10-Q, 8-K, earnings reports), downloaded 2026-09-23 from
  github.com/patronus-ai/financebench, SHA-256 prefixes in `data/pdf_sha256_16.json`.
- Text: PyMuPDF per page, 12,013 pages. Gold evidence pages are 0-based: 189 of 189 evidence
  passages were found on their stated page (checked before this file was written).
- Published comparison rows (Table 2, n=150, human-graded): Llama2 single vector store 41% correct /
  54% incorrect / 5% failed; GPT-4-Turbo single vector store 50% / 11% / 39%; GPT-4-Turbo oracle 85%.

## Pipeline (fixed, no tuning after this commit)

- Retrieval: FinanceBench "single vector store" setting, i.e. search restricted to the question's
  own filing. Chunks of 1,200 characters with 200 overlap, page-tagged. Embeddings:
  `gemini-embedding-001`, 768 dims, cosine, FAISS flat index. **Top k = 5.**
- Generator: **Qwen2.5-7B-Instruct** (open weights, Q4 via Ollama, CPU). One greedy answer
  (temperature 0) is the answer that gets graded. **K = 5 further samples at temperature 0.5**
  (Kuhn 2023's best temperature) feed the uncertainty scores. Same prompt for all six. The greedy
  answer also carries a verbalised confidence 0 to 100 on its last line.
- Judge (grading and clustering): **gemini-3-flash-preview**, a different model family from the
  generator, temperature 0. Grading uses FinanceBench's own three labels: correct (minor rounding or
  unit differences allowed), incorrect, failed to answer (the answer says the information is not
  available). Clustering: one judge call per question groups the 5 samples so that two answers share
  a cluster only if each entails the other (the bidirectional-entailment rule, done by an LLM as the
  Nature paper permits; the one-call grouping is a declared deviation from pairwise checks).

## Scores compared (higher = more likely wrong)

1. **Semantic entropy (primary)**: entropy of the judge's clusters over the 5 samples.
2. Lexical entropy: entropy of exact-match clusters after normalising case, whitespace and number
   formatting. Isolates what the "semantic" step adds.
3. Verbalised confidence: 100 minus the generator's stated confidence.
4. Trivial baselines: (a) negated top-1 retrieval similarity; (b) greedy answer length in characters
   (length heuristics rival detectors, arXiv 2508.08285).
5. Exploratory, local instrument: Laya (`convaiinnovations/laya-typed-decisions`) scoring "is this
   answer supported by the retrieved context". A 5-probe pilot on answer equivalence showed
   overlapping scores (true pairs 0.44 and 0.80, false pairs 0.13, 0.67, 0.44), so it is not used
   for clustering; its grounding score is reported, not relied on.

## Outcomes and metrics

- Primary: **AUROC of each score at separating incorrect from correct greedy answers**, among
  answered questions (failed-to-answer excluded, counted and reported). 95% bootstrap CI,
  2,000 resamples over questions; method differences by paired bootstrap.
- Secondary: AUROC for correct vs not-correct (failed counted as not-correct); selective accuracy at
  50%, 70% and 90% coverage; AURAC; evidence-page recall@5; error attribution (share of incorrect
  answers whose top 5 missed every gold page).
- Judge validity: the author hand-labels a stratified random 30 answers (seed 7) blind to the judge
  label, before reading judge labels for those 30. Agreement is reported with the confusion table.

## Hypotheses, bars and predictions

- **H1 (detection).** PASS if semantic entropy AUROC > 0.5 with the 95% CI lower bound above 0.5.
  FAIL if the CI includes 0.5.
- **H2 (semantic beats lexical).** PASS if SE AUROC exceeds lexical entropy by a paired-bootstrap
  difference whose 95% CI excludes 0. Otherwise "no measurable gain from the semantic step".
- **H3 (beats the trivial).** SE must beat both trivial baselines on AUROC. If it does not, report
  "no skill beyond the trivial baseline" whatever H1 says.
- **Void conditions**, printed beside the headline: fewer than 20 correct or fewer than 20 incorrect
  answered questions (AUROC not estimable); judge agreement with the manual audit below 80% (labels
  not trusted); more than 10% of questions missing any of the six generations.

Predictions (written to be checked, several expected to be wrong):
1. Greedy accuracy 25% to 45% correct (FinanceBench's Llama2 single-store row is 41%).
2. Semantic entropy AUROC 0.60 to 0.72, below the short-QA range, because SE cannot tell
   "the model is unsure" from "retrieval handed it the wrong page".
3. SE beats lexical entropy by less than 0.05 AUROC (numeric answers cluster almost lexically).
4. At least one trivial baseline lands within 0.10 AUROC of SE.
5. Evidence-page recall@5 between 40% and 70%; most incorrect answers are retrieval misses.
6. Answering only the 50% lowest-entropy questions raises accuracy by at least 10 points.

## Fixed in advance

150 questions, one run, no reruns to improve numbers. If a generation fails it is retried up to
3 times for transport errors only. Any change after this commit is logged in `AMENDMENTS.md` with
its reason and the result it could affect.
