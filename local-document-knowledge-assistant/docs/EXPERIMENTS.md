# Experiment guide

No model quality benchmarks or vector database benchmarks have been run during development.
Use this guide when you are ready to test. The default synthetic hash embedding is useful for
checking plumbing, but its results are not evidence of semantic retrieval quality.

## Suggested experiment matrix

| Question | Hold fixed | Change |
| --- | --- | --- |
| Does preserving headings improve retrieval? | Corpus, embeddings, backend, questions, top k | Fixed vs sentence vs Markdown chunking |
| Does hybrid retrieval help? | Index, questions, top k | BM25 vs vector vs RRF hybrid |
| Does diversity remove redundant context? | Index, questions, candidate pool | Vector vs MMR, vary lambda |
| Does keyword expansion help abbreviations? | BM25 settings and corpus | Expansion off vs on |
| How do vector stores compare? | Source hashes, chunks, embedding identity/dimensions, queries | SQLite vs pgvector vs Chroma vs Qdrant vs FAISS |
| What does approximation cost? | Corpus and embeddings | Exact vs HNSW in pgvector or FAISS |
| Does the model use its evidence? | Retrieved passages, questions and prompt version | Installed local chat model or generation parameters |

Build separate snapshots for changes to store, embedding model, chunking, or indexing algorithm.
Retrieval parameters can be changed without rebuilding. The UI supports two pipelines at a time;
`POST /api/compare` accepts two to four configurations. An evaluation batch accepts up to eight.
The API also allows exact `SearchConfig` objects so custom thresholds and weights can be preserved.

Do not call a different-corpus comparison a database benchmark. Compare full config manifests,
embedding identities and index build settings. Approximate indexes can produce different rankings
even with the same vectors. Query-vector generation and connection overhead can dominate small
corpora; inspect component timings and report the actual conditions.

## Fixed question set

`evaluation/cases.json` contains 40 cases:

- 25 answerable policy questions.
- 10 questions with facts absent from the corpus.
- 2 conflicting-version questions requiring both sources.
- 3 malicious-document/instruction cases.

There are 29 development and 11 holdout questions. Tune on development only. Report holdout
separately after settings are chosen. The `all` option reports both splits separately in JSON.
Each supported case has a filename and an exact expected quote, plus written expected behavior.
The evaluation runner requires all 12 unchanged bundled policies, verifies content hashes,
and records the complete selected index manifest. Extra documents are allowed and change that
manifest; disclose them when comparing.

The built-in runner evaluates **retrieval**, not generated answers. Conflict/injection questions
only check whether expected evidence was retrieved. They do not establish whether the model
followed the right policy or resisted the injection.

## Metrics

| Metric | Definition and interpretation |
| --- | --- |
| Passage recall@k | Fraction of expected source quotes found in at least one returned passage |
| MRR@k | Reciprocal rank of the first passage containing expected support; zero if missing |
| nDCG@k | Discounted gain for a newly covered expected source quote; duplicate support has no additional gain |
| Hit rate@k | Fraction of supported cases with at least one expected support match |
| Unsupported-empty rate | Fraction of unsupported questions returning no passages; threshold diagnostic only |
| Median / p95 latency | Retrieval timings from one pass; not statistically stable benchmark estimates |
| Errors | Explicit per-case failures; excluded from scored denominators and counted separately |

Matching normalizes whitespace and case but requires a complete expected quote within a single
chunk from the expected filename. A chunk boundary can therefore yield a false-negative passage
match. Inspect such examples before drawing conclusions about retrieval quality.

Reports include the case-set hash, complete index and retrieval configurations, corpus and
embedding identities, Python/platform/package versions, per-case passages, errors and split
summaries. Reports do not contain invented faithfulness or answer-accuracy numbers. The current
latency pass has no warm-up or confidence intervals; add repeated controlled runs before making
performance claims.

## Later model review

1. Choose installed embedding and chat models suited to the machine, and record their digests.
2. Build a semantic index. Confirm a few questions retrieve the expected passage first.
3. Ask answerable, unsupported, version-conflict, and malicious-document questions using Ollama
   or the explicitly configured API. Record the API endpoint identity and requested/returned models
   when no local model digest is available.
4. For each factual claim, check the supporting source. Record supported/partial/unsupported
   and notes in Runs & reviews. Inspect a rejected draft when citation validation fails.
5. Record answer correctness, abstention behavior, citation support, time to first token and total
   response time. Citation-ID validity alone is not a faithfulness score.
6. Export reviewed runs and document concrete failure examples before tuning prompts.

Ragas or another model-based evaluator is a later optional extension. It requires an explicitly
selected evaluator model, separate costs/hardware choices, and calibration against these manual
judgments. LangSmith is also deferred; current traces stay local.

## Run template

```text
Date and hardware:
Corpus fingerprint:
Index IDs and chunk settings:
Embedding model digest and dimensions:
Store versions and exact/approximate settings:
Retrieval configs:
Chat model digest (if used):
Dataset hash and split:
Cold/warm conditions and repeat count:
Measured results:
Representative failures:
Human review findings:
Decision / next experiment:
```
