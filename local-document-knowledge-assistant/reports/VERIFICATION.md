# Implementation verification — 28 September 2026

This records code checks, not a model-quality evaluation or a vector-database benchmark.

## Environment and constraints

- Used the existing `support-transcript-summarizer/.venv` Python 3.11 environment.
- No Python packages, language/embedding models, runtimes, or container images were downloaded.
- No Ollama or OpenAI-compatible inference requests were sent to a real server.
  No optional database services were started.
- No document corpus was installed into the user's persistent app data. Tests used temporary data.
- Exported the existing core dependency resolution offline to a hash-pinned requirements file.

## Checks

Result after the configurable API addition: **107 passed, 5 opt-in integration tests skipped**.
Ruff lint passes. The added API UI branches also pass a targeted Streamlit check.

```bash
../support-transcript-summarizer/.venv/bin/python -m ruff check .
../support-transcript-summarizer/.venv/bin/python -m pytest -q
```

The test suite covers:

- UTF-8 ingestion, malformed inputs, limits, deduplication and safe filenames.
- Page/offset preservation, source substring coverage and all three built-in chunking methods.
- Deterministic normalized embeddings, invalid vectors, and embedding identity mismatch.
- BM25, vector search, RRF hybrid, MMR, expansion, reranking and document filters.
- Index failure/recovery, separate rebuild snapshots and cascading deletion.
- API upload/build/search/source/stream/review/delete workflows and input validation.
- Evidence-only responses, no-result abstention and persisted follow-up context.
- Cross-thread streaming, deletion during a response, and concurrent conversation revision checks.
- Mocked Ollama chat/embedding contracts, missing models, streamed output, unknown citations,
  uncited paragraphs, interrupted responses and output truncation.
- Mocked Qdrant collection/write/query/delete contracts with explicit vectors.
- Mocked OpenAI-compatible chat/embedding requests, bearer authentication, configurable token
  parameters, usage parsing, reordered embedding results, and endpoint/revision identity checks.
- API URL validation, secret exclusion, safe provider error messages, disabled redirects,
  incomplete/filtered responses, and startup with no API configuration.
- Evaluation formulas, dataset consistency and a tiny synthetic background evaluation fixture.
- Streamlit AppTest rendering of all six pages, with a ready baseline index and mocked API.

Five opt-in backend contract tests skip by default. The SQLite implementation is nevertheless
exercised extensively through the core service/API tests. A dependency deprecation warning notes
that Starlette's HTTPX test transport is deprecated in favor of HTTPX2; the current pinned transport
still passes these tests. No dependency upgrade was installed to suppress that warning.

## Still pending by design

| Path | Status |
| --- | --- |
| Live local chat generation and semantic embeddings | Implemented; no model selected or run |
| OpenAI-compatible API chat and embeddings | Implemented and mock-tested; URL/key/model details pending |
| PDF parsing with pypdf | Implemented; optional package absent, live extraction not verified |
| pgvector | Implemented; extension/service/package integration not exercised |
| Chroma | Implemented; optional package integration not exercised |
| Qdrant | Mock HTTP protocol verified; live service not exercised |
| FAISS exact/HNSW | Implemented; optional package integration not exercised |
| LangChain recursive chunking | Implemented; optional package absent, integration not exercised |
| Full 40-question evaluation and held-out results | Prepared; not run |
| Model factuality, semantic abstention, conflict handling, injection resistance | Not measured |
| Docker/container startup and visual browser QA | Not exercised |
| Ragas, LangSmith, LlamaIndex, llama.cpp | Future extensions; not implemented |

The app should be described as a developed experiment workbench with verified core behavior,
pending live model and optional-backend validation. It should not be described as benchmarked,
production ready, or proven accurate.
