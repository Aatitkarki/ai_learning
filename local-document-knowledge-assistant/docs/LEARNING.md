# Learning and interview notes

This is implementation scaffolding for your learning. Do not claim model accuracy, database
performance, production usage, or advanced framework experience from code that has not been
exercised and explained.

## Draw the two flows

Ingestion: document bytes → pages → chunks → embeddings → vector store plus metadata.

Question answering: question → lexical/vector retrieval → selected evidence → local model →
citation checks → answer plus source passages → human review.

Explain which failures belong to ingestion, retrieval, generation, or display. A correct model
cannot answer from a passage that was never retrieved; a correct retrieved passage does not
guarantee a correct answer.

## Small exercises

1. Inspect `chunk_document`. Change overlap and locate the duplicated character spans in the UI.
2. Run the same keyword question with BM25 and hash-vector retrieval. Explain why the hash
   baseline still cannot understand a synonym it has never seen.
3. Work out one weighted reciprocal rank fusion score on paper and compare it with the trace.
4. Use two near-duplicate passages to explain MMR's relevance/diversity tradeoff.
5. Add a new synthetic rule and supporting evaluation question. Rebuild a snapshot and compare
   corpus fingerprints with the old index.
6. After model setup, produce an answer with a real but irrelevant citation. Explain why the
   citation validator cannot prove that answer correct.
7. Stop an optional database and inspect the explicit failed experiment. Restore it and retry.

## Design tradeoffs to defend

- Separate snapshots prevent accidental embedding and chunk-setting mixing, at the cost of
  duplicate storage. This is appropriate for a small reproducible lab.
- Direct adapters expose score semantics and failure behavior. Framework integration is narrow
  until the underlying pipeline is understandable.
- SQLite owns source metadata regardless of vector backend. This simplifies citation lookup,
  but means external database cleanup is an application responsibility.
- The app uses a single serialized job worker. A production multi-user system would need
  different concurrency, authentication, durable queues and operational controls.
- Character chunk sizes avoid a model tokenizer dependency in the zero-download phase, but
  cannot guarantee token limits across languages or models.

## Before describing it on a CV

Demonstrate the exact path you claim: local inference requires a successful local model run;
pgvector experience requires exercising pgvector; LangChain currently means the optional
splitter adapter. Ragas, LangSmith, LlamaIndex and llama.cpp remain future experiments.

Keep a record of your own contribution, one observed failure, the investigation that explained
it, a fix, and the evidence that the fix helped. The prepared app does not replace that work.
