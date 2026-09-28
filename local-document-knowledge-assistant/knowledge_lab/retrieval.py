import math
from collections import Counter
from time import perf_counter

import numpy as np

from .config import LabError
from .embeddings import Embedder, tokens
from .models import Chunk, Hit, IndexConfig, SearchConfig, SearchResult
from .stores import make_store

EXPANSIONS = {
    "kyc": "identity verification",
    "refund": "reimbursement cancellation",
    "late": "delayed pending transfer",
    "fee": "charge cost",
    "cancel": "cancellation",
    "limit": "maximum cap",
    "eta": "delivery time",
}


def bm25(query: str, chunks: list[Chunk]) -> dict[str, float]:
    bag = [Counter(tokens(c.text)) for c in chunks]
    lengths = [sum(row.values()) for row in bag]
    average = sum(lengths) / max(len(lengths), 1) or 1
    document_frequency = Counter(word for row in bag for word in row)
    scores = {}
    for chunk, row, length in zip(chunks, bag, lengths, strict=True):
        score = 0.0
        for term in set(tokens(query)):
            frequency = row[term]
            if frequency:
                idf = math.log(
                    1
                    + (len(chunks) - document_frequency[term] + 0.5)
                    / (document_frequency[term] + 0.5)
                )
                score += (
                    idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * length / average))
                )
        scores[chunk.id] = score
    return scores


def ranked(scores: dict[str, float], minimum: float, limit: int) -> list[tuple[str, float]]:
    return sorted(
        ((k, v) for k, v in scores.items() if v > 0 and v >= minimum),
        key=lambda item: (-item[1], item[0]),
    )[:limit]


def rrf(lists: list[tuple[list[tuple[str, float]], float]], constant=60) -> dict[str, float]:
    result: dict[str, float] = {}
    for ranking, weight in lists:
        if not weight:
            continue
        for rank, (key, _) in enumerate(ranking, 1):
            result[key] = result.get(key, 0) + weight / (constant + rank)
    return result


def diversify(candidates, query_vector, vectors, limit, relevance):
    selected = []
    remaining = set(candidates)
    while remaining and len(selected) < limit:

        def score(key):
            similarity = float(np.dot(query_vector, vectors[key]))
            redundancy = max(
                (float(np.dot(vectors[key], vectors[other])) for other in selected), default=0.0
            )
            return relevance * similarity - (1 - relevance) * redundancy

        best = min(remaining, key=lambda key: (-score(key), key))
        remaining.remove(best)
        selected.append(best)
    return selected


def search(repository, settings, question: str, config: SearchConfig) -> SearchResult:
    started = perf_counter()
    index = repository.get("indexes", config.index_id)
    if not index:
        raise LabError("Index not found.", 404)
    if index["status"] != "ready":
        raise LabError("Wait for this index to finish building, or select a ready index.", 409)
    rows = repository.chunks(index["id"])
    allowed = set(config.document_ids)
    if allowed - set(index["config"]["document_ids"]):
        raise LabError("Document filter contains a document outside this index.")
    chunks = {
        r["chunk"]["id"]: Chunk(**r["chunk"])
        for r in rows
        if not allowed or r["chunk"]["document_id"] in allowed
    }
    vectors = {
        r["chunk"]["id"]: np.asarray(r["vector"], dtype=np.float32)
        for r in rows
        if r["chunk"]["id"] in chunks
    }
    query = question
    if config.expand_query:
        query += " " + " ".join(
            EXPANSIONS[t] for t in sorted(set(tokens(question))) if t in EXPANSIONS
        )
        query = query.strip()
    warnings = []
    if index["config"]["embedding"] == "hash" and config.method != "bm25":
        warnings.append("Hash embeddings are a deterministic lexical baseline, not semantic AI.")
    timings = {}
    lexical, dense = [], []
    dense_scores = {}
    lexical_scores = {}
    query_vector = None
    if config.method in {"bm25", "hybrid"}:
        tick = perf_counter()
        lexical_scores = bm25(query, list(chunks.values()))
        lexical = ranked(lexical_scores, config.min_bm25_score, config.candidate_k)
        timings["bm25"] = (perf_counter() - tick) * 1000
    if config.method in {"vector", "hybrid", "mmr"} and tokens(query):
        tick = perf_counter()
        embedder = Embedder(IndexConfig(**index["config"]), settings)
        if embedder.identity() != index["embedding_identity"]:
            raise LabError("The embedding model changed. Build a new index before searching.", 409)
        query_vector = embedder.embed([query])[0]
        if len(query_vector) != index["dimensions"]:
            raise LabError("Embedding dimensions changed. Rebuild the index.", 409)
        timings["embedding"] = (perf_counter() - tick) * 1000
        tick = perf_counter()
        try:
            # Request all candidates for filtered searches so filtering cannot starve top-k.
            count = len(rows) if allowed else config.candidate_k
            candidates = make_store(index, settings, repository).search(query_vector, count)
        except LabError:
            raise
        except Exception as exc:
            raise LabError(
                "Vector store query failed. Check its installation and local service.", 503
            ) from exc
        dense_scores = {key: score for key, score in candidates if key in chunks}
        dense = ranked(dense_scores, config.min_vector_score, config.candidate_k)
        timings["vector_store"] = (perf_counter() - tick) * 1000
    if config.method == "bm25":
        scores = dict(lexical)
    elif config.method == "hybrid":
        scores = rrf([(lexical, 1 - config.vector_weight), (dense, config.vector_weight)])
    else:
        scores = dict(dense)
    order = [key for key, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
    if config.method == "mmr" and query_vector is not None:
        order = diversify(order, query_vector, vectors, config.top_k, config.mmr_lambda)
    rerank_scores = {}
    if config.rerank == "lexical":
        query_terms = set(tokens(query))
        for key in order:
            rerank_scores[key] = len(query_terms & set(tokens(chunks[key].text))) / max(
                len(query_terms), 1
            )
        order.sort(key=lambda key: (-rerank_scores[key], -scores[key], key))
        warnings.append("Lexical reranking measures word overlap, not semantic support.")
    hits = []
    for key in order[: config.top_k]:
        component_scores = {"ranking": scores[key]}
        if key in lexical_scores:
            component_scores["bm25"] = lexical_scores[key]
        if key in dense_scores:
            component_scores["cosine"] = dense_scores[key]
        if key in rerank_scores:
            component_scores["lexical_overlap"] = rerank_scores[key]
        hits.append(Hit(chunk=chunks[key], score=scores[key], scores=component_scores))
    timings["total"] = (perf_counter() - started) * 1000
    return SearchResult(
        query=question,
        expanded_query=query,
        config=config,
        hits=hits,
        timings_ms=timings,
        warnings=warnings,
        corpus_fingerprint=index["corpus_fingerprint"],
    )
