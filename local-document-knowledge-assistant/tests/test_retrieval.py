import numpy as np
import pytest

from knowledge_lab.config import LabError
from knowledge_lab.embeddings import Embedder, normalized
from knowledge_lab.models import IndexConfig, SearchConfig
from knowledge_lab.retrieval import diversify, rrf


@pytest.mark.parametrize("method", ["bm25", "vector", "hybrid", "mmr"])
def test_all_methods_retrieve_policy_without_network(indexed, method):
    lab, index, _ = indexed
    result = lab.search(
        "approved cancellation refund original funding source",
        SearchConfig(index_id=index["id"], method=method, top_k=3),
    )
    assert result.hits
    assert result.hits[0].chunk.filename == "04-cancellation.md"
    assert result.timings_ms["total"] >= 0
    assert lab.repo.list("runs")[0]["result"]["config"]["method"] == method


def test_document_filter_cannot_leak_other_sources(indexed):
    lab, index, docs = indexed
    allowed = next(d for d in docs if d["filename"] == "03-fees.md")
    for method in ("bm25", "vector", "hybrid", "mmr"):
        result = lab.search(
            "fee transfer refund",
            SearchConfig(index_id=index["id"], method=method, document_ids=[allowed["id"]]),
        )
        assert all(h.chunk.document_id == allowed["id"] for h in result.hits)
    with pytest.raises(LabError, match="outside"):
        lab.search("fee", SearchConfig(index_id=index["id"], document_ids=["unknown"]))


def test_keyword_search_does_not_require_embedding_service(indexed):
    lab, index, _ = indexed
    index["config"].update(embedding="ollama", embedding_model="absent")
    lab.repo.put("indexes", index)
    assert lab.search(
        "standard transfer fee", SearchConfig(index_id=index["id"], method="bm25")
    ).hits


def test_stopword_query_abstains(indexed):
    lab, index, _ = indexed
    assert not lab.search("is it the", SearchConfig(index_id=index["id"])).hits


def test_query_expansion_and_lexical_reranking_are_visible(indexed):
    lab, index, _ = indexed
    result = lab.search(
        "kyc",
        SearchConfig(index_id=index["id"], method="bm25", expand_query=True, rerank="lexical"),
    )
    assert "identity verification" in result.expanded_query
    assert result.hits[0].chunk.filename == "02-identity-verification.md"
    assert "lexical_overlap" in result.hits[0].scores


def test_embedding_fingerprint_prevents_mixed_spaces(indexed):
    lab, index, _ = indexed
    index["embedding_identity"] = "a-different-model"
    lab.repo.put("indexes", index)
    with pytest.raises(LabError, match="model changed"):
        lab.search("fee", SearchConfig(index_id=index["id"], method="vector"))


def test_hash_is_deterministic_and_normalized(settings):
    embedder = Embedder(IndexConfig(name="x", document_ids=["d"]), settings)
    a, b = embedder.embed(["refund fee", "refund fee"])
    np.testing.assert_array_equal(a, b)
    assert float(np.linalg.norm(a)) == pytest.approx(1)
    assert embedder.identity() == "sha256-word-hash-v1:384"


@pytest.mark.parametrize(
    "vectors,count", [([[float("nan")]], 1), ([[0, 0]], 1), ([[1, 2]], 2), ([1, 2], 1)]
)
def test_invalid_vectors_rejected(vectors, count):
    with pytest.raises(LabError):
        normalized(vectors, count)


def test_rank_fusion_uses_ranks_not_incompatible_raw_scores():
    scores = rrf([([("a", 1000), ("b", 1)], 0.5), ([("b", 0.8)], 0.5)])
    assert scores["b"] > scores["a"]
    assert rrf([([("x", 1)], 0)]) == {}


def test_mmr_diversifies_duplicate_vectors():
    vectors = {"a": np.array([1.0, 0]), "b": np.array([1.0, 0]), "c": np.array([0.6, 0.8])}
    assert diversify(["a", "b", "c"], vectors["a"], vectors, 2, 0.3) == ["a", "c"]
