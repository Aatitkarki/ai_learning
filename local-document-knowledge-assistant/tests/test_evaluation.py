import json
from collections import Counter

import pytest

from evaluation.runner import ROOT, metrics, summarize
from knowledge_lab.models import Chunk, Hit


def hit(filename, text, identifier="1"):
    return Hit(
        chunk=Chunk(
            id=identifier,
            document_id="d",
            filename=filename,
            page=1,
            ordinal=0,
            start=0,
            end=len(text),
            text=text,
        ),
        score=1,
    )


def test_duplicate_passages_do_not_inflate_recall_or_ndcg():
    expected = [
        {"filename": "p.md", "quote": "first rule"},
        {"filename": "q.md", "quote": "second rule"},
    ]
    result = metrics([hit("p.md", "first rule"), hit("p.md", "first rule", "2")], expected)
    assert result["passage_recall"] == 0.5
    assert result["reciprocal_rank"] == 1
    assert 0 < result["ndcg"] < 1


def test_unsupported_empty_result_is_a_separate_metric():
    result = metrics([], [])
    assert result["passage_recall"] is None and result["empty_result"]
    summary = summarize([{"kind": "unsupported", "metrics": result, "latency_ms": 4.0}])
    assert summary["unsupported_empty_rate"] == 1
    assert summary["passage_recall_at_k"] is None


def test_metrics_expected_ranks_and_missing_passages():
    expected = [{"filename": "policy.md", "quote": "golden passage"}]
    result = metrics([hit("noise.md", "irrelevant"), hit("policy.md", "golden passage")], expected)
    assert result["passage_recall"] == 1
    assert result["reciprocal_rank"] == 0.5
    assert result["ndcg"] == pytest.approx(1 / 1.584962500721156)


def test_fixed_dataset_has_real_passages_and_expected_distribution():
    dataset = json.loads((ROOT / "evaluation/cases.json").read_text())
    cases = dataset["cases"]
    assert len(cases) == 40 and len({c["id"] for c in cases}) == 40
    assert Counter(c["kind"] for c in cases) == {
        "answerable": 25,
        "unsupported": 10,
        "conflict": 2,
        "injection": 3,
    }
    assert {c["split"] for c in cases} == {"dev", "holdout"}
    for case in cases:
        for support in case["support"]:
            text = (ROOT / "examples/policies" / support["filename"]).read_text()
            assert support["quote"] in text


def test_background_evaluation_reports_metrics_configs_and_splits(indexed, tmp_path, monkeypatch):
    from knowledge_lab.models import EvaluationRequest

    lab, index, docs = indexed
    root = tmp_path / "fixture"
    (root / "evaluation").mkdir(parents=True)
    (root / "examples" / "policies").mkdir(parents=True)
    doc = next(d for d in docs if d["filename"] == "03-fees.md")
    (root / "examples" / "policies" / doc["filename"]).write_bytes(lab.repo.source(doc["id"]))
    data = {
        "version": "test",
        "cases": [
            {
                "id": "1",
                "question": "standard transfer fee",
                "kind": "answerable",
                "split": "dev",
                "expected_behavior": "Cite the fee",
                "support": [
                    {
                        "filename": "03-fees.md",
                        "quote": "The standard transfer fee is 5 demo credits per transfer.",
                    }
                ],
            },
            {
                "id": "2",
                "question": "xyzzys",
                "kind": "unsupported",
                "split": "holdout",
                "expected_behavior": "Abstain",
                "support": [],
            },
        ],
    }
    (root / "evaluation" / "cases.json").write_text(json.dumps(data))
    monkeypatch.setattr("evaluation.runner.ROOT", root)
    job = lab.create_evaluation(
        EvaluationRequest(split="all", configs=[{"index_id": index["id"], "method": "bm25"}])
    )
    lab.pool.submit(lambda: None).result(timeout=5)
    complete = lab.require("evaluations", job["id"])
    assert complete["status"] == "complete", complete
    result = complete["report"]["results"][0]
    assert result["summary"]["passage_recall_at_k"] == 1
    assert result["summary"]["unsupported_empty_rate"] == 1
    assert set(result["by_split"]) == {"dev", "holdout"}
    assert result["index"]["corpus_fingerprint"] == index["corpus_fingerprint"]
