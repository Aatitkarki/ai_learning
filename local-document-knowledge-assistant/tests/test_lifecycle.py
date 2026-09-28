import pytest

from knowledge_lab.config import LabError
from knowledge_lab.models import ChatRequest, IndexConfig, SearchConfig
from knowledge_lab.repository import Repository


def test_dedup_and_safe_filename(lab):
    first = lab.upload("../../policy.md", b"A useful policy about refunds.")
    second = lab.upload("renamed.md", b"A useful policy about refunds.")
    assert first["filename"] == "policy.md"
    assert first["id"] == second["id"]
    assert second["duplicate"]
    assert len(lab.repo.list("documents")) == 1


def test_new_snapshot_does_not_replace_old_index(indexed):
    lab, original, docs = indexed
    new = lab.create_index(
        IndexConfig(
            name="sentence",
            document_ids=[d["id"] for d in docs],
            chunking={"strategy": "sentence", "size": 300, "overlap": 60},
        ),
        background=False,
    )
    assert new["id"] != original["id"]
    assert new["corpus_fingerprint"] == original["corpus_fingerprint"]
    assert lab.require("indexes", original["id"])["status"] == "ready"
    assert not (
        {r["chunk"]["id"] for r in lab.repo.chunks(new["id"])}
        & {r["chunk"]["id"] for r in lab.repo.chunks(original["id"])}
    )


def test_failed_index_is_not_searchable_and_cleans_up(lab, monkeypatch):
    doc = lab.upload("a.md", b"Policy refunds are reviewed in five business days.")

    def fail(*args):
        raise LabError("Simulated backend outage", 503)

    monkeypatch.setattr("knowledge_lab.service.Embedder.embed", fail)
    index = lab.create_index(IndexConfig(name="fails", document_ids=[doc["id"]]), background=False)
    assert index["status"] == "failed"
    assert index["error"] == "Simulated backend outage"
    assert not lab.repo.chunks(index["id"])
    with pytest.raises(LabError, match="finish building"):
        lab.search("policy", SearchConfig(index_id=index["id"]))


def test_document_deletion_removes_indexes_sources_and_related_history(indexed):
    lab, index, docs = indexed
    list(lab.chat(ChatRequest(question="standard transfer fee", config={"index_id": index["id"]})))
    assert lab.repo.list("runs") and lab.repo.list("conversations")
    lab.delete_document(docs[0]["id"])
    assert lab.repo.source(docs[0]["id"]) is None
    assert not lab.repo.list("indexes")
    assert not lab.repo.chunks(index["id"])
    assert not lab.repo.list("runs") and not lab.repo.list("conversations")
    assert len(lab.repo.list("documents")) == len(docs) - 1


def test_recovery_marks_interrupted_jobs_failed(settings):
    repo = Repository(settings.data_dir / "recovery.sqlite3")
    repo.put("indexes", {"id": "a", "status": "building", "stage": "storing"})
    repo.put("evaluations", {"id": "b", "status": "running"})
    repo.recover()
    assert repo.get("indexes", "a")["status"] == "failed"
    assert repo.get("indexes", "a")["cleanup_pending"]
    assert repo.get("evaluations", "b")["status"] == "failed"


def test_compare_preserves_partial_failure(indexed):
    lab, index, _ = indexed
    comparison = lab.compare(
        "transfer fees", [SearchConfig(index_id=index["id"]), SearchConfig(index_id="missing")]
    )
    assert comparison["results"][0]["ok"]
    assert not comparison["results"][1]["ok"]
