import json
import os

import httpx
import numpy as np
import pytest

from knowledge_lab.config import LabError
from knowledge_lab.models import IndexConfig, SearchConfig
from knowledge_lab.stores import PgvectorStore, QdrantStore, make_store


def test_qdrant_uses_explicit_cosine_vectors_and_uuid_ids(settings, mock_http):
    calls = []
    identifier = "6dbef03b-4fe9-438b-a028-f3b3fba4d675"

    def handler(request):
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, body))
        if request.url.path.endswith("/points/query"):
            return httpx.Response(
                200, json={"result": {"points": [{"id": identifier, "score": 1}]}}
            )
        return httpx.Response(200, json={"result": True})

    mock_http(handler)
    store = QdrantStore("ldk_" + "a" * 32, settings, None, "native")
    store.build([identifier], np.array([[1.0, 0.0]], dtype=np.float32))
    assert store.search(np.array([1.0, 0.0]), 3) == [(identifier, 1)]
    store.drop()
    assert calls[0][2]["vectors"] == {"size": 2, "distance": "Cosine"}
    assert calls[1][2]["points"][0] == {"id": identifier, "vector": [1.0, 0.0]}
    assert calls[-1][0] == "DELETE"


def test_qdrant_delete_missing_collection_is_idempotent(settings, mock_http):
    mock_http(lambda request: httpx.Response(404, json={"status": {"error": "Not found"}}))
    QdrantStore("ldk_" + "a" * 32, settings, None, "native").drop()


def test_sql_identifier_cannot_be_user_controlled(settings):
    with pytest.raises(LabError, match="identifier"):
        make_store(
            {"id": "x; DROP TABLE anything", "config": {"store": "pgvector"}}, settings, None
        )
    assert PgvectorStore.literal([0.25, -1]) == "[0.25,-1.0]"


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["sqlite", "faiss", "chroma", "qdrant", "pgvector"])
def test_opt_in_backend_contract(lab, backend):
    enabled = os.environ.get("LAB_TEST_STORES", "").split(",")
    if backend not in enabled:
        pytest.skip(f"Set LAB_TEST_STORES={backend} after installing or starting that backend.")
    doc = lab.upload(
        "contract.md",
        b"# Fees\n\nThe standard fee is five credits.\n\n"
        b"# Hours\n\nSupport operates Monday through Saturday.",
    )
    index = lab.create_index(
        IndexConfig(name="contract", document_ids=[doc["id"]], store=backend), background=False
    )
    assert index["status"] == "ready", index.get("error")
    result = lab.search(
        "standard fee five credits", SearchConfig(index_id=index["id"], method="vector", top_k=1)
    )
    assert "five credits" in result.hits[0].chunk.text
    lab.delete_index(index["id"])
    assert not lab.repo.chunks(index["id"])
