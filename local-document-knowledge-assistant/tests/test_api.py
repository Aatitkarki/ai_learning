import json
import time

import pytest


def ready_index(client):
    document = client.post(
        "/api/documents",
        files={
            "file": (
                "policy.md",
                b"# Refunds\n\nApproved cancellation refunds take five business days. "
                b"Fees are five credits.",
            )
        },
    )
    assert document.status_code == 201
    response = client.post(
        "/api/indexes", json={"name": "API test", "document_ids": [document.json()["id"]]}
    )
    assert response.status_code == 202
    identifier = response.json()["id"]
    for _ in range(100):
        index = client.get(f"/api/indexes/{identifier}").json()
        if index["status"] in {"ready", "failed"}:
            break
        time.sleep(0.01)
    assert index["status"] == "ready", index
    return index, document.json()


def test_health_does_not_contact_models_or_databases(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["downloads_enabled"] is False
    assert len(response.json()["stores"]) == 5


def test_upload_build_search_stream_inspect_and_delete(client):
    index, document = ready_index(client)
    config = {"index_id": index["id"], "method": "hybrid"}
    result = client.post("/api/search", json={"question": "refund cancellation", "config": config})
    assert result.status_code == 200 and result.json()["hits"]
    chunks = client.get(f"/api/indexes/{index['id']}/chunks").json()
    assert chunks["total"] > 0 and chunks["chunks"][0]["page"] == 1
    stream = client.post("/api/chat", json={"question": "refund", "config": config})
    assert stream.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in stream.text.splitlines()]
    assert events[-1]["type"] == "complete"
    run_id = events[-1]["result"]["id"]
    review = client.post(
        f"/api/runs/{run_id}/review", json={"rating": "supported", "notes": "Checked"}
    )
    assert review.json()["review"]["rating"] == "supported"
    original = client.get(f"/api/documents/{document['id']}/source")
    assert original.status_code == 200 and b"Refunds" in original.content
    assert "text/plain" in original.headers["content-type"]
    assert client.delete(f"/api/documents/{document['id']}").status_code == 204
    assert client.get(f"/api/runs/{run_id}").status_code == 404
    assert client.get("/api/indexes").json() == []


@pytest.mark.parametrize("question", ["", "   ", "x" * 2001])
def test_input_limits(client, question):
    response = client.post(
        "/api/search", json={"question": question, "config": {"index_id": "anything"}}
    )
    assert response.status_code == 422


def test_unknown_index_and_invalid_chunk_parameters(client):
    assert client.get("/api/indexes/missing").status_code == 404
    index, _ = ready_index(client)
    assert client.get(f"/api/indexes/{index['id']}/chunks?offset=-1").status_code == 400
    assert client.get(f"/api/indexes/{index['id']}/chunks?limit=1000").status_code == 400


def test_browser_cross_origin_mutation_rejected(client):
    result = client.post("/api/documents/seed", headers={"Origin": "https://unrelated.example"})
    assert result.status_code == 403
    assert client.get("/api/documents").json() == []


def test_same_origin_api_documentation_can_submit_requests(client):
    result = client.post("/api/documents/seed", headers={"Origin": "http://testserver"})
    assert result.status_code == 201


def test_invalid_models_and_extra_fields_rejected(client):
    response = client.post(
        "/api/indexes",
        json={"name": "test", "document_ids": ["x"], "store": "pinecone", "download": True},
    )
    assert response.status_code == 422


def test_seed_is_idempotent(client):
    first = client.post("/api/documents/seed")
    second = client.post("/api/documents/seed")
    assert len(first.json()) == 12
    assert all(d.get("duplicate") for d in second.json())
    assert len(client.get("/api/documents").json()) == 12
