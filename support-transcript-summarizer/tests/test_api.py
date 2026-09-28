import pytest
from fastapi.testclient import TestClient

from summarizer.api import create_app
from summarizer.config import Settings
from summarizer.provider import ModelError
from tests.conftest import TRANSCRIPT


def create(client, title="Example"):
    response = client.post("/api/summaries", json={"transcript": TRANSCRIPT, "title": title})
    assert response.status_code == 201, response.text
    return response.json()


def test_create_edit_export_delete(client):
    record = create(client)
    record_id = record["id"]
    assert record["summary"]["delay_reason"]["value"] is None
    assert client.get("/api/summaries").json()[0]["id"] == record_id
    edited = client.patch(
        f"/api/summaries/{record_id}",
        json={"draft_markdown": "Human-reviewed handover", "revision": 1},
    )
    assert edited.status_code == 200
    assert edited.json()["summary"] == record["summary"]
    assert edited.json()["is_edited"] is True
    assert edited.json()["revision"] == 2
    stale = client.patch(
        f"/api/summaries/{record_id}", json={"draft_markdown": "Stale edit", "revision": 1}
    )
    assert stale.status_code == 409
    exported = client.get(f"/api/summaries/{record_id}/export").text
    assert "Human-edited draft" in exported and "L2: Agent:" in exported
    assert client.delete(f"/api/summaries/{record_id}").status_code == 204
    assert client.get(f"/api/summaries/{record_id}").status_code == 404
    assert client.get("/api/summaries").json() == []


@pytest.mark.parametrize("text", ["", " \n\t ", "x" * 6001, "x\n" * 81, "hi\x00there"])
def test_invalid_input_never_calls_model(client, provider, text):
    assert client.post("/api/summaries", json={"transcript": text}).status_code == 422
    assert provider.calls == 0


def test_request_validation_does_not_echo_transcript(client):
    response = client.post("/api/summaries", json={"transcript": "PRIVATE-TEXT", "title": ""})
    assert response.status_code == 422
    assert "PRIVATE-TEXT" not in response.text


def test_txt_upload_and_encoding(client):
    response = client.post(
        "/api/summaries/upload",
        files={"file": ("call.TXT", b"\xef\xbb\xbf" + TRANSCRIPT.encode(), "text/plain")},
    )
    assert response.status_code == 201
    assert response.json()["transcript"] == TRANSCRIPT
    assert (
        client.post("/api/summaries/upload", files={"file": ("call.pdf", b"text")}).status_code
        == 415
    )
    assert (
        client.post("/api/summaries/upload", files={"file": ("call.txt", b"\xff\xfe")}).status_code
        == 415
    )
    assert (
        client.post("/api/summaries/upload", files={"file": ("call.txt", b"x" * 24001)}).status_code
        == 413
    )
    assert (
        client.post("/api/summaries/upload", files={"file": ("call.txt", b"x" * 6001)}).status_code
        == 422
    )


def test_request_body_limit(client, provider):
    response = client.post(
        "/api/summaries", content=b"x" * 65537, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413
    assert provider.calls == 0


@pytest.mark.parametrize(
    "code,status",
    [("model_unavailable", 503), ("model_timeout", 504), ("invalid_model_output", 502)],
)
def test_failed_generation_not_saved(client, provider, code, status):
    provider.error = ModelError(code, "Safe error", status)
    response = client.post("/api/summaries", json={"transcript": TRANSCRIPT})
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert client.get("/api/summaries").json() == []


def test_history_persists_between_app_instances(tmp_path, provider):
    settings = Settings(database_path=tmp_path / "persist.sqlite3")
    with TestClient(create_app(settings, provider)) as first:
        record = create(first)
    with TestClient(create_app(settings, provider)) as second:
        assert second.get(f"/api/summaries/{record['id']}").json()["transcript"] == TRANSCRIPT


def test_pagination_and_missing_records(client):
    first = create(client, "First")
    second = create(client, "Second")
    assert client.get("/api/summaries?limit=1").json()[0]["id"] == second["id"]
    assert client.get("/api/summaries?limit=1&offset=1").json()[0]["id"] == first["id"]
    assert client.get("/api/summaries?limit=101").status_code == 422
    assert client.get("/api/summaries?offset=99999999999999999999999").status_code == 422
    assert (
        client.patch(
            "/api/summaries/missing", json={"draft_markdown": "Hello", "revision": 1}
        ).status_code
        == 404
    )
    assert client.delete("/api/summaries/missing").status_code == 404


def test_health_reports_provider(client):
    assert client.get("/health").json()["model_status"]["available"] is True
