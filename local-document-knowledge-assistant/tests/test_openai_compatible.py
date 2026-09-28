import json

import httpx
import numpy as np
import pytest
from pydantic import SecretStr, ValidationError

from knowledge_lab.config import LabError, Settings
from knowledge_lab.embeddings import Embedder
from knowledge_lab.models import ChatRequest, IndexConfig, SearchConfig
from knowledge_lab.openai_compatible import sse_data


def configured(settings):
    settings.openai_base_url = "https://example.invalid/custom/v1"
    settings.openai_api_key = SecretStr("test-key-not-a-real-credential")
    settings.openai_chat_model = "chat-model"
    settings.openai_embed_model = "embedding-model"
    return settings


def stream_response(answer="A refund takes five business days. [S1]", finish="stop"):
    parts = [
        {"model": "resolved-model", "choices": [{"index": 0, "delta": {"role": "assistant"}}]},
        {"choices": [{"index": 0, "delta": {"content": answer}, "finish_reason": None}]},
    ]
    if finish:
        parts += [
            {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 40, "completion_tokens": 9, "total_tokens": 49},
            },
        ]
    data = ": keepalive\n\n" + "".join(f"data: {json.dumps(p)}\n\n" for p in parts)
    if finish:
        data += "data: [DONE]\n\n"
    return httpx.Response(200, text=data, headers={"content-type": "text/event-stream"})


@pytest.mark.parametrize(
    "url",
    [
        "",
        "https://provider.example/v1/",
        "https://gateway.example/api/v1",
        "http://localhost:1234/v1",
        "http://[::1]:8080/v1",
    ],
)
def test_api_base_url_accepts_blank_https_or_local_http(url):
    settings = Settings(_env_file=None, openai_base_url=url)
    assert settings.openai_base_url == url.rstrip("/")


@pytest.mark.parametrize(
    "url",
    [
        "http://provider.example/v1",
        "ftp://localhost/v1",
        "https://user:secret@provider.example/v1",
        "https://provider.example/v1?api_key=secret",
        "https://provider.example/v1#key",
        "https://provider.example/v1/chat/completions",
        "https://provider.example/v1/embeddings",
        "https://provider.example/v1/responses",
    ],
)
def test_unsafe_or_endpoint_urls_are_rejected(url):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, openai_base_url=url)


def test_startup_status_never_contacts_provider_or_exposes_key(lab):
    configured(lab.settings)
    health = lab.catalog()
    assert health["openai"]["base_url_configured"]
    assert health["openai"]["api_key_configured"]
    for text in [json.dumps(health), lab.settings.model_dump_json(), repr(lab.settings)]:
        assert "test-key-not-a-real-credential" not in text


def test_missing_configuration_leaves_evidence_mode_usable(indexed):
    lab, index, _ = indexed
    request = ChatRequest(
        question="refund",
        mode="openai_compatible",
        config={"index_id": index["id"], "method": "bm25"},
    )
    assert "LAB_OPENAI_BASE_URL" in list(lab.chat(request))[-1]["message"]
    assert list(lab.chat(request.model_copy(update={"mode": "evidence"})))[-1]["type"] == "complete"


def test_api_chat_stream_authentication_usage_and_local_citations(indexed, mock_http):
    lab, index, _ = indexed
    configured(lab.settings)

    def handler(request):
        assert str(request.url) == "https://example.invalid/custom/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key-not-a-real-credential"
        payload = json.loads(request.content)
        assert payload["model"] == "chat-model"
        assert payload["stream"] and payload["stream_options"] == {"include_usage": True}
        assert payload["max_completion_tokens"] == 1000
        assert "temperature" not in payload and "max_tokens" not in payload
        assert "untrusted" in payload["messages"][0]["content"]
        assert json.loads(payload["messages"][1]["content"])["evidence"]
        return stream_response()

    mock_http(handler)
    events = list(
        lab.chat(
            ChatRequest(
                question="refund",
                mode="openai_compatible",
                config={"index_id": index["id"], "method": "bm25"},
            )
        )
    )
    assert events[-1]["type"] == "complete"
    result = events[-1]["result"]
    assert result["status"] == "answered" and result["citation_valid"]
    assert result["model_identity"].startswith("openai-compatible:chat:")
    assert result["usage"]["eval_count"] == 9
    assert result["usage"]["response_model"] == "resolved-model"
    assert result["ttft_ms"] is not None
    assert "test-key-not-a-real-credential" not in json.dumps(lab.repo.list("runs"))
    assert "https://example.invalid" not in json.dumps(lab.repo.list("runs"))


def test_legacy_provider_options_and_keyless_local_server(indexed, mock_http):
    lab, index, _ = indexed
    configured(lab.settings)
    lab.settings.openai_base_url = "http://localhost:1234/v1"
    lab.settings.openai_api_key = SecretStr("")
    lab.settings.openai_token_parameter = "max_tokens"
    lab.settings.openai_send_temperature = True
    lab.settings.openai_stream_usage = False

    def handler(request):
        assert "authorization" not in request.headers
        body = json.loads(request.content)
        assert body["max_tokens"] == 1000 and body["temperature"] == 0.25
        assert "max_completion_tokens" not in body and "stream_options" not in body
        return stream_response()

    mock_http(handler)
    events = list(
        lab.chat(
            ChatRequest(
                question="refund",
                mode="openai_compatible",
                temperature=0.25,
                config={"index_id": index["id"], "method": "bm25"},
            )
        )
    )
    assert events[-1]["type"] == "complete"


@pytest.mark.parametrize(
    "status,phrase",
    [
        (401, "authentication"),
        (403, "authentication"),
        (429, "quota"),
        (404, "not found"),
        (500, "HTTP 500"),
        (302, "redirects"),
    ],
)
def test_provider_errors_are_actionable_and_do_not_echo_secrets(indexed, mock_http, status, phrase):
    lab, index, _ = indexed
    configured(lab.settings)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            status,
            json={"error": "private text and test-key-not-a-real-credential"},
            headers={"location": "https://elsewhere.invalid/collect"},
        )

    mock_http(handler)
    events = list(
        lab.chat(
            ChatRequest(
                question="refund",
                mode="openai_compatible",
                config={"index_id": index["id"], "method": "bm25"},
            )
        )
    )
    assert events[-1]["type"] == "error" and phrase in events[-1]["message"]
    assert "private text" not in json.dumps(
        events
    ) and "test-key-not-a-real-credential" not in json.dumps(events)
    assert not lab.repo.list("runs") and len(calls) == 1


@pytest.mark.parametrize(
    "finish,status",
    [
        ("length", "truncated"),
        (None, "error"),
        ("content_filter", "error"),
        ("tool_calls", "error"),
    ],
)
def test_incomplete_filtered_or_truncated_api_output_is_not_accepted(
    indexed, mock_http, finish, status
):
    lab, index, _ = indexed
    configured(lab.settings)
    mock_http(lambda request: stream_response(finish=finish))
    event = list(
        lab.chat(
            ChatRequest(
                question="refund",
                mode="openai_compatible",
                config={"index_id": index["id"], "method": "bm25"},
            )
        )
    )[-1]
    if status == "error":
        assert event["type"] == "error" and not lab.repo.list("runs")
    else:
        assert event["result"]["status"] == status


def test_embedding_response_order_normalization_and_payload(settings, mock_http):
    configured(settings)

    def handler(request):
        assert request.url.path == "/custom/v1/embeddings"
        assert request.headers["authorization"].startswith("Bearer ")
        assert json.loads(request.content) == {
            "model": "embedding-model",
            "input": ["one", "two"],
            "encoding_format": "float",
        }
        return httpx.Response(
            200,
            json={"data": [{"index": 1, "embedding": [0, 2]}, {"index": 0, "embedding": [3, 0]}]},
        )

    mock_http(handler)
    embedder = Embedder(
        IndexConfig(name="api", document_ids=["a"], embedding="openai_compatible"), settings
    )
    assert embedder.identity().startswith("openai-compatible:embedding:")
    np.testing.assert_array_equal(embedder.embed(["one", "two"]), [[1, 0], [0, 1]])


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"index": 1, "embedding": [1, 0]}],
        [{"index": 0, "embedding": [float("nan"), 1]}],
        [{"index": 0, "embedding": [0, 0]}],
    ],
)
def test_invalid_api_embeddings_are_rejected(settings, mock_http, rows):
    configured(settings)
    # text= permits a malformed NaN fixture, which production normalization must reject.
    mock_http(lambda request: httpx.Response(200, text=json.dumps({"data": rows})))
    embedder = Embedder(
        IndexConfig(name="api", document_ids=["a"], embedding="openai_compatible"), settings
    )
    with pytest.raises(LabError):
        embedder.embed(["one"])


def test_api_index_freezes_model_and_rejects_changed_endpoint_or_revision(lab, mock_http):
    configured(lab.settings)

    def handler(request):
        assert request.url.path == "/custom/v1/embeddings"
        inputs = json.loads(request.content)["input"]
        return httpx.Response(
            200, json={"data": [{"index": i, "embedding": [1.0, 0.0]} for i in range(len(inputs))]}
        )

    mock_http(handler)
    doc = lab.upload("rule.md", b"Refunds take five days.")
    index = lab.create_index(
        IndexConfig(name="API", document_ids=[doc["id"]], embedding="openai_compatible"),
        background=False,
    )
    assert index["status"] == "ready", index
    assert index["config"]["embedding_model"] == "embedding-model"
    config = SearchConfig(index_id=index["id"], method="vector")
    assert lab.search("refund", config).hits
    original = lab.settings.openai_base_url
    lab.settings.openai_base_url = "https://different.invalid/v1"
    with pytest.raises(LabError, match="model changed"):
        lab.search("refund", config)
    lab.settings.openai_base_url = original
    lab.settings.openai_embed_revision = "new-version"
    with pytest.raises(LabError, match="model changed"):
        lab.search("refund", config)


def test_sse_multiline_json_comments_and_event_fields():
    frames = list(
        sse_data([": heartbeat", "", "event: message", "data: {", 'data: "choices": []}', ""])
    )
    assert json.loads(frames[0]) == {"choices": []}
