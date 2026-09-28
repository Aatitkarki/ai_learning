import json

import httpx
import pytest

from summarizer.config import Settings
from summarizer.models import Handover
from summarizer.provider import ModelError, OllamaProvider
from tests.conftest import TRANSCRIPT


def test_real_adapter_request_and_response_contract(summary):
    def handle(request):
        payload = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert payload["stream"] is False
        assert payload["format"] == Handover.model_json_schema()
        assert payload["options"]["temperature"] == 0
        assert '"line": 2' in payload["messages"][-1]["content"]
        assert "Agent: I checked TX-01." in payload["messages"][-1]["content"]
        return httpx.Response(
            200,
            json={
                "done": True,
                "done_reason": "stop",
                "message": {"content": summary.model_dump_json()},
                "prompt_eval_count": 101,
                "eval_count": 99,
            },
        )

    provider = OllamaProvider(Settings(), httpx.MockTransport(handle))
    result = provider.generate(TRANSCRIPT)
    assert result.summary == summary
    assert result.metrics.input_tokens == 101
    assert result.metrics.output_tokens == 99
    assert result.metrics.latency_ms >= 0


@pytest.mark.parametrize(
    "status,body,code",
    [
        (404, {}, "model_missing"),
        (500, {"error": "PRIVATE PROVIDER DETAILS"}, "model_unavailable"),
        (200, {"done": True, "message": {"content": "not JSON"}}, "invalid_model_output"),
        (200, {"done": True, "message": {"content": "{}"}}, "invalid_model_output"),
        (200, {"done": False, "message": {"content": "{}"}}, "invalid_model_output"),
        (200, {"done": True, "done_reason": "length"}, "invalid_model_output"),
        (200, [], "invalid_model_output"),
    ],
)
def test_provider_errors(status, body, code):
    provider = OllamaProvider(
        Settings(), httpx.MockTransport(lambda _: httpx.Response(status, json=body))
    )
    with pytest.raises(ModelError) as caught:
        provider.generate(TRANSCRIPT)
    assert caught.value.code == code
    assert "PRIVATE" not in str(caught.value)


@pytest.mark.parametrize(
    "exception,code",
    [(httpx.ConnectError, "model_unavailable"), (httpx.ReadTimeout, "model_timeout")],
)
def test_connection_and_timeout(exception, code):
    def handle(request):
        raise exception("connection failed", request=request)

    provider = OllamaProvider(Settings(), httpx.MockTransport(handle))
    with pytest.raises(ModelError) as caught:
        provider.generate(TRANSCRIPT)
    assert caught.value.code == code


def test_invalid_evidence_never_reaches_display(summary):
    summary.amount.value = "QAR 999"
    provider = OllamaProvider(
        Settings(),
        httpx.MockTransport(
            lambda _: httpx.Response(
                200, json={"done": True, "message": {"content": summary.model_dump_json()}}
            )
        ),
    )
    with pytest.raises(ModelError, match="unsupported"):
        provider.generate(TRANSCRIPT)


def test_health_checks_model_presence():
    provider = OllamaProvider(
        Settings(),
        httpx.MockTransport(
            lambda _: httpx.Response(200, json={"models": [{"name": "unrelated-model:latest"}]})
        ),
    )
    assert provider.health()["available"] is False
