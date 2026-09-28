"""Explicit OpenAI-compatible HTTP requests; no SDK, discovery calls, or model downloads."""

import hashlib
import json

import httpx

from .config import LabError


def identity(settings, model, kind):
    if not settings.openai_base_url or not model:
        variable = "LAB_OPENAI_CHAT_MODEL" if kind == "chat" else "LAB_OPENAI_EMBED_MODEL"
        raise LabError(f"Configure LAB_OPENAI_BASE_URL and {variable} in .env first.", 503)
    # Keep endpoint identity in reproducible runs without exporting private URLs or credentials.
    endpoint_hash = hashlib.sha256(settings.openai_base_url.encode()).hexdigest()
    revision = settings.openai_embed_revision if kind == "embedding" else ""
    return f"openai-compatible:{kind}:{endpoint_hash}:{model}:{revision}"


def headers(settings):
    key = settings.openai_api_key.get_secret_value().strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def check_response(response):
    status = response.status_code
    if 200 <= status < 300:
        return
    if status in {401, 403}:
        message = "API authentication failed. Check LAB_OPENAI_API_KEY and model access."
    elif status == 429:
        message = "API rate or quota limit reached. Retry later or check your provider account."
    elif status == 404:
        message = "API endpoint or model not found. Check the base URL and configured model name."
    elif 300 <= status < 400:
        message = "API redirects are disabled. Configure the final API base URL directly."
    else:
        message = f"API request failed (HTTP {status}). Check the model and compatibility settings."
    # Do not persist provider response bodies: they can echo keys or document text.
    raise LabError(message, 503 if status in {401, 403, 429} or status >= 500 else 502)


def sse_data(lines):
    """Parse SSE data frames including comments, CRLF, and multiline JSON events."""
    parts, size = [], 0
    for line in lines:
        if not line:
            if parts:
                yield "\n".join(parts)
            parts, size = [], 0
        elif line.startswith("data:"):
            value = line[5:].removeprefix(" ")
            size += len(value)
            if size > 1_000_000:
                raise LabError("API returned an oversized stream event.", 502)
            parts.append(value)
    if parts:
        yield "\n".join(parts)


def embed(settings, model, texts):
    identity(settings, model, "embedding")
    vectors = []
    try:
        with httpx.Client(
            timeout=settings.request_timeout,
            trust_env=False,
            follow_redirects=False,
            headers=headers(settings),
        ) as client:
            for start in range(0, len(texts), 32):
                batch = texts[start : start + 32]
                response = client.post(
                    f"{settings.openai_base_url}/embeddings",
                    json={"model": model, "input": batch, "encoding_format": "float"},
                )
                check_response(response)
                payload = response.json()
                rows = payload["data"]
                if len(rows) != len(batch) or any(type(row["index"]) is not int for row in rows):
                    raise ValueError("wrong embedding count or index type")
                ordered = sorted(rows, key=lambda row: row["index"])
                if [row["index"] for row in ordered] != list(range(len(batch))):
                    raise ValueError("duplicate or missing embedding index")
                vectors.extend(row["embedding"] for row in ordered)
        return vectors
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise LabError(
            "API embedding request failed or returned malformed vectors. "
            "Check the endpoint's embeddings support and configured model.",
            502,
        ) from exc


def chat_stream(settings, messages, request):
    model = settings.openai_chat_model
    configured_identity = identity(settings, model, "chat")
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        settings.openai_token_parameter: request.max_tokens,
    }
    if settings.openai_send_temperature:
        payload["temperature"] = request.temperature
    if settings.openai_stream_usage:
        payload["stream_options"] = {"include_usage": True}
    yield {"type": "model", "identity": configured_identity}
    try:
        with (
            httpx.Client(
                timeout=settings.request_timeout,
                trust_env=False,
                follow_redirects=False,
                headers=headers(settings),
            ) as client,
            client.stream(
                "POST", f"{settings.openai_base_url}/chat/completions", json=payload
            ) as response,
        ):
            check_response(response)
            finish_reason, usage, response_model, system_fingerprint = None, {}, None, None
            for frame in sse_data(response.iter_lines()):
                if frame.strip() == "[DONE]":
                    break
                part = json.loads(frame)
                if part.get("error"):
                    raise LabError("The API reported an error during generation.", 502)
                if part.get("model"):
                    response_model = part["model"]
                if part.get("system_fingerprint"):
                    system_fingerprint = part["system_fingerprint"]
                if part.get("usage"):
                    usage = part["usage"]
                for choice in part.get("choices", []):
                    if choice.get("index", 0) != 0:
                        continue
                    delta = choice.get("delta") or {}
                    if delta.get("refusal"):
                        raise LabError(
                            "The API declined to generate an answer for this request.", 502
                        )
                    token = delta.get("content")
                    if token is not None:
                        if not isinstance(token, str):
                            raise ValueError("non-text output")
                        if token:
                            yield {"type": "token", "text": token}
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
            if finish_reason not in {"stop", "length"}:
                raise LabError(
                    "API response did not complete as a text answer. "
                    "It may have been interrupted or filtered.",
                    502,
                )
            yield {
                "type": "usage",
                "usage": {
                    "prompt_eval_count": usage.get("prompt_tokens"),
                    "eval_count": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "done_reason": finish_reason,
                    "response_model": response_model,
                    "system_fingerprint": system_fingerprint,
                    "token_parameter": settings.openai_token_parameter,
                    "temperature_sent": settings.openai_send_temperature,
                },
            }
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise LabError(
            "API generation failed or returned an invalid stream. Check the URL "
            "and Chat Completions streaming support.",
            502,
        ) from exc
