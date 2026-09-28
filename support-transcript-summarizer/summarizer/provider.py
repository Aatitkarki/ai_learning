from dataclasses import dataclass
from time import perf_counter

import httpx
from pydantic import ValidationError

from summarizer.config import Settings
from summarizer.models import EvidenceError, Handover, Metrics, validate_evidence
from summarizer.prompts import PROMPT_VERSION, messages_for


class ModelError(Exception):
    def __init__(self, code: str, message: str, status: int = 502):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


@dataclass
class Generation:
    summary: Handover
    metrics: Metrics


class OllamaProvider:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        self.transport = transport

    def _client(self, timeout: float) -> httpx.Client:
        return httpx.Client(
            base_url=self.settings.ollama_base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=5),
            transport=self.transport,
            trust_env=False,
        )

    def health(self) -> dict:
        try:
            with self._client(3) as client:
                response = client.get("/api/tags")
                response.raise_for_status()
                names = {model["name"] for model in response.json()["models"]}
            available = self.settings.ollama_model in names
            return {
                "available": available,
                "message": "Ready" if available else "Configured model is not downloaded.",
            }
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return {"available": False, "message": "Ollama is unavailable. Start it and retry."}

    def generate(self, transcript: str) -> Generation:
        started = perf_counter()
        try:
            with self._client(self.settings.model_timeout_seconds) as client:
                response = client.post(
                    "/api/chat",
                    json={
                        "model": self.settings.ollama_model,
                        "messages": messages_for(transcript),
                        "format": Handover.model_json_schema(),
                        "stream": False,
                        "options": {
                            "temperature": 0,
                            "seed": 42,
                            "num_ctx": 16384,
                            "num_predict": 2400,
                        },
                        "keep_alive": "10m",
                    },
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ModelError(
                "model_timeout", "The model timed out. Try a shorter transcript.", 504
            ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise ModelError(
                    "model_missing",
                    "Model not found. Download the configured Ollama model and retry.",
                    503,
                ) from exc
            raise ModelError(
                "model_unavailable", "Ollama could not process this request.", 503
            ) from exc
        except httpx.RequestError as exc:
            raise ModelError(
                "model_unavailable", "Cannot reach Ollama. Start it and retry.", 503
            ) from exc
        try:
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Invalid provider envelope")
            if payload.get("done") is not True or payload.get("done_reason") == "length":
                raise ValueError("Incomplete model output")
            summary = Handover.model_validate_json(payload["message"]["content"])
            validate_evidence(summary, transcript)
            metrics = Metrics(
                model=self.settings.ollama_model,
                latency_ms=round((perf_counter() - started) * 1000, 2),
                input_tokens=payload.get("prompt_eval_count"),
                output_tokens=payload.get("eval_count"),
                prompt_version=PROMPT_VERSION,
            )
        except (ValueError, KeyError, TypeError, ValidationError, EvidenceError) as exc:
            # Never return an unvalidated response or store it as a successful handover.
            raise ModelError(
                "invalid_model_output",
                "The model returned incomplete or unsupported "
                "output. No handover was saved. Review the input and retry.",
            ) from exc
        return Generation(summary=summary, metrics=metrics)
