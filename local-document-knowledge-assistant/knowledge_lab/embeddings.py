import hashlib
import re

import httpx
import numpy as np

from . import openai_compatible
from .config import LabError, Settings
from .models import IndexConfig

STOPWORDS = set(
    "a an the is are was were be been do does did to of in on at for from with and or "
    "what which who how when where can could should would i we you it my me our "
    "this that these those please about have has had by as".split()
)


def tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"\w+", text.casefold()) if w not in STOPWORDS]


def normalized(vectors, expected_count: int) -> np.ndarray:
    try:
        array = np.asarray(vectors, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise LabError("Embedding provider returned malformed vectors.", 502) from exc
    if array.ndim != 2 or array.shape[0] != expected_count or not array.shape[1]:
        raise LabError("Embedding provider returned the wrong vector shape.", 502)
    if not np.isfinite(array).all():
        raise LabError("Embedding provider returned non-finite values.", 502)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if (norms == 0).any():
        raise LabError("Cannot embed text without searchable terms.")
    return array / norms


class Embedder:
    def __init__(self, config: IndexConfig, settings: Settings):
        self.config, self.settings = config, settings

    def identity(self) -> str:
        if self.config.embedding == "hash":
            return f"sha256-word-hash-v1:{self.config.hash_dimensions}"
        if self.config.embedding == "openai_compatible":
            model = self.config.embedding_model or self.settings.openai_embed_model
            return openai_compatible.identity(self.settings, model, "embedding")
        model = self.config.embedding_model or self.settings.embed_model
        if not model:
            raise LabError("Set an already-installed Ollama embedding model before building.")
        try:
            with httpx.Client(timeout=self.settings.request_timeout, trust_env=False) as client:
                response = client.get(f"{self.settings.ollama_url}/api/tags")
                response.raise_for_status()
                found = next(
                    (
                        m
                        for m in response.json()["models"]
                        if m["name"] in {model, f"{model}:latest"}
                    ),
                    None,
                )
                if not found or not found.get("digest"):
                    raise LabError(
                        f"Embedding model '{model}' is not installed. No download made.", 503
                    )
                return f"ollama:{found['name']}:{found['digest']}"
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise LabError("Could not inspect the local Ollama models.", 503) from exc

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.config.embedding == "hash":
            vectors = np.zeros((len(texts), self.config.hash_dimensions), dtype=np.float32)
            for row, text in enumerate(texts):
                for token in tokens(text):
                    digest = hashlib.sha256(token.encode()).digest()
                    index = int.from_bytes(digest[:4], "little") % self.config.hash_dimensions
                    vectors[row, index] += 1 if digest[4] % 2 else -1
            return normalized(vectors, len(texts))
        if self.config.embedding == "openai_compatible":
            model = self.config.embedding_model or self.settings.openai_embed_model
            return normalized(openai_compatible.embed(self.settings, model, texts), len(texts))
        model = self.config.embedding_model or self.settings.embed_model
        batches = []
        try:
            with httpx.Client(timeout=self.settings.request_timeout, trust_env=False) as client:
                for start in range(0, len(texts), 32):
                    batch = texts[start : start + 32]
                    response = client.post(
                        f"{self.settings.ollama_url}/api/embed",
                        json={"model": model, "input": batch, "truncate": False},
                    )
                    response.raise_for_status()
                    batches.append(normalized(response.json()["embeddings"], len(batch)))
            return np.concatenate(batches)
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise LabError(
                "Local embedding request failed. Check the installed model and its "
                "context limit; no model was downloaded.",
                503,
            ) from exc
