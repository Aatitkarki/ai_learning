import hashlib
import importlib.metadata
import importlib.util
import json
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

from .config import LabError, Settings
from .embeddings import Embedder, tokens
from .generation import (
    ABSTENTION,
    PROMPT_VERSION,
    context_sources,
    generate_stream,
    validate_citations,
)
from .ingestion import chunk_document, extract
from .models import ChatRequest, IndexConfig, SearchConfig
from .repository import Repository, now
from .retrieval import search
from .stores import make_store

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Lab:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.repo = Repository(settings.data_dir / "lab.sqlite3")
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lab-job")
        self.repo.recover()

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=False)

    def require(self, kind, identifier):
        item = self.repo.get(kind, identifier)
        if not item:
            raise LabError(f"{kind.capitalize().rstrip('s')} not found.", 404)
        return item

    def catalog(self):
        optional = {
            "pdf": "pypdf",
            "chroma": "chromadb",
            "faiss": "faiss",
            "pgvector": "psycopg",
            "langchain": "langchain_text_splitters",
        }
        return {
            "version": "0.1.0",
            "downloads_enabled": False,
            "optional_packages": {
                k: importlib.util.find_spec(v) is not None for k, v in optional.items()
            },
            "chat_model": self.settings.chat_model,
            "embedding_model": self.settings.embed_model,
            "openai": {
                "base_url_configured": bool(self.settings.openai_base_url),
                "api_key_configured": bool(self.settings.openai_api_key.get_secret_value()),
                "chat_model": self.settings.openai_chat_model,
                "embedding_model": self.settings.openai_embed_model,
                "send_temperature": self.settings.openai_send_temperature,
            },
            "stores": ["sqlite", "pgvector", "chroma", "qdrant", "faiss"],
            "methods": ["bm25", "vector", "hybrid", "mmr"],
            "chunkers": ["fixed", "sentence", "markdown", "langchain"],
            "note": "Service connections are checked only when explicitly used.",
        }

    def upload(self, filename: str, content: bytes):
        filename = Path(filename.replace("\\", "/")).name
        if not filename or len(filename) > 180 or any(ord(c) < 32 for c in filename):
            raise LabError("Use a filename of 1–180 printable characters.")
        digest = hashlib.sha256(content).hexdigest()
        with self.lock:
            existing = next((d for d in self.repo.list("documents") if d["sha256"] == digest), None)
            if existing:
                return {**existing, "duplicate": True}
            pages = extract(filename, content, self.settings)
            record = {
                "id": uuid.uuid4().hex,
                "filename": filename,
                "sha256": digest,
                "bytes": len(content),
                "created_at": now(),
                "status": "ready",
                "pages": [p.model_dump() for p in pages],
                "page_count": len(pages),
                "characters": sum(len(p.text) for p in pages),
                "warnings": (
                    ["Some pages contain no text; scanned content was not OCR'd."]
                    if any(not p.text.strip() for p in pages)
                    else []
                ),
            }
            self.repo.add_document(record, content)
            return record

    def seed(self):
        return [
            self.upload(path.name, path.read_bytes())
            for path in sorted((PROJECT_ROOT / "examples" / "policies").glob("*.md"))
        ]

    def create_index(self, config: IndexConfig, background=True):
        with self.lock:
            documents = [self.require("documents", key) for key in config.document_ids]
            if config.embedding == "ollama" and not config.embedding_model:
                config = config.model_copy(update={"embedding_model": self.settings.embed_model})
            if config.embedding == "openai_compatible" and not config.embedding_model:
                config = config.model_copy(
                    update={"embedding_model": self.settings.openai_embed_model}
                )
            fingerprint = hashlib.sha256(
                json.dumps(sorted((d["filename"], d["sha256"]) for d in documents)).encode()
            ).hexdigest()
            record = {
                "id": uuid.uuid4().hex,
                "created_at": now(),
                "status": "queued",
                "config": config.model_dump(),
                "corpus_fingerprint": fingerprint,
                "document_manifest": [
                    {k: d[k] for k in ("id", "filename", "sha256")} for d in documents
                ],
                "chunk_count": 0,
                "dimensions": None,
                "error": None,
            }
            self.repo.put("indexes", record)
        if background:
            self.pool.submit(self.build_index, record["id"])
        else:
            self.build_index(record["id"])
        return self.require("indexes", record["id"])

    def build_index(self, identifier):
        started = perf_counter()
        with self.lock:
            record = self.require("indexes", identifier)
            record.update(status="building", stage="chunking")
            self.repo.put("indexes", record)
            store = None
            try:
                config = IndexConfig(**record["config"])
                chunks = []
                for key in config.document_ids:
                    chunks.extend(
                        chunk_document(self.require("documents", key), identifier, config.chunking)
                    )
                # Punctuation-only headings are not searchable and cannot produce valid vectors.
                chunks = [c for c in chunks if tokens(c.text)]
                if not chunks or len(chunks) > self.settings.max_chunks:
                    raise LabError(f"Index needs 1–{self.settings.max_chunks} searchable chunks.")
                record.update(stage="embedding", chunk_count=len(chunks))
                self.repo.put("indexes", record)
                embedder = Embedder(config, self.settings)
                identity = embedder.identity()
                vectors = embedder.embed([c.text for c in chunks])
                if embedder.identity() != identity:
                    raise LabError("Embedding model changed during indexing. Build again.", 409)
                record.update(
                    stage="storing", dimensions=vectors.shape[1], embedding_identity=identity
                )
                self.repo.put("indexes", record)
                store = make_store(record, self.settings, self.repo)
                store.build([c.id for c in chunks], vectors)
                self.repo.save_chunks(identifier, chunks, vectors.tolist())
                record.update(
                    status="ready",
                    stage="complete",
                    finished_at=now(),
                    build_ms=(perf_counter() - started) * 1000,
                )
            except Exception as exc:
                self.repo.remove_chunks(identifier)
                record.update(
                    status="failed",
                    stage="failed",
                    error=(
                        exc.message
                        if isinstance(exc, LabError)
                        else "Index build failed. Check the optional package or local "
                        "database service."
                    ),
                )
                if store:
                    try:
                        store.drop()
                    except Exception:
                        record["cleanup_pending"] = True
                        record["error"] += (
                            " Backend cleanup is pending; delete this index to retry."
                        )
            self.repo.put("indexes", record)

    def search(self, question, config: SearchConfig, persist=True):
        with self.lock:
            result = search(self.repo, self.settings, question, config)
            if persist:
                self.repo.put(
                    "runs",
                    {
                        "id": uuid.uuid4().hex,
                        "type": "search",
                        "created_at": now(),
                        "index_ids": [config.index_id],
                        "result": result.model_dump(),
                    },
                )
            return result

    def compare(self, question, configs):
        results = []
        with self.lock:
            for config in configs:
                try:
                    results.append(
                        {
                            "ok": True,
                            "result": self.search(question, config, persist=False).model_dump(),
                        }
                    )
                except LabError as exc:
                    results.append(
                        {"ok": False, "config": config.model_dump(), "error": exc.message}
                    )
            fingerprints = {r["result"]["corpus_fingerprint"] for r in results if r["ok"]}
            record = {
                "id": uuid.uuid4().hex,
                "created_at": now(),
                "type": "compare",
                "question": question,
                "index_ids": [c.index_id for c in configs],
                "same_corpus": len(fingerprints) == 1,
                "results": results,
            }
            self.repo.put("runs", record)
            return record

    def delete_index(self, identifier):
        with self.lock:
            record = self.require("indexes", identifier)
            if record["status"] in {"queued", "building"}:
                raise LabError("Wait for this index build to finish before deleting it.", 409)
            if any(
                e["status"] in {"queued", "running"} and identifier in e["index_ids"]
                for e in self.repo.list("evaluations")
            ):
                raise LabError("Wait for evaluations using this index to finish.", 409)
            if record["status"] == "ready" or record.get("cleanup_pending"):
                try:
                    make_store(record, self.settings, self.repo).drop()
                except Exception as exc:
                    raise LabError(
                        "Cannot clean up the vector backend. Restore its service or "
                        "package and retry deletion.",
                        503,
                    ) from exc
            self.repo.remove_chunks(identifier)
            self.repo.delete("indexes", identifier)
            for kind in ("runs", "conversations", "evaluations"):
                for item in self.repo.list(kind):
                    if identifier in item.get("index_ids", []):
                        self.repo.delete(kind, item["id"])

    def delete_document(self, identifier):
        with self.lock:
            self.require("documents", identifier)
            affected = [
                i for i in self.repo.list("indexes") if identifier in i["config"]["document_ids"]
            ]
            if any(i["status"] in {"queued", "building"} for i in affected):
                raise LabError("Wait for indexes using this document to finish building.", 409)
            for index in affected:
                self.delete_index(index["id"])
            self.repo.remove_document(identifier)

    def chat(self, request: ChatRequest):
        started = perf_counter()
        try:
            with self.lock:
                index = self.require("indexes", request.config.index_id)
                conversation = (
                    self.require("conversations", request.conversation_id)
                    if request.conversation_id
                    else {
                        "id": uuid.uuid4().hex,
                        "created_at": now(),
                        "index_ids": [index["id"]],
                        "turns": [],
                    }
                )
                if conversation["index_ids"] != [index["id"]]:
                    raise LabError("Start a new conversation when switching indexes.", 409)
                previous = [t["question"] for t in conversation["turns"][-2:]]
                effective_query = request.question
                followup = re.search(
                    r"\b(it|its|that|they|those|them|these)\b|^and\b|^what about\b",
                    request.question,
                    re.IGNORECASE,
                )
                if request.use_followup_context and previous and followup:
                    effective_query = previous[-1] + "\n" + request.question
                retrieval = self.search(effective_query, request.config, persist=False)
                sources = context_sources(retrieval.hits, request.context_chars)
            yield {
                "type": "retrieval",
                "result": retrieval.model_dump(),
                "sources": sources,
                "conversation_id": conversation["id"],
            }
            answer, draft, identity, usage = "", "", None, {}
            first_token_ms = None
            citation_valid = True
            warning = ""
            if not sources:
                answer, status = ABSTENTION, "abstained"
                yield {"type": "token", "text": answer}
            elif request.mode == "evidence":
                answer = "Retrieved passages (verbatim excerpts; no AI answer generated):\n\n"
                answer += "\n\n".join(f"[{s['label']}] {s['text']}" for s in sources)
                status = "evidence_only"
                yield {"type": "token", "text": answer}
            else:
                for event in generate_stream(
                    self.settings,
                    request.question,
                    previous if request.use_followup_context else [],
                    sources,
                    request,
                ):
                    if event["type"] == "token":
                        if first_token_ms is None:
                            first_token_ms = (perf_counter() - started) * 1000
                        draft += event["text"]
                        if len(draft) > 50000:
                            raise LabError("Model response exceeded the output safety limit.", 502)
                        yield event
                    elif event["type"] == "model":
                        identity = event["identity"]
                    else:
                        usage = event["usage"]
                citation_valid, warning = validate_citations(draft, sources)
                answer = draft if citation_valid else ABSTENTION
                status = "abstained" if answer.strip() == ABSTENTION else "answered"
                if not citation_valid:
                    status = "citation_rejected"
                if usage.get("done_reason") == "length":
                    answer = ABSTENTION
                    status = "truncated"
                    warning = (
                        "The model reached its output limit. Its draft was withheld; "
                        "reduce the question scope or increase the output limit."
                    )
            result = {
                "id": uuid.uuid4().hex,
                "type": "chat",
                "created_at": now(),
                "index_ids": [index["id"]],
                "question": request.question,
                "answer": answer,
                "draft": draft,
                "status": status,
                "sources": sources,
                "mode": request.mode,
                "request": request.model_dump(),
                "retrieval": retrieval.model_dump(),
                "model_identity": identity,
                "prompt_version": PROMPT_VERSION,
                "citation_valid": citation_valid,
                "warning": warning,
                "usage": usage,
                "review": None,
                "ttft_ms": first_token_ms,
                "total_ms": (perf_counter() - started) * 1000,
                "conversation_id": conversation["id"],
            }
            # A synchronous streaming iterator may resume on different worker threads.
            # Never hold a thread-owned lock across a yield or a model network request.
            with self.lock:
                self.require("indexes", index["id"])
                if request.conversation_id:
                    current = self.require("conversations", conversation["id"])
                    if current.get("revision", 0) != conversation.get("revision", 0):
                        raise LabError(
                            "This conversation changed during generation. Retry the question.", 409
                        )
                conversation["turns"].append(
                    {"question": request.question, "answer": answer, "run_id": result["id"]}
                )
                conversation["turns"] = conversation["turns"][-30:]
                conversation["revision"] = conversation.get("revision", 0) + 1
                result["total_ms"] = (perf_counter() - started) * 1000
                self.repo.save_chat(result, conversation)
            yield {"type": "complete", "result": result}
        except LabError as exc:
            yield {"type": "error", "message": exc.message}

    def create_evaluation(self, request):
        with self.lock:
            for config in request.configs:
                index = self.require("indexes", config.index_id)
                if index["status"] != "ready":
                    raise LabError("Evaluations require ready indexes.", 409)
            record = {
                "id": uuid.uuid4().hex,
                "created_at": now(),
                "status": "queued",
                "index_ids": [c.index_id for c in request.configs],
                "request": request.model_dump(),
                "progress": 0,
            }
            self.repo.put("evaluations", record)
        self.pool.submit(self.run_evaluation, record["id"])
        return record

    def run_evaluation(self, identifier):
        from evaluation.runner import evaluate

        with self.lock:
            record = self.require("evaluations", identifier)
            record["status"] = "running"
            self.repo.put("evaluations", record)
            try:
                report = evaluate(
                    self,
                    record["request"],
                    lambda progress: self.repo.put("evaluations", {**record, "progress": progress}),
                )
                record.update(status="complete", report=report, progress=1, finished_at=now())
            except LabError as exc:
                record.update(status="failed", error=exc.message)
            except Exception:
                record.update(
                    status="failed", error="Evaluation failed. Check the corpus and setup."
                )
            self.repo.put("evaluations", record)
