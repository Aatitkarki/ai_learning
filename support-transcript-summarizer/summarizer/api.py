import sqlite3
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from summarizer.config import Settings
from summarizer.middleware import BodyLimitMiddleware
from summarizer.models import (
    MAX_UPLOAD_BYTES,
    DraftUpdate,
    HistoryItem,
    SavedSummary,
    SummaryRequest,
)
from summarizer.provider import ModelError, OllamaProvider
from summarizer.rendering import export_markdown
from summarizer.repository import Repository, RevisionConflict


def create_app(settings: Settings | None = None, provider=None) -> FastAPI:
    settings = settings or Settings()
    repository = Repository(settings.database_path)
    provider = provider or OllamaProvider(settings)

    @asynccontextmanager
    async def lifespan(app):
        repository.initialize()
        yield

    app = FastAPI(title="Support Transcript Summarizer", version="0.1.0", lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware)
    app.state.repository = repository

    @app.exception_handler(ModelError)
    async def model_error(_request, exc):
        return JSONResponse(
            status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message}}
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, exc):
        # Do not echo the submitted transcript into error responses or logs.
        details = [
            {"field": ".".join(str(x) for x in err["loc"]), "message": err["msg"]}
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "invalid_input",
                    "message": "Check the input and try again.",
                    "details": details,
                }
            },
        )

    @app.exception_handler(sqlite3.Error)
    async def storage_error(_request, _exc):
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "storage_unavailable",
                    "message": "Local history is unavailable. Retry later.",
                }
            },
        )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "provider": "ollama",
            "model": settings.ollama_model,
            "model_status": provider.health(),
        }

    @app.post("/api/summaries", response_model=SavedSummary, status_code=201)
    def summarize(request: SummaryRequest):
        return repository.create(request, provider.generate(request.transcript))

    @app.post("/api/summaries/upload", response_model=SavedSummary, status_code=201)
    async def summarize_upload(
        file: Annotated[UploadFile, File()],
        title: Annotated[str, Form(max_length=100)] = "Uploaded conversation",
    ):
        try:
            if not file.filename or not file.filename.lower().endswith(".txt"):
                raise HTTPException(415, "Upload a UTF-8 .txt file.")
            contents = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(contents) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"File exceeds {MAX_UPLOAD_BYTES:,} bytes.")
            try:
                transcript = contents.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise HTTPException(415, "Save the file with UTF-8 encoding and retry.") from exc
            try:
                request = SummaryRequest(transcript=transcript, title=title)
            except ValidationError as exc:
                raise RequestValidationError(exc.errors()) from exc
            return await run_in_threadpool(summarize, request)
        finally:
            await file.close()

    @app.get("/api/summaries", response_model=list[HistoryItem])
    def history(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0, le=1_000_000)):
        return repository.list(limit, offset)

    @app.get("/api/summaries/{record_id}", response_model=SavedSummary)
    def get_summary(record_id: str):
        record = repository.get(record_id)
        if not record:
            raise HTTPException(404, "Handover not found.")
        return record

    @app.patch("/api/summaries/{record_id}", response_model=SavedSummary)
    def update_draft(record_id: str, update: DraftUpdate):
        try:
            record = repository.update(record_id, update.draft_markdown, update.revision)
        except RevisionConflict as exc:
            raise HTTPException(409, "This handover changed. Reopen it before saving.") from exc
        if not record:
            raise HTTPException(404, "Handover not found.")
        return record

    @app.delete("/api/summaries/{record_id}", status_code=204)
    def delete_summary(record_id: str):
        if not repository.delete(record_id):
            raise HTTPException(404, "Handover not found.")

    @app.get("/api/summaries/{record_id}/export", response_class=PlainTextResponse)
    def export(record_id: str):
        return PlainTextResponse(
            export_markdown(get_summary(record_id)),
            headers={"Content-Disposition": 'attachment; filename="support-handover.md"'},
        )

    return app


app = create_app()
