import json
from contextlib import asynccontextmanager
from typing import Annotated
from urllib.parse import quote

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import LabError, Settings
from .models import (
    ChatRequest,
    CompareRequest,
    EvaluationRequest,
    IndexConfig,
    ReviewRequest,
    SearchRequest,
)
from .service import Lab


def create_app(settings: Settings | None = None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.lab = Lab(settings or Settings())
        yield
        app.state.lab.close()

    app = FastAPI(
        title="Local Document Knowledge Lab",
        version="0.1.0",
        lifespan=lifespan,
        description="Local retrieval experiments. No automatic model downloads.",
    )

    @app.exception_handler(LabError)
    async def lab_error(request, exc):
        return JSONResponse({"detail": exc.message}, status_code=exc.status)

    @app.middleware("http")
    async def local_browser_guard(request: Request, call_next):
        # No CORS: a web page on another origin cannot mutate this unauthenticated local API.
        origin = request.headers.get("origin")
        unsafe_origin = origin and origin.rstrip("/") != str(request.base_url).rstrip("/")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and (
            unsafe_origin or request.headers.get("sec-fetch-site") == "cross-site"
        ):
            return JSONResponse({"detail": "Use the local app or API client."}, status_code=403)
        content_length = request.headers.get("content-length", "0")
        try:
            if int(content_length) > 11 * 1024 * 1024:
                return JSONResponse({"detail": "Request exceeds 11 MB."}, status_code=413)
        except ValueError:
            return JSONResponse({"detail": "Invalid Content-Length."}, status_code=400)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/health")
    def health(request: Request):
        return {"status": "ok", **request.app.state.lab.catalog()}

    @app.get("/api/documents")
    def documents(request: Request):
        return [
            {k: v for k, v in doc.items() if k != "pages"}
            for doc in request.app.state.lab.repo.list("documents")
        ]

    @app.post("/api/documents", status_code=201)
    def upload(request: Request, file: Annotated[UploadFile, File()]):
        lab = request.app.state.lab
        content = file.file.read(lab.settings.max_upload_bytes + 1)
        doc = lab.upload(file.filename or "", content)
        return {k: v for k, v in doc.items() if k != "pages"}

    @app.post("/api/documents/seed", status_code=201)
    def seed(request: Request):
        return [
            {k: v for k, v in doc.items() if k != "pages"} for doc in request.app.state.lab.seed()
        ]

    @app.get("/api/documents/{identifier}")
    def document(identifier: str, request: Request):
        return request.app.state.lab.require("documents", identifier)

    @app.get("/api/documents/{identifier}/source")
    def source(identifier: str, request: Request):
        lab = request.app.state.lab
        document = lab.require("documents", identifier)
        pdf = document["filename"].lower().endswith(".pdf")
        disposition = f"inline; filename*=UTF-8''{quote(document['filename'], safe='')}"
        return Response(
            lab.repo.source(identifier),
            media_type="application/pdf" if pdf else "text/plain; charset=utf-8",
            headers={
                "Content-Disposition": disposition,
                "Content-Security-Policy": "sandbox",
            },
        )

    @app.delete("/api/documents/{identifier}", status_code=204)
    def delete_document(identifier: str, request: Request):
        request.app.state.lab.delete_document(identifier)
        return Response(status_code=204)

    @app.get("/api/indexes")
    def indexes(request: Request):
        return request.app.state.lab.repo.list("indexes")

    @app.post("/api/indexes", status_code=202)
    def create_index(config: IndexConfig, request: Request):
        return request.app.state.lab.create_index(config)

    @app.get("/api/indexes/{identifier}")
    def index(identifier: str, request: Request):
        return request.app.state.lab.require("indexes", identifier)

    @app.get("/api/indexes/{identifier}/chunks")
    def chunks(identifier: str, request: Request, offset: int = 0, limit: int = 30):
        lab = request.app.state.lab
        lab.require("indexes", identifier)
        if offset < 0 or not 1 <= limit <= 200:
            raise LabError("Use offset >= 0 and limit between 1 and 200.")
        rows = lab.repo.chunks(identifier)
        return {"total": len(rows), "chunks": [r["chunk"] for r in rows[offset : offset + limit]]}

    @app.post("/api/indexes/{identifier}/rebuild", status_code=202)
    def rebuild(identifier: str, request: Request):
        lab = request.app.state.lab
        config = IndexConfig(**lab.require("indexes", identifier)["config"])
        return lab.create_index(config.model_copy(update={"name": config.name[:68] + " (rebuild)"}))

    @app.delete("/api/indexes/{identifier}", status_code=204)
    def delete_index(identifier: str, request: Request):
        request.app.state.lab.delete_index(identifier)
        return Response(status_code=204)

    @app.post("/api/search")
    def search(body: SearchRequest, request: Request):
        return request.app.state.lab.search(body.question, body.config)

    @app.post("/api/compare")
    def compare(body: CompareRequest, request: Request):
        return request.app.state.lab.compare(body.question, body.configs)

    @app.post("/api/chat")
    def chat(body: ChatRequest, request: Request):
        # NDJSON events preserve tokens, retrieval metadata, terminal results and errors.
        def events():
            for event in request.app.state.lab.chat(body):
                yield json.dumps(event, ensure_ascii=False) + "\n"

        return StreamingResponse(
            events(), media_type="application/x-ndjson", headers={"X-Accel-Buffering": "no"}
        )

    @app.get("/api/conversations")
    def conversations(request: Request):
        return request.app.state.lab.repo.list("conversations")

    @app.delete("/api/conversations/{identifier}", status_code=204)
    def delete_conversation(identifier: str, request: Request):
        lab = request.app.state.lab
        with lab.lock:
            lab.require("conversations", identifier)
            for run in lab.repo.list("runs"):
                if run.get("conversation_id") == identifier:
                    lab.repo.delete("runs", run["id"])
            lab.repo.delete("conversations", identifier)
        return Response(status_code=204)

    @app.get("/api/runs")
    def runs(request: Request):
        return request.app.state.lab.repo.list("runs")

    @app.get("/api/runs/{identifier}")
    def run(identifier: str, request: Request):
        return request.app.state.lab.require("runs", identifier)

    @app.post("/api/runs/{identifier}/review")
    def review(identifier: str, body: ReviewRequest, request: Request):
        lab = request.app.state.lab
        with lab.lock:
            run = lab.require("runs", identifier)
            run["review"] = body.model_dump()
            lab.repo.put("runs", run)
        return run

    @app.get("/api/evaluations")
    def evaluations(request: Request):
        return request.app.state.lab.repo.list("evaluations")

    @app.post("/api/evaluations", status_code=202)
    def evaluation(body: EvaluationRequest, request: Request):
        return request.app.state.lab.create_evaluation(body)

    @app.get("/api/evaluations/{identifier}")
    def evaluation_report(identifier: str, request: Request):
        return request.app.state.lab.require("evaluations", identifier)

    return app


app = create_app()
