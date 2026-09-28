import json

import httpx
import pytest

from knowledge_lab.config import LabError
from knowledge_lab.embeddings import Embedder
from knowledge_lab.generation import ABSTENTION, context_sources, validate_citations
from knowledge_lab.models import ChatRequest, IndexConfig


def test_evidence_mode_and_followup_history_need_no_model(indexed):
    lab, index, _ = indexed
    request = ChatRequest(question="cancellation refund", config={"index_id": index["id"]})
    events = list(lab.chat(request))
    assert [e["type"] for e in events] == ["retrieval", "token", "complete"]
    result = events[-1]["result"]
    assert result["status"] == "evidence_only" and result["model_identity"] is None
    assert result["sources"] and "no AI answer generated" in result["answer"]
    followup = request.model_copy(
        update={"question": "How long does it take?", "conversation_id": result["conversation_id"]}
    )
    next_result = list(lab.chat(followup))[-1]["result"]
    assert "cancellation refund" in next_result["retrieval"]["query"]
    assert len(lab.repo.list("conversations")[0]["turns"]) == 2


def test_no_evidence_abstains_without_requesting_model(indexed):
    lab, index, _ = indexed
    result = list(
        lab.chat(
            ChatRequest(
                question="xyzzyplugh",
                mode="ollama",
                config={"index_id": index["id"], "method": "bm25"},
            )
        )
    )[-1]["result"]
    assert result["answer"] == ABSTENTION
    assert result["status"] == "abstained"


def test_missing_chat_model_is_actionable_not_a_fake_answer(indexed):
    lab, index, _ = indexed
    events = list(
        lab.chat(ChatRequest(question="fee", mode="ollama", config={"index_id": index["id"]}))
    )
    assert events[-1]["type"] == "error"
    assert "LAB_CHAT_MODEL" in events[-1]["message"]
    assert not lab.repo.list("runs")


@pytest.mark.parametrize(
    "answer,valid",
    [
        ("Fee is 5 credits. [S1]", True),
        ("Fee is 5 credits. [S2]", False),
        ("Unsupported claim without citation.", False),
        (ABSTENTION, True),
        ("One claim. [S1]\n\nUncited second claim.", False),
        ("A claim. [S1] [Sunknown]", False),
    ],
)
def test_citation_validation(answer, valid):
    assert validate_citations(answer, [{"label": "S1"}])[0] == valid


@pytest.mark.parametrize(
    "answer,expected_status",
    [
        ("The standard fee is 5 demo credits. [S1]", "answered"),
        ("An invented fee is 900 credits. [S99]", "citation_rejected"),
    ],
)
def test_streamed_model_answer_is_finalized_and_identity_saved(
    indexed, mock_http, answer, expected_status
):
    lab, index, _ = indexed
    lab.settings.chat_model = "local-test"
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": "local-test:latest", "digest": "abc123"}]}
            )
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        assert body["options"]["temperature"] == 0
        assert body["stream"]
        assert "untrusted" in body["messages"][0]["content"]
        payload = [
            json.dumps({"message": {"content": answer}, "done": False}),
            json.dumps(
                {"message": {"content": ""}, "done": True, "eval_count": 12, "done_reason": "stop"}
            ),
        ]
        return httpx.Response(200, text="\n".join(payload))

    mock_http(handler)
    result = list(
        lab.chat(
            ChatRequest(
                question="standard transfer fee",
                mode="ollama",
                config={"index_id": index["id"], "method": "bm25"},
            )
        )
    )[-1]["result"]
    assert result["status"] == expected_status
    assert result["model_identity"] == "ollama:local-test:latest:abc123"
    assert result["ttft_ms"] is not None and result["usage"]["eval_count"] == 12
    assert result["answer"] == (answer if expected_status == "answered" else ABSTENTION)
    assert set(paths) == {"/api/tags", "/api/chat"}


def test_embedding_adapter_uses_explicit_vectors_and_never_pulls(settings, mock_http):
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "embed:latest", "digest": "xyz"}]})
        assert request.url.path == "/api/embed"
        body = json.loads(request.content)
        assert body["truncate"] is False
        return httpx.Response(200, json={"embeddings": [[1, 0] for _ in body["input"]]})

    mock_http(handler)
    embedder = Embedder(
        IndexConfig(name="e", document_ids=["a"], embedding="ollama", embedding_model="embed"),
        settings,
    )
    assert embedder.identity() == "ollama:embed:latest:xyz"
    assert embedder.embed(["one", "two"]).shape == (2, 2)
    assert set(paths) == {"/api/tags", "/api/embed"}


def test_absent_embedding_model_is_not_downloaded(settings, mock_http):
    def handler(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": []})

    mock_http(handler)
    embedder = Embedder(
        IndexConfig(name="e", document_ids=["a"], embedding="ollama", embedding_model="absent"),
        settings,
    )
    with pytest.raises(LabError, match="No download made"):
        embedder.identity()


def test_context_budget_preserves_the_cited_substring(indexed):
    from knowledge_lab.models import SearchConfig

    lab, index, _ = indexed
    retrieved = lab.search("cancellation refund", SearchConfig(index_id=index["id"]))
    sources = context_sources(retrieved.hits, 150)
    assert sum(len(source["text"]) for source in sources) <= 150
    for source in sources:
        document = lab.require("documents", source["document_id"])
        page = next(p for p in document["pages"] if p["number"] == source["page"])
        assert source["text"] == page["text"][source["start"] : source["end"]]


@pytest.mark.parametrize("finished", [True, False])
def test_truncated_and_interrupted_streams_are_not_successful_answers(indexed, mock_http, finished):
    lab, index, _ = indexed
    lab.settings.chat_model = "test"

    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "test", "digest": "hash"}]})
        return httpx.Response(
            200,
            text=json.dumps(
                {"message": {"content": "A draft. [S1]"}, "done": finished, "done_reason": "length"}
            ),
        )

    mock_http(handler)
    events = list(
        lab.chat(
            ChatRequest(
                question="refund", mode="ollama", config={"index_id": index["id"], "method": "bm25"}
            )
        )
    )
    if finished:
        assert events[-1]["result"]["status"] == "truncated"
        assert events[-1]["result"]["answer"] == ABSTENTION
    else:
        assert events[-1]["type"] == "error"
        assert not lab.repo.list("runs")
