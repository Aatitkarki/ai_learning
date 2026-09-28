import httpx
import pytest
from fastapi.testclient import TestClient

from knowledge_lab.api import create_app
from knowledge_lab.config import Settings
from knowledge_lab.models import IndexConfig
from knowledge_lab.service import Lab


@pytest.fixture(autouse=True)
def no_external_http(monkeypatch, request):
    if request.node.get_closest_marker("integration"):
        return

    def blocked(*args, **kwargs):
        raise AssertionError("Tests must not make real HTTP requests or download models.")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path, chat_model="", embed_model="")


@pytest.fixture
def lab(settings):
    service = Lab(settings)
    yield service
    service.close()


@pytest.fixture
def indexed(lab):
    documents = lab.seed()
    index = lab.create_index(
        IndexConfig(name="test", document_ids=[d["id"] for d in documents]), background=False
    )
    assert index["status"] == "ready", index
    return lab, index, documents


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as client:
        yield client


@pytest.fixture
def mock_http(monkeypatch):
    original = httpx.Client

    def install(handler):
        monkeypatch.setattr(
            httpx,
            "Client",
            lambda *args, **kwargs: original(
                *args, **kwargs, transport=httpx.MockTransport(handler)
            ),
        )

    return install
