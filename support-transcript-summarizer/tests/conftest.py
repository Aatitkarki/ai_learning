import pytest
from fastapi.testclient import TestClient

from summarizer.api import create_app
from summarizer.config import Settings
from summarizer.models import Handover, Metrics
from summarizer.provider import Generation

TRANSCRIPT = "Customer: I am Maya. My QAR 100 transfer TX-01 is missing.\nAgent: I checked TX-01."


@pytest.fixture
def summary():
    unknown = {"value": None, "source_lines": []}
    return Handover.model_validate(
        {
            "customer_issue": {
                "text": "Customer reports QAR 100 transfer TX-01 is missing.",
                "source_lines": [1],
            },
            "customer_name": {"value": "Maya", "source_lines": [1]},
            "amount": {"value": "QAR 100", "source_lines": [1]},
            "transfer_reference": {"value": "TX-01", "source_lines": [1]},
            "stated_date": unknown,
            "delay_reason": unknown,
            "key_facts": [],
            "actions_taken": [{"text": "Agent checked TX-01.", "source_lines": [2]}],
            "unresolved_questions": ["Why is it delayed?"],
            "contradictions": [],
            "suggested_next_steps": ["Ask for a status update."],
        }
    )


class FakeProvider:
    """Explicit deterministic test double; never selected by application configuration."""

    def __init__(self, summary):
        self.summary = summary
        self.calls = 0
        self.error = None

    def health(self):
        return {"available": True, "message": "Test double"}

    def generate(self, transcript):
        self.calls += 1
        if self.error:
            raise self.error
        return Generation(
            self.summary,
            Metrics(
                model="test-double",
                latency_ms=12.3,
                input_tokens=20,
                output_tokens=30,
                prompt_version="test",
            ),
        )


@pytest.fixture
def provider(summary):
    return FakeProvider(summary)


@pytest.fixture
def client(tmp_path, provider):
    app = create_app(Settings(database_path=tmp_path / "history.sqlite3"), provider)
    with TestClient(app) as client:
        yield client
