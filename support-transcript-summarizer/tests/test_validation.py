import pytest
from pydantic import ValidationError

from summarizer.models import EvidenceError, Handover, SummaryRequest, validate_evidence
from tests.conftest import TRANSCRIPT


def test_evidence_happy_path(summary):
    validate_evidence(summary, TRANSCRIPT)


@pytest.mark.parametrize(
    "change",
    [
        {"amount": {"value": "QAR 999", "source_lines": [1]}},
        {"amount": {"value": "QAR 100", "source_lines": [2]}},
        {"customer_issue": {"text": "A missing transfer", "source_lines": [3]}},
        {"customer_issue": {"text": "A missing transfer", "source_lines": [1, 1]}},
        {"customer_issue": {"text": "Transfer TX-02 is missing", "source_lines": [1]}},
        {"actions_taken": [{"text": "Refunded QAR 999", "source_lines": [2]}]},
    ],
)
def test_unsupported_evidence_rejected(summary, change):
    invalid = Handover.model_validate(summary.model_dump() | change)
    with pytest.raises(EvidenceError):
        validate_evidence(invalid, TRANSCRIPT)


@pytest.mark.parametrize(
    "fact",
    [
        {"value": None, "source_lines": [1]},
        {"value": "Maya", "source_lines": []},
        {"value": "Maya", "source_lines": [True]},
        {"value": "Maya", "source_lines": ["1"]},
    ],
)
def test_unknown_and_known_contract(summary, fact):
    with pytest.raises(ValidationError):
        Handover.model_validate(summary.model_dump() | {"customer_name": fact})


def test_normalization_preserves_internal_lines():
    request = SummaryRequest(transcript="  Customer: Hi\r\n\r\nAgent: Hello\r\n")
    assert request.transcript == "Customer: Hi\n\nAgent: Hello"


def test_blank_line_reference_rejected(summary):
    with pytest.raises(EvidenceError):
        validate_evidence(summary, TRANSCRIPT.split("\n")[0] + "\n\nAgent: I checked TX-01.")


def test_validation_cannot_guarantee_semantics(summary):
    # Deliberately documents the limit: valid evidence bounds do not prove entailment.
    value = summary.model_copy(deep=True)
    value.customer_issue.text = "The customer received the transfer."
    validate_evidence(value, TRANSCRIPT)


@pytest.mark.parametrize("claim_number,source_number", [("-100", "100"), ("100", "-100")])
def test_numeric_sign_is_preserved(summary, claim_number, source_number):
    value = summary.model_copy(deep=True)
    value.customer_issue.text = f"The balance is {claim_number}."
    value.customer_issue.source_lines = [3]
    with pytest.raises(EvidenceError):
        validate_evidence(value, TRANSCRIPT + f"\nCustomer: The balance is {source_number}.")


def test_multibyte_text_has_a_context_budget():
    with pytest.raises(ValidationError, match="UTF-8"):
        SummaryRequest(transcript="🌍" * 2001)
