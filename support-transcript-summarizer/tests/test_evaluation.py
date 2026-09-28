import json

from evaluation.run import CASES_PATH, score
from summarizer.models import SummaryRequest


def test_dataset_is_twenty_unique_valid_handwritten_cases():
    cases = json.loads(CASES_PATH.read_text())
    assert len(cases) == len({case["id"] for case in cases}) == 20
    for case in cases:
        SummaryRequest(transcript=case["transcript"])
        for expected in case["expected_values"].values():
            if expected is not None:
                assert expected in case["transcript"]


def test_evaluator_catches_missing_expected_facts_and_forbidden_actions(summary):
    case = {
        "expected_values": {"amount": "QAR 100", "customer_name": "Someone else"},
        "required_fact_terms": ["TX-01", "invented"],
        "expected_action_terms": ["checked"],
        "forbidden_action_terms": ["checked"],
        "expect_conflict": False,
        "expect_no_actions": True,
    }
    checks = score(case, summary.model_dump())
    assert checks["exact_amount"]
    assert not checks["exact_customer_name"]
    assert not checks["fact_contains_invented"]
    assert not checks["action_excludes_checked"]
    assert not checks["no_invented_completed_actions"]
