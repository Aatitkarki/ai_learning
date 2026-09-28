"""Repeatable live-model evaluation. No test doubles or saved responses are used."""

import argparse
import hashlib
import json
import platform
import statistics
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx

from summarizer.config import Settings
from summarizer.models import EvidenceError, SummaryRequest
from summarizer.prompts import PROMPT_VERSION
from summarizer.provider import ModelError, OllamaProvider

CASES_PATH = Path(__file__).with_name("cases.json")


def score(case, summary):
    checks = {}
    for field, expected in case["expected_values"].items():
        checks[f"exact_{field}"] = summary[field]["value"] == expected
    claims = [*summary["key_facts"], *summary["actions_taken"], *summary["contradictions"]]
    if summary["customer_issue"]:
        claims.append(summary["customer_issue"])
    facts = " ".join(item["text"] for item in claims).casefold()
    actions = " ".join(item["text"] for item in summary["actions_taken"]).casefold()
    for term in case["required_fact_terms"]:
        checks[f"fact_contains_{term}"] = term.casefold() in facts
    for term in case["expected_action_terms"]:
        checks[f"action_contains_{term}"] = term.casefold() in actions
    for term in case["forbidden_action_terms"]:
        checks[f"action_excludes_{term}"] = term.casefold() not in actions
    if case["expect_conflict"] is not None:
        checks["conflict_expected"] = bool(summary["contradictions"]) == case["expect_conflict"]
    if case.get("expect_no_issue"):
        checks["no_invented_issue"] = summary["customer_issue"] is None
    if case.get("expect_no_actions"):
        checks["no_invented_completed_actions"] = not summary["actions_taken"]
    return checks


def run(settings, cases):
    provider = OllamaProvider(settings)
    rows = []
    for case in cases:
        started = perf_counter()
        row = {
            "id": case["id"],
            "category": case["category"],
            "human_review": "pending",
            "review_notes": case["review_notes"],
        }
        try:
            result = provider.generate(SummaryRequest(transcript=case["transcript"]).transcript)
            summary = result.summary.model_dump()
            checks = score(case, summary)
            row.update(
                schema_valid=True,
                evidence_valid=True,
                summary=summary,
                metrics=result.metrics.model_dump(),
                checks=checks,
                passed=all(checks.values()),
            )
        except ModelError as exc:
            row.update(
                schema_valid=isinstance(exc.__cause__, EvidenceError),
                evidence_valid=False,
                passed=False,
                error={"code": exc.code, "message": exc.message, "diagnostic": str(exc.__cause__)},
            )
        row["wall_ms"] = round((perf_counter() - started) * 1000, 2)
        rows.append(row)
        print(
            f"{row['id']}: {'PASS' if row['passed'] else 'FAIL'} ({row['wall_ms'] / 1000:.1f}s)",
            flush=True,
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 20:
        parser.error("--limit must be between 1 and 20")
    settings = Settings()
    cases = json.loads(CASES_PATH.read_text())[: args.limit]
    started = datetime.now(UTC)
    rows = run(settings, cases)
    digest = None
    try:
        response = httpx.get(settings.ollama_base_url + "/api/tags", timeout=5, trust_env=False)
        digest = next(
            (
                item["digest"]
                for item in response.json()["models"]
                if item["name"] == settings.ollama_model
            ),
            None,
        )
    except (httpx.HTTPError, KeyError, ValueError, TypeError):
        pass
    valid = [row for row in rows if row["evidence_valid"]]
    checked = sum(len(row.get("checks", {})) for row in rows)
    matched = sum(sum(row.get("checks", {}).values()) for row in rows)
    report = {
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "provider": "ollama",
        "model": settings.ollama_model,
        "model_digest": digest,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "prompt_version": PROMPT_VERSION,
        "dataset_sha256": hashlib.sha256(CASES_PATH.read_bytes()).hexdigest(),
        "total_cases": len(rows),
        "schema_valid": sum(row["schema_valid"] for row in rows),
        "evidence_valid": len(valid),
        "all_checks_passed": sum(row["passed"] for row in rows),
        "expectations_matched": matched,
        "expectations_checked_on_valid_outputs": checked,
        "median_success_latency_ms": statistics.median(
            row["metrics"]["latency_ms"] for row in valid
        )
        if valid
        else None,
        "input_tokens_on_valid_outputs": sum(row["metrics"]["input_tokens"] or 0 for row in valid),
        "output_tokens_on_valid_outputs": sum(
            row["metrics"]["output_tokens"] or 0 for row in valid
        ),
        "human_review": "Pending. Exact-match and keyword checks do not measure factual accuracy.",
        "cases": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "evaluation-" + started.strftime("%Y%m%dT%H%M%SZ")
    json_path = args.output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = [
        f"# Live Ollama evaluation — {started.date()}",
        "",
        f"Model: `{settings.ollama_model}`. Digest: `{digest}`. Prompt: `{PROMPT_VERSION}`.",
        f"Environment: {report['platform']}; Python {report['python']}.",
        "",
        f"Schema valid: {report['schema_valid']}/{len(rows)}. "
        f"Evidence checks passed: {len(valid)}/{len(rows)}. "
        f"All expected checks passed: {report['all_checks_passed']}/{len(rows)}.",
        f"Expected checks matched on valid outputs: {matched}/{checked}.",
        f"Median successful latency: {report['median_success_latency_ms']} ms. "
        f"Recorded tokens: {report['input_tokens_on_valid_outputs']} input / "
        f"{report['output_tokens_on_valid_outputs']} output (valid outputs only).",
        "",
        "These are mechanical checks on synthetic data, not a factual accuracy score. "
        "Keyword checks can miss paraphrases or unsupported claims. Human review is pending. "
        "Failed generations count as failed cases; token counts for them are unavailable.",
        "",
        "| Case | Result | Wall time | Failed checks |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        failures = ", ".join(key for key, value in row.get("checks", {}).items() if not value)
        failures = failures or row.get("error", {}).get("code", "—")
        lines.append(
            f"| {row['id']} | {'PASS' if row['passed'] else 'FAIL'} | "
            f"{row['wall_ms'] / 1000:.1f}s | {failures} |"
        )
    lines.extend(
        [
            "",
            f"Full outputs and diagnostics: [{json_path.name}]({json_path.name}).",
            "",
            "Review each generated claim and its cited lines before using these results "
            "in a CV or interview. This set is a development baseline, not a held-out benchmark.",
        ]
    )
    (args.output_dir / f"{stem}.md").write_text("\n".join(lines) + "\n")
    print(f"Report: {json_path}")
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
