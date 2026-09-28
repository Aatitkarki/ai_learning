"""Record real API interactions as an asciicast v2 file and a plain-text backup."""

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from summarizer.config import Settings  # noqa: E402


def main():
    root = Path(__file__).resolve().parents[1]
    settings = Settings()
    output_dir = root / "reports"
    output_dir.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    cast = output_dir / f"demo-{stamp}.cast"
    transcript = output_dir / f"demo-{stamp}.txt"
    started = time.monotonic()
    events = []
    text_parts = []
    created = []

    def emit(message):
        events.append(
            [round(time.monotonic() - started, 3), "o", message.replace("\n", "\r\n") + "\r\n"]
        )
        text_parts.append(message)
        print(message, flush=True)

    outcome = 0
    with httpx.Client(
        base_url=settings.api_base_url, trust_env=False, timeout=settings.model_timeout_seconds + 15
    ) as client:
        try:
            emit("SUPPORT TRANSCRIPT SUMMARIZER — recorded live API demonstration\n")
            for example in ["delayed_transfer", "incomplete_conversation"]:
                text = (root / "examples" / f"{example}.txt").read_text()
                emit(f"INPUT: {example}\n{text}\n\nCalling the local model…")
                response = client.post(
                    "/api/summaries", json={"title": f"Recording: {example}", "transcript": text}
                )
                response.raise_for_status()
                record = response.json()
                created.append(record["id"])
                emit("ACTUAL GENERATED HANDOVER:\n" + record["draft_markdown"])
                emit("ACTUAL METRICS:\n" + json.dumps(record["metrics"], indent=2))
                emit("Inspect the [Lx] references against these lines:")
                emit(
                    "\n".join(f"L{i}: {line}" for i, line in enumerate(text.strip().split("\n"), 1))
                )
                emit(
                    "\nOriginal structured missing values:\n"
                    + json.dumps(
                        {
                            field: record["summary"][field]
                            for field in [
                                "customer_name",
                                "amount",
                                "transfer_reference",
                                "delay_reason",
                            ]
                        },
                        indent=2,
                    )
                )
                time.sleep(2)
            latest = client.get(f"/api/summaries/{created[-1]}").json()
            response = client.patch(
                f"/api/summaries/{created[-1]}",
                json={
                    "draft_markdown": latest["draft_markdown"]
                    + "\n\nReviewer: request the receipt.",
                    "revision": latest["revision"],
                },
            )
            response.raise_for_status()
            emit(
                f"Saved human edit: revision {response.json()['revision']}; "
                "original structured evidence preserved."
            )
            exported = client.get(f"/api/summaries/{created[-1]}/export")
            exported.raise_for_status()
            emit("Export label: " + exported.text.splitlines()[0])
            invalid = client.post("/api/summaries", json={"transcript": ""})
            emit(f"Empty-input check: HTTP {invalid.status_code}; no model call.")
            emit("\nReview output against source; schema/citation checks do not guarantee truth.")
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            emit(f"DEMO STOPPED: {exc}")
            outcome = 1
        finally:
            for record_id in created:
                try:
                    response = client.delete(f"/api/summaries/{record_id}")
                    response.raise_for_status()
                    emit(f"Deleted temporary demo record {record_id}.")
                except httpx.HTTPError:
                    emit(f"Could not delete temporary record {record_id}; remove it from History.")
            header = {
                "version": 2,
                "width": 110,
                "height": 36,
                "timestamp": int(time.time()),
                "title": "Live local support handover demo",
                "env": {"TERM": "xterm-256color"},
            }
            cast.write_text("\n".join(json.dumps(item) for item in [header, *events]) + "\n")
            transcript.write_text("\n\n".join(text_parts) + "\n")
    print(f"Recording: {cast}")
    return outcome


if __name__ == "__main__":
    raise SystemExit(main())
