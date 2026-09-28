"""Inspect a single direct model request before using the API or interface."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from summarizer.config import Settings  # noqa: E402
from summarizer.models import SummaryRequest  # noqa: E402
from summarizer.provider import ModelError, OllamaProvider  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path)
    args = parser.parse_args()
    try:
        request = SummaryRequest(transcript=args.transcript.read_text(encoding="utf-8-sig"))
        result = OllamaProvider(Settings()).generate(request.transcript)
        print(
            json.dumps(
                {"summary": result.summary.model_dump(), "metrics": result.metrics.model_dump()},
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    except (OSError, ValueError, ModelError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
