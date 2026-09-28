"""Run a retrieval experiment against the running API; no implicit setup or downloads."""

import argparse
import json
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_id")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["bm25", "vector", "hybrid", "mmr"],
        default=["bm25", "hybrid"],
    )
    parser.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    parser.add_argument("--api", default="http://127.0.0.1:8020")
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    with httpx.Client(base_url=args.api, timeout=120, trust_env=False) as client:
        response = client.post(
            "/api/evaluations",
            json={
                "split": args.split,
                "configs": [
                    {"index_id": args.index_id, "method": method} for method in args.methods
                ],
            },
        )
        response.raise_for_status()
        identifier = response.json()["id"]
        while True:
            response = client.get(f"/api/evaluations/{identifier}")
            response.raise_for_status()
            job = response.json()
            if job["status"] in {"complete", "failed"}:
                break
            time.sleep(1)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"evaluation-{identifier}.json"
    path.write_text(json.dumps(job, indent=2))
    print(f"{job['status']}: {path}")
    if job["status"] == "failed":
        raise SystemExit(job.get("error", "Evaluation failed"))


if __name__ == "__main__":
    main()
