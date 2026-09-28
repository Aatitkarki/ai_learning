import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
from pathlib import Path

from knowledge_lab.config import LabError
from knowledge_lab.models import SearchConfig
from knowledge_lab.repository import now

ROOT = Path(__file__).resolve().parent.parent


def metrics(hits, expected):
    found = set()
    first = None
    gains = []
    for rank, hit in enumerate(hits, 1):
        matched = {
            i
            for i, support in enumerate(expected)
            if hit.chunk.filename == support["filename"]
            and " ".join(support["quote"].split()).casefold()
            in " ".join(hit.chunk.text.split()).casefold()
        }
        new = matched - found
        if matched and first is None:
            first = rank
        gains.append(1 if new else 0)
        found |= matched
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(expected), len(hits))))
    return {
        "passage_recall": len(found) / len(expected) if expected else None,
        "reciprocal_rank": 1 / first if first else 0.0,
        "ndcg": sum(g / math.log2(i + 2) for i, g in enumerate(gains)) / ideal if ideal else 0,
        "hit": bool(found),
        "all_support_found": len(found) == len(expected) if expected else None,
        "empty_result": not hits,
    }


def summarize(rows):
    supported = [r for r in rows if r["kind"] != "unsupported" and "metrics" in r]
    unsupported = [r for r in rows if r["kind"] == "unsupported" and "metrics" in r]
    latencies = sorted(r["latency_ms"] for r in rows if "latency_ms" in r)

    def mean(key):
        return statistics.mean(r["metrics"][key] for r in supported) if supported else None

    return {
        "cases": len(rows),
        "errors": sum("error" in r for r in rows),
        "answerable_cases_scored": len(supported),
        "passage_recall_at_k": mean("passage_recall"),
        "mrr_at_k": mean("reciprocal_rank"),
        "ndcg_at_k": mean("ndcg"),
        "hit_rate_at_k": mean("hit"),
        "unsupported_empty_rate": (
            statistics.mean(r["metrics"]["empty_result"] for r in unsupported)
            if unsupported
            else None
        ),
        "median_ms": statistics.median(latencies) if latencies else None,
        "p95_ms": latencies[max(0, math.ceil(len(latencies) * 0.95) - 1)] if latencies else None,
    }


def evaluate(lab, request, progress=lambda _: None):
    dataset_bytes = (ROOT / "evaluation" / "cases.json").read_bytes()
    dataset = json.loads(dataset_bytes)
    cases = [c for c in dataset["cases"] if request["split"] in {"all", c["split"]}]
    if not cases:
        raise LabError("The evaluation split contains no cases.")
    results = []
    total = len(cases) * len(request["configs"])
    completed = 0
    for config_data in request["configs"]:
        config = SearchConfig(**config_data)
        index = lab.require("indexes", config.index_id)
        manifest = {d["filename"]: d["sha256"] for d in index["document_manifest"]}
        for path in (ROOT / "examples" / "policies").glob("*.md"):
            if manifest.get(path.name) != hashlib.sha256(path.read_bytes()).hexdigest():
                raise LabError(
                    "The bundled evaluation requires an index containing all unchanged "
                    "demo policies. Seed them, then build an index with all 12 files."
                )
        rows = []
        for case in cases:
            row = {k: case[k] for k in ("id", "question", "kind", "split", "expected_behavior")}
            try:
                result = lab.search(case["question"], config, persist=False)
                row.update(
                    metrics=metrics(result.hits, case["support"]),
                    latency_ms=result.timings_ms["total"],
                    retrieved=[
                        {
                            "chunk_id": h.chunk.id,
                            "filename": h.chunk.filename,
                            "page": h.chunk.page,
                            "text": h.chunk.text,
                            "score": h.score,
                        }
                        for h in result.hits
                    ],
                )
            except LabError as exc:
                row["error"] = exc.message
            rows.append(row)
            completed += 1
            progress(completed / total)
        results.append(
            {
                "config": config.model_dump(),
                "index": index,
                "summary": summarize(rows),
                "by_split": {
                    split: summarize([r for r in rows if r["split"] == split])
                    for split in sorted({r["split"] for r in rows})
                },
                "cases": rows,
            }
        )
    packages = {}
    for name in (
        "fastapi",
        "numpy",
        "pypdf",
        "chromadb",
        "faiss-cpu",
        "psycopg",
        "langchain-text-splitters",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "created_at": now(),
        "dataset_version": dataset["version"],
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "results": results,
        "limitations": [
            "Retrieval only: no answer quality or faithfulness was measured.",
            "Unsupported-empty rate is a retrieval threshold diagnostic, not "
            "a semantic abstention score.",
            "Passage match requires a complete expected quote in one result.",
            "Errors are excluded from scored metric denominators and counted separately.",
            "One pass, no warm-up: latency includes connection and cache effects.",
            "Review conflict and injection answers manually after connecting a model.",
        ],
    }


def markdown_report(report):
    lines = [
        "# Retrieval experiment",
        "",
        f"Recorded: {report['created_at']}",
        "",
        "| Store / method | Recall@k | MRR@k | nDCG@k | Median ms | Errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in report["results"]:
        config, index, summary = result["config"], result["index"], result["summary"]
        values = [summary[k] for k in ("passage_recall_at_k", "mrr_at_k", "ndcg_at_k", "median_ms")]
        formatted = [f"{v:.3f}" if v is not None else "n/a" for v in values]
        lines.append(
            f"| {index['config']['store']} / {config['method']} | "
            + " | ".join(formatted)
            + f" | {summary['errors']} |"
        )
    lines += [
        "",
        *[f"- {limit}" for limit in report["limitations"]],
        "",
        "Full configurations, corpus hashes, split results and failures are in the JSON report.",
    ]
    return "\n".join(lines)
