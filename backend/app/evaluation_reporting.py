"""Build reproducible, portfolio-friendly evaluation reports.

The API already stores retrieval runs.  This module deliberately keeps the
reporting layer read-only and pure so a report can be tested without a live
database, provider credentials, or user documents.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any


def _dataset_name(details: Sequence[Mapping[str, Any]]) -> str:
    if details and all(str(item.get("case_id", "")).startswith("hello-agents:") for item in details):
        return "Hello-Agents-30"
    return "custom-evaluation-set"


def _citation_location_success(details: Sequence[Mapping[str, Any]]) -> float | None:
    """Return the rate of document+locator hits recorded by the evaluator.

    The evaluator marks a case as a hit only when both the expected document
    and expected locator are present in Top K.  Keeping this metric explicit
    prevents a resume claim from confusing retrieval recall with citation
    grounding.
    """

    if not details:
        return None
    return round(sum(bool(item.get("hit")) for item in details) / len(details), 4)


def build_evaluation_report(
    run: Mapping[str, Any],
    details: Sequence[Mapping[str, Any]],
    *,
    embedding_model: str,
    chat_model: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Create a stable report payload from one stored evaluation run."""

    detail_list = [dict(item) for item in details]
    timestamp = generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "report_version": "1.0",
        "generated_at": timestamp,
        "dataset": {
            "name": _dataset_name(detail_list),
            "space_id": run["space_id"],
            "case_count": run["case_count"],
        },
        "configuration": {
            "top_k": run["top_k"],
            "retrieval": "SQLite FTS5/BM25 + dense retrieval + RRF",
            "embedding_model": embedding_model,
            "chat_model": chat_model,
        },
        "machine": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count() or 1,
            "interpreter": sys.implementation.name,
        },
        "metrics": {
            "recall": run["recall"],
            "mrr": run["mrr"],
            "ndcg": run["ndcg"],
            "mean_latency_ms": run["mean_latency_ms"],
            "p95_latency_ms": run["p95_latency_ms"],
            "citation_location_success": _citation_location_success(detail_list),
        },
        "bad_cases": [
            {
                "case_id": item.get("case_id"),
                "question": item.get("question"),
                "latency_ms": item.get("latency_ms"),
                "returned": item.get("returned", []),
            }
            for item in detail_list
            if not item.get("hit")
        ],
        "details": detail_list,
    }


def _format_metric(value: Any) -> str:
    if value is None:
        return "未记录"
    if isinstance(value, float) and 0 <= value <= 1:
        return f"{value:.4f}"
    return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the report without hiding missing or unverified values."""

    dataset = report["dataset"]
    config = report["configuration"]
    machine = report["machine"]
    metrics = report["metrics"]
    lines = [
        "# KUN RAG Evaluation Report",
        "",
        f"> Generated: {report['generated_at']}",
        "> This report measures retrieval and citation-location grounding; it is not a claim about production accuracy.",
        "",
        "## Dataset and configuration",
        "",
        f"- Dataset: `{dataset['name']}`",
        f"- Knowledge space: `{dataset['space_id']}`",
        f"- Cases: `{dataset['case_count']}`",
        f"- Top K: `{config['top_k']}`",
        f"- Retrieval: {config['retrieval']}",
        f"- Embedding model: `{config['embedding_model']}`",
        f"- Chat model: `{config['chat_model']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Recall@K | {_format_metric(metrics['recall'])} |",
        f"| MRR | {_format_metric(metrics['mrr'])} |",
        f"| nDCG@K | {_format_metric(metrics['ndcg'])} |",
        f"| Citation-location success | {_format_metric(metrics['citation_location_success'])} |",
        f"| Mean retrieval latency | {_format_metric(metrics['mean_latency_ms'])} ms |",
        f"| P95 retrieval latency | {_format_metric(metrics['p95_latency_ms'])} ms |",
        "",
        "## Machine profile",
        "",
        f"- OS: `{machine['os']}`",
        f"- Python: `{machine['python']}` ({machine['interpreter']})",
        f"- CPU count: `{machine['cpu_count']}`",
        "",
        "## Bad cases",
        "",
    ]
    bad_cases = report.get("bad_cases", [])
    if not bad_cases:
        lines.append("No misses were recorded in this run.")
    else:
        for index, item in enumerate(bad_cases, 1):
            question = str(item.get("question", "")).replace("\n", " ")
            lines.extend([
                f"{index}. `{item.get('case_id')}` — {question}",
                f"   - Latency: `{item.get('latency_ms')} ms`",
                f"   - Returned candidates: `{len(item.get('returned', []))}`",
            ])
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Recall measures whether the expected source appears in Top K.",
        "- Citation-location success requires both the expected document and locator to appear in Top K.",
        "- Use the bad cases to explain chunking, locator parsing, retrieval fusion, or ranking changes.",
    ])
    return "\n".join(lines) + "\n"


def load_details(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Decode the stored per-case JSON while tolerating an empty legacy value."""

    raw = run.get("result_json") or "[]"
    if isinstance(raw, str):
        value = json.loads(raw)
    else:
        value = raw
    return [dict(item) for item in value] if isinstance(value, list) else []
