from __future__ import annotations

import json

from app.evaluation_reporting import build_evaluation_report, load_details, render_markdown


def sample_run() -> dict:
    return {
        "id": "run-1",
        "space_id": "ai-agent-learning",
        "top_k": 5,
        "case_count": 2,
        "recall": 0.5,
        "mrr": 0.5,
        "ndcg": 0.5,
        "mean_latency_ms": 12.4,
        "p95_latency_ms": 18.2,
        "result_json": json.dumps([
            {"case_id": "hello-agents:ai-agent-learning:HA-001", "question": "q1", "hit": True, "rank": 1, "latency_ms": 10, "returned": []},
            {"case_id": "hello-agents:ai-agent-learning:HA-002", "question": "q2", "hit": False, "rank": None, "latency_ms": 15, "returned": [{"title": "wrong"}]},
        ]),
    }


def test_report_separates_citation_location_success_from_retrieval_metrics():
    run = sample_run()
    report = build_evaluation_report(
        run,
        load_details(run),
        embedding_model="test-embedding",
        chat_model="test-chat",
        generated_at="2026-08-04T00:00:00+00:00",
    )

    assert report["dataset"]["name"] == "Hello-Agents-30"
    assert report["metrics"]["recall"] == 0.5
    assert report["metrics"]["citation_location_success"] == 0.5
    assert len(report["bad_cases"]) == 1
    assert report["configuration"]["embedding_model"] == "test-embedding"


def test_markdown_report_keeps_bad_case_and_machine_sections():
    report = build_evaluation_report(
        sample_run(),
        load_details(sample_run()),
        embedding_model="test-embedding",
        chat_model="test-chat",
        generated_at="2026-08-04T00:00:00+00:00",
    )
    markdown = render_markdown(report)

    assert "Citation-location success" in markdown
    assert "hello-agents:ai-agent-learning:HA-002" in markdown
    assert "## Machine profile" in markdown
