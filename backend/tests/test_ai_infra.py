from __future__ import annotations

import time
from pathlib import Path


def _prepare_database(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KUN_DATA_DIR", str(tmp_path))
    from app.database import init_database

    init_database()


def _insert_space_and_document(tmp_path: Path, *, text: str = "") -> tuple[str, str]:
    from app.database import connect, now

    space_id = "infra-test"
    document_id = "doc-infra-test"
    source = tmp_path / "source.md"
    source.write_text(text or "# Test\n" + "检索系统需要可观测性。" * 120, encoding="utf-8")
    stamp = now()
    with connect() as db:
        db.execute(
            "INSERT INTO spaces(id,name,color,created_at,updated_at) VALUES(?,?,?,?,?)",
            (space_id, "Infra Test", "#6655cc", stamp, stamp),
        )
        db.execute(
            """INSERT INTO documents(
               id,space_id,original_name,library_path,file_type,size_bytes,fingerprint,title,summary,
               tags_json,parse_status,index_status,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                document_id, space_id, source.name, str(source), "md", source.stat().st_size,
                "fingerprint-v1", "Infra source", "", "[]", "parsed", "ready", stamp, stamp,
            ),
        )
    return space_id, document_id


def test_trace_redacts_secrets_and_records_waterfall(monkeypatch, tmp_path: Path):
    _prepare_database(monkeypatch, tmp_path)
    from app.infra import create_trace, finish_trace, trace_detail, trace_span

    trace_id = create_trace("retrieval", "test_pipeline", {"api_key": "do-not-store", "query": "hello"})
    with trace_span(trace_id, "bm25_search", "bm25_search", attributes={"candidate_k": 20}) as span:
        span.annotate(output_count=3)
    finish_trace(trace_id, duration_ms=12)

    detail = trace_detail(trace_id)
    assert detail is not None
    assert detail["attributes"]["api_key"] == "[REDACTED]"
    assert detail["status"] == "succeeded"
    assert detail["spans"][0]["operation"] == "bm25_search"
    assert detail["spans"][0]["attributes"]["output_count"] == 3


def test_persisted_runner_is_idempotent_and_completes(monkeypatch, tmp_path: Path):
    _prepare_database(monkeypatch, tmp_path)
    from app.infra import LocalJobRunner

    runner = LocalJobRunner(max_workers=1)
    runner.register("sample", lambda payload, context: {"value": payload["value"] * 2})
    runner.start()
    try:
        first = runner.enqueue("sample", {"value": 4}, idempotency_key="same-work")
        second = runner.enqueue("sample", {"value": 4}, idempotency_key="same-work")
        assert first["id"] == second["id"]
        deadline = time.time() + 5
        state = runner.get(first["id"])
        while state and state["status"] not in {"succeeded", "failed"} and time.time() < deadline:
            time.sleep(0.02)
            state = runner.get(first["id"])
        assert state is not None
        assert state["status"] == "succeeded"
        assert state["result_summary"] == {"value": 8}
    finally:
        runner.shutdown()


def test_index_generation_really_changes_chunk_layout(monkeypatch, tmp_path: Path):
    _prepare_database(monkeypatch, tmp_path)
    space_id, document_id = _insert_space_and_document(tmp_path)
    from app.privacy import save_document_cloud_policy
    from app.retrieval_engine import create_index_generation, estimate_index_generation

    save_document_cloud_policy(document_id, embedding_allowed=True, llm_allowed=False)

    small = create_index_generation(
        space_id=space_id, model="text-embedding-v4", dimension=256,
        strategy="flat", chunk_size=400, chunk_overlap=80,
    )
    large = create_index_generation(
        space_id=space_id, model="text-embedding-v4", dimension=256,
        strategy="flat", chunk_size=1000, chunk_overlap=150,
    )
    small_estimate = estimate_index_generation(small["id"])
    large_estimate = estimate_index_generation(large["id"])

    assert small["config_hash"] != large["config_hash"]
    assert small_estimate["chunk_count"] > large_estimate["chunk_count"]
    assert small_estimate["estimated_batches"] >= large_estimate["estimated_batches"]


def test_document_cloud_policy_is_deny_by_default(monkeypatch, tmp_path: Path):
    _prepare_database(monkeypatch, tmp_path)
    space_id, document_id = _insert_space_and_document(tmp_path)
    from app.privacy import allowed_for_cloud, document_cloud_policies, save_document_cloud_policy

    assert allowed_for_cloud([document_id], "embedding") == set()
    policy = document_cloud_policies(space_id)[0]
    assert policy["embedding_allowed"] == 0
    assert policy["llm_allowed"] == 0
    save_document_cloud_policy(document_id, embedding_allowed=True, llm_allowed=False)
    assert allowed_for_cloud([document_id], "embedding") == {document_id}
    assert allowed_for_cloud([document_id], "llm") == set()


def test_infra_api_exposes_budget_and_observability(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KUN_DATA_DIR", str(tmp_path))
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        response = client.put("/api/infra/budget", json={
            "max_api_requests_per_run": 120,
            "max_embedding_input_characters": 900_000,
            "allow_multi_model_rebuild": False,
        })
        assert response.status_code == 200
        assert response.json()["max_api_requests_per_run"] == 120
        overview = client.get("/api/infra/overview")
        assert overview.status_code == 200
        assert "jobs" in overview.json()
        assert "traces" in overview.json()


def test_quality_experiment_keeps_gold_and_bad_case_metrics(monkeypatch, tmp_path: Path):
    _prepare_database(monkeypatch, tmp_path)
    space_id, document_id = _insert_space_and_document(tmp_path)
    from app import experiments
    from app.database import connect, json_value, now

    dataset = experiments.create_dataset(space_id, "Gold", "v1")
    stamp = now()
    case_id = "gold-case-1"
    with connect() as db:
        db.execute(
            """INSERT INTO eval_dataset_cases(
               id,dataset_version_id,question,split,query_type,difficulty,status,gold_json,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                case_id, dataset["id"], "什么需要可观测性？", "dev", "fact", "easy", "accepted",
                json_value([{"document_id": document_id, "locator": "Test", "relevance": 3}]), stamp, stamp,
            ),
        )
        db.execute("UPDATE eval_dataset_versions SET status='ready',case_count=1 WHERE id=?", (dataset["id"],))

    monkeypatch.setattr(experiments, "pipeline_search", lambda *args, **kwargs: {
        "trace_id": "case-trace", "duration_ms": 7.5, "stages": [{"stage": "bm25", "duration_ms": 2, "count": 1}],
        "results": [{
            "id": "chunk-1", "document_id": document_id, "title": "Infra source", "locator": "Test",
            "score": 0.5, "lexical_rank": 1,
        }],
    })

    run = experiments.create_experiment_run(dataset["id"], "BM25 baseline", {"pipeline": "bm25", "split": "all"})

    class Context:
        def check_cancelled(self):
            return None

        def update(self, **kwargs):
            return None

    result = experiments.run_experiment(run["id"], Context())
    assert result["document_recall"]["5"] == 1.0
    assert result["evidence_recall"]["5"] == 1.0
    assert result["mrr"] == 1.0
    assert result["citation_resolvable_rate"] == 1.0
    stored = experiments.experiment_detail(run["id"])
    assert stored is not None
    assert stored["cases"][0]["rankings"]["returned"][0]["lexical_rank"] == 1
    report, media_type = experiments.render_experiment_report(run["id"])
    assert "Evidence Recall" in report
    assert "not a production accuracy claim" in report
    assert media_type.startswith("text/markdown")
