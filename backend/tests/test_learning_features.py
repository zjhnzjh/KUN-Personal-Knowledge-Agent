import json
import os
import sqlite3
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient


def test_legacy_database_upgrade_preserves_memory(tmp_path: Path):
    os.environ["KUN_DATA_DIR"] = str(tmp_path)
    database = tmp_path / "kun.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute("""CREATE TABLE memories (
            id TEXT PRIMARY KEY, kind TEXT NOT NULL, content TEXT NOT NULL, source TEXT NOT NULL,
            status TEXT NOT NULL, use_count INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        )""")
        db.execute("INSERT INTO memories VALUES(?,?,?,?,?,?,?,?)",
                   ("legacy", "preference", "回答简洁", "manual", "enabled", 2, "2026-01-01", "2026-01-01"))
    from app.database import connect, init_database
    init_database()
    with connect() as db:
        item = db.execute("SELECT * FROM memories WHERE id='legacy'").fetchone()
        columns = {row["name"] for row in db.execute("PRAGMA table_info(memories)")}
    assert item["content"] == "回答简洁"
    assert item["version"] == 1
    assert {"version", "conflict_with_id"}.issubset(columns)


def test_context_budget_keeps_raw_messages_and_versions_summary(tmp_path: Path):
    os.environ["KUN_DATA_DIR"] = str(tmp_path)
    from app.context import build_context
    from app.database import connect, init_database, json_value, now
    init_database()
    conversation_id = "context-test"
    with connect() as db:
        db.execute("INSERT INTO conversations(id,title,space_id,created_at,updated_at) VALUES(?,?,?,?,?)",
                   (conversation_id, "上下文测试", "ai-agent-learning", now(), now()))
        db.execute("INSERT INTO app_settings(key,value_json,updated_at) VALUES(?,?,?)",
                   ("context_policy", json_value({"model_hard_limit": 1000, "input_budget": 320, "safety_margin": 100,
                                                  "output_reserve": 100, "recent_message_minimum": 2}), now()))
        for index in range(18):
            db.execute("INSERT INTO messages(id,conversation_id,role,content,citations_json,created_at) VALUES(?,?,?,?,?,?)",
                       (uuid4().hex, conversation_id, "user" if index % 2 == 0 else "assistant",
                        f"第 {index} 条消息：请保留本地隐私约束并继续完成 Agent 项目。" * 3, "[]", now()))
    plan = build_context(conversation_id, "下一步是什么？")
    assert plan["composition"]["summarized_message_count"] > 0
    assert plan["summary_version"] == 1
    assert plan["summary"]["method"] == "rule_based"
    with connect() as db:
        assert db.execute("SELECT COUNT(*) FROM messages WHERE conversation_id=?", (conversation_id,)).fetchone()[0] == 18
        assert db.execute("SELECT COUNT(*) FROM context_summaries WHERE conversation_id=?", (conversation_id,)).fetchone()[0] == 1


def test_learning_overview_memory_audit_trace_and_progressive_schema(tmp_path: Path):
    os.environ["KUN_DATA_DIR"] = str(tmp_path)
    from app import main
    from app.database import init_database
    init_database()
    with TestClient(main.app) as client:
        catalog = client.get("/api/tools/catalog")
        catalog.raise_for_status()
        assert catalog.json()
        assert "input_schema" not in catalog.json()[0]
        assert catalog.json()[0]["parameter_summary"]
        schema = client.get("/api/tools/rag.search/schema")
        schema.raise_for_status()
        assert "properties" in schema.json()["input_schema"]

        created = client.post("/api/memories", json={"kind": "preference", "content": "回答先给结论"})
        created.raise_for_status()
        memory_id = created.json()["id"]
        disabled = client.patch(f"/api/memories/{memory_id}", json={"status": "disabled"})
        disabled.raise_for_status()
        assert disabled.json()["version"] == 2
        detail = client.get(f"/api/memories/{memory_id}")
        detail.raise_for_status()
        assert [event["event_type"] for event in detail.json()["events"]][:2] == ["disabled", "created"]

        chat = client.post("/api/chat", json={"question": "我目前在杭州", "space_id": "ai-agent-learning"})
        chat.raise_for_status()
        traces = client.get("/api/agent/traces")
        traces.raise_for_status()
        assert traces.json()[0]["status"] == "completed"
        assert [stage["stage_name"] for stage in traces.json()[0]["stages"]] == [
            "understand", "plan", "context_budget", "retrieve", "memory_candidate", "generate", "finalize_plan"
        ]

        overview = client.get("/api/learning/overview")
        overview.raise_for_status()
        data = overview.json()
        assert len(data["workflow"]) == 9
        assert {item["id"] for item in data["capability_matrix"]} >= {"short_memory", "long_memory", "rag", "tools"}
        assert data["agent"]["trace_count"] >= 1
        assert data["evaluation"]["latest"] is None
