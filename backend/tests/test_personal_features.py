import os
from pathlib import Path

from fastapi.testclient import TestClient


def test_conversations_memory_spaces_and_missing_copy(tmp_path: Path, monkeypatch):
    os.environ["KUN_DATA_DIR"] = str(tmp_path)

    from app import main
    from app.database import connect, init_database, json_value, now

    init_database()
    monkeypatch.setattr(
        main,
        "run",
        lambda question, space_id, conversation_id=None: {
            "answer": "已根据资料回答。",
            "citations": [],
            "tool_trace": [],
            "memory_candidate": "回答时先给结论",
        },
    )

    with TestClient(main.app) as client:
        spaces = client.get("/api/spaces")
        spaces.raise_for_status()
        assert len(spaces.json()) == 4

        chat = client.post(
            "/api/chat",
            json={"question": "请记住回答时先给结论", "space_id": "ai-agent-learning"},
        )
        chat.raise_for_status()
        result = chat.json()
        assert result["conversation_id"]
        assert result["memory_suggestion"]["status"] == "pending"

        history = client.get("/api/conversations", params={"q": "先给结论"})
        history.raise_for_status()
        assert history.json()[0]["message_count"] == 2

        detail = client.get(f"/api/conversations/{result['conversation_id']}")
        detail.raise_for_status()
        assert [item["role"] for item in detail.json()["messages"]] == ["user", "assistant"]

        memory_id = result["memory_suggestion"]["id"]
        enabled = client.patch(f"/api/memories/{memory_id}", json={"status": "enabled"})
        enabled.raise_for_status()
        assert enabled.json()["status"] == "enabled"

        document_id = "missing-copy-test"
        stamp = now()
        with connect() as db:
            db.execute(
                """INSERT INTO documents(id,space_id,original_name,library_path,file_type,size_bytes,fingerprint,
                   title,summary,tags_json,parse_status,index_status,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    document_id, "ai-agent-learning", "missing.pdf", str(tmp_path / "missing.pdf"), "pdf", 1,
                    "missing-fingerprint", "缺失测试", "", json_value([]), "parsed", "ready", stamp, stamp,
                ),
            )
        status = client.get(f"/api/documents/{document_id}/status")
        status.raise_for_status()
        assert status.json()["status"] == "missing"
        assert status.json()["library_copy_exists"] is False


def test_explicit_memory_is_really_written_without_document_citations(tmp_path: Path):
    os.environ["KUN_DATA_DIR"] = str(tmp_path)

    from app import main
    from app.database import init_database

    init_database()
    with TestClient(main.app) as client:
        result = client.post(
            "/api/chat",
            json={
                "question": "我现在在海宁，这个是让你记住的信息",
                "space_id": "ai-agent-learning",
            },
        )
        result.raise_for_status()
        data = result.json()
        assert data["citations"] == []
        assert data["memory_suggestion"]["status"] == "enabled"
        assert "真实写入长期 Memory" in data["answer"]

        memories = client.get("/api/memories")
        memories.raise_for_status()
        assert any(
            item["kind"] == "location"
            and item["content"] == "当前所在地：海宁"
            and item["status"] == "enabled"
            for item in memories.json()
        )

        short_term = client.get(
            "/api/memories/short-term",
            params={"conversation_id": data["conversation_id"]},
        )
        short_term.raise_for_status()
        facts = short_term.json()["working_facts"]
        assert facts == [{"kind": "location", "content": "当前所在地：海宁", "status": "enabled"}]


def test_stable_statement_creates_pending_memory_candidate(tmp_path: Path):
    os.environ["KUN_DATA_DIR"] = str(tmp_path)

    from app import main
    from app.database import init_database

    init_database()
    with TestClient(main.app) as client:
        result = client.post(
            "/api/chat",
            json={"question": "我目前在杭州", "space_id": "ai-agent-learning"},
        )
        result.raise_for_status()
        data = result.json()
        assert data["citations"] == []
        assert data["memory_suggestion"]["status"] == "pending"
        assert "短期 Memory" in data["answer"]
