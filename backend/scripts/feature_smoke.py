from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="kun-feature-smoke-") as folder:
        os.environ["KUN_DATA_DIR"] = folder
        from fastapi.testclient import TestClient
        from app import main as api
        from app.database import connect, init_database, json_value, now

        init_database()
        api.run = lambda question, space_id, conversation_id=None: {
            "answer": "已根据资料回答。",
            "citations": [],
            "tool_trace": [],
            "memory_candidate": {"content": "回答时先给结论", "kind": "preference"},
        }
        with TestClient(api.app) as client:
            spaces = client.get("/api/spaces")
            spaces.raise_for_status()
            assert len(spaces.json()) == 4

            chat = client.post(
                "/api/chat",
                json={"question": "请记住回答时先给结论", "space_id": "ai-agent-learning"},
            )
            chat.raise_for_status()
            result = chat.json()
            conversation_id = result["conversation_id"]
            memory_id = result["memory_suggestion"]["id"]

            history = client.get("/api/conversations", params={"q": "先给结论"})
            history.raise_for_status()
            assert history.json()[0]["message_count"] == 2

            detail = client.get(f"/api/conversations/{conversation_id}")
            detail.raise_for_status()
            assert [item["role"] for item in detail.json()["messages"]] == ["user", "assistant"]
            short_term = client.get("/api/memories/short-term", params={"conversation_id": conversation_id})
            short_term.raise_for_status()
            assert short_term.json()["message_count"] == 2

            memory = client.patch(f"/api/memories/{memory_id}", json={"status": "enabled"})
            memory.raise_for_status()
            assert memory.json()["status"] == "enabled"

            document_id = "missing-copy-test"
            stamp = now()
            with connect() as db:
                db.execute(
                    """INSERT INTO documents(id,space_id,original_name,library_path,file_type,size_bytes,fingerprint,
                       title,summary,tags_json,parse_status,index_status,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        document_id, "ai-agent-learning", "missing.pdf", str(Path(folder) / "missing.pdf"), "pdf", 1,
                        "missing-fingerprint", "缺失测试", "", json_value([]), "parsed", "ready", stamp, stamp,
                    ),
                )
            api.search = lambda question, space_id, top_k: [{
                "id": "chunk-1",
                "document_id": document_id,
                "title": "缺失测试",
                "original_name": "missing.pdf",
                "locator": "第 1 页",
                "text": "测试证据",
                "score": 0.1,
            }]
            status = client.get(f"/api/documents/{document_id}/status")
            status.raise_for_status()
            assert status.json()["status"] == "missing"
            evaluation_case = client.post("/api/rag/evaluation/cases", json={
                "space_id": "ai-agent-learning",
                "question": "测试问题",
                "expected_document_id": document_id,
                "expected_locator": "第 1 页",
            })
            evaluation_case.raise_for_status()
            evaluation = client.post("/api/rag/evaluation/run", json={"space_id": "ai-agent-learning", "top_k": 3})
            evaluation.raise_for_status()
            assert evaluation.json()["recall"] == 1.0
            skills = client.get("/api/skills")
            skills.raise_for_status()
            assert any(item["name"] == "document-skill" for item in skills.json())

            print({
                "spaces": len(spaces.json()),
                "conversation_messages": len(detail.json()["messages"]),
                "memory": memory.json()["status"],
                "missing_copy": status.json()["status"],
                "recall": evaluation.json()["recall"],
                "skills": len(skills.json()),
            })


if __name__ == "__main__":
    main()
