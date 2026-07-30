from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("document", type=Path)
    parser.add_argument("--question", default="这份资料主要讲了什么？")
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    document = args.document.resolve()
    if not document.is_file():
        raise SystemExit("测试文档不存在")

    os.environ["KUN_DATA_DIR"] = str(args.data_dir.resolve() if args.data_dir else Path(tempfile.gettempdir()) / "kun-e2e-smoke")

    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        with document.open("rb") as stream:
            stage = client.post(
                "/api/documents/stage",
                files=[("files", (document.name, stream, "application/octet-stream"))],
            )
        stage.raise_for_status()
        item = stage.json()[0]
        print(
            {
                "stage": item["parse_status"],
                "metadata_source": item["metadata_source"],
                "title": item["title"],
                "sections": item["sections"],
            }
        )
        confirm = client.post(
            f"/api/documents/{item['id']}/confirm",
            json={
                "title": item["title"],
                "summary": item["summary"],
                "tags": item["tags"],
                "space_id": "ai-agent-learning",
            },
        )
        confirm.raise_for_status()
        accepted = confirm.json()
        deadline = time.monotonic() + 180
        job = None
        while time.monotonic() < deadline:
            response = client.get(f"/api/index-jobs/{accepted['job_id']}")
            response.raise_for_status()
            job = response.json()
            print(
                {
                    "index": job["status"],
                    "phase": job["phase"],
                    "progress": job["progress"],
                    "completed": job["completed"],
                    "total": job["total"],
                }
            )
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.5)
        if not job or job["status"] != "completed":
            raise SystemExit(f"索引测试未完成：{job}")
        print(
            {
                "confirm": job["status"],
                "chunks": job["total"],
                "progress": job["progress"],
            }
        )
        chat = client.post(
            "/api/chat",
            json={"question": args.question, "space_id": "ai-agent-learning"},
        )
        chat.raise_for_status()
        answer = chat.json()
        print(
            {
                "answer_chars": len(answer["answer"]),
                "citations": len(answer["citations"]),
                "tool": answer["tool_trace"][0]["tool"],
            }
        )


if __name__ == "__main__":
    main()
