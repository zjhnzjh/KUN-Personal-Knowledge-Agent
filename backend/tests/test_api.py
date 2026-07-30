import os
from pathlib import Path

from fastapi.testclient import TestClient


def test_health_uses_isolated_storage(tmp_path: Path):
    os.environ["KUN_DATA_DIR"] = str(tmp_path)
    from app.main import app
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
