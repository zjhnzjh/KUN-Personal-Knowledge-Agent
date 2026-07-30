import os
from pathlib import Path

import pytest


def test_tool_registry_exposes_honest_capabilities(tmp_path: Path):
    os.environ["KUN_DATA_DIR"] = str(tmp_path)
    from app.database import init_database
    from app.tools import REGISTRY

    init_database()
    tools = {item["name"]: item for item in REGISTRY.definitions()}

    assert tools["rag.search"]["availability"] == "available"
    assert "video.probe" in tools
    assert tools["video.transcribe"]["availability"] == "unavailable"
    assert tools["video.transcribe"]["unavailable_reason"]
    assert tools["web.search"]["availability"] in {"available", "unavailable"}
    if tools["web.search"]["availability"] == "unavailable":
        assert tools["web.search"]["unavailable_reason"]
    assert tools["memory.propose"]["availability"] == "available"
    assert tools["memory.write"]["confirmation_required"] is True


def test_rag_tool_enforces_scope_and_records_trace(tmp_path: Path):
    os.environ["KUN_DATA_DIR"] = str(tmp_path)
    from app.database import init_database
    from app.tools import ToolContext, ToolExecutionError, invoke_tool, tool_runs

    init_database()

    with pytest.raises(ToolExecutionError) as denied:
        invoke_tool(
            "rag.search",
            {"query": "Agent", "space_id": "ai-agent-learning", "top_k": 5},
            ToolContext(),
        )
    assert denied.value.code == "read_scope_denied"

    execution = invoke_tool(
        "rag.search",
        {"query": "Agent", "space_id": "ai-agent-learning", "top_k": 5},
        ToolContext(read_scopes={"kun_index"}),
    )
    assert execution["result"] == []
    assert execution["trace"]["tool"] == "rag.search"
    assert execution["trace"]["status"] == "succeeded"
    assert tool_runs(1)[0]["tool_name"] == "rag.search"
