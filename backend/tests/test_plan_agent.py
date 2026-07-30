import os
from pathlib import Path


def test_memory_questions_are_not_extracted_as_facts(tmp_path: Path):
    os.environ["KUN_DATA_DIR"] = str(tmp_path)
    from app.database import init_database
    from app.workflow import detect_memory_candidate, understand
    init_database()
    for question in ("我在哪", "我在哪里", "我喜欢吃什么", "我喜欢什么"):
        assert detect_memory_candidate(question) is None
        assert understand({"question": question, "space_id": "ai-agent-learning"})["intent"] == "memory_query"


def test_compound_request_keeps_main_task_and_atomic_memory(tmp_path: Path):
    os.environ["KUN_DATA_DIR"] = str(tmp_path)
    from app.database import init_database
    from app.workflow import build_plan, detect_memory_candidate, understand
    init_database()
    question = "我现在在杭州实习，我有师兄也在这里，他在阿里，我怎么投阿里的秋招呢"
    route = understand({"question": question, "space_id": "ai-agent-learning"})
    assert route["intent"] == "web_research"
    candidate = detect_memory_candidate(question)
    assert candidate == {
        "content": "当前所在地：杭州", "value": "杭州", "kind": "location", "label": "所在地",
        "status": "pending", "explicit": False,
    }
    plan = build_plan({"question": question, **route})["plan"]
    assert [item["id"] for item in plan["tasks"]] == [
        "understand_request", "retrieve_web", "compose_answer", "propose_memory"
    ]
    assert plan["tasks"][-1]["detail"] == "当前所在地：杭州"


def test_explicit_memory_side_effect_does_not_hijack_answer(tmp_path: Path):
    os.environ["KUN_DATA_DIR"] = str(tmp_path)
    from app.database import init_database
    from app.workflow import detect_memory_candidate, understand
    init_database()
    question = "请记住我在杭州，并告诉我今天的天气"
    route = understand({"question": question, "space_id": "ai-agent-learning"})
    assert route["intent"] == "web_research"
    assert detect_memory_candidate(question)["status"] == "enabled"


def test_plan_is_persisted_with_assistant_message(tmp_path: Path, monkeypatch):
    os.environ["KUN_DATA_DIR"] = str(tmp_path)
    from app import main
    from app.database import init_database
    from fastapi.testclient import TestClient
    init_database()

    def fake_run(question, space_id, conversation_id=None):
        return {
            "answer": "已完成回答", "citations": [], "tool_trace": [], "memory_candidate": None,
            "plan": {"version": 1, "planner": "deterministic_plan_v1", "goal": question,
                     "intent": "knowledge_question", "status": "completed", "tasks": []},
        }

    monkeypatch.setattr(main, "run", fake_run)
    with TestClient(main.app) as client:
        response = client.post("/api/chat", json={"question": "测试任务流", "space_id": "ai-agent-learning"})
        response.raise_for_status()
        conversation_id = response.json()["conversation_id"]
        detail = client.get(f"/api/conversations/{conversation_id}")
        detail.raise_for_status()
        assert detail.json()["messages"][-1]["plan"]["planner"] == "deterministic_plan_v1"

def test_local_recommendation_selects_skill_and_web_tool(tmp_path: Path):
    os.environ["KUN_DATA_DIR"] = str(tmp_path)
    from app.database import init_database
    from app.workflow import build_plan, understand
    init_database()
    question = "我晚上想出去玩，西湖旁边有没有好吃的"
    route = understand({"question": question, "space_id": "ai-agent-learning"})
    assert route["intent"] == "web_research"
    assert route["skill"] == "recommendation_skill"
    plan = build_plan({"question": question, **route})["plan"]
    assert plan["skill"] == "recommendation_skill"
    assert plan["tasks"][1]["id"] == "retrieve_web"
    assert "不能用无关本地资料" in plan["route_reason"]