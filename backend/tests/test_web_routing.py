import os
from pathlib import Path
from uuid import uuid4


def _message(db, conversation_id: str, role: str, content: str, now):
    db.execute(
        "INSERT INTO messages(id,conversation_id,role,content,citations_json,created_at) VALUES(?,?,?,?,?,?)",
        (uuid4().hex, conversation_id, role, content, "[]", now()),
    )


def test_web_route_does_not_pollute_new_query_and_inherits_followup(tmp_path: Path, monkeypatch):
    os.environ["KUN_DATA_DIR"] = str(tmp_path)
    from app import workflow
    from app.database import connect, init_database, now

    init_database()
    conversation_id = "web-route"
    with connect() as db:
        db.execute(
            "INSERT INTO conversations(id,title,space_id,created_at,updated_at) VALUES(?,?,?,?,?)",
            (conversation_id, "联网测试", "ai-agent-learning", now(), now()),
        )
        _message(db, conversation_id, "user", "今天杭州天气多少度", now)
        _message(db, conversation_id, "assistant", "天气回答", now)
        _message(db, conversation_id, "user", "查一查最新的AI新闻，输出1条", now)

    calls = []

    def fake_invoke(name, arguments, context):
        calls.append((name, arguments))
        return {
            "result": {"results": [], "accessed_at": now()},
            "trace": {"tool": name, "status": "succeeded", "duration_ms": 1, "result_count": 0},
        }

    monkeypatch.setattr(workflow, "invoke_tool", fake_invoke)
    state = {"question": "查一查最新的AI新闻，输出1条", "space_id": "ai-agent-learning", "conversation_id": conversation_id}
    route = workflow.understand(state)
    assert route["intent"] == "web_research"
    assert route["inherited_intent"] is False
    workflow.retrieve({**state, **route})
    assert calls[-1][1]["limit"] == 1
    assert "AI新闻" in calls[-1][1]["query"]
    assert "杭州天气" not in calls[-1][1]["query"]

    with connect() as db:
        _message(db, conversation_id, "assistant", "AI 新闻回答", now)
        _message(db, conversation_id, "user", "那 AI 融资呢", now)
    followup = {"question": "那 AI 融资呢", "space_id": "ai-agent-learning", "conversation_id": conversation_id}
    followup_route = workflow.understand(followup)
    assert followup_route["intent"] == "web_research"
    assert followup_route["inherited_intent"] is True
    workflow.retrieve({**followup, **followup_route})
    assert "最新的AI新闻" in calls[-1][1]["query"]
    assert "AI 融资" in calls[-1][1]["query"]
    assert "杭州天气" not in calls[-1][1]["query"]


def test_weather_is_time_sensitive_web_intent(tmp_path: Path):
    os.environ["KUN_DATA_DIR"] = str(tmp_path)
    from app.database import init_database
    from app.workflow import understand
    init_database()
    route = understand({"question": "杭州天气多少度", "space_id": "ai-agent-learning"})
    assert route == {
        "intent": "web_research",
        "skill": "web_research_skill",
        "previous_question": None,
        "inherited_intent": False,
    }

def test_web_citation_ranges_are_expanded():
    from app.workflow import normalize_citation_markers
    assert normalize_citation_markers("来源[1-3]和[5，7]", 6) == "来源[1][2][3]和[5]"


def test_web_retrieval_reads_top_pages_and_degrades_per_source(tmp_path: Path, monkeypatch):
    os.environ["KUN_DATA_DIR"] = str(tmp_path)
    from app import workflow
    from app.database import init_database
    from app.tools import ToolExecutionError
    init_database()

    def fake_invoke(name, arguments, context):
        if name == "web.search":
            return {"result": {"results": [
                {"url": "https://one.example/page", "title": "来源一", "site_name": "one", "snippet": "短摘要一"},
                {"url": "https://two.example/page", "title": "来源二", "site_name": "two", "snippet": "短摘要二"},
            ], "accessed_at": "2026-01-01"}, "trace": {"tool": name, "status": "succeeded", "duration_ms": 2, "result_count": 2}}
        if arguments["url"].startswith("https://one"):
            return {"result": {"text": "这是网页正文证据", "result_count": 1},
                    "trace": {"tool": name, "status": "succeeded", "duration_ms": 3, "result_count": 1}}
        raise ToolExecutionError("source_blocked", "页面禁止读取")

    monkeypatch.setattr(workflow, "invoke_tool", fake_invoke)
    state = {"question": "最新 Agent 新闻", "space_id": "ai-agent-learning", "intent": "web_research",
             "skill": "web_research_skill"}
    result = workflow.retrieve(state)
    assert result["contexts"][0]["text"] == "这是网页正文证据"
    assert result["contexts"][0]["evidence_depth"] == "full_page"
    assert result["contexts"][1]["text"] == "短摘要二"
    assert result["contexts"][1]["evidence_depth"] == "search_snippet"
    assert result["tool_trace"][-1]["status"] == "failed"
    assert result["tool_trace"][-1]["recoverable"] is True

def test_local_miss_falls_back_to_web_and_exposes_route(tmp_path: Path, monkeypatch):
    os.environ["KUN_DATA_DIR"] = str(tmp_path)
    from app import workflow
    from app.database import init_database

    init_database()

    def fake_invoke(name, arguments, context):
        if name == "rag.search":
            return {
                "result": [{
                    "id": "irrelevant", "title": "研究生评优材料", "original_name": "评优.pdf",
                    "locator": "第 1 页", "text": "摄影比赛通知和成长指南",
                    "vector_score": 0.41, "vector_rank": 1,
                }],
                "trace": {"tool": name, "status": "succeeded", "duration_ms": 2, "result_count": 1},
            }
        if name == "web.search":
            return {
                "result": {
                    "results": [{
                        "url": "https://example.com/vibe-coding", "title": "Vibe coding explanation",
                        "site_name": "Example", "snippet": "Vibe coding is an AI-assisted programming approach.",
                    }],
                    "accessed_at": "2026-07-28",
                },
                "trace": {"tool": name, "status": "succeeded", "duration_ms": 3, "result_count": 1},
            }
        return {
            "result": {"text": "Vibe coding 是一种以自然语言驱动 AI 生成和迭代代码的方式。", "result_count": 1},
            "trace": {"tool": name, "status": "succeeded", "duration_ms": 4, "result_count": 1},
        }

    monkeypatch.setattr(workflow, "invoke_tool", fake_invoke)
    result = workflow.retrieve({
        "question": "vibe coding是什么意思", "space_id": "ai-agent-learning",
        "intent": "knowledge_question", "skill": "document_skill",
    })

    assert result["local_miss"] is True
    assert result["retrieval_origin"] == "web_fallback"
    assert [trace["tool"] for trace in result["tool_trace"]] == ["rag.search", "web.search", "web.fetch"]
    assert result["contexts"][0]["kind"] == "web"
    assert result["contexts"][0]["evidence_depth"] == "full_page"

    plan = workflow.finalize_plan({
        "intent": "knowledge_question",
        "plan": workflow.build_plan({
            "question": "vibe coding是什么意思", "intent": "knowledge_question", "skill": "document_skill",
        })["plan"],
        "contexts": result["contexts"], "citations": [{"id": 1}],
        "tool_trace": result["tool_trace"], "local_miss": True,
        "fallback_reason": result["fallback_reason"], "retrieval_origin": "web_fallback",
    })["plan"]
    assert plan["fallback_skill"] == "web_research_skill"
    assert any(task["id"] == "fallback_web" and task["status"] == "completed" for task in plan["tasks"])