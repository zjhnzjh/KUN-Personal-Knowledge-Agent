from __future__ import annotations

import json
import hashlib
import re
from datetime import date
from time import perf_counter
from uuid import uuid4
from typing import TypedDict

import httpx
from langgraph.graph import END, START, StateGraph

from .config import get_settings
from .context import build_context, estimate_tokens
from .database import connect, json_value, now, rows
from .memory import recall_memories
from .privacy import allowed_for_cloud, get_privacy_settings
from .tools import REGISTRY, ToolContext, ToolExecutionError, invoke_tool


class AgentState(TypedDict, total=False):
    question: str
    space_id: str
    intent: str
    skill: str
    contexts: list[dict]
    answer: str
    citations: list[dict]
    tool_trace: list[dict]
    memory_candidate: dict | None
    policy_message: str
    conversation_id: str | None
    context_plan: dict
    memory_recalls: list[dict]
    stage_events: list[dict]
    trace_id: str
    previous_question: str | None
    inherited_intent: bool
    plan: dict
    local_miss: bool
    fallback_reason: str
    retrieval_origin: str


MEMORY_QUERY_MARKERS = (
    "你记得我", "你记住了", "你有没有记住", "关于我", "我是谁", "我的长期记忆",
    "我的偏好是什么", "我有哪些", "我有几个", "我在哪里", "我住哪里", "没有记住",
    "记忆里有什么", "记忆中有什么", "我在哪", "我住在哪", "我喜欢什么", "我喜欢吃什么",
)
MEMORY_EXPLICIT_MARKERS = (
    "请记住", "帮我记住", "让你记住", "这个要记住", "这是让你记住", "长期记住",
    "加入长期记忆", "保存到长期记忆", "以后别忘", "别忘了",
)
MEMORY_KIND_LABELS = {
    "identity": "身份信息",
    "location": "所在地",
    "preference": "偏好",
    "goal": "长期目标",
    "project": "当前项目",
    "relationship": "人物关系",
}


def detect_memory_candidate(question: str) -> dict | None:
    text = re.sub(r"\s+", " ", question).strip()
    if re.search(r"(身份证|银行卡|密码|验证码|精确住址|手机号)", text):
        return None
    explicit = any(marker in text for marker in MEMORY_EXPLICIT_MARKERS)
    patterns = (
        ("location", r"我(?:现在|目前)?(?:正)?在(?P<value>[^，。！？,!?]{1,40}?)(?:实习|工作|学习|生活|出差|旅游|，|,|。|！|？|$)"),
        ("location", r"我住在(?P<value>[^，。！？,!?]{1,80})"),
        ("identity", r"(?:我叫|我的名字是)(?P<value>[^，。！？,!?]{1,80})"),
        ("identity", r"我是(?P<value>[^，。！？,!?]{1,80})"),
        ("preference", r"(?:我喜欢|我更喜欢|我的偏好是)(?P<value>[^。！？!?]{1,140})"),
        ("preference", r"以后回答(?P<value>[^。！？!?]{1,140})"),
        ("goal", r"(?:我的目标是|我计划|我希望以后)(?P<value>[^。！？!?]{1,140})"),
        ("project", r"(?:我正在做|我目前在做|我的项目是)(?P<value>[^。！？!?]{1,140})"),
        ("relationship", r"(?:我的师兄|我的导师|我的同事)(?P<value>[^。！？!?]{1,140})"),
    )
    for kind, pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        value = match.group("value").strip(" ：:，,。")
        if not value or re.search(r"(哪|哪里|什么|谁|怎么|如何|是否|能否|吗|呢)$", value):
            continue
        if kind == "location":
            content = f"当前所在地：{value}"
        elif kind == "identity":
            content = f"姓名或身份：{value}"
        elif kind == "preference" and pattern.startswith("以后回答"):
            content = f"回答偏好：{value}"
        elif kind == "preference":
            content = f"偏好：{value}"
        elif kind == "goal":
            content = f"长期目标：{value}"
        elif kind == "project":
            content = f"当前项目：{value}"
        elif kind == "relationship":
            content = f"人物关系：{value}"
        else:
            content = match.group(0).strip(" ：:，,。")
        return {
            "content": content[:240],
            "value": value[:180],
            "kind": kind,
            "label": MEMORY_KIND_LABELS.get(kind, "长期信息"),
            "status": "enabled" if explicit else "pending",
            "explicit": explicit,
        }
    if explicit:
        cleaned = text
        for marker in MEMORY_EXPLICIT_MARKERS:
            cleaned = cleaned.replace(marker, " ")
        cleaned = re.sub(r"(这个是|这是|的信息|这条信息|：|:)", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ，,。")
        if cleaned:
            return {
                "content": cleaned[:240],
                "kind": "preference",
                "status": "enabled",
                "explicit": True,
            }
    return None


WEB_MARKERS = (
    "联网", "网页", "搜索网络", "上网查", "网上查", "网络搜索", "查一查", "帮我查",
    "最新", "今天", "今日", "实时", "新闻", "天气", "气温", "温度", "空气质量",
    "当前价格", "当前政策", "截至目前", "本周", "本月", "最近发生",
    "秋招", "春招", "校招", "招聘", "投递", "岗位", "职位", "招聘官网",
    "推荐", "附近", "周边", "好吃", "餐厅", "饭店", "咖啡店", "去哪玩", "哪里玩", "营业时间", "地址", "门票",
)
RECOMMENDATION_MARKERS = ("推荐", "附近", "周边", "好吃", "餐厅", "饭店", "咖啡店", "去哪玩", "哪里玩", "本地生活")
WEB_FOLLOWUP_PREFIXES = ("那", "那么", "还有", "再查", "再看看", "换成", "关于", "至于")
WEB_FOLLOWUP_SUFFIXES = ("呢", "怎么样", "如何", "有吗", "还有吗")


def _previous_user_question(conversation_id: str | None, current_question: str) -> str | None:
    if not conversation_id:
        return None
    recent = rows(
        """SELECT content FROM messages WHERE conversation_id=? AND role='user'
           ORDER BY rowid DESC LIMIT 3""",
        (conversation_id,),
    )
    for item in recent:
        if item["content"] != current_question:
            return item["content"]
    return None


def _previous_agent_intent(conversation_id: str | None) -> str | None:
    if not conversation_id:
        return None
    traces = rows(
        """SELECT intent FROM agent_traces WHERE conversation_id=? AND status='completed'
           ORDER BY started_at DESC, rowid DESC LIMIT 1""",
        (conversation_id,),
    )
    return traces[0]["intent"] if traces else None


def _is_web_followup(text: str) -> bool:
    clean = text.strip()
    return len(clean) <= 80 and (clean.startswith(WEB_FOLLOWUP_PREFIXES) or clean.endswith(WEB_FOLLOWUP_SUFFIXES))


def _has_actionable_request(text: str) -> bool:
    return bool(
        re.search(r"[？?]", text)
        or re.search(r"(怎么|如何|哪里|在哪|什么|谁|是否|能否|可以吗|请帮|帮我查|查一查|告诉我|给我|分析|总结|比较|推荐)", text)
    )


def _requested_web_limit(text: str, default: int = 8) -> int:
    match = re.search(r"(?:输出|返回|给我|列出|展示)\s*([1-9]|10)\s*条", text)
    if match:
        return max(1, min(int(match.group(1)), 10))
    chinese = re.search(r"(?:输出|返回|给我|列出|展示)\s*([一二三四五六七八九十])\s*条", text)
    if chinese:
        values = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        return values[chinese.group(1)]
    return default


def understand(state: AgentState) -> AgentState:
    text = state["question"]
    previous_question = _previous_user_question(state.get("conversation_id"), text)
    if any(word in text for word in MEMORY_QUERY_MARKERS):
        return {"intent": "memory_query", "skill": "memory_skill", "previous_question": previous_question}
    candidate = detect_memory_candidate(text)
    if (any(word in text for word in MEMORY_EXPLICIT_MARKERS) or candidate) and not _has_actionable_request(text):
        return {"intent": "memory_setting", "skill": "memory_skill", "previous_question": previous_question}
    if any(word in text for word in ("图片", "截图", "照片")):
        return {"intent": "image_search", "skill": "image_skill", "previous_question": previous_question}
    if any(word in text for word in ("Excel", "表格", "统计", "工作表")):
        return {"intent": "table_analysis", "skill": "excel_skill", "previous_question": previous_question}
    if any(word in text for word in ("视频", "字幕", "关键帧", "时间戳")):
        return {"intent": "video_learning", "skill": "video_skill", "previous_question": previous_question}
    if any(word in text for word in RECOMMENDATION_MARKERS):
        return {"intent": "web_research", "skill": "recommendation_skill", "previous_question": previous_question, "inherited_intent": False}
    if any(word in text for word in WEB_MARKERS):
        return {"intent": "web_research", "skill": "web_research_skill", "previous_question": previous_question, "inherited_intent": False}
    previous_was_web = (_previous_agent_intent(state.get("conversation_id")) == "web_research" or bool(previous_question and any(word in previous_question for word in WEB_MARKERS)))
    if previous_was_web and _is_web_followup(text):
        return {"intent": "web_research", "skill": "web_research_skill", "previous_question": previous_question, "inherited_intent": True}
    return {"intent": "knowledge_question", "skill": "document_skill", "previous_question": previous_question}

def build_plan(state: AgentState) -> AgentState:
    """Build an observable task plan, never private chain-of-thought."""
    intent = state.get("intent", "knowledge_question")
    candidate = detect_memory_candidate(state["question"])
    task_by_intent = {
        "web_research": ("retrieve_web", "搜索并核验公开网页", "public_web", "问题需要当前或外部信息，本地资料不足以可靠回答"),
        "memory_query": ("recall_memory", "读取已确认的长期 Memory", "local_memory", "问题在询问用户此前确认过的信息"),
        "image_search": ("search_images", "搜索本地图片语义索引", "kun_images", "问题明确依赖图片或截图内容"),
        "table_analysis": ("retrieve_tables", "检索表格与单元格资料", "kun_index", "问题需要表格结构或单元格证据"),
        "video_learning": ("retrieve_video", "检索实验性视频学习资料", "kun_index", "问题明确依赖视频、字幕或关键帧"),
        "memory_setting": ("prepare_memory", "整理可确认的长期 Memory", "local_memory", "本轮只有稳定信息陈述，没有需要继续执行的主问题"),
        "knowledge_question": ("retrieve_knowledge", "检索当前知识空间", "kun_index", "问题优先从已确认的本地资料寻找依据"),
    }
    action_id, action_title, source, route_reason = task_by_intent[intent]
    if state.get("skill") == "recommendation_skill":
        action_title = "搜索并筛选当前本地推荐"
        route_reason = "餐饮、出行或附近推荐依赖当前公开信息，不能用无关本地资料代替"
    tasks = [
        {"id": "understand_request", "title": "理解并拆解请求", "status": "completed", "source": "user_message",
         "detail": f"识别为 {intent}"},
        {"id": action_id, "title": action_title, "status": "in_progress", "source": source,
         "detail": "等待 Tool 返回真实结果"},
        {"id": "compose_answer", "title": "基于证据生成回答", "status": "pending", "source": "model",
         "detail": "模型只负责组织语言，事实必须由 Tool 结果或 Memory 支持"},
    ]
    if candidate and intent != "memory_query":
        tasks.append({"id": "propose_memory", "title": f"提议保存{candidate.get('label', '长期信息')}",
                      "status": "pending", "source": "local_memory", "detail": candidate["content"]})
    return {"plan": {
        "version": 2,
        "planner": "deterministic_plan_v2",
        "goal": state["question"].strip(),
        "intent": intent,
        "skill": state.get("skill"),
        "route_reason": route_reason,
        "tasks": tasks,
        "grounding": {"status": "pending", "risk": "unknown", "label": "等待证据", "explanation": "检索尚未完成"},
    }}

def prepare_working_context(state: AgentState) -> AgentState:
    return {"context_plan": build_context(state.get("conversation_id"), state["question"])}


def _local_relevance_gate(question: str, contexts: list[dict]) -> tuple[list[dict], str]:
    """Reject nearest-neighbour false positives without exporting private document text."""
    if not contexts:
        return [], "本地知识库没有返回候选片段"
    latin_terms = {
        term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9.+#-]{1,}", question)
        if term.lower() not in {"the", "a", "an", "is", "are", "app"}
    }
    chinese_runs = re.findall(r"[\u4e00-\u9fff]{2,}", question)
    stop_bigrams = {"什么", "意思", "怎么", "如何", "分别", "一下", "这个", "哪些", "是否", "可以"}
    bigrams = {
        run[index:index + 2]
        for run in chinese_runs
        for index in range(len(run) - 1)
        if run[index:index + 2] not in stop_bigrams
    }
    selected = []
    for item in contexts:
        haystack = f"{item.get('title', '')}\n{item.get('text', '')}".lower()
        latin_ok = not latin_terms or all(term in haystack for term in latin_terms)
        chinese_overlap = (sum(1 for term in bigrams if term in haystack) / len(bigrams)) if bigrams else 1.0
        lexical = item.get("lexical_rank") is not None
        vector_score = float(item.get("vector_score") or 0)
        if latin_ok and ((lexical and chinese_overlap >= 0.2) or vector_score >= 0.55):
            selected.append(item)
    if selected:
        return selected, "本地词项覆盖与检索分数通过相关性门控"
    return [], "候选只是近邻结果，未完整命中问题术语或关键词，因此不作为本地证据"


def _retrieve_public_web(state: AgentState, retrieval_query: str) -> AgentState:
    if state.get("skill") == "recommendation_skill":
        retrieval_query += (
            "。这是本地生活推荐请求：优先返回当前可核验的具体地点、官方地图或商家页面。"
            "可靠本地生活来源；尽量包含店名、地址或商圈。无法核验评分、价格或营业状态时不要编造。"
        )
    dated_query = (
        f"当前日期是 {date.today().isoformat()}。{retrieval_query}。"
        "优先返回最接近当前日期且明确标注发布日期的可靠来源；"
        "不要把旧报道当作今天发生的新闻。"
    )
    execution = invoke_tool(
        "web.search",
        {"query": dated_query, "limit": _requested_web_limit(state["question"])},
        ToolContext(network_scopes={"public_web"}, conversation_id=state.get("conversation_id")),
    )
    result = execution["result"]
    web_traces = [execution["trace"]]
    fetched_text: dict[str, str] = {}
    for item in result.get("results", [])[:2]:
        url = str(item.get("url") or "")
        if not url.startswith("https://"):
            continue
        try:
            fetched = invoke_tool(
                "web.fetch", {"url": url},
                ToolContext(network_scopes={"public_web"}, conversation_id=state.get("conversation_id")),
            )
            web_traces.append(fetched["trace"])
            fetched_text[url] = str(fetched["result"].get("text") or "")[:4000]
        except ToolExecutionError as error:
            web_traces.append({
                "tool": "web.fetch", "status": "failed", "error_code": error.code,
                "recoverable": True, "result_count": 0, "duration_ms": 0,
            })
    contexts = []
    for item in result.get("results", []):
        url = item.get("url", "")
        contexts.append({
            "id": f"web-{hashlib.sha1(url.encode('utf-8')).hexdigest()[:16]}",
            "title": item.get("title") or "网页来源",
            "original_name": item.get("site_name") or "公开网页",
            "locator": item.get("published_at") or "网页",
            "text": fetched_text.get(url) or item.get("snippet") or item.get("title") or "",
            "evidence_depth": "full_page" if fetched_text.get(url) else "search_snippet",
            "url": url, "kind": "web",
            "site_name": item.get("site_name") or "公开网页",
            "accessed_at": result.get("accessed_at"),
        })
    return {"contexts": contexts, "tool_trace": web_traces, "retrieval_origin": "public_web"}

def retrieve(state: AgentState) -> AgentState:
    privacy = get_privacy_settings()
    if state.get("intent") == "web_research" and not privacy["web_search_enabled"]:
        return {
            "contexts": [],
            "tool_trace": [],
            "policy_message": "联网搜索已在“设置 → 隐私与权限”中关闭，本次没有向公网发送查询。",
        }
    if state.get("intent") == "memory_setting" and not privacy["memory_suggestions_enabled"]:
        return {
            "contexts": [],
            "tool_trace": [],
            "policy_message": "对话记忆建议已在“设置 → 隐私与权限”中关闭，本次没有提取或保存个人信息。",
        }
    if state.get("intent") in {"memory_setting", "memory_query"}:
        return {"contexts": [], "tool_trace": []}
    retrieval_query = state["question"]
    if state.get("intent") == "web_research" and state.get("inherited_intent") and state.get("previous_question"):
        retrieval_query = f"{state['previous_question']}\n联网追问：{state['question']}"
    elif state.get("intent") != "web_research" and state.get("previous_question") and len(state["question"]) < 80:
        retrieval_query = f"{state['previous_question']}\n后续问题：{state['question']}"
    if state.get("intent") == "image_search":
        execution = invoke_tool(
            "image.search",
            {"query": retrieval_query, "space_id": state["space_id"], "limit": 5},
            ToolContext(read_scopes={"kun_images"}, conversation_id=state.get("conversation_id")),
        )
        contexts = [{
            **item,
            "id": item.get("chunk_id"),
            "locator": "整张图片",
            "text": "\n".join(part for part in (item.get("description"), item.get("ocr_text")) if part),
        } for item in execution["result"] if item.get("chunk_id")]
        return {"contexts": contexts, "tool_trace": [execution["trace"]]}
    if state.get("intent") == "web_research":
        return _retrieve_public_web(state, retrieval_query)
    top_k = 6 if any(word in state["question"] for word in ("总结", "概括", "对比", "有哪些", "全部")) else 3
    execution = invoke_tool(
        "rag.search",
        {"query": retrieval_query, "space_id": state["space_id"], "top_k": top_k},
        ToolContext(read_scopes={"kun_index"}, conversation_id=state.get("conversation_id")),
    )
    local_contexts, relevance_reason = _local_relevance_gate(state["question"], execution["result"])
    if local_contexts:
        return {
            "contexts": local_contexts,
            "tool_trace": [execution["trace"]],
            "retrieval_origin": "local_knowledge",
        }
    if privacy["web_search_enabled"] and state.get("intent") == "knowledge_question":
        web_result = _retrieve_public_web(
            {**state, "skill": "web_research_skill"},
            f"{state['question']}。请查找可直接解释该问题的可靠公开来源",
        )
        return {
            **web_result,
            "tool_trace": [execution["trace"], *web_result.get("tool_trace", [])],
            "local_miss": True,
            "fallback_reason": relevance_reason,
            "retrieval_origin": "web_fallback",
        }
    return {
        "contexts": [],
        "tool_trace": [execution["trace"]],
        "local_miss": True,
        "fallback_reason": relevance_reason,
        "retrieval_origin": "local_miss",
    }


def normalize_citation_markers(answer: str, maximum: int) -> str:
    """Turn [1-3], [1, 3] and [1、3] into the claim parser's canonical [1][2][3]."""
    def expand(match: re.Match) -> str:
        raw = match.group(1)
        values: list[int] = []
        for part in re.split(r"\s*[,，、]\s*", raw):
            range_match = re.fullmatch(r"(\d+)\s*[-–—]\s*(\d+)", part)
            if range_match:
                start, end = int(range_match.group(1)), int(range_match.group(2))
                if start <= end and end - start <= 20:
                    values.extend(range(start, end + 1))
                continue
            if part.strip().isdigit():
                values.append(int(part.strip()))
        unique = [value for value in dict.fromkeys(values) if 1 <= value <= maximum]
        return "".join(f"[{value}]" for value in unique)
    return re.sub(r"\[(\d+(?:\s*[-–—,，、]\s*\d+)*)\]", expand, answer)

def generate(state: AgentState) -> AgentState:
    if state.get("policy_message"):
        return {"answer": f"**本次操作已被隐私设置拦截。**\n\n{state['policy_message']}", "citations": []}
    contexts = state.get("contexts", [])
    citations = [{
        "id": i,
        "title": item["title"],
        "file": item["original_name"],
        "locator": item["locator"],
        "chunk_id": item["id"],
        "kind": item.get("kind", "document"),
        "url": item.get("url"),
        "site_name": item.get("site_name"),
    } for i, item in enumerate(contexts, 1)]
    if state.get("intent") == "memory_setting":
        candidate = state.get("memory_candidate")
        if not candidate:
            return {
                "answer": "我知道你想让我记住，但还没提取到一条清晰、可保存的信息。可以直接说：**“请记住，我现在在……”**。",
                "citations": [],
            }
        label = MEMORY_KIND_LABELS.get(candidate["kind"], "长期信息")
        if candidate.get("status") == "enabled":
            return {
                "answer": (
                    f"**记住了。** 已真实写入长期 Memory：\n\n"
                    f"- {candidate['content']}\n\n"
                    "这不是文档事实，因此不会附加知识库引用。你可以在 Memory 中心查看、编辑或停用。"
                ),
                "citations": [],
            }
        return {
            "answer": (
                f"收到，这条信息已经进入当前对话的**短期 Memory**：\n\n"
                f"- {candidate['content']}\n\n"
                "我也把它整理成了长期记忆候选；点击下方“记住”后，才会跨对话使用。"
            ),
            "citations": [],
        }
    memories = recall_memories(
        state["question"], state.get("conversation_id"),
        include_all=state.get("intent") == "memory_query",
    )
    if state.get("intent") == "memory_query":
        if not memories:
            return {
                "answer": "我翻了翻自己的小本本，目前还没有你确认过的长期记忆。你可以在对话里告诉我稳定的偏好或背景，我会先生成建议，等你确认后再记住。",
                "citations": [],
            }
        remembered = "\n".join(f"- {item['content']}" for item in memories)
        return {
            "answer": f"当然记得，我的小本本里现在有这些：\n\n{remembered}\n\n这些来自你确认过的长期 Memory，不是文档检索结果，所以这里不会伪造资料引用。",
            "citations": [],
        }
    if not contexts:
        if state.get("local_miss"):
            return {"answer": "**本地知识库未命中。** 本轮没有获得可核验的联网来源，因此不使用模型常识冒充答案。请检查联网权限或换一个更具体的关键词。", "citations": []}
        if state.get("intent") == "web_research":
            return {"answer": "这次联网搜索没有返回可核验的网页来源。阿里联网 Tool 已执行，但我不会把无来源的模型内容当成事实；可以换一个更具体的关键词再试。", "citations": []}
        return {"answer": "当前知识空间里还没有找到足够依据。你可以补充关键词，或先添加相关资料。", "citations": []}
    evidence = "\n\n".join(f"[{i}] {item['title']} / {item['locator']}\n{item['text']}" for i, item in enumerate(contexts, 1))
    settings = get_settings()
    memory_context = "\n".join(f"- {item['content']}" for item in memories)
    context_plan = state.get("context_plan") or build_context(state.get("conversation_id"), state["question"])
    history_context = context_plan.get("history_text", "")
    summary_context = context_plan.get("summary_text", "")
    local_document_ids = list({str(item["document_id"]) for item in contexts if item.get("kind", "document") == "document" and item.get("document_id")})
    llm_allowed_documents = allowed_for_cloud(local_document_ids, "llm")
    cloud_generation_allowed = not local_document_ids or set(local_document_ids).issubset(llm_allowed_documents)
    if settings.deepseek_api_key and cloud_generation_allowed:
        response = httpx.post(
            f"{settings.deepseek_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.deepseek_api_key}"},
            json={"model": settings.deepseek_model, "temperature": 0.2, "messages": [
                {"role": "system", "content": (
                    "你是坤坤，一个原创、机灵但不油腻的个人知识助手。回答要少废话、结论优先、层次清楚。"
                    "使用标准 Markdown：短标题、加粗重点、列表；不要输出裸露的星号，也不要为了俏皮牺牲准确性。"
                    "只根据资料证据回答事实，每个关键事实只引用真正支持它的证据编号 [数字]；"
                    "不要为了凑数量引用，不要引用没有用于回答的片段。证据不足时明确说明。"
                    "网页证据同样必须逐条引用；不要把搜索摘要扩写成来源没有支持的事实。"
                    f"当前日期是 {date.today().isoformat()}。若网页没有明确发布日期，"
                    "不得声称它发生在“今天”；时效证据不足时要直接说明。"
                    "长期记忆只用于调整表达和理解用户，不得把记忆伪装成资料证据。"
                    f"\n\n已确认的用户长期记忆：\n{memory_context or '无'}"
                )},
                {"role": "user", "content": (
                    f"历史摘要（规则式、可从原消息重建）：\n{summary_context or '无'}\n\n"
                    f"最近完整消息（仅用于理解上下文，不作为事实证据）：\n{history_context or '无'}"
                    f"\n\n当前问题：{state['question']}\n\n资料证据：\n{evidence}"
                )},
            ]}, timeout=45,
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"]
    else:
        reason = "这些资料没有授权发送给 DeepSeek" if settings.deepseek_api_key else "尚未配置对话模型"
        answer = f"我找到了以下相关资料，但{reason}，因此先展示可核对的本地证据：\n\n" + "\n\n".join(f"[{i}] {item['text'][:260]}" for i, item in enumerate(contexts, 1))
    answer = normalize_citation_markers(answer, len(citations))
    valid_ids = {item["id"] for item in citations}
    answer = re.sub(
        r"\[(\d+)\]",
        lambda match: match.group(0) if int(match.group(1)) in valid_ids else "",
        answer,
    )
    used_ids = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}
    used_citations = [item for item in citations if item["id"] in used_ids]
    renumber = {item["id"]: index for index, item in enumerate(used_citations, 1)}
    if renumber:
        answer = re.sub(
            r"\[(\d+)\]",
            lambda match: f"[{renumber[int(match.group(1))]}]",
            answer,
        )
        used_citations = [
            {**item, "id": renumber[item["id"]]}
            for item in used_citations
        ]
    if state.get("retrieval_origin") in {"public_web", "web_fallback"} and contexts and not used_citations:
        return {
            "answer": "联网检索已经返回网页，但本次生成结果没有通过引用校验，因此没有展示无来源结论。请点击重新生成，或把问题再限定得更具体。",
            "citations": [],
        }
    if state.get("retrieval_origin") == "web_fallback":
        answer = "**本地知识库未命中，已自动转为联网检索。**\n\n" + answer
    return {"answer": answer, "citations": used_citations}


def finalize_plan(state: AgentState) -> AgentState:
    plan = dict(state.get("plan") or {})
    contexts = state.get("contexts", [])
    citations = state.get("citations", [])
    traces = state.get("tool_trace", [])
    failed_tool = next((item for item in traces if item.get("status") != "succeeded" and not item.get("recoverable")), None)
    policy_blocked = bool(state.get("policy_message"))
    finalized = []
    retrieval_ids = {"retrieve_web", "retrieve_knowledge", "retrieve_tables", "retrieve_video", "search_images", "recall_memory", "prepare_memory"}
    for task in plan.get("tasks", []):
        next_task = dict(task)
        if task["id"] == "propose_memory":
            candidate = state.get("memory_candidate") or {}
            next_task["status"] = "completed" if candidate.get("status") == "enabled" else "awaiting_confirmation"
            next_task["detail"] = "已写入长期 Memory" if candidate.get("status") == "enabled" else "候选已生成，未确认前不会跨对话使用"
        elif task["id"] in retrieval_ids:
            next_task["status"] = "failed" if failed_tool or policy_blocked else "completed"
            if policy_blocked:
                next_task["detail"] = state.get("policy_message")
            elif failed_tool:
                next_task["detail"] = f"{failed_tool.get('tool', 'Tool')} 执行失败：{failed_tool.get('error_code') or 'unknown'}"
            elif state.get("local_miss"):
                next_task["detail"] = f"本地候选未通过相关性门控：{state.get('fallback_reason', '未命中问题')}"
            elif traces:
                tool_names = "、".join(item.get("tool", "Tool") for item in traces)
                next_task["detail"] = f"{tool_names} 已执行，返回 {len(contexts)} 条候选证据"
            elif state.get("intent") == "memory_query":
                next_task["detail"] = "已读取用户确认过的长期 Memory"
            else:
                next_task["detail"] = "本步骤无需调用外部 Tool"
        elif task["id"] == "compose_answer":
            next_task["status"] = "failed" if state.get("intent") in {"web_research", "knowledge_question"} and contexts and not citations else "completed"
            next_task["detail"] = f"保留 {len(citations)} 个实际使用的引用" if citations else "未形成可验证引用，回答已降级或明确证据不足"
        else:
            next_task["status"] = "completed"
        finalized.append(next_task)
    if state.get("local_miss"):
        fallback_task = {
            "id": "fallback_web",
            "title": "本地未命中，回退到公开网页",
            "status": "completed" if contexts and state.get("retrieval_origin") == "web_fallback" else "failed",
            "source": "public_web",
            "detail": (
                "已切换 web_research_skill，并使用公开网页证据"
                if contexts and state.get("retrieval_origin") == "web_fallback"
                else "联网未启用或没有返回可核验来源"
            ),
        }
        compose_index = next((index for index, task in enumerate(finalized) if task["id"] == "compose_answer"), len(finalized))
        finalized.insert(compose_index, fallback_task)
        plan["fallback_skill"] = "web_research_skill"
        plan["fallback_reason"] = state.get("fallback_reason")
        plan["retrieval_origin"] = state.get("retrieval_origin")
    intent = state.get("intent")
    if intent == "memory_query":
        grounding = {"status": "memory", "risk": "low", "label": "来自已确认 Memory",
                     "explanation": "没有使用文档引用；只读取用户确认过的长期 Memory。"}
    elif citations:
        grounding = {"status": "grounded", "risk": "low", "label": "已有可点击证据",
                     "explanation": f"检索 {len(contexts)} 条候选，回答实际采用 {len(citations)} 条引用。引用降低幻觉风险，但不等于自动证明每句话正确。"}
    elif contexts:
        grounding = {"status": "unsupported", "risk": "high", "label": "引用校验未通过",
                     "explanation": f"Tool 返回 {len(contexts)} 条候选，但生成结果没有形成有效引用；系统不会把它冒充为有依据的答案。"}
    else:
        grounding = {"status": "insufficient", "risk": "high", "label": "证据不足",
                     "explanation": "没有检索到足够依据。继续生成具体事实会有较高幻觉风险，因此应拒答或要求补充信息。"}
    plan["tasks"] = finalized
    plan["status"] = "completed"
    plan["grounding"] = grounding
    plan["tool_calls"] = [{"name": item.get("tool"), "status": item.get("status"),
                            "duration_ms": item.get("duration_ms"), "result_count": item.get("result_count"),
                            "error_code": item.get("error_code"), "recoverable": bool(item.get("recoverable"))}
                           for item in traces]
    plan["explanation"] = "Plan 决定做什么，Skill 规定怎么做，Tool 执行原子操作，模型只负责结合证据组织回答。"
    return {"plan": plan}

def propose_memory(state: AgentState) -> AgentState:
    if not get_privacy_settings()["memory_suggestions_enabled"]:
        return {"memory_candidate": None}
    return {"memory_candidate": detect_memory_candidate(state["question"])}


def _observable_node(name: str, function):
    def wrapped(state: AgentState) -> AgentState:
        started = perf_counter()
        try:
            result = function(state)
            summary = {}
            if name == "understand":
                summary = {"intent": result.get("intent"), "skill": result.get("skill")}
            elif name == "plan":
                summary = {"task_count": len((result.get("plan") or {}).get("tasks", []))}
            elif name == "context_budget":
                plan = result.get("context_plan", {})
                summary = {"token_estimate": plan.get("token_estimate"), "summary_version": plan.get("summary_version")}
            elif name == "retrieve":
                summary = {"result_count": len(result.get("contexts", [])), "tool_count": len(result.get("tool_trace", []))}
            elif name == "memory_candidate":
                summary = {"candidate": bool(result.get("memory_candidate"))}
            elif name == "generate":
                summary = {"citation_count": len(result.get("citations", []))}
            elif name == "finalize_plan":
                summary = {"status": (result.get("plan") or {}).get("status")}
            event = {"stage": name, "status": "completed", "duration_ms": round((perf_counter() - started) * 1000), "summary": summary}
            return {**result, "stage_events": [*state.get("stage_events", []), event]}
        except Exception as error:
            event = {"stage": name, "status": "failed", "duration_ms": round((perf_counter() - started) * 1000), "error_type": type(error).__name__, "summary": {}}
            raise
    return wrapped


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("understand", _observable_node("understand", understand))
    graph.add_node("plan", _observable_node("plan", build_plan))
    graph.add_node("context_budget", _observable_node("context_budget", prepare_working_context))
    graph.add_node("retrieve", _observable_node("retrieve", retrieve))
    graph.add_node("generate", _observable_node("generate", generate))
    graph.add_node("propose_memory", _observable_node("memory_candidate", propose_memory))
    graph.add_node("finalize_plan", _observable_node("finalize_plan", finalize_plan))
    graph.add_edge(START, "understand")
    graph.add_edge("understand", "plan")
    graph.add_edge("plan", "context_budget")
    graph.add_edge("context_budget", "retrieve")
    graph.add_edge("retrieve", "propose_memory")
    graph.add_edge("propose_memory", "generate")
    graph.add_edge("generate", "finalize_plan")
    graph.add_edge("finalize_plan", END)
    return graph.compile()


AGENT_GRAPH = build_graph()


def _persist_trace(trace_id: str, result: AgentState, started_at: str, duration_ms: int, status: str, error_type: str | None = None) -> None:
    plan = result.get("context_plan") or {}
    tool_names = [item.get("tool") for item in result.get("tool_trace", []) if item.get("tool")]
    definitions = REGISTRY.definitions()
    selected = [item for item in definitions if item["name"] in tool_names]
    schema_tokens = sum(estimate_tokens(json.dumps(item.get("input_schema", {}), ensure_ascii=False)) for item in selected)
    with connect() as db:
        db.execute("""INSERT INTO agent_traces(id,conversation_id,intent,selected_skill,status,context_tokens,context_budget,
                   summary_version,retrieval_count,citation_count,exposed_tool_count,schema_token_estimate,error_type,
                   started_at,finished_at,duration_ms) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                   (trace_id, result.get("conversation_id"), result.get("intent"), result.get("skill"), status,
                    int(plan.get("token_estimate") or 0), int((plan.get("policy") or {}).get("effective_input_limit") or 0),
                    plan.get("summary_version"), len(result.get("contexts", [])), len(result.get("citations", [])),
                    len(definitions), schema_tokens, error_type, started_at, now(), duration_ms))
        for ordinal, event in enumerate(result.get("stage_events", []), 1):
            db.execute("""INSERT INTO agent_trace_stages(id,trace_id,stage_name,status,duration_ms,result_summary_json,error_type,ordinal)
                          VALUES(?,?,?,?,?,?,?,?)""",
                       (uuid4().hex, trace_id, event["stage"], event["status"], event["duration_ms"],
                        json_value(event.get("summary", {})), event.get("error_type"), ordinal))


def run(question: str, space_id: str, conversation_id: str | None = None) -> AgentState:
    trace_id = uuid4().hex
    started_at = now()
    started = perf_counter()
    initial: AgentState = {"question": question, "space_id": space_id, "conversation_id": conversation_id,
                           "trace_id": trace_id, "stage_events": []}
    try:
        result = AGENT_GRAPH.invoke(initial)
        _persist_trace(trace_id, result, started_at, round((perf_counter() - started) * 1000), "completed")
        return {**result, "trace_id": trace_id}
    except Exception as error:
        failed = {**initial}
        _persist_trace(trace_id, failed, started_at, round((perf_counter() - started) * 1000), "failed", type(error).__name__)
        raise
