from __future__ import annotations

import json
import re
from uuid import uuid4

from .database import connect, json_value, now, rows

DEFAULT_CONTEXT_POLICY = {
    "model_hard_limit": 32_768,
    "input_budget": 24_000,
    "safety_margin": 2_000,
    "output_reserve": 4_000,
    "recent_message_minimum": 6,
    "estimator": "unicode_chars_div_2",
}

def estimate_tokens(text: str) -> int:
    """Deterministic estimate; it is deliberately not reported as tokenizer truth."""
    if not text:
        return 0
    ascii_count = sum(1 for char in text if ord(char) < 128)
    non_ascii_count = len(text) - ascii_count
    return max(1, (ascii_count + 3) // 4 + (non_ascii_count + 1) // 2)

def context_policy() -> dict:
    policy = dict(DEFAULT_CONTEXT_POLICY)
    setting = rows("SELECT value_json FROM app_settings WHERE key='context_policy'")
    if setting:
        try:
            patch = json.loads(setting[0]["value_json"])
        except (TypeError, json.JSONDecodeError):
            patch = {}
        for key in policy:
            if key in patch and isinstance(patch[key], (int, str)):
                policy[key] = patch[key]
    usable = int(policy["model_hard_limit"]) - int(policy["safety_margin"]) - int(policy["output_reserve"])
    policy["effective_input_limit"] = min(int(policy["input_budget"]), usable)
    return policy

def _structured_summary(messages: list[dict]) -> dict:
    user_text = [item["content"].strip() for item in messages if item["role"] == "user"]
    assistant_text = [item["content"].strip() for item in messages if item["role"] == "assistant"]
    goals = [text for text in user_text if re.search(r"目标|希望|想要|需要|请|帮我", text)]
    constraints = [text for text in user_text if re.search(r"不要|必须|只能|不能|本地|隐私|保留", text)]
    entities: list[str] = []
    for text in user_text:
        entities.extend(re.findall(r"[A-Za-z][A-Za-z0-9_.+-]{2,}|[\u4e00-\u9fff]{2,12}(?:项目|文档|页面|接口|模型)", text))
    return {
        "user_goal": goals[-3:] or user_text[-2:],
        "completed": assistant_text[-3:],
        "current_progress": assistant_text[-1:] or ["尚无可归纳的完成事项"],
        "pending": goals[-2:] or ["继续根据用户的新问题推进"],
        "constraints": constraints[-5:],
        "entities": list(dict.fromkeys(entities))[:12],
        "method": "rule_based",
        "notice": "规则式结构化摘要，不代表模型推理；可从原始消息重新生成。",
    }

def _summary_text(summary: dict) -> str:
    labels = (("用户目标", "user_goal"), ("已完成事项", "completed"), ("当前进度", "current_progress"),
              ("未完成事项", "pending"), ("关键约束", "constraints"), ("重要实体或资料", "entities"))
    return "\n".join(f"{label}：" + ("；".join(str(value)[:240] for value in summary.get(key) or []) or "无") for label, key in labels)

def _latest_summary(conversation_id: str) -> dict | None:
    items = rows("SELECT * FROM context_summaries WHERE conversation_id=? ORDER BY version DESC LIMIT 1", (conversation_id,))
    if not items:
        return None
    item = items[0]
    item["summary"] = json.loads(item.pop("summary_json"))
    item["source_message_ids"] = json.loads(item.pop("source_message_ids_json"))
    return item

def build_context(conversation_id: str | None, current_question: str) -> dict:
    policy = context_policy()
    empty_composition = {"current_question_tokens": estimate_tokens(current_question), "summary_tokens": 0,
                         "recent_message_tokens": 0, "recent_message_count": 0, "summarized_message_count": 0}
    if not conversation_id:
        return {"history_text": "", "summary": None, "summary_version": None, "recent_messages": [],
                "token_estimate": estimate_tokens(current_question), "policy": policy, "composition": empty_composition}
    messages = rows("SELECT id,role,content,created_at FROM messages WHERE conversation_id=? ORDER BY rowid", (conversation_id,))
    history = messages[:-1] if messages and messages[-1]["role"] == "user" and messages[-1]["content"] == current_question else messages
    budget = max(256, int(policy["effective_input_limit"]) - estimate_tokens(current_question))
    recent: list[dict] = []
    recent_tokens = 0
    for item in reversed(history):
        item_tokens = estimate_tokens(item["content"]) + 8
        if recent and recent_tokens + item_tokens > budget * 0.62 and len(recent) >= int(policy["recent_message_minimum"]):
            break
        recent.append(item)
        recent_tokens += item_tokens
    recent.reverse()
    recent_ids = {item["id"] for item in recent}
    older = [item for item in history if item["id"] not in recent_ids]
    latest = _latest_summary(conversation_id)
    summary_record = latest
    if older:
        source_ids = [item["id"] for item in older]
        if not latest or latest["source_message_ids"] != source_ids:
            summary = _structured_summary(older)
            summary_text = _summary_text(summary)
            version = (latest["version"] if latest else 0) + 1
            summary_record = {"id": uuid4().hex, "version": version, "source_message_ids": source_ids,
                              "summary": summary, "token_estimate": estimate_tokens(summary_text), "created_at": now()}
            with connect() as db:
                db.execute("""INSERT INTO context_summaries(id,conversation_id,version,first_message_id,last_message_id,
                           source_message_ids_json,summary_json,token_estimate,created_at) VALUES(?,?,?,?,?,?,?,?,?)""",
                           (summary_record["id"], conversation_id, version, source_ids[0], source_ids[-1],
                            json_value(source_ids), json_value(summary), summary_record["token_estimate"], summary_record["created_at"]))
    summary_text = _summary_text(summary_record["summary"]) if summary_record else ""
    history_text = "\n".join(f"{'用户' if item['role'] == 'user' else '坤坤'}：{item['content']}" for item in recent)
    composition = {"current_question_tokens": estimate_tokens(current_question), "summary_tokens": estimate_tokens(summary_text),
                   "recent_message_tokens": estimate_tokens(history_text), "recent_message_count": len(recent),
                   "summarized_message_count": len(older)}
    return {"history_text": history_text, "summary": summary_record["summary"] if summary_record else None,
            "summary_text": summary_text, "summary_version": summary_record["version"] if summary_record else None,
            "recent_messages": recent, "token_estimate": sum((composition["current_question_tokens"], composition["summary_tokens"], composition["recent_message_tokens"])),
            "policy": policy, "composition": composition}
