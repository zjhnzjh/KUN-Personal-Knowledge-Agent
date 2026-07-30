from __future__ import annotations

import json
import re
from uuid import uuid4

from .database import connect, json_value, now, rows

SINGULAR_MEMORY_KINDS = {"identity", "location", "preference"}

def snapshot(item: dict | None) -> dict:
    if not item:
        return {}
    return {key: item.get(key) for key in ("id", "kind", "content", "source", "status", "version", "conflict_with_id")}

def record_memory_event(memory_id: str, event_type: str, source: str, before: dict | None, after: dict | None) -> None:
    version = int((after or before or {}).get("version") or 1)
    with connect() as db:
        db.execute("""INSERT INTO memory_events(id,memory_id,event_type,version,source,before_json,after_json,created_at)
                      VALUES(?,?,?,?,?,?,?,?)""",
                   (uuid4().hex, memory_id, event_type, version, source, json_value(snapshot(before)), json_value(snapshot(after)), now()))

def find_conflict(kind: str, content: str, exclude_id: str | None = None) -> str | None:
    if kind not in SINGULAR_MEMORY_KINDS:
        return None
    params: list[object] = [kind, content]
    sql = "SELECT id FROM memories WHERE kind=? AND status='enabled' AND content<>?"
    if exclude_id:
        sql += " AND id<>?"
        params.append(exclude_id)
    sql += " ORDER BY updated_at DESC LIMIT 1"
    items = rows(sql, tuple(params))
    return items[0]["id"] if items else None

def _terms(text: str) -> set[str]:
    terms = set(re.findall(r"[A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,6}", text.lower()))
    return {term for term in terms if term not in {"什么", "怎么", "可以", "是否", "一个", "这个", "那个"}}

def recall_memories(query: str, conversation_id: str | None, limit: int = 5, include_all: bool = False) -> list[dict]:
    candidates = rows("SELECT * FROM memories WHERE status='enabled' ORDER BY updated_at DESC")
    query_terms = _terms(query)
    kind_hints = {
        "identity": ("名字", "姓名", "我是谁", "身份"), "location": ("哪里", "所在地", "住", "城市"),
        "preference": ("偏好", "喜欢", "回答风格"), "goal": ("目标", "计划", "以后"),
        "project": ("项目", "正在做"), "relationship": ("导师", "同事", "师兄", "关系"),
    }
    scored: list[tuple[float, dict, str]] = []
    for item in candidates:
        content_terms = _terms(item["content"])
        overlap = len(query_terms.intersection(content_terms))
        hinted = any(marker in query for marker in kind_hints.get(item["kind"], ()))
        score = overlap * 1.0 + (1.5 if hinted else 0.0)
        if include_all:
            score = max(score, 0.1)
        if score <= 0:
            continue
        reason = "问题直接询问该类长期信息" if hinted else f"问题与 Memory 共享 {overlap} 个关键词"
        scored.append((score, item, reason))
    scored.sort(key=lambda value: (value[0], value[1]["updated_at"]), reverse=True)
    selected = scored[:limit]
    if selected:
        stamp = now()
        with connect() as db:
            for score, item, reason in selected:
                db.execute("INSERT INTO memory_recalls(id,memory_id,conversation_id,query,reason,score,created_at) VALUES(?,?,?,?,?,?,?)",
                           (uuid4().hex, item["id"], conversation_id, query[:500], reason, score, stamp))
                db.execute("UPDATE memories SET use_count=use_count+1 WHERE id=?", (item["id"],))
    return [{**item, "recall_reason": reason, "recall_score": score} for score, item, reason in selected]

def memory_detail(memory_id: str) -> dict | None:
    items = rows("SELECT * FROM memories WHERE id=?", (memory_id,))
    if not items:
        return None
    item = items[0]
    events = rows("SELECT * FROM memory_events WHERE memory_id=? ORDER BY created_at DESC, rowid DESC", (memory_id,))
    recalls = rows("SELECT * FROM memory_recalls WHERE memory_id=? ORDER BY created_at DESC, rowid DESC LIMIT 20", (memory_id,))
    for event in events:
        event["before"] = json.loads(event.pop("before_json"))
        event["after"] = json.loads(event.pop("after_json"))
    return {**item, "events": events, "recent_recalls": recalls}
