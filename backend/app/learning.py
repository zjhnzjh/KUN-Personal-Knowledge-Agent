from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from .context import context_policy
from .database import rows


def _scalar(query: str, params: tuple = ()) -> int:
    result = rows(query, params)
    return int(result[0]["value"] or 0) if result else 0


def _status(has_evidence: bool, implemented: bool = True) -> str:
    if has_evidence:
        return "verified"
    return "partial" if implemented else "planned"


def learning_overview(skills: list[dict], tools: list[dict]) -> dict:
    documents = {
        "total": _scalar("SELECT COUNT(*) value FROM documents"),
        "ready": _scalar("SELECT COUNT(*) value FROM documents WHERE index_status IN ('ready','lexical_ready')"),
        "failed": _scalar("SELECT COUNT(*) value FROM documents WHERE index_status='failed'"),
        "chunks": _scalar("SELECT COUNT(*) value FROM chunks"),
    }
    memories = {
        "pending": _scalar("SELECT COUNT(*) value FROM memories WHERE status='pending'"),
        "enabled": _scalar("SELECT COUNT(*) value FROM memories WHERE status='enabled'"),
        "disabled": _scalar("SELECT COUNT(*) value FROM memories WHERE status='disabled'"),
        "events": _scalar("SELECT COUNT(*) value FROM memory_events"),
        "recalls": _scalar("SELECT COUNT(*) value FROM memory_recalls"),
        "conflicts": _scalar("SELECT COUNT(*) value FROM memories WHERE conflict_with_id IS NOT NULL"),
    }
    tool_stats = rows("""SELECT COUNT(*) total,
        SUM(CASE WHEN status='succeeded' THEN 1 ELSE 0 END) succeeded,
        SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) failed,
        AVG(CASE WHEN duration_ms IS NOT NULL THEN duration_ms END) average_duration_ms
        FROM (SELECT * FROM tool_runs ORDER BY created_at DESC LIMIT 50)""")[0]
    evaluation_count = _scalar("SELECT COUNT(*) value FROM evaluation_cases")
    latest_evaluation = rows("SELECT * FROM evaluation_runs ORDER BY created_at DESC LIMIT 1")
    if latest_evaluation:
        evaluation_payload = json.loads(latest_evaluation[0].pop("result_json") or "[]")
        if isinstance(evaluation_payload, dict):
            latest_evaluation[0].update(evaluation_payload)
        else:
            latest_evaluation[0]["details"] = evaluation_payload
    trace_count = _scalar("SELECT COUNT(*) value FROM agent_traces")
    completed_traces = _scalar("SELECT COUNT(*) value FROM agent_traces WHERE status='completed'")
    policy = context_policy()
    available_tools = [item["name"] for item in tools if item.get("availability") == "available"]
    tool_names = [item["name"] for item in tools]
    skill_names = [item["name"] for item in skills]
    matrix = [
        {"id": "short_memory", "topic": "短期记忆", "capability": "Token 预算、历史摘要与最近完整消息",
         "status": _status(trace_count > 0), "evidence": f"{trace_count} 条 Agent Trace；预算 {policy['effective_input_limit']} Token",
         "verify": "打开学习地图中的上下文预算，再查看 Memory 中心", "route": "memory", "next": "接入模型原生 tokenizer 与可选模型摘要"},
        {"id": "long_memory", "topic": "长期记忆", "capability": "候选、确认、修改、停用、删除、版本与事件",
         "status": _status(memories["events"] > 0), "evidence": f"{memories['enabled']} 条启用，{memories['events']} 条审计事件",
         "verify": "打开 Memory 中心并查看单条审计详情", "route": "memory", "next": "改进语义冲突合并和过期策略"},
        {"id": "memory_safety", "topic": "记忆安全", "capability": "敏感信息拦截与用户确认边界",
         "status": "partial", "evidence": "规则已实现；尚无独立安全评估集", "verify": "查看隐私设置与 Memory 审计事件", "route": "settings",
         "next": "为规则来源、版本和误报/漏报建立评估"},
        {"id": "raw_data", "topic": "原始数据", "capability": "对话、资料副本与来源映射本地持久化",
         "status": _status(documents["total"] > 0), "evidence": f"{documents['total']} 份资料，{documents['chunks']} 个 Chunk",
         "verify": "打开文件资料库和历史对话", "route": "library", "next": "提供摘要到原消息的 UI 反向定位"},
        {"id": "rag", "topic": "RAG", "capability": "FTS5/BM25、Embedding 与融合检索",
         "status": _status(documents["ready"] > 0), "evidence": f"{documents['ready']} 份可检索资料；{evaluation_count} 道评估题",
         "verify": "打开 RAG 实验室运行人工评估集", "route": "lab", "next": "增加融合参数实验、引用支持率和 Bad Case 归因"},
        {"id": "observability", "topic": "可观测性", "capability": "Tool Run、索引状态、引用与 Agent Trace",
         "status": _status(trace_count > 0), "evidence": f"{trace_count} 条 Agent Trace，{int(tool_stats['total'] or 0)} 条近期 Tool Run",
         "verify": "打开学习地图的最近 Agent Trace", "route": "learning", "next": "增加跨阶段成本与取消原因分析"},
        {"id": "evaluation", "topic": "评估", "capability": "Recall@K、MRR、nDCG、平均与 P95 延迟",
         "status": _status(bool(latest_evaluation)), "evidence": f"{evaluation_count} 道题；" + ("已有真实运行" if latest_evaluation else "尚未评估"),
         "verify": "打开 RAG 实验室运行评估", "route": "lab", "next": "补充引用、拒答、Agent、Tool 和 Memory 指标"},
        {"id": "tools", "topic": "Tool 系统", "capability": "参数、权限、超时、错误、脱敏日志与渐进 Schema",
         "status": _status(int(tool_stats["total"] or 0) > 0), "evidence": f"{len(available_tools)}/{len(tool_names)} 个 Tool 当前可用",
         "verify": "打开 Tool 中心查看目录、Schema 与运行记录", "route": "tools", "next": "补充幂等、重试、最大步数和循环检测"},
        {"id": "context", "topic": "上下文优化", "capability": "Skill 路由、摘要目录与选中后加载完整 Schema",
         "status": _status(trace_count > 0), "evidence": f"{len(skill_names)} 个 Skill，{len(tool_names)} 个 Tool 摘要可路由",
         "verify": "在 Tool 中心预览路由并展开 Schema", "route": "tools", "next": "用离线集测量路由准确率与 Token 节省"},
    ]
    workflow = [
        {"id": "import", "label": "资料导入", "plain": "把允许的本地文件复制到 KUN 暂存区，原文件不被覆盖。", "status": _status(documents["total"] > 0), "route": "library"},
        {"id": "parse", "label": "临时解析", "plain": "先抽取文字、表格或图片含义，生成可修改的标题、摘要和标签。", "status": _status(documents["total"] > 0), "route": "library"},
        {"id": "confirm", "label": "用户确认", "plain": "只有你确认元数据与知识空间后，资料才正式入库。", "status": _status(documents["total"] > 0), "route": "library"},
        {"id": "index", "label": "建立索引", "plain": "把资料切成可定位片段，并分别建立关键词与语义索引。", "status": _status(documents["ready"] > 0), "route": "library"},
        {"id": "retrieve", "label": "混合检索", "plain": "问题到来时，用 BM25 与可用的向量结果寻找证据。", "status": _status(int(tool_stats["total"] or 0) > 0), "route": "lab"},
        {"id": "answer", "label": "生成回答", "plain": "把有限的对话上下文、按需 Memory 和检索证据交给模型。", "status": _status(completed_traces > 0), "route": "learning"},
        {"id": "citation", "label": "引用核验", "plain": "只保留回答真正使用的来源编号，并支持回到原片段。", "status": _status(completed_traces > 0), "route": "chat"},
        {"id": "candidate", "label": "Memory 候选", "plain": "稳定信息先变成候选，敏感信息会被拦截。", "status": _status(memories["pending"] + memories["enabled"] > 0), "route": "memory"},
        {"id": "memory_confirm", "label": "用户确认", "plain": "确认后才跨对话召回，并保留版本、来源和事件。", "status": _status(memories["enabled"] > 0), "route": "memory"},
    ]
    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "documents": documents,
            "memories": memories, "tools": {**tool_stats, "available": available_tools, "all": tool_names},
            "skills": skill_names, "evaluation": {"case_count": evaluation_count, "latest": latest_evaluation[0] if latest_evaluation else None},
            "context_policy": policy, "agent": {"trace_count": trace_count, "completed": completed_traces},
            "workflow": workflow, "capability_matrix": matrix}


def agent_traces(limit: int = 20) -> list[dict]:
    traces = rows("SELECT * FROM agent_traces ORDER BY started_at DESC LIMIT ?", (max(1, min(limit, 100)),))
    for trace in traces:
        stages = rows("SELECT * FROM agent_trace_stages WHERE trace_id=? ORDER BY ordinal", (trace["id"],))
        for stage in stages:
            stage["result_summary"] = json.loads(stage.pop("result_summary_json") or "{}")
        trace["stages"] = stages
    return traces
