from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
import httpx

from .config import get_settings
from .database import connect, json_value, now, rows
from .infra import JobCancelled, JobContext, create_trace, finish_trace, trace_span
from .privacy import allowed_for_cloud, get_privacy_settings
from .retrieval_engine import pipeline_search


def machine_profile() -> dict[str, Any]:
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count() or 1,
        "architecture": platform.machine(),
    }


def git_revision() -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL, timeout=3
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _dataset_hash(cases: list[dict[str, Any]]) -> str:
    stable = [
        {
            "id": item["id"],
            "question": item["question"],
            "split": item["split"],
            "gold": item["gold"],
        }
        for item in sorted(cases, key=lambda value: value["id"])
    ]
    return hashlib.sha256(json_value(stable).encode("utf-8")).hexdigest()


def _case_validation(dataset: dict[str, Any], case: dict[str, Any], *, all_cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Run cheap, deterministic checks before a draft can become Gold."""
    issues: list[str] = []
    warnings: list[str] = []
    question = str(case.get("question", "")).strip()
    answer = str(case.get("answer", case.get("answer_text", ""))).strip()
    gold = case.get("gold") or []
    query_type = str(case.get("query_type", "")).strip()
    is_no_answer = query_type == "no_answer"
    if len(question) < 8:
        issues.append("question_too_short")
    if not gold and not is_no_answer:
        issues.append("missing_gold_evidence")
    document_ids = {item["id"] for item in rows("SELECT id FROM documents WHERE space_id=?", (dataset["space_id"],))}
    missing_documents = [item.get("document_id") for item in gold if item.get("document_id") not in document_ids]
    if missing_documents:
        issues.append("gold_document_not_in_space")
    if not answer:
        if is_no_answer:
            issues.append("missing_refusal_answer")
        elif case.get("source_type") not in {"human", "legacy_import"}:
            issues.append("missing_reference_answer")
    existing = all_cases if all_cases is not None else dataset.get("cases", [])
    normalized_question = " ".join(question.casefold().split())
    duplicate = any(
        item.get("id") != case.get("id")
        and " ".join(str(item.get("question", "")).casefold().split()) == normalized_question
        for item in existing
    )
    if duplicate:
        issues.append("duplicate_question")
    if len(gold) > 1:
        warnings.append("multiple_gold_evidence")
    return {
        "valid": not issues,
        "issues": issues,
        "warnings": warnings,
        "checked_at": now(),
        "validator": "kun-eval-validator-v1",
    }


def _refresh_dataset(dataset_id: str) -> None:
    raw_cases = rows("SELECT * FROM eval_dataset_cases WHERE dataset_version_id=? ORDER BY id", (dataset_id,))
    cases = []
    for item in raw_cases:
        cases.append({
            "id": item["id"], "question": item["question"], "split": item["split"],
            "gold": json.loads(item["gold_json"] or "[]"),
        })
    accepted = sum(item["status"] == "accepted" for item in raw_cases)
    dataset_row = rows("SELECT target_case_count,holdout_count FROM eval_dataset_versions WHERE id=?", (dataset_id,))
    target_case_count = int(dataset_row[0]["target_case_count"] if dataset_row else 100)
    holdout_count = int(dataset_row[0]["holdout_count"] if dataset_row else 30)
    formal_ready = int(
        accepted >= target_case_count
        and sum(item["split"] == "holdout" and item["status"] == "accepted" for item in raw_cases) == holdout_count
    )
    with connect() as db:
        db.execute(
            "UPDATE eval_dataset_versions SET content_hash=?,case_count=?,formal_ready=?,status=?,updated_at=? WHERE id=?",
            (_dataset_hash(cases), len(cases), formal_ready, "ready" if accepted else "draft", now(), dataset_id),
        )


def _stratified_split(dataset_id: str, holdout_count: int = 30, seed: int = 20260814) -> dict[str, Any]:
    """Assign accepted cases to a deterministic 70/30 holdout split."""
    dataset = dataset_detail(dataset_id)
    if not dataset:
        raise ValueError("Evaluation dataset does not exist")
    accepted = [item for item in dataset["cases"] if item["status"] == "accepted"]
    if len(accepted) < holdout_count:
        raise ValueError(f"At least {holdout_count} accepted cases are required for a holdout split")
    ranked = sorted(
        accepted,
        key=lambda item: hashlib.sha256(f"{seed}:{item['id']}".encode("utf-8")).hexdigest(),
    )
    holdout_ids = {item["id"] for item in ranked[:holdout_count]}
    with connect() as db:
        db.executemany(
            "UPDATE eval_dataset_cases SET split=?,updated_at=? WHERE id=? AND dataset_version_id=? AND status='accepted'",
            [("holdout" if item["id"] in holdout_ids else "dev", now(), item["id"], dataset_id) for item in accepted],
        )
        db.execute(
            "UPDATE eval_dataset_versions SET holdout_count=?,split_seed=?,updated_at=? WHERE id=?",
            (holdout_count, seed, now(), dataset_id),
        )
    _refresh_dataset(dataset_id)
    refreshed = dataset_detail(dataset_id) or {}
    return {
        "dataset_id": dataset_id,
        "accepted_count": sum(item["status"] == "accepted" for item in refreshed.get("cases", [])),
        "dev_count": sum(item["status"] == "accepted" and item["split"] == "dev" for item in refreshed.get("cases", [])),
        "holdout_count": sum(item["status"] == "accepted" and item["split"] == "holdout" for item in refreshed.get("cases", [])),
        "formal_ready": bool(refreshed.get("formal_ready")),
    }


def dataset_readiness(dataset_id: str) -> dict[str, Any]:
    dataset = dataset_detail(dataset_id)
    if not dataset:
        raise ValueError("Evaluation dataset does not exist")
    cases = dataset.get("cases", [])
    accepted = [item for item in cases if item["status"] == "accepted"]
    source_documents = {gold.get("document_id") for item in accepted for gold in item.get("gold", []) if gold.get("document_id")}
    accepted_types = {item["query_type"] for item in accepted}
    checks = [
        {"name": "accepted_gold_count", "required": dataset.get("target_case_count", 100), "actual": len(accepted), "passed": len(accepted) >= dataset.get("target_case_count", 100)},
        {"name": "source_document_count", "required": 3, "actual": len(source_documents), "passed": len(source_documents) >= 3},
        {"name": "query_type_coverage", "required": 5, "actual": len(accepted_types), "passed": len(accepted_types) >= 5},
        {"name": "holdout_count", "required": dataset.get("holdout_count", 30), "actual": dataset.get("splits", {}).get("holdout", 0), "passed": dataset.get("splits", {}).get("holdout", 0) == dataset.get("holdout_count", 30)},
    ]
    return {
        "dataset_id": dataset_id,
        "name": dataset["name"],
        "version": dataset["version"],
        "accepted_count": len(accepted),
        "draft_count": sum(item["status"] == "draft" for item in cases),
        "rejected_count": sum(item["status"] == "rejected" for item in cases),
        "dev_count": dataset.get("splits", {}).get("dev", 0),
        "holdout_count": dataset.get("splits", {}).get("holdout", 0),
        "source_document_count": len(source_documents),
        "query_types": dataset.get("case_mix", {}),
        "checks": checks,
        "formal_ready": bool(dataset.get("formal_ready")),
        "next_actions": [
            check["name"] for check in checks if not check["passed"]
        ],
    }


def finalize_dataset(dataset_id: str, holdout_count: int = 30, seed: int = 20260814) -> dict[str, Any]:
    readiness = dataset_readiness(dataset_id)
    if readiness["accepted_count"] < 100:
        raise ValueError(f"Formal benchmark requires 100 accepted Gold cases; currently {readiness['accepted_count']}")
    dataset = dataset_detail(dataset_id)
    source_documents = {gold.get("document_id") for item in dataset.get("cases", []) if item["status"] == "accepted" for gold in item.get("gold", []) if gold.get("document_id")}
    if len(source_documents) < 3:
        raise ValueError("Formal benchmark requires at least 3 source documents")
    _stratified_split(dataset_id, holdout_count=holdout_count, seed=seed)
    _refresh_dataset(dataset_id)
    return dataset_readiness(dataset_id)


def create_dataset(space_id: str, name: str, version: str) -> dict:
    if not rows("SELECT 1 ok FROM spaces WHERE id=?", (space_id,)):
        raise ValueError("Knowledge space does not exist")
    existing = rows(
        "SELECT id FROM eval_dataset_versions WHERE name=? AND version=? AND space_id=?",
        (name, version, space_id),
    )
    if existing:
        return dataset_detail(existing[0]["id"]) or {}
    dataset_id = uuid4().hex
    stamp = now()
    with connect() as db:
        db.execute(
            """INSERT INTO eval_dataset_versions(
               id,name,version,space_id,status,source,content_hash,case_count,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (dataset_id, name[:120], version[:40], space_id, "draft", "candidate_workflow", "", 0, stamp, stamp),
        )
    return dataset_detail(dataset_id) or {}


def _candidate_source_snippets(space_id: str, count: int) -> tuple[list[dict[str, Any]], set[str]]:
    """Select candidate-generation context with document-level coverage.

    Ordering all chunks by document id can accidentally fill the entire context
    from one document. Sample a small quota per allowed document first, then use
    remaining capacity for additional chunks. This keeps the generated draft set
    useful for the multi-document benchmark.
    """
    target = max(12, min(count * 2, 120))
    document_rows = rows("SELECT id FROM documents WHERE space_id=? ORDER BY id", (space_id,))
    allowed_documents = allowed_for_cloud([item["id"] for item in document_rows], "llm")
    per_document_source = max(12, math.ceil(target / max(len(allowed_documents), 1)) * 2)
    raw: list[dict[str, Any]] = []
    for document_id in sorted(allowed_documents):
        raw.extend(rows(
            """SELECT c.id,c.document_id,c.locator,c.text,d.title,d.fingerprint
               FROM chunks c JOIN documents d ON d.id=c.document_id
               WHERE d.id=? AND length(c.text)>=80 ORDER BY c.ordinal LIMIT ?""",
            (document_id, per_document_source),
        ))
    candidates = [item for item in raw if item["document_id"] in allowed_documents]
    if not candidates:
        return [], allowed_documents
    per_document = max(1, math.ceil(target / len(allowed_documents)))
    selected: list[dict[str, Any]] = []
    for document_id in sorted(allowed_documents):
        selected.extend(
            [item for item in candidates if item["document_id"] == document_id][:per_document]
        )
    selected_ids = {item["id"] for item in selected}
    for item in candidates:
        if len(selected) >= target:
            break
        if item["id"] not in selected_ids:
            selected.append(item)
            selected_ids.add(item["id"])
    return selected[:target], allowed_documents


def _parse_candidate_response(content: str) -> list[dict[str, Any]]:
    """Accept strict JSON, fenced JSON, or JSON followed by short prose."""
    text = str(content or "").strip()
    if not text:
        raise ValueError("DeepSeek returned an empty candidate response")
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```"))
    if text.lstrip().lower().startswith("json"):
        text = text.lstrip()[4:].lstrip(" :\n")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        start = text.find("{")
        if start < 0:
            start = text.find("[")
        if start < 0:
            raise ValueError("DeepSeek response did not contain a JSON object") from None
        payload, _ = decoder.raw_decode(text[start:])
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        return payload["cases"]
    raise ValueError("DeepSeek response JSON did not contain a cases array")


def candidate_generation_estimate(dataset_id: str, count: int) -> dict[str, Any]:
    dataset = dataset_detail(dataset_id)
    if not dataset:
        raise ValueError("Evaluation dataset does not exist")
    snippets, allowed_documents = _candidate_source_snippets(dataset["space_id"], count)
    characters = sum(min(len(item["text"]), 700) for item in snippets)
    return {
        "dataset_id": dataset_id,
        "requested_candidates": max(1, min(count, 70)),
        "source_chunk_count": len(snippets),
        "allowed_document_count": len(allowed_documents),
        "estimated_input_characters": characters,
        "cloud_provider": "deepseek",
        "cost_status": "estimated",
        "requires_confirmation": True,
    }


def generate_candidate_cases(dataset_id: str, count: int, context: JobContext) -> dict[str, Any]:
    # Large generations are intentionally serialized into small provider calls.
    # This keeps reasoning-heavy models from exhausting the output budget and
    # makes each successful batch independently recoverable in the trace store.
    if count > 5:
        results: list[dict[str, Any]] = []
        remaining = count
        while remaining > 0:
            batch = min(5, remaining)
            results.append(generate_candidate_cases(dataset_id, batch, context))
            remaining -= batch
        return {
            "dataset_id": dataset_id,
            "generated_count": sum(int(item.get("generated_count", 0)) for item in results),
            "trace_ids": [item.get("trace_id") for item in results if item.get("trace_id")],
            "batches": len(results),
        }
    settings = get_settings()
    privacy = get_privacy_settings()
    if not settings.deepseek_api_key:
        raise ValueError("DeepSeek is not configured")
    if not privacy["cloud_document_analysis_enabled"]:
        raise ValueError("Cloud document analysis is disabled by the privacy policy")
    estimate = candidate_generation_estimate(dataset_id, count)
    dataset = dataset_detail(dataset_id)
    if not dataset:
        raise ValueError("Evaluation dataset does not exist")
    snippets, allowed_documents = _candidate_source_snippets(dataset["space_id"], count)
    if not snippets:
        raise ValueError("No document in this space is allowed for DeepSeek candidate generation")
    source_map = {item["id"]: item for item in snippets}
    source_text = "\n\n".join(
        f"SOURCE_ID={item['id']} | {item['title']} | {item['locator']}\n{item['text'][:700]}"
        for item in snippets
    )
    trace_id = create_trace("dataset", "generate_gold_candidates", {
        "dataset_id": dataset_id, "requested_count": estimate["requested_candidates"], "provider": "deepseek",
    })
    started = perf_counter()
    context.update(progress=10, phase="candidate_generation", message="正在生成候选问题")
    try:
        with trace_span(trace_id, "deepseek_candidate_generation", "llm", attributes={
            "model": settings.deepseek_model, "input_characters": len(source_text),
        }) as span:
            response = httpx.post(
                f"{settings.deepseek_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {settings.deepseek_api_key}"},
                json={
                    "model": settings.deepseek_model,
                    "temperature": 0.2,
                    # Leave enough room for one JSON object per requested case;
                    # the API may still return fewer cases, which remains a draft
                    # and is never counted as accepted Gold automatically.
                    # deepseek-v4-flash may spend a large portion of the budget
                    # on reasoning before emitting the JSON answer.
                    "max_tokens": max(6000, min(12000, estimate["requested_candidates"] * 500)),
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "你是检索评测集设计师。只依据给定 SOURCE 生成可由单个证据片段回答的问题。"
                                "返回 JSON：{\"cases\":[{\"question\":...,\"source_id\":...,"
                                "\"query_type\":\"fact|reasoning|table|cross_page\","
                                "\"difficulty\":\"easy|medium|hard\"}]}。source_id 必须原样复制。"
                            ),
                        },
                        {"role": "user", "content": f"生成 {estimate['requested_candidates']} 条候选题：\n{source_text}"},
                    ],
                },
                timeout=90,
            )
            response.raise_for_status()
            payload = response.json()
            span.annotate(usage=payload.get("usage") or {})
        content = payload["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:].lstrip()
        generated = _parse_candidate_response(content)
        accepted: list[dict[str, Any]] = []
        existing_questions = {
            " ".join(str(item["question"]).casefold().split())
            for item in dataset.get("cases", [])
        }
        for item in generated:
            question = str(item.get("question", "")).strip()
            normalized_question = " ".join(question.casefold().split())
            source_ids = item.get("source_ids") or [item.get("source_id")]
            sources = [source_map.get(str(source_id)) for source_id in source_ids]
            sources = [source for source in sources if source]
            if not sources or len(question) < 8 or normalized_question in existing_questions:
                continue
            existing_questions.add(normalized_question)
            accepted.append({
                "id": uuid4().hex,
                "question": question[:1000],
                "query_type": str(item.get("query_type", "fact"))[:40],
                "difficulty": str(item.get("difficulty", "medium"))[:20],
                "answer": str(item.get("answer", "")).strip()[:2000],
                "gold": [{
                    "document_id": source["document_id"],
                    "source_fingerprint": source["fingerprint"],
                    "title": source["title"],
                    "locator": source["locator"],
                    "text_hash": hashlib.sha256(source["text"].encode("utf-8")).hexdigest(),
                    "relevance": 3,
                } for source in sources],
            })
            if len(accepted) >= estimate["requested_candidates"]:
                break
        stamp = now()
        with connect() as db:
            db.executemany(
                """INSERT INTO eval_dataset_cases(
                   id,dataset_version_id,question,split,query_type,difficulty,status,gold_json,
                   answer_text,source_type,validation_json,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        item["id"], dataset_id, item["question"], "dev", item["query_type"],
                        item["difficulty"], "draft", json_value(item["gold"]), item.get("answer", ""),
                        "deepseek_candidate", json_value({"valid": False, "issues": ["human_review_required"]}), stamp, stamp,
                    )
                    for item in accepted
                ],
            )
        _refresh_dataset(dataset_id)
        context.update(progress=100, phase="draft_ready", message="候选题已生成，等待人工确认")
        finish_trace(trace_id, duration_ms=round((perf_counter() - started) * 1000), attributes={
            "generated_count": len(accepted), "status": "draft",
        })
        return {"dataset_id": dataset_id, "generated_count": len(accepted), "trace_id": trace_id}
    except Exception as error:
        finish_trace(trace_id, "failed", duration_ms=round((perf_counter() - started) * 1000), error_code=type(error).__name__)
        raise


def update_dataset_case(dataset_id: str, case_id: str, patch: dict[str, Any]) -> dict:
    matches = rows("SELECT * FROM eval_dataset_cases WHERE id=? AND dataset_version_id=?", (case_id, dataset_id))
    if not matches:
        raise ValueError("Evaluation case does not exist")
    current = matches[0]
    dataset = dataset_detail(dataset_id)
    if not dataset:
        raise ValueError("Evaluation dataset does not exist")
    status = patch.get("status", current["status"])
    if status not in {"draft", "accepted", "rejected"}:
        raise ValueError("Invalid evaluation case status")
    question = str(patch.get("question", current["question"])).strip()[:1000]
    gold = patch.get("gold")
    gold_json = json_value(gold) if isinstance(gold, list) else current["gold_json"]
    answer = str(patch.get("answer", current["answer_text"])).strip()[:2000]
    query_type = str(patch.get("query_type", current["query_type"])).strip()[:40] or "fact"
    difficulty = str(patch.get("difficulty", current["difficulty"])).strip()[:20] or "medium"
    split = str(patch.get("split", current["split"])).strip()
    if split not in {"dev", "holdout"}:
        raise ValueError("Case split must be dev or holdout")
    case_for_validation = {
        "id": case_id, "question": question, "answer": answer, "query_type": query_type,
        "gold": json.loads(gold_json or "[]"), "source_type": current["source_type"],
    }
    other_cases = [item for item in dataset.get("cases", []) if item["id"] != case_id]
    validation = _case_validation(dataset, case_for_validation, all_cases=other_cases)
    if status == "accepted" and not validation["valid"]:
        raise ValueError(f"Case failed validation: {', '.join(validation['issues'])}")
    with connect() as db:
        db.execute(
            """UPDATE eval_dataset_cases SET question=?,split=?,query_type=?,difficulty=?,status=?,gold_json=?,
               answer_text=?,validation_json=?,updated_at=?
               WHERE id=? AND dataset_version_id=?""",
            (question, split, query_type, difficulty, status, gold_json, answer, json_value(validation), now(), case_id, dataset_id),
        )
    _refresh_dataset(dataset_id)
    return dataset_detail(dataset_id) or {}


def _auto_review_source(case: dict[str, Any]) -> str:
    """Resolve the stored Gold chunk text used for an AI-assisted review."""
    evidence: list[str] = []
    for gold in case.get("gold") or []:
        document_id = str(gold.get("document_id") or "")
        if not document_id:
            continue
        source_rows = rows(
            """SELECT c.locator,c.text,d.title FROM chunks c
               JOIN documents d ON d.id=c.document_id
               WHERE c.document_id=? AND length(c.text)>0 ORDER BY c.ordinal""",
            (document_id,),
        )
        target_hash = str(gold.get("text_hash") or "")
        target_locator = str(gold.get("locator") or "")
        selected = next(
            (item for item in source_rows if hashlib.sha256(item["text"].encode("utf-8")).hexdigest() == target_hash),
            None,
        )
        if selected is None and target_locator:
            selected = next((item for item in source_rows if item["locator"] == target_locator), None)
        if selected is None and source_rows:
            selected = source_rows[0]
        if selected:
            evidence.append(f"{selected['title']} / {selected['locator']}\n{selected['text'][:1800]}")
    return "\n\n".join(evidence)


def _parse_auto_review_response(content: str) -> list[dict[str, Any]]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].lstrip()
    payload = json.loads(cleaned)
    if isinstance(payload, dict):
        answers = payload.get("answers") or []
        return answers if isinstance(answers, list) else []
    return []


def auto_accept_dataset_candidates(dataset_id: str, context: JobContext) -> dict[str, Any]:
    """Use DeepSeek to fill and validate draft answers, then promote only supported cases."""
    dataset = dataset_detail(dataset_id)
    if not dataset:
        raise ValueError("Evaluation dataset does not exist")
    drafts = [item for item in dataset.get("cases", []) if item["status"] == "draft"]
    if not drafts:
        return {"dataset_id": dataset_id, "accepted_count": 0, "remaining_draft_count": 0, "failed_count": 0}
    settings = get_settings()
    if not settings.deepseek_api_key:
        raise ValueError("DeepSeek API 未配置，无法自动补全参考答案")
    trace_id = create_trace("dataset", "auto_accept_gold", {
        "dataset_id": dataset_id, "draft_count": len(drafts), "model": settings.deepseek_model,
    })
    started = perf_counter()
    accepted_count = 0
    failed: list[dict[str, str]] = []
    batch_size = 8
    try:
        for offset in range(0, len(drafts), batch_size):
            context.check_cancelled()
            batch = drafts[offset:offset + batch_size]
            prompt_items = []
            for item in batch:
                evidence = _auto_review_source(item)
                if not evidence:
                    failed.append({"id": item["id"], "reason": "missing_source_text"})
                    continue
                prompt_items.append({"id": item["id"], "question": item["question"], "evidence": evidence})
            if not prompt_items:
                context.update(progress=min(95, round((offset + len(batch)) / len(drafts) * 100)), phase="auto_review", message="部分候选缺少可解析证据")
                continue
            with trace_span(trace_id, "deepseek_auto_review", "llm", attributes={
                "model": settings.deepseek_model, "batch_size": len(prompt_items),
            }) as span:
                response = httpx.post(
                    f"{settings.deepseek_base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {settings.deepseek_api_key}"},
                    json={
                        "model": settings.deepseek_model,
                        "temperature": 0,
                        "max_tokens": 3200,
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "你是严格的检索评测集审核器。只能依据每题给出的 evidence 作答，不能补充常识。"
                                    "如果 evidence 不足以回答，supported 必须为 false。返回 JSON："
                                    "{\"answers\":[{\"id\":\"原样id\",\"supported\":true,\"answer\":\"简洁准确的标准答案\"}]}。"
                                ),
                            },
                            {"role": "user", "content": json.dumps({"cases": prompt_items}, ensure_ascii=False)},
                        ],
                    },
                    timeout=90,
                )
                response.raise_for_status()
                payload = response.json()
                span.annotate(usage=payload.get("usage") or {})
            answers = {str(item.get("id")): item for item in _parse_auto_review_response(payload["choices"][0]["message"]["content"])}
            for item in batch:
                review = answers.get(item["id"])
                answer = str((review or {}).get("answer") or "").strip()
                if not review or not bool(review.get("supported")) or not answer:
                    failed.append({"id": item["id"], "reason": "evidence_not_sufficient"})
                    continue
                try:
                    update_dataset_case(dataset_id, item["id"], {"status": "accepted", "answer": answer})
                    validation_row = rows("SELECT validation_json FROM eval_dataset_cases WHERE id=?", (item["id"],))
                    validation = json.loads(validation_row[0]["validation_json"] or "{}") if validation_row else {}
                    validation.update({"review_mode": "ai_assisted_auto", "review_model": settings.deepseek_model})
                    with connect() as db:
                        db.execute("UPDATE eval_dataset_cases SET validation_json=? WHERE id=?", (json_value(validation), item["id"]))
                    accepted_count += 1
                except ValueError as error:
                    failed.append({"id": item["id"], "reason": str(error)[:240]})
            context.update(progress=min(95, round((offset + len(batch)) / len(drafts) * 100)), phase="auto_review", message=f"已自动审核 {min(offset + len(batch), len(drafts))}/{len(drafts)} 条")
        refreshed = dataset_detail(dataset_id) or {}
        remaining = sum(item["status"] == "draft" for item in refreshed.get("cases", []))
        finish_trace(trace_id, duration_ms=round((perf_counter() - started) * 1000), attributes={
            "accepted_count": accepted_count, "remaining_draft_count": remaining, "failed_count": len(failed),
        })
        context.update(progress=100, phase="auto_review_done", message=f"自动确认完成：{accepted_count} 条，保留 {remaining} 条待处理")
        return {
            "dataset_id": dataset_id, "accepted_count": accepted_count,
            "remaining_draft_count": remaining, "failed_count": len(failed),
            "failed": failed[:20], "trace_id": trace_id,
        }
    except Exception as error:
        finish_trace(trace_id, "failed", duration_ms=round((perf_counter() - started) * 1000), error_code=type(error).__name__)
        raise


def import_legacy_dataset(space_id: str, name: str = "KUN Gold Set", version: str = "v1") -> dict:
    legacy = rows(
        """SELECT e.*,d.fingerprint,d.title FROM evaluation_cases e
           JOIN documents d ON d.id=e.expected_document_id WHERE e.space_id=? ORDER BY e.id""",
        (space_id,),
    )
    if not legacy:
        raise ValueError("当前知识空间还没有可导入的人工评测题")
    cases: list[dict[str, Any]] = []
    for index, item in enumerate(legacy):
        cases.append({
            "id": item["id"],
            "question": item["question"],
            "split": "holdout" if index % 10 >= 7 else "dev",
            "query_type": "legacy",
            "difficulty": "medium",
            "status": "accepted",
            "gold": [{
                "document_id": item["expected_document_id"],
                "source_fingerprint": item["fingerprint"],
                "title": item["title"],
                "locator": item["expected_locator"],
                "text_hash": "",
                "relevance": 3,
            }],
        })
    content_hash = _dataset_hash(cases)
    existing = rows(
        "SELECT * FROM eval_dataset_versions WHERE name=? AND version=? AND space_id=?",
        (name, version, space_id),
    )
    dataset_id = existing[0]["id"] if existing else uuid4().hex
    stamp = now()
    with connect() as db:
        db.execute(
            """INSERT INTO eval_dataset_versions(
               id,name,version,space_id,status,source,content_hash,case_count,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(name,version,space_id) DO UPDATE SET
                 status=excluded.status,source=excluded.source,content_hash=excluded.content_hash,
                 case_count=excluded.case_count,updated_at=excluded.updated_at""",
            (dataset_id, name, version, space_id, "ready", "legacy_import", content_hash, len(cases), stamp, stamp),
        )
        db.execute("DELETE FROM eval_dataset_cases WHERE dataset_version_id=?", (dataset_id,))
        db.executemany(
            """INSERT INTO eval_dataset_cases(
               id,dataset_version_id,question,split,query_type,difficulty,status,gold_json,
               answer_text,source_type,validation_json,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    f"{dataset_id}:{item['id']}", dataset_id, item["question"], item["split"],
                    item["query_type"], item["difficulty"], item["status"], json_value(item["gold"]), "",
                    "human", json_value({"valid": True, "validator": "legacy_import"}), stamp, stamp,
                )
                for item in cases
            ],
        )
    _refresh_dataset(dataset_id)
    return dataset_detail(dataset_id) or {}


def list_datasets(space_id: str | None = None) -> list[dict]:
    clause = "WHERE v.space_id=?" if space_id else ""
    params: tuple[Any, ...] = (space_id,) if space_id else ()
    return rows(
        f"""SELECT v.*,
            SUM(CASE WHEN c.status='accepted' THEN 1 ELSE 0 END) accepted_count,
            SUM(CASE WHEN c.status='draft' THEN 1 ELSE 0 END) draft_count,
            SUM(CASE WHEN c.status='rejected' THEN 1 ELSE 0 END) rejected_count
            FROM eval_dataset_versions v LEFT JOIN eval_dataset_cases c ON c.dataset_version_id=v.id
            {clause} GROUP BY v.id ORDER BY v.created_at DESC""",
        params,
    )


def dataset_detail(dataset_id: str) -> dict | None:
    matches = rows("SELECT * FROM eval_dataset_versions WHERE id=?", (dataset_id,))
    if not matches:
        return None
    dataset = matches[0]
    cases = rows("SELECT * FROM eval_dataset_cases WHERE dataset_version_id=? ORDER BY id", (dataset_id,))
    for item in cases:
        item["gold"] = json.loads(item.pop("gold_json") or "[]")
        item["validation"] = json.loads(item.pop("validation_json") or "{}")
        item["answer"] = item.pop("answer_text", "")
    dataset["cases"] = cases
    dataset["splits"] = {
        "dev": sum(item["split"] == "dev" and item["status"] == "accepted" for item in cases),
        "holdout": sum(item["split"] == "holdout" and item["status"] == "accepted" for item in cases),
    }
    dataset["case_mix"] = {
        key: sum(item["query_type"] == key and item["status"] == "accepted" for item in cases)
        for key in sorted({item["query_type"] for item in cases})
    }
    dataset["source_documents"] = sorted({gold.get("title", "") for item in cases for gold in item["gold"] if gold.get("title")})
    return dataset


def create_experiment_run(
    dataset_version_id: str,
    name: str,
    config: dict[str, Any],
    parent_run_id: str | None = None,
) -> dict:
    if not rows("SELECT 1 ok FROM eval_dataset_versions WHERE id=? AND status='ready'", (dataset_version_id,)):
        raise ValueError("评测集不存在或尚未就绪")
    stable_config = normalize_experiment_config(config)
    config_hash = hashlib.sha256(json_value(stable_config).encode("utf-8")).hexdigest()
    run_id = uuid4().hex
    with connect() as db:
        db.execute(
            """INSERT INTO experiment_runs(
               id,dataset_version_id,name,status,config_json,config_hash,parent_run_id,git_revision,machine_json,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id, dataset_version_id, name[:120], "queued", json_value(stable_config), config_hash,
                parent_run_id, git_revision(), json_value(machine_profile()), now(),
            ),
        )
    return experiment_detail(run_id) or {"id": run_id, "status": "queued"}


def normalize_experiment_config(config: dict[str, Any]) -> dict[str, Any]:
    """Create the immutable, hashable configuration used by a run and a sweep."""
    stable_config = {
        "pipeline": config.get("pipeline", "bm25"),
        "generation_id": config.get("generation_id"),
        "candidate_k": max(5, min(int(config.get("candidate_k", 20)), 100)),
        "top_k": max(5, min(int(config.get("top_k", 10)), 20)),
        "reranker_top_n": max(10, min(int(config.get("reranker_top_n", 10)), 50)),
        "rrf_k": max(1, min(int(config.get("rrf_k", 60)), 200)),
        "reranker_model": config.get("reranker_model"),
        "split": config.get("split", "dev"),
        "case_status": config.get("case_status", "accepted"),
    }
    if stable_config["pipeline"] not in {"bm25", "dense", "hybrid", "hybrid_rerank"}:
        raise ValueError("Invalid experiment pipeline")
    if stable_config["split"] not in {"dev", "holdout", "all"}:
        raise ValueError("Experiment split must be dev, holdout, or all")
    if stable_config["case_status"] not in {"accepted", "exploratory"}:
        raise ValueError("Experiment case status must be accepted or exploratory")
    if stable_config["pipeline"] == "bm25":
        stable_config["generation_id"] = None
    elif not stable_config["generation_id"]:
        raise ValueError("Dense or hybrid experiments require an index generation")
    if stable_config["generation_id"]:
        generation_matches = rows(
            "SELECT * FROM index_generations WHERE id=? AND status='ready'",
            (stable_config["generation_id"],),
        )
        if not generation_matches:
            raise ValueError("Selected index generation is not ready")
        generation = generation_matches[0]
        stable_config["index_snapshot"] = {
            "id": generation["id"], "provider": generation["provider"], "model": generation["model"],
            "dimension": generation["dimension"], "strategy": generation["strategy"],
            "chunk_size": generation["chunk_size"], "chunk_overlap": generation["chunk_overlap"],
            "config_hash": generation["config_hash"], "parser_version": generation["parser_version"],
            "chunker_version": generation["chunker_version"],
        }
    return stable_config


def _locator_matches(expected: str, actual: str) -> bool:
    if not expected:
        return True
    import re
    expected_pages = set(re.findall(r"第\s*(\d+)\s*页", expected))
    actual_pages = set(re.findall(r"第\s*(\d+)\s*页", actual))
    if expected_pages:
        return bool(expected_pages.intersection(actual_pages))
    return expected.strip().lower() in actual.strip().lower()


def _relevance(item: dict[str, Any], gold: list[dict[str, Any]]) -> int:
    scores = [
        int(target.get("relevance", 3))
        for target in gold
        if item.get("document_id") == target.get("document_id")
        and _locator_matches(str(target.get("locator", "")), str(item.get("locator", "")))
    ]
    return max(scores, default=0)


def _dcg(relevances: list[int], k: int) -> float:
    return sum((2**value - 1) / math.log2(index + 2) for index, value in enumerate(relevances[:k]))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * percentile) - 1))
    return round(float(ordered[index]), 2)


def _bootstrap_ci(values: list[float], seed: int = 20260814, samples: int = 1000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = [
        mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(samples)
    ]
    means.sort()
    return [round(means[24], 4), round(means[974], 4)]


def run_experiment(run_id: str, context: JobContext) -> dict[str, Any]:
    matches = rows("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
    if not matches:
        raise ValueError("Experiment run does not exist")
    run = matches[0]
    config = json.loads(run["config_json"])
    dataset = rows("SELECT * FROM eval_dataset_versions WHERE id=?", (run["dataset_version_id"],))[0]
    split_clause = "" if config["split"] == "all" else "AND split=?"
    params: tuple[Any, ...] = (run["dataset_version_id"],) if config["split"] == "all" else (run["dataset_version_id"], config["split"])
    case_status_clause = "status='accepted'" if config.get("case_status", "accepted") == "accepted" else "status IN ('accepted','draft')"
    cases = rows(
        f"""SELECT * FROM eval_dataset_cases WHERE dataset_version_id=? AND {case_status_clause} {split_clause}
            ORDER BY id""",
        params,
    )
    if not cases:
        raise ValueError("Selected dataset split has no accepted cases")
    trace_id = create_trace("evaluation", "retrieval_evaluation", {
        "run_id": run_id, "dataset": dataset["name"], "dataset_version": dataset["version"], **config,
    })
    started = perf_counter()
    with connect() as db:
        db.execute("UPDATE experiment_runs SET status='running',started_at=? WHERE id=?", (now(), run_id))
        db.execute("DELETE FROM experiment_case_results WHERE run_id=?", (run_id,))
    try:
        latencies: list[float] = []
        stage_latencies: dict[str, list[float]] = {}
        api_totals = {
            "embedding_requests": 0, "rerank_requests": 0, "embedding_cache_hits": 0,
            "rerank_cache_hits": 0, "embedding_input_characters": 0, "rerank_input_characters": 0,
        }
        doc_hits = {1: 0, 5: 0, 10: 0}
        evidence_hits = {1: 0, 5: 0, 10: 0}
        case_metric_values = {
            "document_recall": {1: [], 5: [], 10: []},
            "evidence_recall": {1: [], 5: [], 10: []},
        }
        stage_doc_hits: dict[str, dict[int, int]] = {}
        stage_evidence_hits: dict[str, dict[int, int]] = {}
        reciprocal_ranks: list[float] = []
        ndcgs: list[float] = []
        citation_total = 0
        citation_resolvable = 0
        failure_counts: dict[str, int] = {}
        type_totals: dict[str, list[int]] = {}
        answerable_case_count = 0
        no_answer_case_count = 0
        no_answer_abstentions = 0
        for case_index, case in enumerate(cases, 1):
            context.check_cancelled()
            context.update(
                progress=max(1, int((case_index - 1) / len(cases) * 100)),
                phase="evaluation",
                message=f"正在评测 {case_index} / {len(cases)}",
            )
            gold = json.loads(case["gold_json"] or "[]")
            is_no_answer = case["query_type"] == "no_answer"
            if is_no_answer:
                no_answer_case_count += 1
            else:
                answerable_case_count += 1
            gold_documents = {item.get("document_id") for item in gold}
            result = pipeline_search(case["question"], dataset["space_id"], config, trace_type="evaluation_case")
            returned = result["results"]
            if is_no_answer and not returned:
                no_answer_abstentions += 1
            stage_results = result.get("stage_results", {})
            for stage_name, stage_items in stage_results.items():
                if stage_name in {"final", "evaluation"} or not stage_items:
                    continue
                stage_doc_hits.setdefault(stage_name, {1: 0, 5: 0, 10: 0})
                stage_evidence_hits.setdefault(stage_name, {1: 0, 5: 0, 10: 0})
                for k in (1, 5, 10):
                    stage_slice = stage_items[:k]
                    if any(item.get("document_id") in gold_documents for item in stage_slice):
                        stage_doc_hits[stage_name][k] += 1
                    if any(_relevance(item, gold) > 0 for item in stage_slice):
                        stage_evidence_hits[stage_name][k] += 1
            for key, value in result.get("latency_breakdown", {}).items():
                stage_latencies.setdefault(key, []).append(float(value or 0))
            for key, value in result.get("api_stats", {}).items():
                if key in api_totals:
                    api_totals[key] += int(value or 0)
            if not is_no_answer:
                citation_total += len(returned)
                citation_resolvable += sum(bool(item.get("id") and item.get("locator")) for item in returned)
            latencies.append(float(result["duration_ms"]))
            evaluation_items = stage_results.get("evaluation") or returned
            evaluation_relevances = [_relevance(item, gold) for item in evaluation_items]
            returned_relevances = [_relevance(item, gold) for item in returned]
            first_rank = next((index for index, relevance in enumerate(evaluation_relevances, 1) if relevance > 0), None)
            for k in (1, 5, 10):
                document_hit = any(item.get("document_id") in gold_documents for item in evaluation_items[:k])
                evidence_hit = any(value > 0 for value in evaluation_relevances[:k])
                if not is_no_answer:
                    if document_hit:
                        doc_hits[k] += 1
                    if evidence_hit:
                        evidence_hits[k] += 1
                    case_metric_values["document_recall"][k].append(float(document_hit))
                    case_metric_values["evidence_recall"][k].append(float(evidence_hit))
            if not is_no_answer:
                reciprocal_ranks.append(1 / first_rank if first_rank else 0.0)
            ideal = sorted([int(item.get("relevance", 3)) for item in gold], reverse=True)
            ideal_dcg = _dcg(ideal, 10)
            ndcg = _dcg(evaluation_relevances, 10) / ideal_dcg if ideal_dcg else 0.0
            if not is_no_answer:
                ndcgs.append(ndcg)
            document_found = any(item.get("document_id") in gold_documents for item in evaluation_items)
            if is_no_answer:
                failure = "negative_query_not_abstained" if returned else None
            elif first_rank:
                failure = None
            elif document_found:
                failure = "document_found_evidence_missed"
            elif returned:
                failure = "not_retrieved"
            else:
                failure = "empty_result"
            if failure:
                failure_counts[failure] = failure_counts.get(failure, 0) + 1
            if not is_no_answer:
                type_bucket = type_totals.setdefault(case["query_type"], [0, 0])
                type_bucket[1] += 1
                type_bucket[0] += int(bool(first_rank))
            metrics = {
                "document_hit": document_found,
                "evidence_hit": bool(first_rank),
                "first_evidence_rank": first_rank,
                "reciprocal_rank": round(1 / first_rank, 6) if first_rank else 0,
                "ndcg_10": round(ndcg, 6),
                "evaluation_result_count": len(evaluation_items),
                "context_result_count": len(returned),
                "negative_case": is_no_answer,
                "abstained": not returned if is_no_answer else None,
            }
            rankings = {
                "gold": gold,
                "returned": [
                    {
                        "chunk_id": item["id"],
                        "document_id": item["document_id"],
                        "title": item["title"],
                        "locator": item["locator"],
                        "lexical_rank": item.get("lexical_rank"),
                        "vector_rank": item.get("vector_rank"),
                        "fusion_rank": item.get("fusion_rank"),
                        "rerank_rank": item.get("rerank_rank"),
                        "score": item.get("score"),
                        "relevance": relevance,
                    }
                    for item, relevance in zip(returned, returned_relevances)
                ],
                "evaluation": [
                    {
                        "chunk_id": item.get("chunk_id") or item.get("id"),
                        "document_id": item.get("document_id"),
                        "locator": item.get("locator"),
                        "rank": index,
                        "relevance": relevance,
                    }
                    for index, (item, relevance) in enumerate(zip(evaluation_items, evaluation_relevances), 1)
                ],
                "stages": result["stages"],
                "stage_results": stage_results,
                "latency_breakdown": result.get("latency_breakdown", {}),
                "api_stats": result.get("api_stats", {}),
            }
            with connect() as db:
                db.execute(
                    """INSERT INTO experiment_case_results(
                       run_id,case_id,latency_ms,metrics_json,rankings_json,failure_category,trace_id
                       ) VALUES(?,?,?,?,?,?,?)""",
                    (
                        run_id, case["id"], result["duration_ms"], json_value(metrics), json_value(rankings),
                        failure, result["trace_id"],
                    ),
                )
        case_count = len(cases)
        quality_denominator = max(answerable_case_count, 1)
        summary = {
            "case_count": case_count,
            "answerable_case_count": answerable_case_count,
            "no_answer_case_count": no_answer_case_count,
            "no_answer_abstention_rate": round(no_answer_abstentions / no_answer_case_count, 4) if no_answer_case_count else None,
            "document_recall": {str(k): round(doc_hits[k] / quality_denominator, 4) for k in (1, 5, 10)},
            "evidence_recall": {str(k): round(evidence_hits[k] / quality_denominator, 4) for k in (1, 5, 10)},
            "mrr": round(mean(reciprocal_ranks), 4),
            "ndcg_10": round(mean(ndcgs), 4),
            "citation_resolvable_rate": round(citation_resolvable / citation_total, 4) if citation_total else 0,
            "latency_ms": {
                "mean": round(mean(latencies), 2),
                "p50": _percentile(latencies, 0.5),
                "p95": _percentile(latencies, 0.95),
                "p99": _percentile(latencies, 0.99),
            },
            "failure_counts": failure_counts,
            "query_types": {
                key: {"evidence_recall": round(value[0] / value[1], 4), "case_count": value[1]}
                for key, value in type_totals.items()
            },
            "stage_recall": {
                stage: {
                    "document_recall": {str(k): round(stage_doc_hits[stage][k] / quality_denominator, 4) for k in (1, 5, 10)},
                    "evidence_recall": {str(k): round(stage_evidence_hits[stage][k] / quality_denominator, 4) for k in (1, 5, 10)},
                }
                for stage in stage_doc_hits
            },
            "latency_breakdown": {
                key: {
                    "mean": round(mean(values), 2) if values else 0,
                    "p50": _percentile(values, 0.5),
                    "p95": _percentile(values, 0.95),
                }
                for key, values in stage_latencies.items()
            },
            "api_stats": api_totals,
            "confidence_intervals": {
                metric: {str(k): _bootstrap_ci(values) for k, values in buckets.items()}
                for metric, buckets in case_metric_values.items()
            },
            "evaluation_scope": "accepted" if config.get("case_status", "accepted") == "accepted" else "exploratory",
            "dataset": {"name": dataset["name"], "version": dataset["version"], "content_hash": dataset["content_hash"]},
            "config": config,
        }
        with connect() as db:
            db.execute(
                """UPDATE experiment_runs SET status='succeeded',summary_json=?,finished_at=?,error_code=NULL
                   WHERE id=?""",
                (json_value(summary), now(), run_id),
            )
        context.update(progress=100, phase="completed", message="评测运行完成")
        finish_trace(trace_id, duration_ms=round((perf_counter() - started) * 1000), attributes={
            "run_id": run_id, "case_count": case_count, "evidence_recall_5": summary["evidence_recall"]["5"],
        })
        return {"run_id": run_id, "trace_id": trace_id, **summary}
    except Exception as error:
        with connect() as db:
            db.execute(
                "UPDATE experiment_runs SET status='failed',error_code=?,finished_at=? WHERE id=?",
                (type(error).__name__, now(), run_id),
            )
        finish_trace(
            trace_id, "failed", duration_ms=round((perf_counter() - started) * 1000), error_code=type(error).__name__
        )
        raise


def list_experiments(limit: int = 100) -> list[dict]:
    runs = rows("SELECT * FROM experiment_runs ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 500)),))
    return [_decode_run(item) for item in runs]


def experiment_detail(run_id: str) -> dict | None:
    matches = rows("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
    if not matches:
        return None
    run = _decode_run(matches[0])
    results = rows(
        """SELECT r.*,c.question,c.query_type,c.difficulty FROM experiment_case_results r
           JOIN eval_dataset_cases c ON c.id=r.case_id WHERE r.run_id=? ORDER BY c.id""",
        (run_id,),
    )
    for item in results:
        item["metrics"] = json.loads(item.pop("metrics_json") or "{}")
        item["rankings"] = json.loads(item.pop("rankings_json") or "{}")
    run["cases"] = results
    return run


def experiment_display_name(config: dict[str, Any], summary: dict[str, Any] | None = None, fallback: str = "") -> str:
    """Return one stable, human-readable name without changing persisted run names."""
    summary = summary or {}
    pipeline = str(config.get("pipeline") or "bm25")
    generation = config.get("index_snapshot") or {}
    model = str(generation.get("model") or config.get("embedding_model") or "embedding")
    dimension = generation.get("dimension") or config.get("dimension")
    model_label = f"{model}/{dimension}d" if dimension else model
    pipeline_labels = {
        "bm25": "BM25 · 本地词法检索",
        "dense": "Dense · 向量检索",
        "hybrid": "Hybrid · BM25 + Dense + RRF",
        "hybrid_rerank": "Hybrid + Rerank · RRF 后重排",
    }
    headline = pipeline_labels.get(pipeline, pipeline)
    if pipeline != "bm25":
        headline = f"{headline} · {model_label}"

    details: list[str] = []
    chunk_size = generation.get("chunk_size")
    chunk_overlap = generation.get("chunk_overlap")
    if chunk_size is not None and chunk_overlap is not None:
        details.append(f"Chunk {chunk_size}/{chunk_overlap}")
    strategy = generation.get("strategy")
    if strategy:
        details.append(str(strategy).upper())
    details.append(f"cand {int(config.get('candidate_k', 20))}")
    details.append(f"top {int(config.get('top_k', 10))}")
    if pipeline in {"hybrid", "hybrid_rerank"}:
        details.append(f"RRF {int(config.get('rrf_k', 60))}")
    if pipeline == "hybrid_rerank":
        details.append(f"rerank {int(config.get('reranker_top_n', 10))}")
    case_count = summary.get("case_count")
    if isinstance(case_count, (int, float)) and case_count > 0:
        scope = "exploratory" if config.get("case_status") == "exploratory" else "Gold"
        details.append(f"{int(case_count)} {scope}")
    return " · ".join([headline, *details]) if details else (headline or fallback)


def render_experiment_report(run_id: str, report_format: str = "markdown") -> tuple[str, str]:
    run = experiment_detail(run_id)
    if not run:
        raise ValueError("Experiment run does not exist")
    if report_format == "json":
        return json.dumps(run, ensure_ascii=False, indent=2), "application/json"
    if report_format != "markdown":
        raise ValueError("Report format must be markdown or json")
    summary = run["summary"]
    dataset = rows("SELECT * FROM eval_dataset_versions WHERE id=?", (run["dataset_version_id"],))[0]
    bad_cases = [item for item in run.get("cases", []) if item.get("failure_category")]
    lines = [
        f"# {run.get('display_name') or run['name']}",
        "",
        "> This is a reproducible local evaluation report. It is not a production accuracy claim.",
        "",
        "## Reproducibility",
        "",
        f"- Run ID: `{run['id']}`",
        f"- Status: `{run['status']}`",
        f"- Dataset: `{dataset['name']} {dataset['version']}` (`{dataset['content_hash']}`)",
        f"- Cases measured: {summary.get('case_count', 0)} (answerable: {summary.get('answerable_case_count', summary.get('case_count', 0))}; no-answer: {summary.get('no_answer_case_count', 0)})",
        f"- Git revision: `{run['git_revision']}`",
        f"- Created at: {run['created_at']}",
        f"- Original name: `{run['name']}`",
        f"- Machine: `{json.dumps(run['machine'], ensure_ascii=False, sort_keys=True)}`",
        f"- Config: `{json.dumps(run['config'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Quality metrics",
        "",
        "| Metric | @1 | @5 | @10 |",
        "| --- | ---: | ---: | ---: |",
        f"| Document Recall | {summary.get('document_recall', {}).get('1', 0):.4f} | {summary.get('document_recall', {}).get('5', 0):.4f} | {summary.get('document_recall', {}).get('10', 0):.4f} |",
        f"| Evidence Recall | {summary.get('evidence_recall', {}).get('1', 0):.4f} | {summary.get('evidence_recall', {}).get('5', 0):.4f} | {summary.get('evidence_recall', {}).get('10', 0):.4f} |",
        "",
        f"- MRR: {summary.get('mrr', 0):.4f}",
        f"- nDCG@10: {summary.get('ndcg_10', 0):.4f}",
        f"- Citation resolvable rate: {summary.get('citation_resolvable_rate', 0):.4f}",
        f"- No-answer abstention rate: {summary.get('no_answer_abstention_rate', 'n/a')}",
        "",
        "## 95% confidence intervals (paired bootstrap)",
        "",
    ]
    for metric_name, label in (("evidence_recall", "Evidence Recall"), ("document_recall", "Document Recall")):
        intervals = summary.get("confidence_intervals", {}).get(metric_name, {})
        lines.append("- " + label + ": " + ", ".join(
            f"@{k} [{value[0]:.4f}, {value[1]:.4f}]" for k, value in intervals.items()
        ))
    lines.extend([
        "",
        "## Retrieval latency",
        "",
        f"- Mean: {summary.get('latency_ms', {}).get('mean', 0)} ms",
        f"- P50: {summary.get('latency_ms', {}).get('p50', 0)} ms",
        f"- P95: {summary.get('latency_ms', {}).get('p95', 0)} ms",
        f"- P99: {summary.get('latency_ms', {}).get('p99', 0)} ms",
        "",
        "## Bad cases",
        "",
    ])
    if not bad_cases:
        lines.append("No bad cases in this run.")
    else:
        for item in bad_cases:
            lines.extend([
                f"### {item['question']}",
                "",
                f"- Category: `{item['failure_category']}`",
                f"- Latency: {item['latency_ms']} ms",
                f"- Trace ID: `{item.get('trace_id') or 'n/a'}`",
                "",
            ])
    lines.extend([
        "## Interpretation boundary",
        "",
        "Quality metrics come only from accepted human Gold cases. Synthetic-vector performance benchmarks are reported separately and must not be presented as retrieval quality.",
        "",
    ])
    return "\n".join(lines), "text/markdown; charset=utf-8"


def _decode_run(item: dict) -> dict:
    item["config"] = json.loads(item.pop("config_json") or "{}")
    item["machine"] = json.loads(item.pop("machine_json") or "{}")
    item["summary"] = json.loads(item.pop("summary_json") or "{}")
    item["display_name"] = experiment_display_name(item["config"], item["summary"], item.get("name", ""))
    return item


def compare_experiments(baseline_id: str, candidate_id: str) -> dict[str, Any]:
    baseline = experiment_detail(baseline_id)
    candidate = experiment_detail(candidate_id)
    if not baseline or not candidate:
        raise ValueError("Baseline or candidate experiment does not exist")
    if baseline["dataset_version_id"] != candidate["dataset_version_id"]:
        raise ValueError("Experiments must use the same dataset version")
    if baseline["status"] != "succeeded" or candidate["status"] != "succeeded":
        raise ValueError("Both experiments must have succeeded")
    base_summary = baseline["summary"]
    candidate_summary = candidate["summary"]
    base_evidence = float(base_summary["evidence_recall"]["5"])
    candidate_evidence = float(candidate_summary["evidence_recall"]["5"])
    base_document = float(base_summary["document_recall"]["5"])
    candidate_document = float(candidate_summary["document_recall"]["5"])
    base_p95 = float(base_summary["latency_ms"]["p95"])
    candidate_p95 = float(candidate_summary["latency_ms"]["p95"])
    checks = [
        {
            "name": "Evidence Recall@5",
            "baseline": base_evidence,
            "candidate": candidate_evidence,
            "delta": round(candidate_evidence - base_evidence, 4),
            "status": "failed" if candidate_evidence < base_evidence - 0.01 else "passed",
            "rule": "绝对下降不得超过 0.01",
        },
        {
            "name": "Document Recall@5",
            "baseline": base_document,
            "candidate": candidate_document,
            "delta": round(candidate_document - base_document, 4),
            "status": "failed" if candidate_document < base_document - 0.01 else "passed",
            "rule": "绝对下降不得超过 0.01",
        },
        {
            "name": "Retrieval P95",
            "baseline": base_p95,
            "candidate": candidate_p95,
            "delta": round(candidate_p95 - base_p95, 2),
            "status": "advisory" if base_p95 and candidate_p95 > base_p95 * 1.2 else "informational",
            "rule": "增长不得超过 20%",
        },
    ]
    for check in checks:
        if check["name"] == "Retrieval P95":
            check["rule"] = "Provider and total latency are advisory; quality is not blocked."
    baseline_cases = {item["case_id"]: int(bool(item["metrics"].get("evidence_hit"))) for item in baseline["cases"]}
    candidate_cases = {item["case_id"]: int(bool(item["metrics"].get("evidence_hit"))) for item in candidate["cases"]}
    paired = [(baseline_cases[key], candidate_cases[key]) for key in baseline_cases if key in candidate_cases]
    rng = random.Random(20260813)
    bootstrap: list[float] = []
    if paired:
        for _ in range(1000):
            sample = [paired[rng.randrange(len(paired))] for _ in paired]
            bootstrap.append(mean(right - left for left, right in sample))
    ci = sorted(bootstrap)
    confidence = {
        "method": "paired_bootstrap",
        "samples": 1000,
        "seed": 20260813,
        "evidence_recall_delta_95_ci": [round(ci[24], 4), round(ci[974], 4)] if ci else [0, 0],
    }
    return {
        "baseline": {"id": baseline_id, "name": baseline["name"], "display_name": baseline["display_name"], "config": baseline["config"]},
        "candidate": {"id": candidate_id, "name": candidate["name"], "display_name": candidate["display_name"], "config": candidate["config"]},
        "status": "failed" if any(item["status"] == "failed" for item in checks) else "passed",
        "checks": checks,
        "confidence": confidence,
    }


def create_experiment_sweep(
    dataset_version_id: str,
    name: str,
    configs: list[dict[str, Any]],
    case_status: str = "accepted",
) -> dict[str, Any]:
    dataset = dataset_detail(dataset_version_id)
    if not dataset or dataset["status"] != "ready":
        raise ValueError("Evaluation dataset does not exist or is not ready")
    if case_status not in {"accepted", "exploratory"}:
        raise ValueError("Sweep case status must be accepted or exploratory")
    if not configs or len(configs) > 100:
        raise ValueError("Sweep must contain between 1 and 100 configurations")
    normalized: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for ordinal, raw in enumerate(configs, 1):
        config = {**raw, "case_status": case_status}
        stable = normalize_experiment_config(config)
        config_hash = hashlib.sha256(json_value(stable).encode("utf-8")).hexdigest()
        if config_hash in seen_hashes:
            continue
        seen_hashes.add(config_hash)
        normalized.append({"ordinal": ordinal, "name": raw.get("name") or f"{name} #{ordinal}", "config": stable, "config_hash": config_hash})
    if not normalized:
        raise ValueError("Sweep has no unique configurations")
    eligible_cases = [
        item for item in dataset["cases"]
        if item["status"] == "accepted" and (case_status == "exploratory" or item["status"] == "accepted")
    ]
    if case_status == "exploratory":
        eligible_cases = [item for item in dataset["cases"] if item["status"] in {"accepted", "draft"}]
    if not eligible_cases:
        raise ValueError("Selected dataset has no eligible cases")
    request_count = 0
    for item in normalized:
        pipeline = item["config"]["pipeline"]
        request_count += len(eligible_cases) * (
            int(pipeline in {"dense", "hybrid", "hybrid_rerank"}) + int(pipeline == "hybrid_rerank")
        )
    if request_count > 0:
        sample_chars = rows(
            "SELECT COALESCE(SUM(length(text)),0) total FROM chunks WHERE document_id IN (SELECT id FROM documents WHERE space_id=?)",
            (dataset["space_id"],),
        )[0]["total"]
    else:
        sample_chars = 0
    sweep_config = {
        "case_status": case_status,
        "dataset_version_id": dataset_version_id,
        "items": normalized,
        "execution": "serial",
    }
    sweep_hash = hashlib.sha256(json_value(sweep_config).encode("utf-8")).hexdigest()
    existing = rows("SELECT id FROM experiment_sweeps WHERE config_hash=?", (sweep_hash,))
    if existing:
        return sweep_detail(existing[0]["id"]) or {}
    sweep_id = uuid4().hex
    stamp = now()
    with connect() as db:
        db.execute(
            """INSERT INTO experiment_sweeps(
               id,dataset_version_id,name,status,config_json,config_hash,case_status,
               estimated_requests,estimated_input_characters,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                sweep_id, dataset_version_id, name[:120], "queued", json_value(sweep_config), sweep_hash,
                case_status, request_count, int(sample_chars or 0), stamp,
            ),
        )
    for item in normalized:
        existing_run = rows(
            "SELECT id,status FROM experiment_runs WHERE dataset_version_id=? AND config_hash=? ORDER BY created_at DESC LIMIT 1",
            (dataset_version_id, item["config_hash"]),
        )
        if existing_run:
            run_id = existing_run[0]["id"]
            item_status = existing_run[0]["status"]
        else:
            run = create_experiment_run(dataset_version_id, item["name"], item["config"])
            run_id = run["id"]
            item_status = run["status"]
        with connect() as db:
            db.execute(
                "INSERT INTO experiment_sweep_items(sweep_id,ordinal,run_id,status) VALUES(?,?,?,?)",
                (sweep_id, item["ordinal"], run_id, item_status if item_status in {"succeeded", "running", "queued", "failed"} else "queued"),
            )
    return sweep_detail(sweep_id) or {"id": sweep_id, "status": "queued"}


def list_experiment_sweeps(limit: int = 30) -> list[dict[str, Any]]:
    return [sweep_summary(item) for item in rows(
        "SELECT * FROM experiment_sweeps ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 100)),)
    )]


def sweep_summary(item: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(item)
    decoded["config"] = json.loads(decoded.pop("config_json") or "{}")
    counts = rows(
        "SELECT status,COUNT(*) count FROM experiment_sweep_items WHERE sweep_id=? GROUP BY status",
        (decoded["id"],),
    )
    decoded["item_counts"] = {row["status"]: row["count"] for row in counts}
    decoded["total_items"] = sum(decoded["item_counts"].values())
    decoded["completed_items"] = sum(decoded["item_counts"].get(value, 0) for value in ("succeeded", "failed", "cancelled"))
    return decoded


def sweep_detail(sweep_id: str) -> dict[str, Any] | None:
    matches = rows("SELECT * FROM experiment_sweeps WHERE id=?", (sweep_id,))
    if not matches:
        return None
    sweep = sweep_summary(matches[0])
    item_rows = rows(
        """SELECT i.*,r.name,r.status run_status,r.config_json,r.summary_json,r.error_code run_error
           FROM experiment_sweep_items i JOIN experiment_runs r ON r.id=i.run_id
           WHERE i.sweep_id=? ORDER BY i.ordinal""",
        (sweep_id,),
    )
    items: list[dict[str, Any]] = []
    for item in item_rows:
        item["config"] = json.loads(item.pop("config_json") or "{}")
        item["summary"] = json.loads(item.pop("summary_json") or "{}")
        item["display_name"] = experiment_display_name(item["config"], item["summary"], item.get("name", ""))
        item["status"] = item["run_status"] if item["run_status"] in {"succeeded", "running", "queued", "failed"} else item["status"]
        item.pop("run_status", None)
        items.append(item)
    sweep["items"] = items
    return sweep


def run_experiment_sweep(sweep_id: str, context: JobContext) -> dict[str, Any]:
    sweep = sweep_detail(sweep_id)
    if not sweep:
        raise ValueError("Experiment sweep does not exist")
    started = perf_counter()
    with connect() as db:
        db.execute("UPDATE experiment_sweeps SET status='running',started_at=?,error_code=NULL WHERE id=?", (now(), sweep_id))
    failed = 0
    cancelled = False
    try:
        items = sweep["items"]
        for index, item in enumerate(items, 1):
            context.check_cancelled()
            if item["status"] == "succeeded":
                context.update(progress=int(index / len(items) * 100), phase="sweep", message=f"跳过已完成实验 {index}/{len(items)}")
                continue
            with connect() as db:
                db.execute("UPDATE experiment_sweep_items SET status='running',started_at=?,error_code=NULL WHERE sweep_id=? AND ordinal=?", (now(), sweep_id, item["ordinal"]))
                db.execute("UPDATE experiment_runs SET status='queued',error_code=NULL WHERE id=? AND status='failed'", (item["run_id"],))
            try:
                run_experiment(item["run_id"], context)
                item_status = "succeeded"
                error_code = None
            except JobCancelled:
                cancelled = True
                with connect() as db:
                    db.execute("UPDATE experiment_sweep_items SET status='cancelled',finished_at=? WHERE sweep_id=? AND ordinal=?", (now(), sweep_id, item["ordinal"]))
                raise
            except Exception as error:
                failed += 1
                item_status = "failed"
                error_code = type(error).__name__
            with connect() as db:
                db.execute("UPDATE experiment_sweep_items SET status=?,error_code=?,finished_at=? WHERE sweep_id=? AND ordinal=?", (item_status, error_code, now(), sweep_id, item["ordinal"]))
            context.update(progress=int(index / len(items) * 100), phase="sweep", message=f"实验完成 {index}/{len(items)}")
        final_status = "partial_failed" if failed else "succeeded"
        with connect() as db:
            db.execute("UPDATE experiment_sweeps SET status=?,finished_at=?,error_code=? WHERE id=?", (final_status, now(), "child_failed" if failed else None, sweep_id))
        return {"sweep_id": sweep_id, "status": final_status, "failed_items": failed, "duration_ms": round((perf_counter() - started) * 1000, 2)}
    except JobCancelled:
        with connect() as db:
            db.execute("UPDATE experiment_sweeps SET status='cancelled',finished_at=?,error_code='cancelled' WHERE id=?", (now(), sweep_id))
        raise
    except Exception as error:
        with connect() as db:
            db.execute("UPDATE experiment_sweeps SET status='failed',finished_at=?,error_code=? WHERE id=?", (now(), type(error).__name__, sweep_id))
        raise


def create_performance_benchmark(config: dict[str, Any]) -> dict:
    stable = {
        "sizes": [int(value) for value in config.get("sizes", [1000, 10000]) if int(value) in {1000, 10000, 100000}],
        "dimension": int(config.get("dimension", 256)),
        "query_count": max(10, min(int(config.get("query_count", 100)), 500)),
        "seed": int(config.get("seed", 20260813)),
    }
    if not stable["sizes"]:
        stable["sizes"] = [1000, 10000]
    if stable["dimension"] not in {128, 256, 512, 768, 1024}:
        raise ValueError("Unsupported benchmark dimension")
    benchmark_id = uuid4().hex
    with connect() as db:
        db.execute(
            """INSERT INTO performance_benchmarks(id,status,config_json,machine_json,created_at)
               VALUES(?,?,?,?,?)""",
            (benchmark_id, "queued", json_value(stable), json_value(machine_profile()), now()),
        )
    return performance_detail(benchmark_id) or {"id": benchmark_id, "status": "queued"}


def run_performance_benchmark(benchmark_id: str, context: JobContext) -> dict[str, Any]:
    import faiss

    matches = rows("SELECT * FROM performance_benchmarks WHERE id=?", (benchmark_id,))
    if not matches:
        raise ValueError("Performance benchmark does not exist")
    config = json.loads(matches[0]["config_json"])
    with connect() as db:
        db.execute("UPDATE performance_benchmarks SET status='running',started_at=? WHERE id=?", (now(), benchmark_id))
    trace_id = create_trace("performance", "faiss_scale_benchmark", config)
    started = perf_counter()
    results: list[dict[str, Any]] = []
    try:
        for size_index, size in enumerate(config["sizes"], 1):
            context.check_cancelled()
            context.update(
                progress=int((size_index - 1) / len(config["sizes"]) * 100),
                phase="vector_generation",
                message=f"正在准备 {size:,} 条确定性向量",
            )
            estimated_bytes = size * config["dimension"] * 4
            if estimated_bytes > 1_200_000_000:
                results.append({"size": size, "status": "skipped", "reason": "预计向量内存超过 1.2 GB 安全阈值"})
                continue
            rng = np.random.default_rng(config["seed"] + size)
            vectors = rng.standard_normal((size, config["dimension"]), dtype=np.float32)
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            query_count = min(config["query_count"], size)
            queries = vectors[:query_count].copy()
            flat = faiss.IndexFlatIP(config["dimension"])
            flat_started = perf_counter()
            flat.add(vectors)
            flat_build_ms = round((perf_counter() - flat_started) * 1000, 2)
            flat_latencies: list[float] = []
            flat_ids: list[list[int]] = []
            for query in queries:
                query_started = perf_counter()
                _, identifiers = flat.search(query.reshape(1, -1), 10)
                flat_latencies.append((perf_counter() - query_started) * 1000)
                flat_ids.append([int(value) for value in identifiers[0]])
            hnsw = faiss.IndexHNSWFlat(config["dimension"], 32, faiss.METRIC_INNER_PRODUCT)
            hnsw.hnsw.efConstruction = 80
            hnsw.hnsw.efSearch = 64
            hnsw_started = perf_counter()
            hnsw.add(vectors)
            hnsw_build_ms = round((perf_counter() - hnsw_started) * 1000, 2)
            hnsw_latencies: list[float] = []
            overlaps: list[float] = []
            for query, expected in zip(queries, flat_ids):
                query_started = perf_counter()
                _, identifiers = hnsw.search(query.reshape(1, -1), 10)
                hnsw_latencies.append((perf_counter() - query_started) * 1000)
                actual = {int(value) for value in identifiers[0]}
                overlaps.append(len(set(expected).intersection(actual)) / 10)
            results.append({
                "size": size,
                "status": "measured",
                "dimension": config["dimension"],
                "estimated_vector_bytes": estimated_bytes,
                "flat": {
                    "build_ms": flat_build_ms,
                    "p50_ms": _percentile(flat_latencies, 0.5),
                    "p95_ms": _percentile(flat_latencies, 0.95),
                    "p99_ms": _percentile(flat_latencies, 0.99),
                    "qps": round(1000 / mean(flat_latencies), 2) if mean(flat_latencies) else 0,
                },
                "hnsw": {
                    "build_ms": hnsw_build_ms,
                    "p50_ms": _percentile(hnsw_latencies, 0.5),
                    "p95_ms": _percentile(hnsw_latencies, 0.95),
                    "p99_ms": _percentile(hnsw_latencies, 0.99),
                    "qps": round(1000 / mean(hnsw_latencies), 2) if mean(hnsw_latencies) else 0,
                    "ann_recall_10": round(mean(overlaps), 4),
                },
            })
            del vectors, queries, flat, hnsw
        payload = {
            "benchmark_id": benchmark_id,
            "quality_claim": False,
            "note": "确定性向量仅用于基础设施性能测试，不代表真实检索准确率。",
            "results": results,
            "trace_id": trace_id,
        }
        with connect() as db:
            db.execute(
                """UPDATE performance_benchmarks SET status='succeeded',result_json=?,finished_at=?,error_code=NULL
                   WHERE id=?""",
                (json_value(payload), now(), benchmark_id),
            )
        finish_trace(trace_id, duration_ms=round((perf_counter() - started) * 1000), attributes={"sizes": config["sizes"]})
        context.update(progress=100, phase="completed", message="性能压测完成")
        return payload
    except Exception as error:
        with connect() as db:
            db.execute(
                "UPDATE performance_benchmarks SET status='failed',error_code=?,finished_at=? WHERE id=?",
                (type(error).__name__, now(), benchmark_id),
            )
        finish_trace(trace_id, "failed", duration_ms=round((perf_counter() - started) * 1000), error_code=type(error).__name__)
        raise


def list_performance_benchmarks(limit: int = 30) -> list[dict]:
    return [_decode_performance(item) for item in rows(
        "SELECT * FROM performance_benchmarks ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 100)),)
    )]


def performance_detail(benchmark_id: str) -> dict | None:
    matches = rows("SELECT * FROM performance_benchmarks WHERE id=?", (benchmark_id,))
    return _decode_performance(matches[0]) if matches else None


def _decode_performance(item: dict) -> dict:
    item["config"] = json.loads(item.pop("config_json") or "{}")
    item["result"] = json.loads(item.pop("result_json") or "{}")
    item["machine"] = json.loads(item.pop("machine_json") or "{}")
    return item
