from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .config import get_settings
from .privacy import get_privacy_settings
from .credentials import available as credential_store_available
from .credentials import set_secret
from .database import connect, now, rows


def provider_statuses() -> list[dict]:
    settings = get_settings()
    persisted = {item["provider"]: item for item in rows("SELECT * FROM provider_status")}
    definitions = [
        {
            "provider": "deepseek",
            "label": "DeepSeek",
            "capability": "chat",
            "configured": bool(settings.deepseek_api_key),
            "model": settings.deepseek_model,
            "base_url_host": urlparse(settings.deepseek_base_url).hostname,
        },
        {
            "provider": "dashscope",
            "label": "阿里云百炼",
            "capability": "embedding",
            "configured": bool(settings.dashscope_api_key),
            "model": settings.embedding_model,
            "base_url_host": urlparse(settings.dashscope_base_url).hostname,
        },
        {
            "provider": "dashscope-rerank",
            "label": "百炼 Rerank",
            "capability": "rerank",
            "configured": bool(settings.dashscope_api_key and settings.dashscope_rerank_base_url),
            "model": settings.rerank_model,
            "base_url_host": urlparse(settings.dashscope_rerank_base_url).hostname,
        },
    ]
    for item in definitions:
        state = persisted.get(item["provider"], {})
        item.update(
            {
                "connection_status": state.get("status", "not_tested") if item["configured"] else "not_configured",
                "last_checked_at": state.get("last_checked_at"),
                "error_code": state.get("error_code"),
                "credential_store": "windows_credential_manager" if credential_store_available() else "environment_only",
            }
        )
    return definitions


def save_provider_key(provider: str, api_key: str) -> None:
    set_secret(provider, api_key)
    _save_status(provider, "not_tested", None)


def test_provider(provider: str) -> dict:
    settings = get_settings()
    try:
        if provider == "deepseek":
            if not settings.deepseek_api_key:
                return _failure(provider, "not_configured", "尚未配置 DeepSeek API Key")
            response = httpx.post(
                f"{settings.deepseek_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {settings.deepseek_api_key}"},
                json={
                    "model": settings.deepseek_model,
                    "temperature": 0,
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": "只回复 pong"}],
                },
                timeout=30,
            )
            response.raise_for_status()
            if not response.json().get("choices"):
                return _failure(provider, "invalid_response", "模型返回格式异常")
            return _success(provider, settings.deepseek_model)
        if provider == "dashscope":
            if not settings.dashscope_api_key:
                return _failure(provider, "not_configured", "尚未配置阿里云百炼 API Key")
            response = httpx.post(
                f"{settings.dashscope_base_url}/embeddings",
                headers={"Authorization": f"Bearer {settings.dashscope_api_key}"},
                json={"model": settings.embedding_model, "input": ["KUN connection test"], "dimensions": 1024},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json().get("data", [])
            if not data or not data[0].get("embedding"):
                return _failure(provider, "invalid_response", "Embedding 返回格式异常")
            return _success(provider, settings.embedding_model, dimension=len(data[0]["embedding"]))
        if provider == "dashscope-rerank":
            if not settings.dashscope_api_key or not settings.dashscope_rerank_base_url:
                return _failure(provider, "not_configured", "请先配置百炼 API Key 和 Rerank 工作空间地址")
            response = httpx.post(
                f"{settings.dashscope_rerank_base_url}/reranks",
                headers={"Authorization": f"Bearer {settings.dashscope_api_key}"},
                json={
                    "model": settings.rerank_model,
                    "query": "KUN connection test",
                    "documents": ["KUN connection test", "unrelated document"],
                    "top_n": 2,
                },
                timeout=30,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            if not results or "relevance_score" not in results[0]:
                return _failure(provider, "invalid_response", "Rerank 返回格式异常")
            return _success(provider, settings.rerank_model, result_count=len(results))
        raise ValueError("未知模型提供方")
    except httpx.HTTPStatusError as error:
        status = error.response.status_code
        code = "authentication_failed" if status in {401, 403} else f"http_{status}"
        return _failure(provider, code, "认证失败" if code == "authentication_failed" else "服务请求失败")
    except httpx.TimeoutException:
        return _failure(provider, "timeout", "连接服务超时")
    except httpx.HTTPError:
        return _failure(provider, "network_error", "无法连接模型服务")


def generate_document_metadata(filename: str, file_type: str, preview: str) -> dict:
    settings = get_settings()
    fallback_title = Path(filename).stem.replace("_", " ").replace("-", " ")
    fallback_summary = (preview[:180] + "…") if len(preview) > 180 else (preview or f"一份待识别的{file_type.upper()}资料")
    fallback = {
        "title": fallback_title,
        "summary": fallback_summary,
        "tags": [file_type.upper(), "待确认"],
        "metadata_source": "local_heuristic",
    }
    if (
        not settings.deepseek_api_key
        or not preview.strip()
        or not get_privacy_settings()["cloud_document_analysis_enabled"]
    ):
        return fallback
    try:
        response = httpx.post(
            f"{settings.deepseek_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.deepseek_api_key}"},
            json={
                "model": settings.deepseek_model,
                "temperature": 0.1,
                "max_tokens": 500,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "你是个人知识资料整理器。根据文件名和节选生成通俗、准确、不过度推断的元数据。"
                            "只返回 JSON：title（不超过40字）、summary（不超过180字）、tags（2到5个短标签）。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"文件名：{filename}\n文件类型：{file_type}\n内容节选：\n{preview[:6000]}",
                    },
                ],
            },
            timeout=45,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.IGNORECASE).strip()
        data = json.loads(content)
        title = str(data.get("title", "")).strip()[:180] or fallback_title
        summary = str(data.get("summary", "")).strip()[:1200] or fallback_summary
        tags = [str(item).strip()[:30] for item in data.get("tags", []) if str(item).strip()][:5]
        return {
            "title": title,
            "summary": summary,
            "tags": tags or fallback["tags"],
            "metadata_source": "deepseek",
        }
    except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return fallback


def _save_status(provider: str, status: str, error_code: str | None) -> None:
    with connect() as db:
        db.execute(
            """INSERT INTO provider_status(provider,status,last_checked_at,error_code)
               VALUES(?,?,?,?) ON CONFLICT(provider) DO UPDATE SET
               status=excluded.status,last_checked_at=excluded.last_checked_at,error_code=excluded.error_code""",
            (provider, status, now(), error_code),
        )


def _success(provider: str, model: str, **extra: object) -> dict:
    _save_status(provider, "connected", None)
    return {"provider": provider, "status": "connected", "model": model, **extra}


def _failure(provider: str, code: str, message: str) -> dict:
    _save_status(provider, "failed", code)
    return {"provider": provider, "status": "failed", "error_code": code, "message": message}
