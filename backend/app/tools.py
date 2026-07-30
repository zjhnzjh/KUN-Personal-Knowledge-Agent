from __future__ import annotations

import json
import ipaddress
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin, urlparse
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field, ValidationError

from .config import get_settings
from .database import connect, json_value, now, rows
from .images import search_images
from .memory import find_conflict, record_memory_event
from .rag import parse_document, search


class ToolContext(BaseModel):
    read_scopes: set[str] = Field(default_factory=set)
    write_scopes: set[str] = Field(default_factory=set)
    network_scopes: set[str] = Field(default_factory=set)
    confirmed: bool = False
    conversation_id: str | None = None


class ToolDefinition(BaseModel):
    name: str
    description: str
    read_scopes: set[str] = Field(default_factory=set)
    write_scopes: set[str] = Field(default_factory=set)
    network_scopes: set[str] = Field(default_factory=set)
    timeout_seconds: int = 30
    confirmation_required: bool = False
    availability: str = "available"
    unavailable_reason: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)


class RagSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=12_000)
    space_id: str = Field(default="ai-agent-learning", min_length=1, max_length=120)
    top_k: int = Field(default=5, ge=1, le=20)


class FileSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    space_id: str | None = Field(default=None, max_length=120)
    limit: int = Field(default=20, ge=1, le=100)


class ImageSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    space_id: str = Field(default="ai-agent-learning", min_length=1, max_length=120)
    limit: int = Field(default=10, ge=1, le=30)


class LocalPathInput(BaseModel):
    path: str = Field(min_length=1, max_length=2048)


class MemorySearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=50)


class MemoryWriteInput(BaseModel):
    kind: str = Field(pattern=r"^(identity|location|preference|goal|project|relationship)$")
    content: str = Field(min_length=1, max_length=240)
    source: str = Field(default="conversation", max_length=180)


class WebSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=5, ge=1, le=20)


class WebFetchInput(BaseModel):
    url: str = Field(pattern=r"^https://", max_length=2048)


class ToolExecutionError(RuntimeError):
    def __init__(self, code: str, message: str, *, recoverable: bool = True):
        super().__init__(message)
        self.code = code
        self.recoverable = recoverable


class _ReadableHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = 0
        self.title = ""
        self._in_title = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self.hidden += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self.hidden:
            self.hidden -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        clean = " ".join(data.split())
        if not clean:
            return
        if self._in_title:
            self.title = f"{self.title} {clean}".strip()
        if not self.hidden:
            self.parts.append(clean)


@dataclass
class RegisteredTool:
    definition: ToolDefinition
    input_model: type[BaseModel]
    handler: Callable[[BaseModel], Any] | None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        definition: ToolDefinition,
        input_model: type[BaseModel],
        handler: Callable[[BaseModel], Any] | None,
    ) -> None:
        definition.input_schema = input_model.model_json_schema()
        self._tools[definition.name] = RegisteredTool(definition, input_model, handler)

    def definitions(self) -> list[dict]:
        return [item.definition.model_dump(mode="json") for item in self._tools.values()]

    def routing_definitions(self) -> list[dict]:
        items = []
        for registered in self._tools.values():
            definition = registered.definition
            properties = definition.input_schema.get("properties", {})
            required = set(definition.input_schema.get("required", []))
            items.append({
                "name": definition.name,
                "description": definition.description,
                "read_scopes": sorted(definition.read_scopes),
                "write_scopes": sorted(definition.write_scopes),
                "network_scopes": sorted(definition.network_scopes),
                "timeout_seconds": definition.timeout_seconds,
                "confirmation_required": definition.confirmation_required,
                "availability": definition.availability,
                "unavailable_reason": definition.unavailable_reason,
                "parameter_summary": [
                    {"name": name, "type": schema.get("type", "value"), "required": name in required}
                    for name, schema in properties.items()
                ],
            })
        return items

    def schema(self, name: str) -> dict:
        registered = self._tools.get(name)
        if not registered:
            raise ToolExecutionError("tool_not_found", f"未知 Tool：{name}", recoverable=False)
        return {"name": name, "input_schema": registered.definition.input_schema}

    def invoke(self, name: str, arguments: dict[str, Any], context: ToolContext) -> dict:
        registered = self._tools.get(name)
        if not registered:
            raise ToolExecutionError("tool_not_found", f"未知 Tool：{name}", recoverable=False)
        definition = registered.definition
        if definition.availability != "available" or registered.handler is None:
            raise ToolExecutionError(
                "tool_unavailable",
                definition.unavailable_reason or f"{name} 当前不可用",
            )
        if not definition.read_scopes.issubset(context.read_scopes):
            raise ToolExecutionError("read_scope_denied", f"{name} 缺少读取权限", recoverable=False)
        if not definition.write_scopes.issubset(context.write_scopes):
            raise ToolExecutionError("write_scope_denied", f"{name} 缺少写入权限", recoverable=False)
        if not definition.network_scopes.issubset(context.network_scopes):
            raise ToolExecutionError("network_scope_denied", f"{name} 缺少联网权限", recoverable=False)
        if definition.confirmation_required and not context.confirmed:
            raise ToolExecutionError("confirmation_required", f"{name} 需要用户确认")

        run_id = uuid4().hex
        started = time.perf_counter()
        safe_input = _safe_summary(arguments)
        _record_run(run_id, context.conversation_id, name, "running", safe_input)
        try:
            validated = registered.input_model.model_validate(arguments)
            result = registered.handler(validated)
            duration_ms = round((time.perf_counter() - started) * 1000)
            output_summary = _result_summary(result)
            _finish_run(run_id, "succeeded", duration_ms, output_summary)
            return {
                "result": result,
                "trace": {
                    "run_id": run_id,
                    "tool": name,
                    "status": "succeeded",
                    "duration_ms": duration_ms,
                    **output_summary,
                },
            }
        except ValidationError as error:
            duration_ms = round((time.perf_counter() - started) * 1000)
            _finish_run(run_id, "failed", duration_ms, {}, "invalid_input")
            raise ToolExecutionError("invalid_input", str(error), recoverable=False) from error
        except ToolExecutionError as error:
            duration_ms = round((time.perf_counter() - started) * 1000)
            _finish_run(run_id, "failed", duration_ms, {}, error.code)
            raise
        except Exception as error:
            duration_ms = round((time.perf_counter() - started) * 1000)
            _finish_run(run_id, "failed", duration_ms, {}, "tool_failed")
            raise ToolExecutionError("tool_failed", f"{name} 执行失败：{type(error).__name__}") from error


def _safe_summary(value: Any) -> Any:
    if isinstance(value, dict):
        safe = {}
        for key, item in value.items():
            lowered = key.lower()
            if any(word in lowered for word in ("key", "token", "secret", "authorization")):
                safe[key] = "<redacted>"
            elif "path" in lowered and isinstance(item, str):
                safe[key] = Path(item).name
            else:
                safe[key] = _safe_summary(item)
        return safe
    if isinstance(value, list):
        return [_safe_summary(item) for item in value[:20]]
    if isinstance(value, str) and len(value) > 500:
        return value[:500] + "…"
    return value


def _result_summary(result: Any) -> dict:
    if isinstance(result, list):
        return {"result_count": len(result)}
    if isinstance(result, dict):
        count = result.get("result_count")
        if count is None and isinstance(result.get("sections"), list):
            count = len(result["sections"])
        return {"result_count": count} if count is not None else {"result_type": "object"}
    return {"result_type": type(result).__name__}


def _record_run(run_id: str, conversation_id: str | None, name: str, status: str, safe_input: Any) -> None:
    with connect() as db:
        db.execute(
            """INSERT INTO tool_runs(id,conversation_id,tool_name,status,input_summary_json,output_summary_json,
               error_code,duration_ms,created_at,finished_at) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (run_id, conversation_id, name, status, json_value(safe_input), "{}", None, None, now(), None),
        )


def _finish_run(run_id: str, status: str, duration_ms: int, output: dict, error_code: str | None = None) -> None:
    with connect() as db:
        db.execute(
            """UPDATE tool_runs SET status=?,output_summary_json=?,error_code=?,duration_ms=?,finished_at=?
               WHERE id=?""",
            (status, json_value(output), error_code, duration_ms, now(), run_id),
        )


def _allowed_local_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    settings = get_settings()
    roots = [(settings.data_dir / folder).resolve() for folder in ("library", "staging")]
    if not any(path.is_relative_to(root) for root in roots):
        raise ToolExecutionError("path_denied", "Tool 只能读取 KUN 资料库或暂存区中的文件", recoverable=False)
    if not path.is_file():
        raise ToolExecutionError("file_not_found", "目标文件不存在")
    return path


def _rag_search(payload: BaseModel) -> list[dict]:
    value = RagSearchInput.model_validate(payload)
    return search(value.query, value.space_id, value.top_k)


def _file_search(payload: BaseModel) -> list[dict]:
    value = FileSearchInput.model_validate(payload)
    pattern = f"%{value.query}%"
    sql = """SELECT id,space_id,original_name,file_type,title,summary,index_status,updated_at
             FROM documents WHERE (title LIKE ? OR original_name LIKE ? OR summary LIKE ?)"""
    params: list[Any] = [pattern, pattern, pattern]
    if value.space_id:
        sql += " AND space_id=?"
        params.append(value.space_id)
    sql += " ORDER BY updated_at DESC LIMIT ?"
    params.append(value.limit)
    return rows(sql, tuple(params))


def _image_search(payload: BaseModel) -> list[dict]:
    value = ImageSearchInput.model_validate(payload)
    return search_images(value.query, value.space_id, value.limit)


def _document_parse(payload: BaseModel) -> dict:
    value = LocalPathInput.model_validate(payload)
    path = _allowed_local_path(value.path)
    sections = parse_document(path)
    return {
        "file": path.name,
        "sections": [
            {"locator": item.locator, "heading": item.heading, "preview": item.text[:500]}
            for item in sections[:100]
        ],
    }


def _memory_search(payload: BaseModel) -> list[dict]:
    value = MemorySearchInput.model_validate(payload)
    pattern = f"%{value.query}%"
    return rows(
        """SELECT id,kind,content,source,status,use_count,updated_at FROM memories
           WHERE status='enabled' AND content LIKE ? ORDER BY updated_at DESC LIMIT ?""",
        (pattern, value.limit),
    )


def _upsert_memory(payload: BaseModel, status: str) -> dict:
    value = MemoryWriteInput.model_validate(payload)
    stamp = now()
    before = None
    event_type = "proposed" if status == "pending" else "confirmed"
    with connect() as db:
        existing = db.execute(
            """SELECT * FROM memories WHERE kind=? AND content=? AND status IN ('pending','enabled','disabled')
               ORDER BY updated_at DESC LIMIT 1""",
            (value.kind, value.content.strip()),
        ).fetchone()
        if existing:
            before = dict(existing)
            memory_id = existing["id"]
            next_status = "enabled" if status == "enabled" else existing["status"]
            version = int(existing["version"] or 1) + (1 if next_status != existing["status"] else 0)
            conflict_id = find_conflict(value.kind, value.content.strip(), memory_id) if next_status == "enabled" else existing["conflict_with_id"]
            db.execute(
                "UPDATE memories SET source=?,status=?,version=?,conflict_with_id=?,updated_at=? WHERE id=?",
                (value.source, next_status, version, conflict_id, stamp, memory_id),
            )
            event_type = "confirmed" if next_status == "enabled" and existing["status"] != "enabled" else "refreshed"
        else:
            memory_id = uuid4().hex
            next_status = status
            conflict_id = find_conflict(value.kind, value.content.strip()) if next_status == "enabled" else None
            db.execute(
                """INSERT INTO memories(id,kind,content,source,status,use_count,version,conflict_with_id,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (memory_id, value.kind, value.content.strip(), value.source, next_status, 0, 1, conflict_id, stamp, stamp),
            )
        item = dict(db.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone())
    record_memory_event(memory_id, event_type, value.source, before, item)
    return item

def _memory_propose(payload: BaseModel) -> dict:
    return _upsert_memory(payload, "pending")


def _memory_write(payload: BaseModel) -> dict:
    return _upsert_memory(payload, "enabled")


def _video_probe(payload: BaseModel) -> dict:
    value = LocalPathInput.model_validate(payload)
    path = _allowed_local_path(value.path)
    if path.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm", ".m4v"}:
        raise ToolExecutionError("unsupported_video", "不支持的视频格式")
    executable = shutil.which("ffprobe")
    if not executable:
        raise ToolExecutionError("dependency_missing", "尚未安装 FFmpeg/ffprobe")
    completed = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-show_entries",
            "format=duration,size,format_name:stream=index,codec_type,codec_name,width,height",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if completed.returncode != 0:
        raise ToolExecutionError("video_probe_failed", "无法读取视频媒体信息")
    metadata = json.loads(completed.stdout)
    return {"file": path.name, "metadata": metadata}


def _validate_public_https(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ToolExecutionError("invalid_url", "只允许读取公开 HTTPS 网页", recoverable=False)
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        raise ToolExecutionError("dns_failed", "网页域名无法解析") from error
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ToolExecutionError("private_address_denied", "禁止访问本机、局域网或保留地址", recoverable=False)


def _web_fetch(payload: BaseModel) -> dict:
    value = WebFetchInput.model_validate(payload)
    current = value.url
    response = None
    with httpx.Client(
        timeout=httpx.Timeout(20, connect=8),
        headers={"User-Agent": "KUN-Personal-Knowledge-Agent/0.1 (+private-study)"},
        follow_redirects=False,
    ) as client:
        for _ in range(4):
            _validate_public_https(current)
            response = client.get(current)
            if response.status_code in {301, 302, 303, 307, 308}:
                target = response.headers.get("location")
                if not target:
                    raise ToolExecutionError("redirect_failed", "网页重定向缺少目标地址")
                current = urljoin(current, target)
                continue
            response.raise_for_status()
            break
        else:
            raise ToolExecutionError("too_many_redirects", "网页重定向次数过多")
    if response is None:
        raise ToolExecutionError("fetch_failed", "网页读取失败")
    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type and "text/plain" not in content_type:
        raise ToolExecutionError("unsupported_content", "当前只读取 HTML 或纯文本网页")
    if len(response.content) > 2 * 1024 * 1024:
        raise ToolExecutionError("page_too_large", "网页正文超过 2 MB 限制")
    parser = _ReadableHTML()
    parser.feed(response.text)
    text = "\n".join(parser.parts)
    text = "\n".join(dict.fromkeys(line for line in text.splitlines() if line))[:30_000]
    return {
        "url": str(response.url),
        "title": parser.title[:300] or urlparse(str(response.url)).hostname,
        "text": text,
        "content_type": content_type.split(";")[0],
        "accessed_at": now(),
        "result_count": 1,
    }


def _web_search(payload: BaseModel) -> dict:
    value = WebSearchInput.model_validate(payload)
    settings = get_settings()
    if not settings.dashscope_api_key:
        raise ToolExecutionError("provider_not_configured", "尚未配置阿里云百炼 API Key")
    endpoint = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    response = httpx.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {settings.dashscope_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "qwen-plus",
            "input": {"messages": [{"role": "user", "content": value.query}]},
            "parameters": {
                "result_format": "message",
                "enable_search": True,
                "search_options": {
                    "forced_search": True,
                    "enable_source": True,
                    "enable_citation": True,
                    "citation_format": "[<number>]",
                },
            },
        },
        timeout=45,
    )
    if response.status_code >= 400:
        try:
            detail = response.json().get("message") or response.text
        except ValueError:
            detail = response.text
        raise ToolExecutionError("web_search_failed", f"百炼联网搜索失败：{str(detail)[:240]}")
    data = response.json()
    output = data.get("output") or {}
    search_info = output.get("search_info") or {}
    raw_results = search_info.get("search_results") or []
    results = []
    for index, item in enumerate(raw_results[: value.limit], 1):
        url = str(item.get("url") or "").strip()
        if not url.startswith(("https://", "http://")):
            continue
        results.append({
            "index": int(item.get("index") or index),
            "title": str(item.get("title") or urlparse(url).hostname or "网页")[:300],
            "url": url,
            "site_name": str(item.get("site_name") or urlparse(url).hostname or "网页")[:120],
            "snippet": str(item.get("snippet") or item.get("content") or "")[:1500],
            "published_at": item.get("publish_time") or item.get("published_at"),
        })
    choices = output.get("choices") or []
    answer = ""
    if choices:
        answer = str((choices[0].get("message") or {}).get("content") or "")
    return {
        "query": value.query,
        "answer": answer,
        "results": results,
        "result_count": len(results),
        "provider": "aliyun_bailian",
        "accessed_at": now(),
    }


REGISTRY = ToolRegistry()


def _register_tools() -> None:
    ffprobe_ready = bool(shutil.which("ffprobe"))
    web_search_ready = bool(get_settings().dashscope_api_key)
    definitions: list[tuple[ToolDefinition, type[BaseModel], Callable[[BaseModel], Any] | None]] = [
        (
            ToolDefinition(
                name="rag.search",
                description="在指定知识空间中执行 BM25 与可用的向量融合检索。",
                read_scopes={"kun_index"},
                timeout_seconds=45,
            ),
            RagSearchInput,
            _rag_search,
        ),
        (
            ToolDefinition(
                name="file.search",
                description="按文件名、标题和摘要搜索已确认的本地资料。",
                read_scopes={"kun_library"},
            ),
            FileSearchInput,
            _file_search,
        ),
        (
            ToolDefinition(
                name="image.search",
                description="按画面描述、OCR 文字和语义向量搜索当前知识空间中的本地图片。",
                read_scopes={"kun_images"},
                timeout_seconds=45,
            ),
            ImageSearchInput,
            _image_search,
        ),
        (
            ToolDefinition(
                name="document.parse",
                description="解析 KUN 资料库或暂存区中的受支持文档。",
                read_scopes={"kun_library"},
                timeout_seconds=120,
            ),
            LocalPathInput,
            _document_parse,
        ),
        (
            ToolDefinition(
                name="memory.search",
                description="搜索用户可见且处于启用状态的长期记忆。",
                read_scopes={"memory"},
            ),
            MemorySearchInput,
            _memory_search,
        ),
        (
            ToolDefinition(
                name="memory.propose",
                description="把对话中识别到的稳定信息保存为待确认候选，不参与跨对话回答。",
                write_scopes={"memory"},
            ),
            MemoryWriteInput,
            _memory_propose,
        ),
        (
            ToolDefinition(
                name="memory.write",
                description="将用户明确要求记住的信息写入可见、可编辑的长期记忆。",
                write_scopes={"memory"},
                confirmation_required=True,
            ),
            MemoryWriteInput,
            _memory_write,
        ),
        (
            ToolDefinition(
                name="video.probe",
                description="读取本地视频的时长、编码和画面尺寸，不上传视频。",
                read_scopes={"kun_library"},
                timeout_seconds=20,
                availability="available" if ffprobe_ready else "unavailable",
                unavailable_reason=None if ffprobe_ready else "需要安装 FFmpeg/ffprobe",
            ),
            LocalPathInput,
            _video_probe if ffprobe_ready else None,
        ),
        (
            ToolDefinition(
                name="video.transcribe",
                description="将已确认的本地视频转写为带时间戳文本。",
                read_scopes={"kun_library"},
                write_scopes={"kun_derived"},
                timeout_seconds=1800,
                confirmation_required=True,
                availability="unavailable",
                unavailable_reason="本地语音转写引擎尚未接入",
            ),
            LocalPathInput,
            None,
        ),
        (
            ToolDefinition(
                name="video.sample_frames",
                description="按时间轴抽取关键帧，供 OCR 和画面理解使用。",
                read_scopes={"kun_library"},
                write_scopes={"kun_derived"},
                timeout_seconds=300,
                confirmation_required=True,
                availability="unavailable",
                unavailable_reason="关键帧任务队列尚未接入",
            ),
            LocalPathInput,
            None,
        ),
        (
            ToolDefinition(
                name="web.search",
                description="搜索公开互联网并返回来源元数据。",
                network_scopes={"public_web"},
                timeout_seconds=45,
                availability="available" if web_search_ready else "unavailable",
                unavailable_reason=None if web_search_ready else "需要先配置阿里云百炼 API Key",
            ),
            WebSearchInput,
            _web_search if web_search_ready else None,
        ),
        (
            ToolDefinition(
                name="web.fetch",
                description="读取用户明确提供或搜索结果中的许可网页正文。",
                network_scopes={"public_web"},
                timeout_seconds=25,
            ),
            WebFetchInput,
            _web_fetch,
        ),
    ]
    for definition, input_model, handler in definitions:
        REGISTRY.register(definition, input_model, handler)


_register_tools()


def invoke_tool(name: str, arguments: dict[str, Any], context: ToolContext) -> dict:
    return REGISTRY.invoke(name, arguments, context)


def tool_runs(limit: int = 50) -> list[dict]:
    return rows(
        """SELECT id,conversation_id,tool_name,status,input_summary_json,output_summary_json,
           error_code,duration_ms,created_at,finished_at FROM tool_runs
           ORDER BY created_at DESC LIMIT ?""",
        (max(1, min(limit, 200)),),
    )
