from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import zipfile
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .context import build_context
from .database import connect, db_path, init_database, json_value, now, rows
from .evaluation_dataset import HELLO_AGENTS_DOCUMENT, HELLO_AGENTS_EVALUATION_CASES
from .agent_evaluation import GATE_TARGET, REFUSAL_CASES, ROUTE_CASES
from .images import IMAGE_TYPES, index_image_document, search_images, understand_image
from .learning import agent_traces, learning_overview
from .memory import find_conflict, memory_detail, record_memory_event
from .providers import generate_document_metadata, provider_statuses, save_provider_key, test_provider
from .privacy import get_privacy_settings, save_privacy_settings
from .rag import DashScopeEmbeddings, ParsedSection, SUPPORTED, chunk_sections, fingerprint, lexical_text, parse_document, search
from .tools import REGISTRY, ToolContext, tool_runs
from .workflow import build_plan, detect_memory_candidate, generate, run, understand


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_database()
    settings = get_settings()
    for folder in ("staging", "library", "indexes", "exports"):
        (settings.data_dir / folder).mkdir(parents=True, exist_ok=True)
    with connect() as db:
        interrupted = db.execute(
            """SELECT j.id,j.document_id,d.library_path FROM indexing_jobs j
               JOIN documents d ON d.id=j.document_id WHERE j.status IN ('queued','running')"""
        ).fetchall()
        for job in interrupted:
            db.execute(
                "UPDATE indexing_jobs SET status='queued',phase='queued',message='服务恢复，索引任务已重新排队',updated_at=? WHERE id=?",
                (now(), job["id"]),
            )
    for job in interrupted:
        INDEX_EXECUTOR.submit(_run_index_job, job["id"], job["document_id"], job["library_path"])
    with connect() as db:
        pending_images = db.execute(
            """SELECT d.id FROM documents d LEFT JOIN image_assets i ON i.document_id=d.id
               WHERE d.file_type IN ('png','jpg','jpeg') AND (i.document_id IS NULL OR i.status='failed')"""
        ).fetchall()
    for image in pending_images:
        INDEX_EXECUTOR.submit(index_image_document, image["id"])
    yield


app = FastAPI(title="KUN Local API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:3000", "http://localhost:3000", "tauri://localhost"], allow_methods=["*"], allow_headers=["*"])
INDEX_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="kun-index")
EMBEDDING_WORKERS = 3


class ConfirmDocument(BaseModel):
    title: str = Field(min_length=1, max_length=180)
    summary: str = Field(max_length=1200)
    tags: list[str] = Field(default_factory=list, max_length=20)
    space_id: str = "ai-agent-learning"


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=12_000)
    space_id: str = "ai-agent-learning"
    conversation_id: str | None = None


class ChatRegenerateRequest(BaseModel):
    conversation_id: str
    assistant_message_id: str


class SpaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    color: str = Field(default="#6d5ce7", pattern=r"^#[0-9a-fA-F]{6}$")


class SpaceUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=60)


class ConversationUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=80)


class MemoryCreate(BaseModel):
    content: str = Field(min_length=2, max_length=500)
    kind: str = Field(default="preference", max_length=40)


class MemoryUpdate(BaseModel):
    content: str | None = Field(default=None, min_length=2, max_length=500)
    status: str | None = None


class ProviderKeyUpdate(BaseModel):
    api_key: str = Field(min_length=16, max_length=512)


class PrivacySettingsUpdate(BaseModel):
    web_search_enabled: bool | None = None
    cloud_document_analysis_enabled: bool | None = None
    cloud_image_analysis_enabled: bool | None = None
    memory_suggestions_enabled: bool | None = None


class BackupRestoreRequest(BaseModel):
    backup_id: str = Field(min_length=10, max_length=180)
    confirmation: str


class EvaluationCaseCreate(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    expected_document_id: str
    expected_locator: str = Field(default="", max_length=200)
    space_id: str = "ai-agent-learning"


class EvaluationRunRequest(BaseModel):
    space_id: str = "ai-agent-learning"
    top_k: int = Field(default=5, ge=1, le=10)
    limit: int = Field(default=2, ge=1, le=100)


class SkillCreate(BaseModel):
    name: str = Field(min_length=2, max_length=63, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    description: str = Field(min_length=10, max_length=600)
    instructions: str = Field(min_length=10, max_length=8000)
    tools: list[str] = Field(default_factory=list, max_length=20)


class WebFetchRequest(BaseModel):
    url: str = Field(pattern=r"^https://", max_length=2048)


@app.get("/api/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "storage": "local",
        "chat_model": settings.deepseek_model,
        "embedding_model": settings.embedding_model,
        "chat_ready": bool(settings.deepseek_api_key),
        "embedding_ready": bool(settings.dashscope_api_key),
    }


@app.get("/api/settings/providers")
def get_provider_statuses() -> list[dict]:
    return provider_statuses()


@app.put("/api/settings/providers/{provider}")
def update_provider_key(provider: str, payload: ProviderKeyUpdate) -> dict:
    if provider not in {"deepseek", "dashscope"}:
        raise HTTPException(404, "未知模型提供方")
    save_provider_key(provider, payload.api_key)
    return {"provider": provider, "configured": True, "connection_status": "not_tested"}


@app.post("/api/settings/providers/{provider}/test")
def connect_provider(provider: str) -> dict:
    if provider not in {"deepseek", "dashscope"}:
        raise HTTPException(404, "未知模型提供方")
    result = test_provider(provider)
    if result["status"] != "connected":
        raise HTTPException(400, detail=result)
    return result


def _folder_size(folder: Path) -> int:
    if not folder.exists():
        return 0
    total = 0
    for path in folder.rglob("*"):
        if path.is_file() and not path.is_symlink():
            try:
                total += path.stat().st_size
            except OSError:
                continue
    return total


def _backup_folder() -> Path:
    folder = get_settings().data_dir / "backups"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _backup_path(backup_id: str) -> Path:
    if Path(backup_id).name != backup_id or not backup_id.endswith(".zip"):
        raise HTTPException(400, "备份标识不合法")
    path = (_backup_folder() / backup_id).resolve()
    if path.parent != _backup_folder().resolve():
        raise HTTPException(400, "备份路径不合法")
    return path


def _create_backup(reason: str = "manual") -> dict:
    settings = get_settings()
    backup_dir = _backup_folder()
    stamp = now()
    safe_stamp = re.sub(r"[^0-9]", "", stamp)[:14]
    backup_id = f"KUN-backup-{safe_stamp}-{uuid4().hex[:8]}.zip"
    destination = backup_dir / backup_id
    snapshot = backup_dir / f".snapshot-{uuid4().hex}.sqlite3"
    source_db = sqlite3.connect(db_path())
    snapshot_db = sqlite3.connect(snapshot)
    try:
        source_db.backup(snapshot_db)
    finally:
        snapshot_db.close()
        source_db.close()
    manifest = {
        "format": "kun-local-backup",
        "format_version": 1,
        "created_at": stamp,
        "reason": reason,
        "contains": ["database", "library", "indexes", "skills", "exports", "staging"],
    }
    try:
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            archive.write(snapshot, "kun.sqlite3")
            for folder_name in ("library", "indexes", "skills", "exports", "staging"):
                folder = settings.data_dir / folder_name
                if not folder.exists():
                    continue
                for path in folder.rglob("*"):
                    if path.is_file() and not path.is_symlink():
                        archive.write(path, path.relative_to(settings.data_dir).as_posix())
    finally:
        snapshot.unlink(missing_ok=True)
    return {
        "id": backup_id,
        "created_at": stamp,
        "reason": reason,
        "size_bytes": destination.stat().st_size,
        "status": "ready",
    }


def _backup_items() -> list[dict]:
    items: list[dict] = []
    for path in sorted(_backup_folder().glob("KUN-backup-*.zip"), key=lambda item: item.stat().st_mtime, reverse=True):
        created_at = ""
        reason = "manual"
        try:
            with zipfile.ZipFile(path) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                created_at = str(manifest.get("created_at", ""))
                reason = str(manifest.get("reason", "manual"))
        except (OSError, KeyError, zipfile.BadZipFile, json.JSONDecodeError):
            reason = "invalid"
        items.append({
            "id": path.name,
            "created_at": created_at,
            "reason": reason,
            "size_bytes": path.stat().st_size,
            "status": "invalid" if reason == "invalid" else "ready",
        })
    return items


@app.get("/api/settings/storage")
def storage_status() -> dict:
    settings = get_settings()
    folders = {
        name: _folder_size(settings.data_dir / name)
        for name in ("library", "indexes", "staging", "exports", "skills", "backups")
    }
    database_size = db_path().stat().st_size if db_path().exists() else 0
    counts = {
        "documents": rows("SELECT COUNT(*) count FROM documents")[0]["count"],
        "chunks": rows("SELECT COUNT(*) count FROM chunks")[0]["count"],
        "conversations": rows("SELECT COUNT(*) count FROM conversations")[0]["count"],
        "memories": rows("SELECT COUNT(*) count FROM memories WHERE status!='dismissed'")[0]["count"],
    }
    return {
        "data_dir": str(settings.data_dir),
        "database_bytes": database_size,
        "folders": folders,
        "total_bytes": database_size + sum(folders.values()),
        "counts": counts,
        "backups": _backup_items(),
    }


@app.post("/api/settings/storage/open")
def open_storage_folder() -> dict:
    folder = get_settings().data_dir.resolve()
    if os.name != "nt":
        raise HTTPException(501, "当前仅支持在 Windows 中打开资料位置")
    os.startfile(str(folder))  # type: ignore[attr-defined]
    return {"opened": True, "path": str(folder)}


@app.post("/api/settings/backups")
def create_backup() -> dict:
    return _create_backup()


@app.get("/api/settings/backups/{backup_id}")
def download_backup(backup_id: str) -> FileResponse:
    path = _backup_path(backup_id)
    if not path.is_file():
        raise HTTPException(404, "备份不存在")
    return FileResponse(path, media_type="application/zip", filename=path.name)


@app.post("/api/settings/backups/restore")
def restore_backup(payload: BackupRestoreRequest) -> dict:
    if payload.confirmation != "恢复此备份":
        raise HTTPException(400, "需要明确确认后才能恢复备份")
    archive_path = _backup_path(payload.backup_id)
    if not archive_path.is_file():
        raise HTTPException(404, "备份不存在")
    settings = get_settings()
    restore_root = settings.data_dir / f".restore-{uuid4().hex}"
    restore_root.mkdir(parents=False)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            for member in members:
                target = (restore_root / member.filename).resolve()
                if restore_root.resolve() not in target.parents and target != restore_root.resolve():
                    raise HTTPException(400, "备份中包含不安全路径")
            archive.extractall(restore_root)
        manifest_path = restore_root / "manifest.json"
        restored_db = restore_root / "kun.sqlite3"
        if not manifest_path.is_file() or not restored_db.is_file():
            raise HTTPException(400, "不是有效的 KUN 备份")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "kun-local-backup" or manifest.get("format_version") != 1:
            raise HTTPException(400, "备份格式或版本不受支持")
        safety_backup = _create_backup(reason="pre-restore")
        for folder_name in ("library", "indexes", "skills", "exports", "staging"):
            source = restore_root / folder_name
            target = (settings.data_dir / folder_name).resolve()
            if target.parent != settings.data_dir.resolve():
                raise HTTPException(400, "恢复目标不安全")
            if target.exists():
                shutil.rmtree(target)
            if source.exists():
                shutil.copytree(source, target)
            else:
                target.mkdir(parents=True)
        for sidecar in (Path(f"{db_path()}-wal"), Path(f"{db_path()}-shm")):
            sidecar.unlink(missing_ok=True)
        os.replace(restored_db, db_path())
        init_database()
        return {"restored": True, "backup_id": payload.backup_id, "safety_backup": safety_backup}
    finally:
        shutil.rmtree(restore_root, ignore_errors=True)


@app.get("/api/settings/privacy")
def privacy_status() -> dict:
    return {
        **get_privacy_settings(),
        "fixed_boundaries": {
            "local_api": "仅监听 127.0.0.1",
            "credentials": "Windows Credential Manager",
            "original_files": "从不覆盖",
            "sensitive_memory": "不自动保存密码、验证码、银行卡、手机号或精确住址",
        },
    }


@app.put("/api/settings/privacy")
def update_privacy(payload: PrivacySettingsUpdate) -> dict:
    patch = {key: value for key, value in payload.model_dump().items() if value is not None}
    return {
        **save_privacy_settings(patch),
        "fixed_boundaries": {
            "local_api": "仅监听 127.0.0.1",
            "credentials": "Windows Credential Manager",
            "original_files": "从不覆盖",
            "sensitive_memory": "不自动保存密码、验证码、银行卡、手机号或精确住址",
        },
    }


@app.get("/api/tools")
def list_tools() -> list[dict]:
    return REGISTRY.definitions()


@app.get("/api/tools/catalog")
def tool_catalog() -> list[dict]:
    return REGISTRY.routing_definitions()


@app.get("/api/tools/{tool_name}/schema")
def tool_schema(tool_name: str) -> dict:
    try:
        return REGISTRY.schema(tool_name)
    except Exception as error:
        raise HTTPException(404, str(error)) from error


@app.get("/api/learning/overview")
def get_learning_overview() -> dict:
    return learning_overview(list_skills(), REGISTRY.definitions())


@app.get("/api/agent/traces")
def get_agent_traces(limit: int = 20) -> list[dict]:
    return agent_traces(limit)


@app.get("/api/tools/runs")
def list_tool_runs(limit: int = 50) -> list[dict]:
    return [
        {
            **item,
            "input_summary": json.loads(item.pop("input_summary_json") or "{}"),
            "output_summary": json.loads(item.pop("output_summary_json") or "{}"),
        }
        for item in tool_runs(limit)
    ]


@app.get("/api/agent/route")
def preview_agent_route(q: str) -> dict:
    question = q.strip()
    if not question:
        raise HTTPException(400, "请输入一个问题")
    route = understand({"question": question, "space_id": "preview"})
    plan = build_plan({"question": question, "space_id": "preview", **route})["plan"]
    memory_candidate = detect_memory_candidate(question)
    skill_tools = {
        "document_skill": ["rag.search", "document.parse"],
        "image_skill": ["image.search"],
        "excel_skill": ["document.parse", "rag.search"],
        "video_skill": ["video.probe", "video.transcribe", "video.sample_frames"],
        "memory_skill": [
            "memory.write" if memory_candidate and memory_candidate.get("status") == "enabled"
            else "memory.propose" if memory_candidate
            else "memory.search"
        ],
        "web_research_skill": ["web.search", "web.fetch"], 
        "recommendation_skill": ["web.search", "web.fetch"],
    }
    return {
        **route,
        "question": question,
        "tools": skill_tools.get(route["skill"], []),
        "memory_candidate": memory_candidate,
        "plan": plan,
    }


@app.post("/api/web/fetch")
def fetch_webpage(payload: WebFetchRequest) -> dict:
    execution = REGISTRY.invoke(
        "web.fetch",
        {"url": payload.url},
        ToolContext(network_scopes={"public_web"}),
    )
    return {**execution["result"], "trace": execution["trace"]}


@app.get("/api/spaces")
def list_spaces() -> list[dict]:
    return rows("""SELECT s.*,COUNT(DISTINCT d.id) document_count,COUNT(c.id) chunk_count
                   FROM spaces s LEFT JOIN documents d ON d.space_id=s.id
                   LEFT JOIN chunks c ON c.document_id=d.id GROUP BY s.id ORDER BY s.created_at""")


@app.post("/api/spaces", status_code=201)
def create_space(payload: SpaceCreate) -> dict:
    space_id = f"space-{uuid4().hex[:12]}"
    stamp = now()
    try:
        with connect() as db:
            db.execute(
                "INSERT INTO spaces(id,name,color,created_at,updated_at) VALUES(?,?,?,?,?)",
                (space_id, payload.name.strip(), payload.color, stamp, stamp),
            )
    except Exception as error:
        if "UNIQUE" in str(error):
            raise HTTPException(409, "知识空间名称已经存在") from error
        raise
    return {"id": space_id, "name": payload.name.strip(), "color": payload.color, "document_count": 0, "chunk_count": 0}


@app.patch("/api/spaces/{space_id}")
def update_space(space_id: str, payload: SpaceUpdate) -> dict:
    with connect() as db:
        found = db.execute("SELECT id FROM spaces WHERE id=?", (space_id,)).fetchone()
        if not found:
            raise HTTPException(404, "知识空间不存在")
        db.execute("UPDATE spaces SET name=?,updated_at=? WHERE id=?", (payload.name.strip(), now(), space_id))
    return {"id": space_id, "name": payload.name.strip()}


@app.get("/api/documents")
def list_documents(space_id: str | None = None) -> list[dict]:
    sql = "SELECT * FROM documents"
    params: tuple = ()
    if space_id:
        sql, params = sql + " WHERE space_id=?", (space_id,)
    documents = rows(sql + " ORDER BY updated_at DESC", params)
    with connect() as db:
        for document in documents:
            stats = db.execute(
                """SELECT COUNT(*) chunk_count,
                   SUM(CASE WHEN embedding_json IS NOT NULL THEN 1 ELSE 0 END) embedding_count
                   FROM chunks WHERE document_id=?""",
                (document["id"],),
            ).fetchone()
            document["chunk_count"] = stats["chunk_count"] or 0
            document["embedding_count"] = stats["embedding_count"] or 0
            document["library_copy_exists"] = Path(document["library_path"]).is_file()
            if not document["library_copy_exists"]:
                document["effective_index_status"] = "missing"
            else:
                document["effective_index_status"] = document["index_status"]
            document.pop("library_path", None)
    return documents


@app.get("/api/documents/{document_id}/status")
def document_status(document_id: str) -> dict:
    with connect() as db:
        document = db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        if not document:
            raise HTTPException(404, "资料不存在")
        stats = db.execute(
            """SELECT COUNT(*) chunk_count,
               SUM(CASE WHEN embedding_json IS NOT NULL THEN 1 ELSE 0 END) embedding_count,
               MAX(embedding_model) embedding_model FROM chunks WHERE document_id=?""",
            (document_id,),
        ).fetchone()
        job = db.execute(
            "SELECT * FROM indexing_jobs WHERE document_id=? ORDER BY created_at DESC LIMIT 1",
            (document_id,),
        ).fetchone()
    exists = Path(document["library_path"]).is_file()
    return {
        "id": document_id,
        "title": document["title"],
        "original_name": document["original_name"],
        "status": document["index_status"] if exists else "missing",
        "library_copy_exists": exists,
        "library_path": document["library_path"],
        "chunk_count": stats["chunk_count"] or 0,
        "embedding_count": stats["embedding_count"] or 0,
        "embedding_model": stats["embedding_model"],
        "updated_at": document["updated_at"],
        "latest_job": dict(job) if job else None,
    }


@app.get("/api/files/{document_id}")
def local_file(document_id: str) -> FileResponse:
    matches = rows("SELECT library_path,original_name,file_type FROM documents WHERE id=?", (document_id,))
    if not matches:
        raise HTTPException(404, "资料不存在")
    path = Path(matches[0]["library_path"])
    if not path.is_file():
        raise HTTPException(404, "KUN 本地副本缺失")
    media = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "pdf": "application/pdf"}.get(matches[0]["file_type"])
    return FileResponse(path, media_type=media, filename=matches[0]["original_name"])


@app.post("/api/documents/stage")
async def stage_documents(files: list[UploadFile] = File(...)) -> list[dict]:
    if len(files) > 50:
        raise HTTPException(400, "单次最多上传 50 个文件")
    settings = get_settings()
    results = []
    for upload in files:
        name = Path(upload.filename or "unnamed").name
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED:
            raise HTTPException(400, f"不支持的文件类型：{suffix}")
        document_id = uuid4().hex
        target = settings.data_dir / "staging" / f"{document_id}{suffix}"
        total = 0
        with target.open("wb") as output:
            while block := await upload.read(1024 * 1024):
                total += len(block)
                if total > 500 * 1024 * 1024:
                    output.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(413, f"{name} 超过 500 MB")
                output.write(block)
        vision_analysis = None
        if suffix in IMAGE_TYPES:
            try:
                vision_data, width, height, vision_model = understand_image(target)
                preview = "\n".join(
                    part for part in (
                        vision_data["description"],
                        vision_data["ocr_text"],
                        " ".join(vision_data["tags"]),
                    ) if part
                )
                metadata = generate_document_metadata(name, suffix.lstrip("."), preview)
                metadata["tags"] = list(dict.fromkeys([*vision_data["tags"], *metadata["tags"]]))[:8]
                metadata["metadata_source"] = "qwen_vision+deepseek"
                vision_analysis = (width, height, vision_data, vision_model)
                sections = [ParsedSection("整张图片", "图片理解", preview)]
            except Exception:
                sections = parse_document(target)
                preview = re.sub(r"\s+", " ", " ".join(section.text for section in sections)).strip()
                metadata = generate_document_metadata(name, suffix.lstrip("."), preview)
        else:
            sections = parse_document(target)
            preview = re.sub(r"\s+", " ", " ".join(section.text for section in sections)).strip()
            metadata = generate_document_metadata(name, suffix.lstrip("."), preview)
        title = metadata["title"]
        summary = metadata["summary"]
        item = {
            "id": document_id,
            "original_name": name,
            "file_type": suffix.lstrip("."),
            "size_bytes": total,
            "fingerprint": fingerprint(target),
            "title": title,
            "summary": summary,
            "tags": metadata["tags"],
            "metadata_source": metadata["metadata_source"],
            "parse_status": "awaiting_confirmation",
            "sections": len(sections),
        }
        with connect() as db:
            db.execute("""INSERT INTO staged_documents(id,original_name,staged_path,file_type,size_bytes,fingerprint,title,summary,tags_json,parse_status,created_at)
                          VALUES(?,?,?,?,?,?,?,?,?,?,?)""", (document_id, name, str(target), item["file_type"], total, item["fingerprint"], title, summary, json_value(item["tags"]), item["parse_status"], now()))
            if vision_analysis:
                width, height, vision_data, vision_model = vision_analysis
                db.execute(
                    """INSERT OR REPLACE INTO staged_image_analysis(
                       document_id,width,height,description,ocr_text,tags_json,vision_model,created_at
                       ) VALUES(?,?,?,?,?,?,?,?)""",
                    (document_id, width, height, vision_data["description"], vision_data["ocr_text"],
                     json_value(vision_data["tags"]), vision_model, now()),
                )
        results.append(item)
    return results


def _update_index_job(job_id: str, *, status: str | None = None, phase: str | None = None,
                      progress: int | None = None, total: int | None = None,
                      completed: int | None = None, message: str | None = None,
                      error_message: str | None = None) -> None:
    fields: list[str] = ["updated_at=?"]
    values: list[object] = [now()]
    for column, value in (
        ("status", status), ("phase", phase), ("progress", progress), ("total", total),
        ("completed", completed), ("message", message), ("error_message", error_message),
    ):
        if value is not None:
            fields.append(f"{column}=?")
            values.append(value)
    values.append(job_id)
    with connect() as db:
        db.execute(f"UPDATE indexing_jobs SET {','.join(fields)} WHERE id=?", tuple(values))


def _embed_batch(texts: list[str]) -> list[list[float]]:
    return DashScopeEmbeddings().encode(texts)


def _run_index_job(job_id: str, document_id: str, library_path: str) -> None:
    settings = get_settings()
    try:
        _update_index_job(job_id, status="running", phase="parsing", progress=2, message="正在解析文档结构")
        if Path(library_path).suffix.lower() in IMAGE_TYPES:
            _update_index_job(job_id, phase="vision", progress=10, total=1, message="正在识别图片内容和文字")
            index_image_document(document_id)
            _update_index_job(
                job_id, status="completed", phase="ready", progress=100, total=1, completed=1,
                message="图片语义索引已完成，可以自然语言搜索",
            )
            return
        with connect() as db:
            old_ids = [row["id"] for row in db.execute("SELECT id FROM chunks WHERE document_id=?", (document_id,))]
            for start in range(0, len(old_ids), 400):
                group = old_ids[start:start + 400]
                placeholders = ",".join("?" for _ in group)
                db.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})", tuple(group))
            db.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
        def parsing_progress(current: int, page_total: int) -> None:
            _update_index_job(
                job_id,
                progress=min(5, 2 + int(current / max(page_total, 1) * 3)),
                message=f"正在解析第 {current} / {page_total} 页",
            )

        chunks = chunk_sections(parse_document(Path(library_path), parsing_progress))
        total = len(chunks)
        stamp = now()
        _update_index_job(job_id, phase="lexical", progress=5, total=total, message=f"正在建立本地关键词索引，共 {total} 个 Chunk")

        chunk_rows: list[tuple[str, ParsedSection, str]] = []
        for ordinal, chunk in enumerate(chunks):
            chunk_id = hashlib.sha1(f"{document_id}:{ordinal}:{chunk.text}".encode()).hexdigest()[:24]
            content_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
            chunk_rows.append((chunk_id, chunk, content_hash))
        with connect() as db:
            for ordinal, (chunk_id, chunk, _) in enumerate(chunk_rows):
                db.execute(
                    "INSERT INTO chunks(id,document_id,ordinal,locator,heading,text,embedding_json,embedding_model,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (chunk_id, document_id, ordinal, chunk.locator, chunk.heading, chunk.text, None, None, stamp),
                )
                db.execute("INSERT INTO chunks_fts(chunk_id,text) VALUES(?,?)", (chunk_id, lexical_text(chunk.text)))
            db.execute("UPDATE documents SET index_status=?,updated_at=? WHERE id=?", ("lexical_ready", now(), document_id))

        provider = DashScopeEmbeddings()
        if not provider.available or not chunk_rows:
            with connect() as db:
                db.execute("UPDATE documents SET index_status=?,updated_at=? WHERE id=?", ("ready", now(), document_id))
            _update_index_job(job_id, status="completed", phase="ready", progress=100, completed=total, message="本地 BM25 索引已完成")
            return

        cached: dict[str, str] = {}
        hashes = [row[2] for row in chunk_rows]
        with connect() as db:
            for start in range(0, len(hashes), 400):
                group = hashes[start:start + 400]
                placeholders = ",".join("?" for _ in group)
                for row in db.execute(
                    f"SELECT content_hash,embedding_json FROM embedding_cache WHERE model=? AND content_hash IN ({placeholders})",
                    (settings.embedding_model, *group),
                ):
                    cached[row["content_hash"]] = row["embedding_json"]

        uncached = [(chunk_id, chunk, content_hash) for chunk_id, chunk, content_hash in chunk_rows if content_hash not in cached]
        if cached:
            with connect() as db:
                for chunk_id, _, content_hash in chunk_rows:
                    if content_hash in cached:
                        db.execute(
                            "UPDATE chunks SET embedding_json=?,embedding_model=? WHERE id=?",
                            (cached[content_hash], settings.embedding_model, chunk_id),
                        )

        completed = total - len(uncached)
        batch_size = 20 if settings.embedding_model == "qwen3.7-text-embedding" else 10
        batches = [uncached[start:start + batch_size] for start in range(0, len(uncached), batch_size)]
        _update_index_job(
            job_id, phase="embedding", progress=max(8, int(completed / max(total, 1) * 100)),
            completed=completed, message=f"正在生成语义向量：{completed} / {total}",
        )
        with ThreadPoolExecutor(max_workers=EMBEDDING_WORKERS, thread_name_prefix="kun-embedding") as pool:
            future_batches = {pool.submit(_embed_batch, [item[1].text for item in batch]): batch for batch in batches}
            for future in as_completed(future_batches):
                batch = future_batches[future]
                vectors = future.result()
                with connect() as db:
                    for (chunk_id, _, content_hash), vector in zip(batch, vectors):
                        encoded = json_value(vector)
                        db.execute(
                            "UPDATE chunks SET embedding_json=?,embedding_model=? WHERE id=?",
                            (encoded, settings.embedding_model, chunk_id),
                        )
                        db.execute(
                            "INSERT OR REPLACE INTO embedding_cache(content_hash,model,embedding_json,created_at) VALUES(?,?,?,?)",
                            (content_hash, settings.embedding_model, encoded, now()),
                        )
                completed += len(batch)
                _update_index_job(
                    job_id, progress=min(99, max(8, int(completed / max(total, 1) * 100))),
                    completed=completed, message=f"正在生成语义向量：{completed} / {total}",
                )

        with connect() as db:
            db.execute("UPDATE documents SET index_status=?,updated_at=? WHERE id=?", ("ready", now(), document_id))
        _update_index_job(job_id, status="completed", phase="ready", progress=100, completed=total, message="文档索引已完成，可以开始提问")
    except Exception as error:
        with connect() as db:
            db.execute("UPDATE documents SET index_status=?,updated_at=? WHERE id=?", ("failed", now(), document_id))
        _update_index_job(job_id, status="failed", phase="failed", message="建立索引失败", error_message=str(error)[:500])


@app.post("/api/documents/{document_id}/confirm", status_code=202)
def confirm_document(document_id: str, payload: ConfirmDocument) -> dict:
    settings = get_settings()
    with connect() as db:
        staged = db.execute("SELECT * FROM staged_documents WHERE id=?", (document_id,)).fetchone()
        if not staged:
            raise HTTPException(404, "待确认文件不存在")
        duplicate = db.execute("SELECT id FROM documents WHERE fingerprint=? AND space_id=?", (staged["fingerprint"], payload.space_id)).fetchone()
        if duplicate:
            raise HTTPException(409, "相同内容已经存在于该知识空间")
        source = Path(staged["staged_path"])
        destination_dir = settings.data_dir / "library" / payload.space_id / document_id
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / staged["original_name"]
        shutil.copy2(source, destination)
        stamp = now()
        db.execute("""INSERT INTO documents(id,space_id,original_name,library_path,file_type,size_bytes,fingerprint,title,summary,tags_json,parse_status,index_status,created_at,updated_at)
                      VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (document_id, payload.space_id, staged["original_name"], str(destination), staged["file_type"], staged["size_bytes"], staged["fingerprint"], payload.title, payload.summary, json_value(payload.tags), "parsed", "indexing", stamp, stamp))
        staged_vision = db.execute(
            "SELECT * FROM staged_image_analysis WHERE document_id=?",
            (document_id,),
        ).fetchone()
        if staged_vision:
            db.execute(
                """INSERT INTO image_assets(
                   document_id,width,height,description,ocr_text,tags_json,search_text,
                   embedding_json,embedding_model,vision_model,status,error_message,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (document_id, staged_vision["width"], staged_vision["height"],
                 staged_vision["description"], staged_vision["ocr_text"], staged_vision["tags_json"],
                 "", None, None, staged_vision["vision_model"], "staged_ready", None, stamp, stamp),
            )
        job_id = uuid4().hex
        db.execute(
            """INSERT INTO indexing_jobs(id,document_id,status,phase,progress,total,completed,message,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (job_id, document_id, "queued", "queued", 0, 0, 0, "已加入索引队列", stamp, stamp),
        )
        db.execute("DELETE FROM staged_documents WHERE id=?", (document_id,))
        db.execute("DELETE FROM staged_image_analysis WHERE document_id=?", (document_id,))
    source.unlink(missing_ok=True)
    INDEX_EXECUTOR.submit(_run_index_job, job_id, document_id, str(destination))
    return {"id": document_id, "job_id": job_id, "title": payload.title, "status": "queued"}


@app.post("/api/documents/{document_id}/reindex", status_code=202)
def reindex_document(document_id: str) -> dict:
    with connect() as db:
        document = db.execute("SELECT id,library_path FROM documents WHERE id=?", (document_id,)).fetchone()
        if not document:
            raise HTTPException(404, "资料不存在")
        if not Path(document["library_path"]).is_file():
            raise HTTPException(409, "KUN 本地副本缺失，无法重新索引")
        job_id = uuid4().hex
        stamp = now()
        db.execute(
            """INSERT INTO indexing_jobs(id,document_id,status,phase,progress,total,completed,message,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (job_id, document_id, "queued", "queued", 0, 0, 0, "已加入重新索引队列", stamp, stamp),
        )
    INDEX_EXECUTOR.submit(_run_index_job, job_id, document_id, document["library_path"])
    return {"id": document_id, "job_id": job_id, "status": "queued"}


@app.get("/api/index-jobs/{job_id}")
def get_index_job(job_id: str) -> dict:
    matches = rows("SELECT * FROM indexing_jobs WHERE id=?", (job_id,))
    if not matches:
        raise HTTPException(404, "索引任务不存在")
    return matches[0]


@app.get("/api/search")
def retrieve(q: str, space_id: str = "ai-agent-learning", top_k: int = 5) -> dict:
    return {"query": q, "results": search(q, space_id, max(1, min(top_k, 20)))}


@app.get("/api/chunks/{chunk_id}")
def get_chunk(chunk_id: str) -> dict:
    matches = rows(
        """SELECT c.id,c.locator,c.heading,c.text,d.title,d.original_name
           FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.id=?""",
        (chunk_id,),
    )
    if not matches:
        raise HTTPException(404, "引用片段不存在")
    return matches[0]


def _conversation_for(payload: ChatRequest) -> str:
    stamp = now()
    conversation_id = payload.conversation_id
    with connect() as db:
        if conversation_id:
            conversation = db.execute("SELECT id FROM conversations WHERE id=?", (conversation_id,)).fetchone()
            if not conversation:
                raise HTTPException(404, "对话不存在")
        else:
            conversation_id = uuid4().hex
            title = re.sub(r"\s+", " ", payload.question).strip()[:32]
            db.execute(
                "INSERT INTO conversations(id,title,space_id,created_at,updated_at) VALUES(?,?,?,?,?)",
                (conversation_id, title, payload.space_id, stamp, stamp),
            )
        db.execute(
            "INSERT INTO messages(id,conversation_id,role,content,citations_json,created_at) VALUES(?,?,?,?,?,?)",
            (uuid4().hex, conversation_id, "user", payload.question, "[]", stamp),
        )
        db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (stamp, conversation_id))
    return conversation_id


def _apply_memory_candidate(result: dict, conversation_id: str) -> dict | None:
    memory_suggestion = None
    candidate = result.get("memory_candidate")
    if candidate:
        candidate_content = candidate["content"] if isinstance(candidate, dict) else str(candidate)
        candidate_kind = candidate.get("kind", "preference") if isinstance(candidate, dict) else "preference"
        requested_status = candidate.get("status", "pending") if isinstance(candidate, dict) else "pending"
        tool_name = "memory.write" if requested_status == "enabled" else "memory.propose"
        execution = REGISTRY.invoke(
            tool_name,
            {
                "kind": candidate_kind,
                "content": candidate_content,
                "source": f"conversation:{conversation_id}",
            },
            ToolContext(
                write_scopes={"memory"},
                confirmed=requested_status == "enabled",
                conversation_id=conversation_id,
            ),
        )
        memory_item = execution["result"]
        memory_suggestion = {
            "id": memory_item["id"],
            "content": memory_item["content"],
            "kind": memory_item["kind"],
            "status": memory_item["status"],
        }
        result.setdefault("tool_trace", []).append(execution["trace"])
    return memory_suggestion


def _run_chat(payload: ChatRequest) -> dict:
    conversation_id = _conversation_for(payload)
    result = run(payload.question, payload.space_id, conversation_id)
    memory_suggestion = _apply_memory_candidate(result, conversation_id)
    assistant_message_id = uuid4().hex
    with connect() as db:
        db.execute(
            "INSERT INTO messages(id,conversation_id,role,content,citations_json,plan_json,created_at) VALUES(?,?,?,?,?,?,?)",
            (assistant_message_id, conversation_id, "assistant", result["answer"], json_value(result.get("citations", [])),
             json_value(result.get("plan", {})), now()),
        )
        db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now(), conversation_id))
    return {
        **result,
        "conversation_id": conversation_id,
        "message_id": assistant_message_id,
        "memory_suggestion": memory_suggestion,
    }


@app.get("/api/conversations")
def conversations(q: str = "") -> list[dict]:
    pattern = f"%{q.strip()}%"
    return rows(
        """SELECT c.*,
           (SELECT content FROM messages m WHERE m.conversation_id=c.id ORDER BY m.rowid DESC LIMIT 1) preview,
           (SELECT COUNT(*) FROM messages m WHERE m.conversation_id=c.id) message_count
           FROM conversations c
           WHERE ?='' OR c.title LIKE ? OR EXISTS(
             SELECT 1 FROM messages m WHERE m.conversation_id=c.id AND m.content LIKE ?
           ) ORDER BY c.updated_at DESC LIMIT 100""",
        (q.strip(), pattern, pattern),
    )


@app.get("/api/conversations/{conversation_id}")
def conversation_detail(conversation_id: str) -> dict:
    with connect() as db:
        conversation = db.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,)).fetchone()
        if not conversation:
            raise HTTPException(404, "对话不存在")
        messages = [
            {**dict(item), "citations": json.loads(item["citations_json"] or "[]"),
             "plan": json.loads(item["plan_json"] or "{}")}
            for item in db.execute(
                "SELECT id,role,content,citations_json,plan_json,created_at FROM messages WHERE conversation_id=? ORDER BY rowid",
                (conversation_id,),
            )
        ]
    return {"conversation": dict(conversation), "messages": messages}


@app.patch("/api/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, payload: ConversationUpdate) -> dict:
    with connect() as db:
        result = db.execute(
            "UPDATE conversations SET title=?,updated_at=? WHERE id=?",
            (payload.title.strip(), now(), conversation_id),
        )
        if not result.rowcount:
            raise HTTPException(404, "对话不存在")
    return {"id": conversation_id, "title": payload.title.strip()}


@app.delete("/api/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str) -> None:
    with connect() as db:
        result = db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
        if not result.rowcount:
            raise HTTPException(404, "对话不存在")


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict:
    return _run_chat(payload)


@app.post("/api/chat/regenerate")
def regenerate_chat(payload: ChatRegenerateRequest) -> dict:
    with connect() as db:
        assistant = db.execute(
            """SELECT rowid,id,conversation_id,role FROM messages
               WHERE id=? AND conversation_id=?""",
            (payload.assistant_message_id, payload.conversation_id),
        ).fetchone()
        if not assistant or assistant["role"] != "assistant":
            raise HTTPException(404, "找不到要重新生成的回答")
        latest = db.execute(
            "SELECT id FROM messages WHERE conversation_id=? ORDER BY rowid DESC LIMIT 1",
            (payload.conversation_id,),
        ).fetchone()
        if not latest or latest["id"] != payload.assistant_message_id:
            raise HTTPException(409, "目前只支持重新生成最后一条回答")
        user = db.execute(
            """SELECT content FROM messages WHERE conversation_id=? AND role='user' AND rowid<?
               ORDER BY rowid DESC LIMIT 1""",
            (payload.conversation_id, assistant["rowid"]),
        ).fetchone()
        conversation = db.execute(
            "SELECT space_id FROM conversations WHERE id=?",
            (payload.conversation_id,),
        ).fetchone()
    if not user or not conversation:
        raise HTTPException(409, "这条回答缺少可重新执行的用户问题")
    result = run(user["content"], conversation["space_id"], payload.conversation_id)
    memory_suggestion = _apply_memory_candidate(result, payload.conversation_id)
    stamp = now()
    with connect() as db:
        db.execute(
            """UPDATE messages SET content=?,citations_json=?,plan_json=?,created_at=? WHERE id=?""",
            (
                result["answer"],
                json_value(result.get("citations", [])),
                json_value(result.get("plan", {})),
                stamp,
                payload.assistant_message_id,
            ),
        )
        db.execute(
            "UPDATE conversations SET updated_at=? WHERE id=?",
            (stamp, payload.conversation_id),
        )
    return {
        **result,
        "conversation_id": payload.conversation_id,
        "message_id": payload.assistant_message_id,
        "memory_suggestion": memory_suggestion,
    }


@app.post("/api/chat/stream")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    def events():
        route = understand({"question": payload.question, "space_id": payload.space_id})
        if route["intent"] == "memory_setting":
            statuses = ("正在识别需要记住的信息", "正在更新 Memory", "正在核对写入结果")
        elif route["intent"] == "memory_query":
            statuses = ("正在理解你的问题", "正在读取长期 Memory", "正在组织回答")
        elif route["intent"] == "web_research":
            statuses = ("正在判断是否需要联网", "正在搜索公开网页", "正在核对网页来源")
        else:
            statuses = ("正在理解你的问题", "正在检索知识空间", "正在核对引用")
        for status in statuses:
            yield f"data: {json.dumps({'type':'status','message':status}, ensure_ascii=False)}\n\n"
        try:
            result = _run_chat(payload)
            for trace in result.get("tool_trace", []):
                yield f"data: {json.dumps({'type':'tool','data':trace}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type':'result','data':result}, ensure_ascii=False)}\n\n"
        except Exception as error:
            yield f"data: {json.dumps({'type':'error','message':str(error)}, ensure_ascii=False)}\n\n"
        yield "data: {\"type\":\"done\"}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/memories")
def memories() -> list[dict]:
    return rows("SELECT * FROM memories WHERE status!='dismissed' ORDER BY updated_at DESC")


@app.get("/api/memories/short-term")
def short_term_memory(conversation_id: str | None = None) -> dict:
    with connect() as db:
        conversation = None
        if conversation_id:
            conversation = db.execute(
                "SELECT * FROM conversations WHERE id=?",
                (conversation_id,),
            ).fetchone()
        if not conversation:
            conversation = db.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        if not conversation:
            return {
                "conversation": None,
                "messages": [],
                "message_count": 0,
                "scope": "conversation",
                "window_size": 10,
                "previous_message_limit": 9,
                "storage_policy": "完整对话保存在本地；模型只滚动读取最近窗口",
            }
        messages = [
            dict(item) for item in db.execute(
                """SELECT id,role,content,created_at FROM messages
                   WHERE conversation_id=? ORDER BY rowid DESC LIMIT 10""",
                (conversation["id"],),
            ).fetchall()
        ]
        messages.reverse()
        working_facts: list[dict] = []
        seen_facts: set[tuple[str, str]] = set()
        for message in messages:
            if message["role"] != "user":
                continue
            candidate = detect_memory_candidate(message["content"])
            if not candidate:
                continue
            key = (candidate["kind"], candidate["content"])
            if key in seen_facts:
                continue
            seen_facts.add(key)
            saved = db.execute(
                """SELECT status FROM memories
                   WHERE kind=? AND content=? AND status IN ('pending','enabled','disabled')
                   ORDER BY updated_at DESC LIMIT 1""",
                key,
            ).fetchone()
            working_facts.append({
                "kind": candidate["kind"],
                "content": candidate["content"],
                "status": saved["status"] if saved else "short_term",
            })
        total = db.execute(
            "SELECT COUNT(*) count FROM messages WHERE conversation_id=?",
            (conversation["id"],),
        ).fetchone()["count"]
    context_plan = build_context(conversation["id"], "")
    return {
        "conversation": dict(conversation),
        "messages": messages,
        "working_facts": working_facts,
        "message_count": total,
        "scope": "conversation",
        "window_size": context_plan["composition"]["recent_message_count"],
        "previous_message_limit": context_plan["composition"]["recent_message_count"],
        "storage_policy": "完整对话保存在本地；发送给模型的是结构化历史摘要与预算内最近完整消息",
        "expires": "切换或删除对话后不再作为当前工作记忆",
        "context": context_plan,
    }


@app.post("/api/memories", status_code=201)
def create_memory(payload: MemoryCreate) -> dict:
    memory_id = uuid4().hex
    stamp = now()
    conflict_id = find_conflict(payload.kind, payload.content.strip())
    with connect() as db:
        db.execute(
            """INSERT INTO memories(id,kind,content,source,status,use_count,version,conflict_with_id,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (memory_id, payload.kind, payload.content.strip(), "manual", "enabled", 0, 1, conflict_id, stamp, stamp),
        )
        item = dict(db.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone())
    record_memory_event(memory_id, "created", "manual", None, item)
    return item


@app.get("/api/memories/{memory_id}")
def get_memory_detail(memory_id: str) -> dict:
    item = memory_detail(memory_id)
    if not item:
        raise HTTPException(404, "Memory 不存在")
    return item


@app.patch("/api/memories/{memory_id}")
def update_memory(memory_id: str, payload: MemoryUpdate) -> dict:
    if payload.status is not None and payload.status not in {"pending", "enabled", "disabled", "dismissed"}:
        raise HTTPException(400, "不支持的 Memory 状态")
    with connect() as db:
        existing = db.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        if not existing:
            raise HTTPException(404, "Memory 不存在")
        before = dict(existing)
        content = payload.content.strip() if payload.content is not None else existing["content"]
        status = payload.status if payload.status is not None else existing["status"]
        conflict_id = find_conflict(existing["kind"], content, memory_id) if status == "enabled" else existing["conflict_with_id"]
        version = int(existing["version"] or 1) + 1
        db.execute("UPDATE memories SET content=?,status=?,version=?,conflict_with_id=?,updated_at=? WHERE id=?",
                   (content, status, version, conflict_id, now(), memory_id))
        item = dict(db.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone())
    if payload.content is not None:
        event_type = "edited"
    elif payload.status == "enabled":
        event_type = "confirmed" if before["status"] == "pending" else "enabled"
    elif payload.status == "disabled":
        event_type = "disabled"
    elif payload.status == "dismissed":
        event_type = "dismissed"
    else:
        event_type = "updated"
    record_memory_event(memory_id, event_type, "user", before, item)
    return item


@app.delete("/api/memories/{memory_id}", status_code=204)
def delete_memory(memory_id: str) -> None:
    with connect() as db:
        existing = db.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        if not existing:
            raise HTTPException(404, "Memory 不存在")
        before = dict(existing)
        db.execute("DELETE FROM memories WHERE id=?", (memory_id,))
    record_memory_event(memory_id, "deleted", "user", before, None)

def _skill_roots() -> list[tuple[Path, str]]:
    project_root = Path(__file__).resolve().parents[2]
    user_root = get_settings().data_dir / "skills"
    user_root.mkdir(parents=True, exist_ok=True)
    return [(project_root / "skills", "builtin"), (user_root, "user")]


def _read_skill(folder: Path, source: str) -> dict | None:
    skill_md = folder / "SKILL.md"
    contract_file = folder / "skill.json"
    contract: dict = {}
    if contract_file.is_file():
        try:
            contract = json.loads(contract_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            contract = {}
    content = skill_md.read_text(encoding="utf-8") if skill_md.is_file() else ""
    frontmatter: dict[str, str] = {}
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, flags=re.DOTALL)
    if match:
        for line in match.group(1).splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                frontmatter[key.strip()] = value.strip().strip("\"'")
    if not content and not contract:
        return None
    return {
        "id": folder.name,
        "name": frontmatter.get("name") or contract.get("name") or folder.name,
        "description": frontmatter.get("description") or contract.get("description") or "尚未填写触发说明",
        "tools": contract.get("tools", []),
        "read_scope": contract.get("read_scope", []),
        "write_scope": contract.get("write_scope", []),
        "timeout_seconds": contract.get("timeout_seconds"),
        "requires_confirmation_for_write": contract.get("requires_confirmation_for_write", False),
        "source": source,
        "has_skill_md": skill_md.is_file(),
        "content": content,
        "steps": [match.group(1).strip() for match in re.finditer(r"(?m)^\s*\d+\.\s+(.+)$", content)],
        "recoverable_errors": contract.get("recoverable_errors", []),
        "output_fields": list((contract.get("output_schema", {}).get("properties") or {}).keys()),
    }


@app.get("/api/skills")
def list_skills() -> list[dict]:
    items: list[dict] = []
    for root, source in _skill_roots():
        if not root.is_dir():
            continue
        for folder in sorted(path for path in root.iterdir() if path.is_dir()):
            item = _read_skill(folder, source)
            if item:
                items.append(item)
    return items


@app.post("/api/skills", status_code=201)
def create_skill(payload: SkillCreate) -> dict:
    user_root = _skill_roots()[1][0]
    folder = (user_root / payload.name).resolve()
    if folder.parent != user_root.resolve():
        raise HTTPException(400, "Skill 名称不安全")
    if folder.exists():
        raise HTTPException(409, "同名 Skill 已存在")
    folder.mkdir(parents=False)
    tool_lines = "\n".join(f"- `{name}`" for name in payload.tools) or "- 暂不调用 Tool"
    content = (
        f"---\nname: {payload.name}\ndescription: {payload.description.strip()}\n---\n\n"
        f"# {payload.name}\n\n## Instructions\n\n{payload.instructions.strip()}\n\n"
        f"## Tools\n\n{tool_lines}\n"
    )
    (folder / "SKILL.md").write_text(content, encoding="utf-8")
    contract = {
        "name": payload.name,
        "version": "0.1.0",
        "description": payload.description.strip(),
        "tools": payload.tools,
        "read_scope": [],
        "write_scope": [],
        "requires_confirmation_for_write": True,
        "timeout_seconds": 120,
        "recoverable_errors": ["tool_unavailable", "permission_denied"],
    }
    (folder / "skill.json").write_text(json_value(contract), encoding="utf-8")
    return _read_skill(folder, "user") or {}


@app.post("/api/agent/evaluation/run")
def run_agent_quality_evaluation() -> dict:
    route_details = []
    for question, expected_intent, expected_skill in ROUTE_CASES:
        actual = understand({"question": question, "space_id": "evaluation"})
        passed = actual.get("intent") == expected_intent and actual.get("skill") == expected_skill
        route_details.append({"question": question, "expected_intent": expected_intent, "expected_skill": expected_skill,
                              "actual_intent": actual.get("intent"), "actual_skill": actual.get("skill"), "passed": passed})
    refusal_details = []
    for question in REFUSAL_CASES:
        result = generate({"question": question, "space_id": "evaluation-empty", "intent": "knowledge_question",
                           "skill": "document_skill", "contexts": [], "tool_trace": []})
        answer = result.get("answer", "")
        passed = "没有找到足够依据" in answer or "没有返回可核验" in answer or "证据不足" in answer
        refusal_details.append({"question": question, "passed": passed, "behavior": "refused" if passed else "answered_without_evidence"})
    route_passed = sum(1 for item in route_details if item["passed"])
    refusal_passed = sum(1 for item in refusal_details if item["passed"])
    latest = rows("SELECT * FROM evaluation_runs ORDER BY created_at DESC LIMIT 1")
    citation_metric = None
    citation_cases = 0
    if latest:
        citation_metric = float(latest[0]["recall"])
        citation_cases = int(latest[0]["case_count"])
    automatic_passed = route_passed + refusal_passed
    automatic_total = len(route_details) + len(refusal_details)
    if citation_metric is not None:
        automatic_passed += round(citation_metric * citation_cases)
        automatic_total += citation_cases
    task_success = automatic_passed / automatic_total if automatic_total else 0.0
    metrics = {
        "routing_accuracy": {"value": round(route_passed / len(route_details), 4), "case_count": len(route_details), "target": GATE_TARGET},
        "refusal_accuracy": {"value": round(refusal_passed / len(refusal_details), 4), "case_count": len(refusal_details), "target": GATE_TARGET},
        "citation_location_success": {"value": citation_metric, "case_count": citation_cases, "target": GATE_TARGET,
                                      "status": "measured" if citation_metric is not None else "needs_rag_run"},
        "agent_task_success": {"value": round(task_success, 4), "case_count": automatic_total, "target": GATE_TARGET},
        "claim_support_rate": {"value": None, "case_count": 0, "target": GATE_TARGET, "status": "needs_human_labels"},
    }
    publish_ready = all(item.get("value") is not None and item["value"] >= GATE_TARGET for key, item in metrics.items() if key != "claim_support_rate") and citation_cases >= 100
    return {"target": GATE_TARGET, "metrics": metrics, "publish_ready": publish_ready,
            "publication_blockers": (["引用定位人工集少于 100 题"] if citation_cases < 100 else []) + ["逐条 Claim—Citation 支持率尚未人工标注"],
            "route_details": route_details, "refusal_details": refusal_details,
            "notes": "路由与空证据拒答为确定性自动测试；引用定位读取最近一次人工 RAG 评估；Claim 支持率必须人工标注。"}

@app.get("/api/rag/evaluation/cases")
def evaluation_cases(space_id: str = "ai-agent-learning") -> list[dict]:
    return rows(
        """SELECT e.*,d.title expected_document_title,d.original_name expected_original_name
           FROM evaluation_cases e JOIN documents d ON d.id=e.expected_document_id
           WHERE e.space_id=? ORDER BY e.updated_at DESC""",
        (space_id,),
    )


@app.post("/api/rag/evaluation/import/hello-agents")
def import_hello_agents_evaluation(space_id: str = "ai-agent-learning") -> dict:
    with connect() as db:
        document = db.execute(
            """SELECT id,title,original_name FROM documents
               WHERE space_id=? AND lower(original_name)=lower(?)
               ORDER BY updated_at DESC LIMIT 1""",
            (space_id, HELLO_AGENTS_DOCUMENT),
        ).fetchone()
        if not document:
            raise HTTPException(404, f"请先把 {HELLO_AGENTS_DOCUMENT} 加入当前知识空间并完成索引")
        stamp = now()
        imported = 0
        for code, question, locator in HELLO_AGENTS_EVALUATION_CASES:
            case_id = f"hello-agents:{space_id}:{code}"
            exists = db.execute("SELECT 1 FROM evaluation_cases WHERE id=?", (case_id,)).fetchone()
            db.execute(
                """INSERT INTO evaluation_cases(
                   id,space_id,question,expected_document_id,expected_locator,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     question=excluded.question,
                     expected_document_id=excluded.expected_document_id,
                     expected_locator=excluded.expected_locator,
                     updated_at=excluded.updated_at""",
                (case_id, space_id, question, document["id"], locator, stamp, stamp),
            )
            if not exists:
                imported += 1
    return {
        "dataset": "Hello-Agents-30",
        "document_id": document["id"],
        "document_title": document["title"],
        "total": len(HELLO_AGENTS_EVALUATION_CASES),
        "imported": imported,
        "status": "ready",
    }


@app.post("/api/rag/evaluation/cases", status_code=201)
def create_evaluation_case(payload: EvaluationCaseCreate) -> dict:
    with connect() as db:
        document = db.execute(
            "SELECT id,title,original_name FROM documents WHERE id=? AND space_id=?",
            (payload.expected_document_id, payload.space_id),
        ).fetchone()
        if not document:
            raise HTTPException(400, "预期来源不属于当前知识空间")
        case_id = uuid4().hex
        stamp = now()
        db.execute(
            """INSERT INTO evaluation_cases(
               id,space_id,question,expected_document_id,expected_locator,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?)""",
            (case_id, payload.space_id, payload.question.strip(), payload.expected_document_id,
             payload.expected_locator.strip(), stamp, stamp),
        )
    return {
        "id": case_id,
        "space_id": payload.space_id,
        "question": payload.question.strip(),
        "expected_document_id": payload.expected_document_id,
        "expected_document_title": document["title"],
        "expected_original_name": document["original_name"],
        "expected_locator": payload.expected_locator.strip(),
        "created_at": stamp,
        "updated_at": stamp,
    }


@app.delete("/api/rag/evaluation/cases/{case_id}", status_code=204)
def delete_evaluation_case(case_id: str) -> None:
    with connect() as db:
        result = db.execute("DELETE FROM evaluation_cases WHERE id=?", (case_id,))
        if not result.rowcount:
            raise HTTPException(404, "评估问题不存在")


@app.post("/api/rag/evaluation/run")
def run_evaluation(payload: EvaluationRunRequest) -> dict:
    cases = rows(
        """SELECT * FROM evaluation_cases WHERE space_id=?
           ORDER BY CASE WHEN id LIKE 'hello-agents:%' THEN 0 ELSE 1 END, id
           LIMIT ?""",
        (payload.space_id, payload.limit),
    )
    if not cases:
        raise HTTPException(400, "请先添加至少 1 条人工标注问题")
    details: list[dict] = []
    reciprocal_ranks: list[float] = []
    gains: list[float] = []
    latencies: list[float] = []
    hits = 0
    for case in cases:
        started = perf_counter()
        results = search(case["question"], payload.space_id, payload.top_k)
        latency_ms = round((perf_counter() - started) * 1000, 2)
        latencies.append(latency_ms)
        rank = None
        for index, item in enumerate(results, 1):
            document_match = item["document_id"] == case["expected_document_id"]
            expected_pages = set(re.findall(r"第\s*(\d+)\s*页", case["expected_locator"]))
            returned_pages = set(re.findall(r"第\s*(\d+)\s*页", item["locator"]))
            locator_match = (
                not case["expected_locator"]
                or bool(expected_pages.intersection(returned_pages))
                or (not expected_pages and case["expected_locator"] in item["locator"])
            )
            if document_match and locator_match:
                rank = index
                break
        if rank:
            hits += 1
        reciprocal = 1 / rank if rank else 0.0
        gain = 1 / math.log2(rank + 1) if rank else 0.0
        reciprocal_ranks.append(reciprocal)
        gains.append(gain)
        failure_category = None if rank else (
            "retrieved_but_not_at_expected_location"
            if any(item["document_id"] == case["expected_document_id"] for item in results)
            else "not_retrieved"
        )
        details.append({
            "case_id": case["id"],
            "question": case["question"],
            "hit": bool(rank),
            "rank": rank,
            "failure_category": failure_category,
            "latency_ms": latency_ms,
            "returned": [
                {
                    "document_id": item["document_id"],
                    "title": item["title"],
                    "locator": item["locator"],
                    "score": item["score"],
                }
                for item in results
            ],
        })
    sorted_latency = sorted(latencies)
    p95_index = max(0, math.ceil(len(sorted_latency) * 0.95) - 1)
    result = {
        "id": uuid4().hex,
        "space_id": payload.space_id,
        "top_k": payload.top_k,
        "case_count": len(cases),
        "recall": round(hits / len(cases), 4),
        "mrr": round(sum(reciprocal_ranks) / len(cases), 4),
        "ndcg": round(sum(gains) / len(cases), 4),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2),
        "p95_latency_ms": round(sorted_latency[p95_index], 2),
        "details": details,
        "bad_case_counts": {
            "not_retrieved": sum(1 for item in details if item.get("failure_category") == "not_retrieved"),
            "retrieved_but_not_at_expected_location": sum(1 for item in details if item.get("failure_category") == "retrieved_but_not_at_expected_location"),
        },
        "extended_metrics": {
            "citation_support_rate": {"status": "not_evaluated", "reason": "当前检索集没有逐条答案与引用支持标注"},
            "unsupported_question_refusal": {"status": "not_evaluated", "reason": "当前评估集没有无依据问题标注"},
            "agent_task_success_rate": {"status": "not_evaluated", "reason": "需要独立端到端任务集"},
            "tool_success_rate": {"status": "operational_only", "reason": "Tool Run 可观测，但尚未按评估任务标注期望结果"},
            "memory_quality": {"status": "not_evaluated", "reason": "需要候选准确性、确认率和冲突人工标注"},
        },
        "created_at": now(),
    }
    with connect() as db:
        db.execute(
            """INSERT INTO evaluation_runs(
               id,space_id,top_k,case_count,recall,mrr,ndcg,mean_latency_ms,p95_latency_ms,result_json,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (result["id"], result["space_id"], result["top_k"], result["case_count"],
             result["recall"], result["mrr"], result["ndcg"], result["mean_latency_ms"],
             result["p95_latency_ms"], json_value({"details": result["details"], "bad_case_counts": result["bad_case_counts"], "extended_metrics": result["extended_metrics"]}), result["created_at"]),
        )
    return result


@app.get("/api/images")
def images(q: str = "", space_id: str = "ai-agent-learning") -> list[dict]:
    return search_images(q, space_id)


@app.get("/api/images/status")
def image_statuses(space_id: str = "ai-agent-learning") -> list[dict]:
    return rows(
        """SELECT d.id document_id,d.title,d.original_name,d.space_id,d.index_status,
           COALESCE(i.status,'pending') vision_status,i.error_message,i.vision_model,i.updated_at
           FROM documents d LEFT JOIN image_assets i ON i.document_id=d.id
           WHERE d.file_type IN ('png','jpg','jpeg') AND d.space_id=? ORDER BY d.updated_at DESC""",
        (space_id,),
    )


@app.post("/api/images/{document_id}/analyze", status_code=202)
def analyze_image(document_id: str) -> dict:
    matches = rows("SELECT id,file_type FROM documents WHERE id=?", (document_id,))
    if not matches or matches[0]["file_type"] not in {"png", "jpg", "jpeg"}:
        raise HTTPException(404, "图片资料不存在")
    INDEX_EXECUTOR.submit(index_image_document, document_id)
    return {"document_id": document_id, "status": "queued"}
