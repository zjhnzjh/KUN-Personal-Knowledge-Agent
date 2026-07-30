from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image, ImageOps

from .config import get_settings
from .database import connect, json_value, now
from .providers import generate_document_metadata
from .privacy import get_privacy_settings
from .rag import DashScopeEmbeddings, lexical_text


IMAGE_TYPES = {".png", ".jpg", ".jpeg"}


def _image_data_url(path: Path) -> tuple[str, int, int]:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        width, height = image.size
        if image.mode not in {"RGB", "L"}:
            background = Image.new("RGB", image.size, "white")
            if "A" in image.getbands():
                background.paste(image, mask=image.getchannel("A"))
            else:
                background.paste(image)
            image = background
        elif image.mode == "L":
            image = image.convert("RGB")
        image.thumbnail((2048, 2048))
        output = BytesIO()
        image.save(output, format="JPEG", quality=84, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}", width, height


def _parse_json(content: str) -> dict:
    clean = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.IGNORECASE).strip()
    data = json.loads(clean)
    return {
        "description": str(data.get("description", "")).strip()[:2000],
        "ocr_text": str(data.get("ocr_text", "")).strip()[:8000],
        "tags": [str(item).strip()[:30] for item in data.get("tags", []) if str(item).strip()][:8],
    }


def understand_image(path: Path) -> tuple[dict, int, int, str]:
    settings = get_settings()
    if not get_privacy_settings()["cloud_image_analysis_enabled"]:
        raise RuntimeError("云端图片理解已在“隐私与权限”中关闭")
    data_url, width, height = _image_data_url(path)
    if not settings.dashscope_api_key:
        raise RuntimeError("尚未配置阿里云百炼 API Key")
    response = httpx.post(
        f"{settings.dashscope_base_url}/chat/completions",
        headers={"Authorization": f"Bearer {settings.dashscope_api_key}"},
        json={
            "model": settings.vision_model,
            "temperature": 0.1,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": (
                        "为个人学习资料库分析这张图片。只返回 JSON："
                        "description（客观描述画面、图表结构和重要信息），"
                        "ocr_text（完整提取可见文字，没有则为空），"
                        "tags（2到8个中文短标签）。不要猜测看不清的内容。"
                    )},
                ],
            }],
        },
        timeout=90,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return _parse_json(content), width, height, settings.vision_model


def index_image_document(document_id: str) -> dict:
    settings = get_settings()
    with connect() as db:
        document = db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        if not document:
            raise ValueError("图片资料不存在")
        cached_analysis = db.execute(
            "SELECT * FROM image_assets WHERE document_id=? AND (description!='' OR ocr_text!='')",
            (document_id,),
        ).fetchone()
        stamp = now()
        db.execute(
            """INSERT INTO image_assets(document_id,status,created_at,updated_at)
               VALUES(?,?,?,?) ON CONFLICT(document_id) DO UPDATE SET
               status=excluded.status,error_message=NULL,updated_at=excluded.updated_at""",
            (document_id, "processing", stamp, stamp),
        )
        db.execute("UPDATE documents SET index_status='indexing',updated_at=? WHERE id=?", (stamp, document_id))
    try:
        if cached_analysis:
            data = {
                "description": cached_analysis["description"],
                "ocr_text": cached_analysis["ocr_text"],
                "tags": json.loads(cached_analysis["tags_json"] or "[]"),
            }
            width = cached_analysis["width"]
            height = cached_analysis["height"]
            vision_model = cached_analysis["vision_model"] or settings.vision_model
        else:
            data, width, height, vision_model = understand_image(Path(document["library_path"]))
        search_text = "\n".join(
            part for part in (data["description"], data["ocr_text"], " ".join(data["tags"])) if part
        ).strip() or f"图片文件 {document['original_name']}"
        metadata = generate_document_metadata(document["original_name"], document["file_type"], search_text)
        combined_tags = list(dict.fromkeys([*data["tags"], *metadata["tags"]]))[:8]
        vectors = DashScopeEmbeddings().encode([search_text])
        embedding = json_value(vectors[0]) if vectors else None
        chunk_id = hashlib.sha1(f"{document_id}:image:{search_text}".encode("utf-8")).hexdigest()[:24]
        with connect() as db:
            old_ids = [row["id"] for row in db.execute("SELECT id FROM chunks WHERE document_id=?", (document_id,))]
            for old_id in old_ids:
                db.execute("DELETE FROM chunks_fts WHERE chunk_id=?", (old_id,))
            db.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            db.execute(
                """INSERT INTO chunks(id,document_id,ordinal,locator,heading,text,embedding_json,embedding_model,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (chunk_id, document_id, 0, "整张图片", "图片理解", search_text, embedding,
                 settings.embedding_model if embedding else None, now()),
            )
            db.execute("INSERT INTO chunks_fts(chunk_id,text) VALUES(?,?)", (chunk_id, lexical_text(search_text)))
            db.execute(
                """INSERT INTO image_assets(document_id,width,height,description,ocr_text,tags_json,search_text,
                   embedding_json,embedding_model,vision_model,status,error_message,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(document_id) DO UPDATE SET width=excluded.width,height=excluded.height,
                   description=excluded.description,ocr_text=excluded.ocr_text,tags_json=excluded.tags_json,
                   search_text=excluded.search_text,embedding_json=excluded.embedding_json,
                   embedding_model=excluded.embedding_model,vision_model=excluded.vision_model,
                   status='ready',error_message=NULL,updated_at=excluded.updated_at""",
                (document_id, width, height, data["description"], data["ocr_text"], json_value(combined_tags),
                 search_text, embedding, settings.embedding_model if embedding else None, vision_model,
                 "ready", None, now(), now()),
            )
            db.execute(
                """UPDATE documents SET title=?,summary=?,tags_json=?,index_status='ready',updated_at=?
                   WHERE id=?""",
                (metadata["title"], metadata["summary"], json_value(combined_tags), now(), document_id),
            )
        return {"document_id": document_id, "status": "ready", "description": data["description"]}
    except Exception as error:
        with connect() as db:
            db.execute(
                "UPDATE image_assets SET status='failed',error_message=?,updated_at=? WHERE document_id=?",
                (str(error)[:500], now(), document_id),
            )
            db.execute("UPDATE documents SET index_status='failed',updated_at=? WHERE id=?", (now(), document_id))
        raise


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0


def search_images(query: str, space_id: str, limit: int = 30) -> list[dict]:
    with connect() as db:
        items = [
            dict(row) for row in db.execute(
                """SELECT i.*,d.title,d.original_name,d.file_type,d.space_id,d.updated_at,
                   (SELECT id FROM chunks c WHERE c.document_id=d.id ORDER BY ordinal LIMIT 1) chunk_id
                   FROM image_assets i JOIN documents d ON d.id=i.document_id
                   WHERE i.status='ready' AND d.space_id=? ORDER BY i.updated_at DESC""",
                (space_id,),
            )
        ]
    query = query.strip()
    query_vector: list[float] | None = None
    if query:
        vectors = DashScopeEmbeddings().encode([query], "query")
        query_vector = vectors[0] if vectors else None
    tokens = lexical_text(query).split()
    for item in items:
        text = item["search_text"].lower()
        lexical_score = sum(1 for token in tokens if token in text) / max(len(tokens), 1) if tokens else 0.0
        vector_score = _cosine(query_vector, json.loads(item["embedding_json"])) if query_vector and item["embedding_json"] else 0.0
        item["score"] = round(vector_score * 0.8 + lexical_score * 0.2, 4) if query else 1.0
        item["tags"] = json.loads(item["tags_json"] or "[]")
        item.pop("embedding_json", None)
        item.pop("search_text", None)
    if query:
        items.sort(key=lambda item: item["score"], reverse=True)
    return items[:limit]
