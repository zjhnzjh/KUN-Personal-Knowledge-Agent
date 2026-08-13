from __future__ import annotations

import json

from .database import connect, now


DEFAULT_PRIVACY_SETTINGS = {
    "web_search_enabled": True,
    "cloud_document_analysis_enabled": True,
    "cloud_image_analysis_enabled": True,
    "memory_suggestions_enabled": True,
    "sensitive_data_protection_enabled": True,
}


def get_privacy_settings() -> dict[str, bool]:
    settings = dict(DEFAULT_PRIVACY_SETTINGS)
    with connect() as db:
        row = db.execute("SELECT value_json FROM app_settings WHERE key='privacy'").fetchone()
    if not row:
        return settings
    try:
        stored = json.loads(row["value_json"])
    except (TypeError, json.JSONDecodeError):
        return settings
    for key in settings:
        if key in stored and isinstance(stored[key], bool):
            settings[key] = stored[key]
    # Sensitive-data filtering is a fixed safety boundary in the first release.
    settings["sensitive_data_protection_enabled"] = True
    return settings


def save_privacy_settings(patch: dict[str, bool]) -> dict[str, bool]:
    settings = get_privacy_settings()
    for key, value in patch.items():
        if key in settings and key != "sensitive_data_protection_enabled":
            settings[key] = bool(value)
    settings["sensitive_data_protection_enabled"] = True
    with connect() as db:
        db.execute(
            """INSERT INTO app_settings(key,value_json,updated_at) VALUES('privacy',?,?)
               ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,updated_at=excluded.updated_at""",
            (json.dumps(settings, ensure_ascii=False), now()),
        )
    return settings


def document_cloud_policies(space_id: str) -> list[dict]:
    with connect() as db:
        return [dict(item) for item in db.execute(
            """SELECT d.id document_id,d.title,d.original_name,d.file_type,
               COALESCE(p.embedding_allowed,0) embedding_allowed,
               COALESCE(p.llm_allowed,0) llm_allowed,p.updated_at
               FROM documents d LEFT JOIN document_cloud_policies p ON p.document_id=d.id
               WHERE d.space_id=? ORDER BY d.updated_at DESC""",
            (space_id,),
        ).fetchall()]


def save_document_cloud_policy(document_id: str, *, embedding_allowed: bool, llm_allowed: bool) -> dict:
    with connect() as db:
        document = db.execute("SELECT id FROM documents WHERE id=?", (document_id,)).fetchone()
        if not document:
            raise ValueError("Document does not exist")
        db.execute(
            """INSERT INTO document_cloud_policies(document_id,embedding_allowed,llm_allowed,updated_at)
               VALUES(?,?,?,?) ON CONFLICT(document_id) DO UPDATE SET
               embedding_allowed=excluded.embedding_allowed,llm_allowed=excluded.llm_allowed,
               updated_at=excluded.updated_at""",
            (document_id, int(embedding_allowed), int(llm_allowed), now()),
        )
        item = db.execute(
            """SELECT d.id document_id,d.title,d.original_name,d.file_type,
               p.embedding_allowed,p.llm_allowed,p.updated_at
               FROM documents d JOIN document_cloud_policies p ON p.document_id=d.id WHERE d.id=?""",
            (document_id,),
        ).fetchone()
    return dict(item)


def allowed_for_cloud(document_ids: list[str], capability: str) -> set[str]:
    if not document_ids:
        return set()
    column = "embedding_allowed" if capability == "embedding" else "llm_allowed"
    placeholders = ",".join("?" for _ in document_ids)
    with connect() as db:
        return {
            item["document_id"]
            for item in db.execute(
                f"SELECT document_id FROM document_cloud_policies WHERE {column}=1 AND document_id IN ({placeholders})",
                tuple(document_ids),
            )
        }
