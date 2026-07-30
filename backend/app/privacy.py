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
