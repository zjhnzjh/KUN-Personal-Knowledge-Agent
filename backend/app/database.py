from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .config import get_settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations (
  id TEXT PRIMARY KEY, applied_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS app_settings (
  key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS spaces (
  id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, color TEXT NOT NULL,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS staged_documents (
  id TEXT PRIMARY KEY, original_name TEXT NOT NULL, staged_path TEXT NOT NULL,
  file_type TEXT NOT NULL, size_bytes INTEGER NOT NULL, fingerprint TEXT NOT NULL,
  title TEXT NOT NULL, summary TEXT NOT NULL, tags_json TEXT NOT NULL,
  parse_status TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY, space_id TEXT NOT NULL REFERENCES spaces(id),
  original_name TEXT NOT NULL, library_path TEXT NOT NULL, file_type TEXT NOT NULL,
  size_bytes INTEGER NOT NULL, fingerprint TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 1,
  title TEXT NOT NULL, summary TEXT NOT NULL, tags_json TEXT NOT NULL,
  parse_status TEXT NOT NULL, index_status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_document_fingerprint_space ON documents(fingerprint, space_id);
CREATE TABLE IF NOT EXISTS document_cloud_policies (
  document_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
  embedding_allowed INTEGER NOT NULL DEFAULT 0,
  llm_allowed INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL, locator TEXT NOT NULL, heading TEXT, text TEXT NOT NULL,
  embedding_json TEXT, embedding_model TEXT, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id, ordinal);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, text, tokenize='unicode61');
CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, content TEXT NOT NULL, source TEXT NOT NULL,
  status TEXT NOT NULL, use_count INTEGER NOT NULL DEFAULT 0, version INTEGER NOT NULL DEFAULT 1,
  conflict_with_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_events (
  id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, event_type TEXT NOT NULL, version INTEGER NOT NULL,
  source TEXT NOT NULL, before_json TEXT NOT NULL, after_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_events_memory ON memory_events(memory_id, created_at DESC);
CREATE TABLE IF NOT EXISTS memory_recalls (
  id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, conversation_id TEXT, query TEXT NOT NULL,
  reason TEXT NOT NULL, score REAL NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY, title TEXT NOT NULL, space_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role TEXT NOT NULL, content TEXT NOT NULL, citations_json TEXT NOT NULL,
  plan_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tool_runs (
  id TEXT PRIMARY KEY, conversation_id TEXT,
  tool_name TEXT NOT NULL, status TEXT NOT NULL,
  input_summary_json TEXT NOT NULL, output_summary_json TEXT NOT NULL,
  error_code TEXT, duration_ms INTEGER,
  created_at TEXT NOT NULL, finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_tool_runs_created ON tool_runs(created_at DESC);
CREATE TABLE IF NOT EXISTS context_summaries (
  id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  version INTEGER NOT NULL, first_message_id TEXT, last_message_id TEXT,
  source_message_ids_json TEXT NOT NULL, summary_json TEXT NOT NULL, token_estimate INTEGER NOT NULL, created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_context_summary_version ON context_summaries(conversation_id, version);
CREATE TABLE IF NOT EXISTS agent_traces (
  id TEXT PRIMARY KEY, conversation_id TEXT, intent TEXT, selected_skill TEXT, status TEXT NOT NULL,
  context_tokens INTEGER NOT NULL DEFAULT 0, context_budget INTEGER NOT NULL DEFAULT 0, summary_version INTEGER,
  retrieval_count INTEGER NOT NULL DEFAULT 0, citation_count INTEGER NOT NULL DEFAULT 0,
  exposed_tool_count INTEGER NOT NULL DEFAULT 0, schema_token_estimate INTEGER NOT NULL DEFAULT 0,
  error_type TEXT, started_at TEXT NOT NULL, finished_at TEXT, duration_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_agent_traces_created ON agent_traces(started_at DESC);
CREATE TABLE IF NOT EXISTS agent_trace_stages (
  id TEXT PRIMARY KEY, trace_id TEXT NOT NULL REFERENCES agent_traces(id) ON DELETE CASCADE,
  stage_name TEXT NOT NULL, status TEXT NOT NULL, duration_ms INTEGER NOT NULL,
  result_summary_json TEXT NOT NULL, error_type TEXT, ordinal INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS provider_status (
  provider TEXT PRIMARY KEY, status TEXT NOT NULL,
  last_checked_at TEXT, error_code TEXT
);
CREATE TABLE IF NOT EXISTS indexing_jobs (
  id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  status TEXT NOT NULL, phase TEXT NOT NULL, progress INTEGER NOT NULL DEFAULT 0,
  total INTEGER NOT NULL DEFAULT 0, completed INTEGER NOT NULL DEFAULT 0,
  message TEXT NOT NULL, error_message TEXT,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_indexing_jobs_document ON indexing_jobs(document_id, created_at DESC);
CREATE TABLE IF NOT EXISTS embedding_cache (
  content_hash TEXT NOT NULL, model TEXT NOT NULL, embedding_json TEXT NOT NULL,
  created_at TEXT NOT NULL, PRIMARY KEY(content_hash, model)
);
CREATE TABLE IF NOT EXISTS image_assets (
  document_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
  width INTEGER NOT NULL DEFAULT 0, height INTEGER NOT NULL DEFAULT 0,
  description TEXT NOT NULL DEFAULT '', ocr_text TEXT NOT NULL DEFAULT '',
  tags_json TEXT NOT NULL DEFAULT '[]', search_text TEXT NOT NULL DEFAULT '',
  embedding_json TEXT, embedding_model TEXT, vision_model TEXT,
  status TEXT NOT NULL, error_message TEXT,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_image_assets_status ON image_assets(status, updated_at DESC);
CREATE TABLE IF NOT EXISTS staged_image_analysis (
  document_id TEXT PRIMARY KEY,
  width INTEGER NOT NULL DEFAULT 0, height INTEGER NOT NULL DEFAULT 0,
  description TEXT NOT NULL DEFAULT '', ocr_text TEXT NOT NULL DEFAULT '',
  tags_json TEXT NOT NULL DEFAULT '[]', vision_model TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluation_cases (
  id TEXT PRIMARY KEY, space_id TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
  question TEXT NOT NULL, expected_document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  expected_locator TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evaluation_cases_space ON evaluation_cases(space_id, updated_at DESC);
CREATE TABLE IF NOT EXISTS evaluation_runs (
  id TEXT PRIMARY KEY, space_id TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
  top_k INTEGER NOT NULL, case_count INTEGER NOT NULL,
  recall REAL NOT NULL, mrr REAL NOT NULL, ndcg REAL NOT NULL,
  mean_latency_ms REAL NOT NULL, p95_latency_ms REAL NOT NULL,
  result_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evaluation_runs_space ON evaluation_runs(space_id, created_at DESC);
CREATE TABLE IF NOT EXISTS infra_traces (
  id TEXT PRIMARY KEY, trace_type TEXT NOT NULL, name TEXT NOT NULL, status TEXT NOT NULL,
  root_attributes_json TEXT NOT NULL DEFAULT '{}', error_code TEXT,
  started_at TEXT NOT NULL, finished_at TEXT, duration_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_infra_traces_started ON infra_traces(started_at DESC);
CREATE TABLE IF NOT EXISTS infra_spans (
  id TEXT PRIMARY KEY, trace_id TEXT NOT NULL REFERENCES infra_traces(id) ON DELETE CASCADE,
  parent_span_id TEXT, operation TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL,
  attributes_json TEXT NOT NULL DEFAULT '{}', error_code TEXT,
  started_at TEXT NOT NULL, finished_at TEXT, duration_ms REAL
);
CREATE INDEX IF NOT EXISTS idx_infra_spans_trace ON infra_spans(trace_id, started_at);
CREATE TABLE IF NOT EXISTS infra_jobs (
  id TEXT PRIMARY KEY, job_type TEXT NOT NULL, status TEXT NOT NULL,
  payload_json TEXT NOT NULL, result_summary_json TEXT NOT NULL DEFAULT '{}',
  idempotency_key TEXT, worker_id TEXT, attempt INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL DEFAULT 3, progress INTEGER NOT NULL DEFAULT 0,
  phase TEXT NOT NULL DEFAULT 'queued', message TEXT NOT NULL DEFAULT '', error_code TEXT,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
  heartbeat_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_infra_jobs_idempotency ON infra_jobs(idempotency_key) WHERE idempotency_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_infra_jobs_status ON infra_jobs(status, updated_at DESC);
CREATE TABLE IF NOT EXISTS embedding_vectors (
  chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  content_hash TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
  dimension INTEGER NOT NULL, embedding_json TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY(chunk_id,provider,model,dimension)
);
CREATE INDEX IF NOT EXISTS idx_embedding_vectors_lookup ON embedding_vectors(provider,model,dimension,content_hash);
CREATE TABLE IF NOT EXISTS generation_embedding_vectors (
  content_hash TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
  dimension INTEGER NOT NULL, embedding_json TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY(content_hash,provider,model,dimension)
);
CREATE TABLE IF NOT EXISTS index_generations (
  id TEXT PRIMARY KEY, space_id TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
  status TEXT NOT NULL, is_active INTEGER NOT NULL DEFAULT 0,
  provider TEXT NOT NULL, model TEXT NOT NULL, dimension INTEGER NOT NULL,
  strategy TEXT NOT NULL, chunk_size INTEGER NOT NULL, chunk_overlap INTEGER NOT NULL,
  parser_version TEXT NOT NULL, chunker_version TEXT NOT NULL, config_hash TEXT NOT NULL,
  manifest_path TEXT, vector_count INTEGER NOT NULL DEFAULT 0, index_bytes INTEGER NOT NULL DEFAULT 0,
  error_code TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, activated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_index_generations_space ON index_generations(space_id, created_at DESC);
CREATE TABLE IF NOT EXISTS index_generation_chunks (
  id TEXT PRIMARY KEY, generation_id TEXT NOT NULL REFERENCES index_generations(id) ON DELETE CASCADE,
  document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL, locator TEXT NOT NULL, heading TEXT, text TEXT NOT NULL,
  content_hash TEXT NOT NULL, UNIQUE(generation_id,document_id,ordinal)
);
CREATE INDEX IF NOT EXISTS idx_generation_chunks_generation ON index_generation_chunks(generation_id,document_id,ordinal);
CREATE VIRTUAL TABLE IF NOT EXISTS index_generation_chunks_fts USING fts5(
  generation_id UNINDEXED, chunk_id UNINDEXED, text, tokenize='unicode61'
);
CREATE TABLE IF NOT EXISTS index_generation_items (
  generation_id TEXT NOT NULL REFERENCES index_generations(id) ON DELETE CASCADE,
  vector_id INTEGER NOT NULL, chunk_id TEXT NOT NULL,
  PRIMARY KEY(generation_id,vector_id), UNIQUE(generation_id,chunk_id)
);
CREATE TABLE IF NOT EXISTS eval_dataset_versions (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, version TEXT NOT NULL, space_id TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
  status TEXT NOT NULL, source TEXT NOT NULL, content_hash TEXT NOT NULL,
  case_count INTEGER NOT NULL DEFAULT 0, target_case_count INTEGER NOT NULL DEFAULT 100,
  holdout_count INTEGER NOT NULL DEFAULT 30, formal_ready INTEGER NOT NULL DEFAULT 0,
  split_seed INTEGER NOT NULL DEFAULT 20260814, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  UNIQUE(name,version,space_id)
);
CREATE TABLE IF NOT EXISTS eval_dataset_cases (
  id TEXT PRIMARY KEY, dataset_version_id TEXT NOT NULL REFERENCES eval_dataset_versions(id) ON DELETE CASCADE,
  question TEXT NOT NULL, split TEXT NOT NULL, query_type TEXT NOT NULL,
  difficulty TEXT NOT NULL, status TEXT NOT NULL, gold_json TEXT NOT NULL,
  answer_text TEXT NOT NULL DEFAULT '', source_type TEXT NOT NULL DEFAULT 'human',
  validation_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_dataset_cases_version ON eval_dataset_cases(dataset_version_id,status,split);
CREATE TABLE IF NOT EXISTS experiment_runs (
  id TEXT PRIMARY KEY, dataset_version_id TEXT NOT NULL REFERENCES eval_dataset_versions(id),
  name TEXT NOT NULL, status TEXT NOT NULL, config_json TEXT NOT NULL, config_hash TEXT NOT NULL,
  parent_run_id TEXT, git_revision TEXT NOT NULL, machine_json TEXT NOT NULL,
  summary_json TEXT NOT NULL DEFAULT '{}', error_code TEXT,
  created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_experiment_runs_created ON experiment_runs(created_at DESC);
CREATE TABLE IF NOT EXISTS experiment_sweeps (
  id TEXT PRIMARY KEY, dataset_version_id TEXT NOT NULL REFERENCES eval_dataset_versions(id),
  name TEXT NOT NULL, status TEXT NOT NULL, config_json TEXT NOT NULL,
  config_hash TEXT NOT NULL, case_status TEXT NOT NULL DEFAULT 'accepted',
  estimated_requests INTEGER NOT NULL DEFAULT 0, estimated_input_characters INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, error_code TEXT
);
CREATE INDEX IF NOT EXISTS idx_experiment_sweeps_created ON experiment_sweeps(created_at DESC);
CREATE TABLE IF NOT EXISTS experiment_sweep_items (
  sweep_id TEXT NOT NULL REFERENCES experiment_sweeps(id) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL, run_id TEXT NOT NULL REFERENCES experiment_runs(id) ON DELETE CASCADE,
  status TEXT NOT NULL, error_code TEXT, started_at TEXT, finished_at TEXT,
  PRIMARY KEY(sweep_id,ordinal), UNIQUE(sweep_id,run_id)
);
CREATE INDEX IF NOT EXISTS idx_experiment_sweep_items_status ON experiment_sweep_items(sweep_id,status,ordinal);
CREATE TABLE IF NOT EXISTS experiment_case_results (
  run_id TEXT NOT NULL REFERENCES experiment_runs(id) ON DELETE CASCADE,
  case_id TEXT NOT NULL REFERENCES eval_dataset_cases(id) ON DELETE CASCADE,
  latency_ms REAL NOT NULL, metrics_json TEXT NOT NULL, rankings_json TEXT NOT NULL,
  failure_category TEXT, trace_id TEXT, PRIMARY KEY(run_id,case_id)
);
CREATE TABLE IF NOT EXISTS performance_benchmarks (
  id TEXT PRIMARY KEY, status TEXT NOT NULL, config_json TEXT NOT NULL,
  result_json TEXT NOT NULL DEFAULT '{}', machine_json TEXT NOT NULL,
  error_code TEXT, created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT
);
CREATE TABLE IF NOT EXISTS eval_query_embedding_cache (
  query_hash TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
  dimension INTEGER NOT NULL, text_type TEXT NOT NULL, embedding_json TEXT NOT NULL,
  created_at TEXT NOT NULL, PRIMARY KEY(query_hash,provider,model,dimension,text_type)
);
CREATE TABLE IF NOT EXISTS eval_rerank_cache (
  query_hash TEXT NOT NULL, candidate_hash TEXT NOT NULL, model TEXT NOT NULL,
  top_n INTEGER NOT NULL, result_json TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY(query_hash,candidate_hash,model,top_n)
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db_path() -> Path:
    return get_settings().data_dir / "kun.sqlite3"


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(db_path())
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_database() -> None:
    with connect() as db:
        db.executescript(SCHEMA)
        memory_columns = {row["name"] for row in db.execute("PRAGMA table_info(memories)").fetchall()}
        if "version" not in memory_columns:
            db.execute("ALTER TABLE memories ADD COLUMN version INTEGER NOT NULL DEFAULT 1")
        if "conflict_with_id" not in memory_columns:
            db.execute("ALTER TABLE memories ADD COLUMN conflict_with_id TEXT")
        message_columns = {row["name"] for row in db.execute("PRAGMA table_info(messages)").fetchall()}
        if "plan_json" not in message_columns:
            db.execute("ALTER TABLE messages ADD COLUMN plan_json TEXT NOT NULL DEFAULT '{}'")
        dataset_columns = {row["name"] for row in db.execute("PRAGMA table_info(eval_dataset_versions)").fetchall()}
        for column, definition in {
            "target_case_count": "INTEGER NOT NULL DEFAULT 100",
            "holdout_count": "INTEGER NOT NULL DEFAULT 30",
            "formal_ready": "INTEGER NOT NULL DEFAULT 0",
            "split_seed": "INTEGER NOT NULL DEFAULT 20260814",
        }.items():
            if column not in dataset_columns:
                db.execute(f"ALTER TABLE eval_dataset_versions ADD COLUMN {column} {definition}")
        case_columns = {row["name"] for row in db.execute("PRAGMA table_info(eval_dataset_cases)").fetchall()}
        for column, definition in {
            "answer_text": "TEXT NOT NULL DEFAULT ''",
            "source_type": "TEXT NOT NULL DEFAULT 'human'",
            "validation_json": "TEXT NOT NULL DEFAULT '{}'",
        }.items():
            if column not in case_columns:
                db.execute(f"ALTER TABLE eval_dataset_cases ADD COLUMN {column} {definition}")
        legacy_citation_migration = "2026-07-25-remove-fixed-five-citations"
        applied = db.execute(
            "SELECT 1 FROM schema_migrations WHERE id=?",
            (legacy_citation_migration,),
        ).fetchone()
        if not applied:
            # Old builds attached exactly five retrieved chunks to every answer,
            # even when the answer did not use them. Keep the conversation text,
            # but remove that unverified provenance once. New answers may still
            # legitimately contain five citations after claim-level validation.
            legacy_ids: list[str] = []
            for row in db.execute(
                "SELECT id, citations_json FROM messages WHERE role='assistant'"
            ).fetchall():
                try:
                    citations = json.loads(row["citations_json"] or "[]")
                except (TypeError, json.JSONDecodeError):
                    citations = []
                if isinstance(citations, list) and len(citations) == 5:
                    legacy_ids.append(row["id"])
            db.executemany(
                "UPDATE messages SET citations_json='[]' WHERE id=?",
                [(message_id,) for message_id in legacy_ids],
            )
            db.execute(
                "INSERT INTO schema_migrations(id,applied_at) VALUES(?,?)",
                (legacy_citation_migration, now()),
            )
        # Remove legacy source cards from obvious Memory-only answers. Earlier
        # builds retrieved five document chunks for every question, including
        # personal-memory questions; those citations were not valid evidence.
        db.execute(
            """UPDATE messages AS assistant SET citations_json='[]'
               WHERE assistant.role='assistant' AND assistant.citations_json!='[]'
               AND EXISTS(
                 SELECT 1 FROM messages AS user
                 WHERE user.conversation_id=assistant.conversation_id
                   AND user.role='user'
                   AND user.rowid=(
                     SELECT MAX(previous.rowid) FROM messages AS previous
                     WHERE previous.conversation_id=assistant.conversation_id
                       AND previous.rowid<assistant.rowid
                   )
                   AND (
                     user.content LIKE '%你记得我%' OR user.content LIKE '%我的偏好%'
                     OR user.content LIKE '%我有哪个%' OR user.content LIKE '%我有几个%'
                     OR user.content LIKE '%长期记忆%' OR user.content LIKE '%memory%'
                   )
               )"""
        )
        stamp = now()
        defaults = (
            ("ai-agent-learning", "AI Agent 学习", "#6d5ce7"),
            ("courses-and-reading", "课程与读书", "#4e8d78"),
            ("personal-projects", "个人项目", "#b4784d"),
            ("career-growth", "求职与成长", "#5576b8"),
        )
        db.executemany(
            "INSERT OR IGNORE INTO spaces(id,name,color,created_at,updated_at) VALUES(?,?,?,?,?)",
            [(space_id, name, color, stamp, stamp) for space_id, name, color in defaults],
        )


def rows(query: str, params: tuple = ()) -> list[dict]:
    with connect() as db:
        return [dict(item) for item in db.execute(query, params).fetchall()]


def json_value(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
