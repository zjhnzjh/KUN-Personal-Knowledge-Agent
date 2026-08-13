from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import httpx
import numpy as np

from .config import get_settings
from .database import connect, json_value, now, rows
from .infra import JobContext, create_trace, finish_trace, trace_span
from .privacy import allowed_for_cloud
from .rag import chunk_sections, fts_query, lexical_text, parse_document


PARSER_VERSION = "kun-parser-v1"
CHUNKER_VERSION = "kun-character-v1"


class CapabilityUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class RetrievalConfig:
    pipeline: str = "hybrid"
    generation_id: str | None = None
    candidate_k: int = 20
    top_k: int = 5
    reranker_top_n: int = 10
    rrf_k: int = 60
    reranker_model: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RetrievalConfig":
        pipeline = str(value.get("pipeline", "hybrid"))
        if pipeline not in {"bm25", "dense", "hybrid", "hybrid_rerank"}:
            raise ValueError("Unsupported retrieval pipeline")
        return cls(
            pipeline=pipeline,
            generation_id=value.get("generation_id"),
            candidate_k=max(5, min(int(value.get("candidate_k", 20)), 100)),
            top_k=max(1, min(int(value.get("top_k", 5)), 20)),
            reranker_top_n=max(10, min(int(value.get("reranker_top_n", 10)), 50)),
            rrf_k=max(1, min(int(value.get("rrf_k", 60)), 200)),
            reranker_model=value.get("reranker_model") or ("qwen3-rerank" if pipeline == "hybrid_rerank" else None),
        )


class DashScopeEmbeddingAdapter:
    provider = "dashscope"

    def __init__(self, model: str, dimension: int) -> None:
        self.settings = get_settings()
        self.model = model
        self.dimension = dimension

    @property
    def available(self) -> bool:
        return bool(self.settings.dashscope_api_key)

    def encode(self, texts: list[str], text_type: str) -> tuple[list[list[float]], dict[str, Any]]:
        if not self.available:
            raise CapabilityUnavailable("百炼 Embedding 未配置")
        input_characters = sum(len(text) for text in texts)
        query_hash = hashlib.sha256(texts[0].encode("utf-8")).hexdigest() if text_type == "query" and len(texts) == 1 else None
        if query_hash:
            cached = rows(
                "SELECT embedding_json FROM eval_query_embedding_cache WHERE query_hash=? AND provider=? AND model=? AND dimension=? AND text_type=?",
                (query_hash, self.provider, self.model, self.dimension, text_type),
            )
            if cached:
                return [json.loads(cached[0]["embedding_json"])], {
                    "provider": self.provider,
                    "model": self.model,
                    "dimension": self.dimension,
                    "input_count": len(texts),
                    "input_characters": input_characters,
                    "usage": {},
                    "usage_source": "cache",
                    "cache_hit": True,
                    "request_count": 0,
                    "provider_duration_ms": 0,
                    "text_type": text_type,
                }
        last_error: Exception | None = None
        provider_started = perf_counter()
        for attempt in range(3):
            try:
                response = httpx.post(
                    f"{self.settings.dashscope_base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.settings.dashscope_api_key}"},
                    json={"model": self.model, "input": texts, "dimensions": self.dimension},
                    timeout=90,
                )
                response.raise_for_status()
                payload = response.json()
                ordered = sorted(payload["data"], key=lambda item: item["index"])
                vectors = [item["embedding"] for item in ordered]
                if any(len(vector) != self.dimension for vector in vectors):
                    raise ValueError("Embedding dimension does not match the requested index dimension")
                usage = payload.get("usage") or {}
                if query_hash:
                    with connect() as db:
                        db.execute(
                            "INSERT OR REPLACE INTO eval_query_embedding_cache(query_hash,provider,model,dimension,text_type,embedding_json,created_at) VALUES(?,?,?,?,?,?,?)",
                            (query_hash, self.provider, self.model, self.dimension, text_type, json_value(vectors[0]), now()),
                        )
                return vectors, {
                    "provider": self.provider,
                    "model": self.model,
                    "dimension": self.dimension,
                    "input_count": len(texts),
                    "input_characters": input_characters,
                    "usage": usage,
                    "usage_source": "provider" if usage else "unavailable",
                    "cache_hit": False,
                    "request_count": 1,
                    "provider_duration_ms": round((perf_counter() - provider_started) * 1000, 2),
                    "text_type": text_type,
                }
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
                last_error = error
                if attempt < 2:
                    import time
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError("百炼 Embedding 连续三次请求失败") from last_error


class DashScopeRerankerAdapter:
    provider = "dashscope"

    def __init__(self, model: str = "qwen3-rerank") -> None:
        self.settings = get_settings()
        self.model = model

    @property
    def available(self) -> bool:
        return bool(self.settings.dashscope_api_key and self.settings.dashscope_rerank_base_url)

    def rerank(self, query: str, candidates: list[dict], top_n: int) -> tuple[list[dict], dict[str, Any]]:
        if not self.available:
            raise CapabilityUnavailable("百炼 Reranker 地址尚未配置")
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        candidate_hash = hashlib.sha256(
            json_value([str(item.get("id")) for item in candidates]).encode("utf-8")
        ).hexdigest()
        bounded_top_n = min(top_n, len(candidates))
        cached = rows(
            "SELECT result_json FROM eval_rerank_cache WHERE query_hash=? AND candidate_hash=? AND model=? AND top_n=?",
            (query_hash, candidate_hash, self.model, bounded_top_n),
        )
        input_characters = sum(len(str(item.get("text", ""))) for item in candidates)
        if cached:
            cached_results = json.loads(cached[0]["result_json"] or "[]")
            ranked = []
            for rank, result in enumerate(cached_results, 1):
                index = int(result["index"])
                if 0 <= index < len(candidates):
                    ranked.append({
                        **candidates[index],
                        "rerank_rank": rank,
                        "rerank_score": round(float(result.get("relevance_score", 0)), 6),
                    })
            return ranked, {
                "provider": self.provider,
                "model": self.model,
                "input_count": len(candidates),
                "output_count": len(ranked),
                "input_characters": input_characters,
                "usage": {},
                "cache_hit": True,
                "request_count": 0,
                "provider_duration_ms": 0,
            }
        provider_started = perf_counter()
        response = httpx.post(
            f"{self.settings.dashscope_rerank_base_url}/reranks",
            headers={"Authorization": f"Bearer {self.settings.dashscope_api_key}"},
            json={
                "model": self.model,
                "query": query,
                "documents": [item["text"] for item in candidates],
                "top_n": bounded_top_n,
                "instruct": "Given a search query, retrieve passages that directly answer the query.",
            },
            timeout=90,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results") or []
        with connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO eval_rerank_cache(query_hash,candidate_hash,model,top_n,result_json,created_at) VALUES(?,?,?,?,?,?)",
                (query_hash, candidate_hash, self.model, bounded_top_n, json_value(results), now()),
            )
        ranked: list[dict] = []
        for rank, result in enumerate(results, 1):
            index = int(result["index"])
            if index < 0 or index >= len(candidates):
                continue
            ranked.append({
                **candidates[index],
                "rerank_rank": rank,
                "rerank_score": round(float(result.get("relevance_score", 0)), 6),
            })
        return ranked, {
            "provider": self.provider,
            "model": self.model,
            "input_count": len(candidates),
            "output_count": len(ranked),
            "usage": payload.get("usage") or {},
            "input_characters": input_characters,
            "cache_hit": False,
            "request_count": 1,
            "provider_duration_ms": round((perf_counter() - provider_started) * 1000, 2),
        }


def _normalise(matrix: np.ndarray) -> np.ndarray:
    matrix = np.ascontiguousarray(matrix, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def _generation_folder(generation_id: str, space_id: str) -> Path:
    safe_space = "".join(character for character in space_id if character.isalnum() or character in "-_")[:80]
    folder = get_settings().data_dir / "indexes" / safe_space / generation_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _source_documents(space_id: str) -> list[dict]:
    return rows(
        """SELECT id,library_path,fingerprint,title,original_name FROM documents
           WHERE space_id=? ORDER BY id""",
        (space_id,),
    )


def _generation_chunks(generation: dict, *, persist: bool) -> list[dict]:
    chunks: list[dict] = []
    for document in _source_documents(generation["space_id"]):
        path = Path(document["library_path"])
        if not path.is_file():
            raise ValueError(f"Document copy is missing: {document['original_name']}")
        parsed = chunk_sections(
            parse_document(path),
            size=int(generation["chunk_size"]),
            overlap=int(generation["chunk_overlap"]),
        )
        for ordinal, chunk in enumerate(parsed):
            content_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
            chunk_id = hashlib.sha1(
                f"{generation['id']}:{document['id']}:{ordinal}:{content_hash}".encode("utf-8")
            ).hexdigest()[:24]
            chunks.append({
                "id": chunk_id,
                "document_id": document["id"],
                "ordinal": ordinal,
                "locator": chunk.locator,
                "heading": chunk.heading,
                "text": chunk.text,
                "content_hash": content_hash,
                "title": document["title"],
                "original_name": document["original_name"],
            })
    if persist:
        with connect() as db:
            db.execute("DELETE FROM index_generation_chunks_fts WHERE generation_id=?", (generation["id"],))
            db.execute("DELETE FROM index_generation_chunks WHERE generation_id=?", (generation["id"],))
            db.executemany(
                """INSERT INTO index_generation_chunks(
                   id,generation_id,document_id,ordinal,locator,heading,text,content_hash
                   ) VALUES(?,?,?,?,?,?,?,?)""",
                [
                    (
                        item["id"], generation["id"], item["document_id"], item["ordinal"],
                        item["locator"], item["heading"], item["text"], item["content_hash"],
                    )
                    for item in chunks
                ],
            )
            db.executemany(
                "INSERT INTO index_generation_chunks_fts(generation_id,chunk_id,text) VALUES(?,?,?)",
                [(generation["id"], item["id"], lexical_text(item["text"])) for item in chunks],
            )
    return chunks


def create_index_generation(
    *,
    space_id: str,
    model: str,
    dimension: int,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
) -> dict:
    if strategy not in {"flat", "hnsw"}:
        raise ValueError("Index strategy must be flat or hnsw")
    if dimension not in {256, 512, 768, 1024, 1536, 2048, 2560}:
        raise ValueError("Unsupported embedding dimension")
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size")
    documents = _source_documents(space_id)
    config = {
        "space_id": space_id,
        "provider": "dashscope",
        "model": model,
        "dimension": dimension,
        "strategy": strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "parser_version": PARSER_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "source_fingerprints": [item["fingerprint"] for item in documents],
    }
    config_hash = hashlib.sha256(json_value(config).encode("utf-8")).hexdigest()
    existing = rows(
        "SELECT * FROM index_generations WHERE config_hash=? AND status IN ('building','ready') ORDER BY created_at DESC LIMIT 1",
        (config_hash,),
    )
    if existing:
        return existing[0]
    generation_id = uuid4().hex
    stamp = now()
    with connect() as db:
        space = db.execute("SELECT id FROM spaces WHERE id=?", (space_id,)).fetchone()
        if not space:
            raise ValueError("Knowledge space does not exist")
        db.execute(
            """INSERT INTO index_generations(
               id,space_id,status,provider,model,dimension,strategy,chunk_size,chunk_overlap,
               parser_version,chunker_version,config_hash,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                generation_id, space_id, "building", "dashscope", model, dimension, strategy,
                chunk_size, chunk_overlap, PARSER_VERSION, CHUNKER_VERSION, config_hash, stamp, stamp,
            ),
        )
    return rows("SELECT * FROM index_generations WHERE id=?", (generation_id,))[0]


def estimate_index_generation(generation_id: str) -> dict:
    matches = rows("SELECT * FROM index_generations WHERE id=?", (generation_id,))
    if not matches:
        raise ValueError("Index generation does not exist")
    generation = matches[0]
    chunks = _generation_chunks(generation, persist=False)
    document_ids = list(dict.fromkeys(item["document_id"] for item in chunks))
    allowed_documents = allowed_for_cloud(document_ids, "embedding")
    blocked_documents = set(document_ids) - allowed_documents
    chunks = [item for item in chunks if item["document_id"] in allowed_documents]
    hits = 0
    for chunk in chunks:
        cached = rows(
            """SELECT 1 found FROM generation_embedding_vectors WHERE content_hash=? AND provider=? AND model=?
               AND dimension=?""",
            (chunk["content_hash"], generation["provider"], generation["model"], generation["dimension"]),
        )
        hits += int(bool(cached))
    missing = len(chunks) - hits
    batch_size = 20 if generation["model"] == "qwen3.7-text-embedding" else 10
    return {
        "generation_id": generation_id,
        "chunk_count": len(chunks),
        "source_document_count": len(document_ids),
        "allowed_document_count": len(allowed_documents),
        "blocked_document_count": len(blocked_documents),
        "cache_hits": hits,
        "cache_misses": missing,
        "cache_hit_rate": round(hits / len(chunks), 4) if chunks else 0,
        "estimated_batches": math.ceil(missing / batch_size) if missing else 0,
        "estimated_input_characters": sum(
            len(item["text"])
            for item in chunks
            if not rows(
                """SELECT 1 found FROM generation_embedding_vectors
                   WHERE content_hash=? AND provider=? AND model=? AND dimension=?""",
                (item["content_hash"], generation["provider"], generation["model"], generation["dimension"]),
            )
        ),
        "cost_status": "estimated",
        "requires_confirmation": missing > 0,
    }


def get_index_generation(generation_id: str) -> dict | None:
    matches = rows("SELECT * FROM index_generations WHERE id=?", (generation_id,))
    if not matches:
        return None
    generation = matches[0]
    generation["estimate"] = estimate_index_generation(generation_id)
    return generation


def build_index_generation(generation_id: str, context: JobContext) -> dict[str, Any]:
    import faiss

    matches = rows("SELECT * FROM index_generations WHERE id=?", (generation_id,))
    if not matches:
        raise ValueError("Index generation does not exist")
    generation = matches[0]
    started = perf_counter()
    trace_id = create_trace("index", "build_index_generation", {
        "generation_id": generation_id,
        "space_id": generation["space_id"],
        "model": generation["model"],
        "dimension": generation["dimension"],
        "strategy": generation["strategy"],
    })
    try:
        context.update(progress=2, phase="loading_chunks", message="正在读取当前知识空间的 Chunk")
        with trace_span(trace_id, "load_chunks", "index_build") as span:
            all_chunks = _generation_chunks(generation, persist=False)
            document_ids = list(dict.fromkeys(item["document_id"] for item in all_chunks))
            allowed_documents = allowed_for_cloud(document_ids, "embedding")
            chunks = [item for item in all_chunks if item["document_id"] in allowed_documents]
            if not chunks:
                raise CapabilityUnavailable("No document in this space is allowed for cloud Embedding")
            # Persist only the explicitly allowed corpus into this cloud-backed generation.
            with connect() as db:
                db.execute("DELETE FROM index_generation_chunks_fts WHERE generation_id=?", (generation["id"],))
                db.execute("DELETE FROM index_generation_chunks WHERE generation_id=?", (generation["id"],))
                db.executemany(
                    """INSERT INTO index_generation_chunks(
                       id,generation_id,document_id,ordinal,locator,heading,text,content_hash
                       ) VALUES(?,?,?,?,?,?,?,?)""",
                    [
                        (item["id"], generation["id"], item["document_id"], item["ordinal"], item["locator"],
                         item["heading"], item["text"], item["content_hash"])
                        for item in chunks
                    ],
                )
                db.executemany(
                    "INSERT INTO index_generation_chunks_fts(generation_id,chunk_id,text) VALUES(?,?,?)",
                    [(generation["id"], item["id"], lexical_text(item["text"])) for item in chunks],
                )
            span.annotate(output_count=len(chunks))
        if not chunks:
            raise ValueError("Current knowledge space has no chunks")
        context.check_cancelled()
        adapter = DashScopeEmbeddingAdapter(generation["model"], generation["dimension"])
        vectors_by_chunk: dict[str, list[float]] = {}
        misses: list[dict] = []
        with trace_span(trace_id, "embedding_cache_lookup", "cache") as span:
            for chunk in chunks:
                match = rows(
                    """SELECT embedding_json FROM generation_embedding_vectors WHERE content_hash=?
                       AND provider=? AND model=? AND dimension=?""",
                    (chunk["content_hash"], generation["provider"], generation["model"], generation["dimension"]),
                )
                if match:
                    vectors_by_chunk[chunk["id"]] = json.loads(match[0]["embedding_json"])
                else:
                    misses.append(chunk)
            span.annotate(cache_hits=len(vectors_by_chunk), cache_misses=len(misses), output_count=len(vectors_by_chunk))
        batch_size = 20 if generation["model"] == "qwen3.7-text-embedding" else 10
        for start in range(0, len(misses), batch_size):
            context.check_cancelled()
            batch = misses[start:start + batch_size]
            context.update(
                progress=5 + int(start / max(len(misses), 1) * 70),
                phase="embedding",
                message=f"正在生成向量 {min(start + len(batch), len(misses))} / {len(misses)}",
            )
            with trace_span(trace_id, "embed_documents", "embedding", attributes={
                "provider": "dashscope", "model": generation["model"], "dimension": generation["dimension"],
            }) as span:
                encoded, usage = adapter.encode([item["text"] for item in batch], "document")
                span.annotate(**usage)
            with connect() as db:
                for item, vector in zip(batch, encoded):
                    vectors_by_chunk[item["id"]] = vector
                    db.execute(
                        """INSERT OR REPLACE INTO generation_embedding_vectors(
                           content_hash,provider,model,dimension,embedding_json,created_at
                           ) VALUES(?,?,?,?,?,?)""",
                        (
                            item["content_hash"], generation["provider"], generation["model"],
                            generation["dimension"], json_value(vector), now(),
                        ),
                    )
        context.check_cancelled()
        context.update(progress=80, phase="faiss_build", message="正在构建 FAISS 索引")
        ordered_vectors = _normalise(np.asarray([vectors_by_chunk[item["id"]] for item in chunks], dtype="float32"))
        vector_ids = np.arange(len(chunks), dtype="int64")
        with trace_span(trace_id, "faiss_build", "index_build", attributes={"strategy": generation["strategy"]}) as span:
            if generation["strategy"] == "hnsw":
                base = faiss.IndexHNSWFlat(generation["dimension"], 32, faiss.METRIC_INNER_PRODUCT)
                base.hnsw.efConstruction = 80
                base.hnsw.efSearch = 64
            else:
                base = faiss.IndexFlatIP(generation["dimension"])
            index = faiss.IndexIDMap2(base)
            index.add_with_ids(ordered_vectors, vector_ids)
            span.annotate(vector_count=len(chunks), dimension=generation["dimension"])
        folder = _generation_folder(generation_id, generation["space_id"])
        index_path = folder / "index.faiss"
        temp_path = folder / "index.faiss.tmp"
        faiss.write_index(index, str(temp_path))
        os.replace(temp_path, index_path)
        manifest = {
            "format": "kun-index-generation",
            "version": 1,
            "generation_id": generation_id,
            "space_id": generation["space_id"],
            "provider": generation["provider"],
            "model": generation["model"],
            "dimension": generation["dimension"],
            "strategy": generation["strategy"],
            "parser_version": generation["parser_version"],
            "chunker_version": generation["chunker_version"],
            "chunk_size": generation["chunk_size"],
            "chunk_overlap": generation["chunk_overlap"],
            "config_hash": generation["config_hash"],
            "vector_count": len(chunks),
            "created_at": now(),
            "document_fingerprints": [item["fingerprint"] for item in rows(
                "SELECT fingerprint FROM documents WHERE space_id=? ORDER BY id", (generation["space_id"],)
            )],
        }
        manifest_path = folder / "manifest.json"
        manifest_temp = folder / "manifest.json.tmp"
        manifest_temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(manifest_temp, manifest_path)
        with connect() as db:
            db.execute("DELETE FROM index_generation_items WHERE generation_id=?", (generation_id,))
            db.executemany(
                "INSERT INTO index_generation_items(generation_id,vector_id,chunk_id) VALUES(?,?,?)",
                [(generation_id, index, item["id"]) for index, item in enumerate(chunks)],
            )
            db.execute(
                """UPDATE index_generations SET status='ready',manifest_path=?,vector_count=?,index_bytes=?,
                   error_code=NULL,updated_at=? WHERE id=?""",
                (str(manifest_path), len(chunks), index_path.stat().st_size, now(), generation_id),
            )
        context.update(progress=100, phase="ready", message="索引代次已构建并验证")
        duration_ms = round((perf_counter() - started) * 1000)
        finish_trace(trace_id, duration_ms=duration_ms, attributes={**manifest, "index_bytes": index_path.stat().st_size})
        return {
            "generation_id": generation_id,
            "vector_count": len(chunks),
            "index_bytes": index_path.stat().st_size,
            "trace_id": trace_id,
            "cache_misses": len(misses),
        }
    except Exception as error:
        with connect() as db:
            db.execute(
                "UPDATE index_generations SET status='failed',error_code=?,updated_at=? WHERE id=?",
                (type(error).__name__, now(), generation_id),
            )
        finish_trace(
            trace_id,
            "failed",
            duration_ms=round((perf_counter() - started) * 1000),
            error_code=type(error).__name__,
        )
        raise


def _validated_manifest(generation: dict) -> tuple[dict, Path]:
    manifest_path = Path(generation["manifest_path"] or "")
    index_path = manifest_path.parent / "index.faiss"
    if not manifest_path.is_file() or not index_path.is_file():
        raise ValueError("Index files are missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Index manifest is unreadable") from error
    expected = {
        "generation_id": generation["id"],
        "space_id": generation["space_id"],
        "model": generation["model"],
        "dimension": generation["dimension"],
        "strategy": generation["strategy"],
        "config_hash": generation["config_hash"],
        "vector_count": generation["vector_count"],
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Index manifest does not match the database generation")
    return manifest, index_path


def activate_generation(generation_id: str) -> dict:
    matches = rows("SELECT * FROM index_generations WHERE id=?", (generation_id,))
    if not matches:
        raise ValueError("Index generation does not exist")
    generation = matches[0]
    if generation["status"] != "ready":
        raise ValueError("Only a ready index generation can be activated")
    _validated_manifest(generation)
    with connect() as db:
        db.execute("UPDATE index_generations SET is_active=0,updated_at=? WHERE space_id=?", (now(), generation["space_id"]))
        db.execute(
            "UPDATE index_generations SET is_active=1,activated_at=?,updated_at=? WHERE id=?",
            (now(), now(), generation_id),
        )
    return rows("SELECT * FROM index_generations WHERE id=?", (generation_id,))[0]


def _load_generation(generation_id: str, space_id: str) -> tuple[dict, Any]:
    import faiss

    matches = rows(
        "SELECT * FROM index_generations WHERE id=? AND space_id=? AND status='ready'",
        (generation_id, space_id),
    )
    if not matches:
        raise CapabilityUnavailable("所选向量索引尚未就绪")
    generation = matches[0]
    try:
        _, index_path = _validated_manifest(generation)
        index = faiss.read_index(str(index_path))
    except (ValueError, RuntimeError) as error:
        raise CapabilityUnavailable(f"索引验证失败，需要重新构建：{error}") from error
    if index.d != generation["dimension"] or index.ntotal != generation["vector_count"]:
        raise CapabilityUnavailable("FAISS 索引维度或向量数量与代次清单不一致")
    return generation, index


def _lexical_candidates(query: str, space_id: str, limit: int, generation_id: str | None = None) -> list[dict]:
    expression = fts_query(query)
    if not expression:
        return []
    if generation_id:
        return rows(
            """SELECT c.id,c.document_id,c.locator,c.heading,c.text,d.title,d.original_name,
                      bm25(index_generation_chunks_fts) AS lexical_score
               FROM index_generation_chunks_fts
               JOIN index_generation_chunks c ON c.id=index_generation_chunks_fts.chunk_id
               JOIN documents d ON d.id=c.document_id
               WHERE index_generation_chunks_fts MATCH ? AND index_generation_chunks_fts.generation_id=?
                 AND d.space_id=? ORDER BY lexical_score LIMIT ?""",
            (expression, generation_id, space_id, limit),
        )
    return rows(
        """SELECT c.id,c.document_id,c.locator,c.heading,c.text,d.title,d.original_name,
                  bm25(chunks_fts) AS lexical_score
           FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.chunk_id
           JOIN documents d ON d.id=c.document_id
           WHERE chunks_fts MATCH ? AND d.space_id=? ORDER BY lexical_score LIMIT ?""",
        (expression, space_id, limit),
    )


def _dense_candidates(query: str, space_id: str, generation_id: str, limit: int) -> tuple[list[dict], dict]:
    generation, index = _load_generation(generation_id, space_id)
    adapter = DashScopeEmbeddingAdapter(generation["model"], generation["dimension"])
    vectors, usage = adapter.encode([query], "query")
    query_vector = _normalise(np.asarray(vectors, dtype="float32"))
    scores, identifiers = index.search(query_vector, limit)
    vector_ids = [int(item) for item in identifiers[0] if int(item) >= 0]
    if not vector_ids:
        return [], usage
    mapping_rows = rows(
        f"""SELECT i.vector_id,c.id,c.document_id,c.locator,c.heading,c.text,d.title,d.original_name
            FROM index_generation_items i JOIN index_generation_chunks c ON c.id=i.chunk_id
            JOIN documents d ON d.id=c.document_id
            WHERE i.generation_id=? AND i.vector_id IN ({','.join('?' for _ in vector_ids)})""",
        (generation_id, *vector_ids),
    )
    by_id = {int(item["vector_id"]): item for item in mapping_rows}
    candidates = []
    for vector_id, score in zip(identifiers[0], scores[0]):
        if int(vector_id) not in by_id:
            continue
        candidates.append({**by_id[int(vector_id)], "vector_score": round(float(score), 6)})
    return candidates, usage


def pipeline_search(
    query: str,
    space_id: str,
    config_value: dict[str, Any],
    *,
    trace_type: str = "retrieval",
) -> dict[str, Any]:
    config = RetrievalConfig.from_dict(config_value)
    trace_id = create_trace(trace_type, "retrieval_pipeline", {"space_id": space_id, **asdict(config)})
    started = perf_counter()
    stage_details: list[dict[str, Any]] = []
    latency_breakdown: dict[str, float] = {
        "local_bm25_ms": 0,
        "local_faiss_ms": 0,
        "embedding_provider_ms": 0,
        "local_rrf_ms": 0,
        "rerank_provider_ms": 0,
        "total_ms": 0,
    }
    embedding_usage: dict[str, Any] = {}
    rerank_usage: dict[str, Any] = {}

    def snapshot(items: list[dict], rank_field: str | None = None) -> list[dict]:
        result: list[dict] = []
        for rank, item in enumerate(items, 1):
            result.append({
                "chunk_id": item.get("id"),
                "document_id": item.get("document_id"),
                "locator": item.get("locator"),
                "rank": int(item.get(rank_field, rank)) if rank_field and item.get(rank_field) else rank,
                "score": item.get("score") or item.get("vector_score") or item.get("rerank_score"),
            })
        return result

    try:
        lexical: list[dict] = []
        dense: list[dict] = []
        if config.pipeline in {"bm25", "hybrid", "hybrid_rerank"}:
            stage_started = perf_counter()
            with trace_span(trace_id, "bm25_search", "bm25_search", attributes={"candidate_k": config.candidate_k}) as span:
                lexical = _lexical_candidates(query, space_id, config.candidate_k, config.generation_id)
                span.annotate(output_count=len(lexical))
            latency_breakdown["local_bm25_ms"] = round((perf_counter() - stage_started) * 1000, 2)
            stage_details.append({"stage": "bm25", "duration_ms": latency_breakdown["local_bm25_ms"], "count": len(lexical), "cache_hit": False})
        if config.pipeline in {"dense", "hybrid", "hybrid_rerank"}:
            if not config.generation_id:
                raise CapabilityUnavailable("Dense pipeline requires an index generation")
            stage_started = perf_counter()
            with trace_span(trace_id, "vector_search", "vector_search", attributes={"candidate_k": config.candidate_k}) as span:
                dense, embedding_usage = _dense_candidates(query, space_id, config.generation_id, config.candidate_k)
                span.annotate(output_count=len(dense), **embedding_usage)
            dense_total_ms = round((perf_counter() - stage_started) * 1000, 2)
            latency_breakdown["embedding_provider_ms"] = float(embedding_usage.get("provider_duration_ms") or 0)
            latency_breakdown["local_faiss_ms"] = round(max(0, dense_total_ms - latency_breakdown["embedding_provider_ms"]), 2)
            stage_details.append({
                "stage": "dense", "duration_ms": dense_total_ms, "count": len(dense),
                "cache_hit": bool(embedding_usage.get("cache_hit")),
                "provider_duration_ms": latency_breakdown["embedding_provider_ms"],
            })
        candidates: dict[str, dict] = {}
        for rank, item in enumerate(lexical, 1):
            candidates[item["id"]] = {**item, "lexical_rank": rank}
        for rank, item in enumerate(dense, 1):
            candidate = candidates.setdefault(item["id"], dict(item))
            candidate["vector_rank"] = rank
            candidate["vector_score"] = item.get("vector_score")
        stage_started = perf_counter()
        with trace_span(trace_id, "rrf_fusion", "fusion", attributes={"rrf_k": config.rrf_k}) as span:
            fused: list[dict] = []
            for item in candidates.values():
                rrf_score = sum(
                    1 / (config.rrf_k + int(item[key]))
                    for key in ("lexical_rank", "vector_rank")
                    if key in item
                )
                fused.append({**item, "score": round(rrf_score, 8)})
            fused.sort(key=lambda item: item["score"], reverse=True)
            for rank, item in enumerate(fused, 1):
                item["fusion_rank"] = rank
            span.annotate(input_count=len(candidates), output_count=len(fused))
        latency_breakdown["local_rrf_ms"] = round((perf_counter() - stage_started) * 1000, 2)
        stage_details.append({"stage": "fusion", "duration_ms": latency_breakdown["local_rrf_ms"], "count": len(fused), "cache_hit": False})
        fusion_snapshot = snapshot(fused, "fusion_rank")
        if config.reranker_model:
            stage_started = perf_counter()
            reranker = DashScopeRerankerAdapter(config.reranker_model)
            with trace_span(trace_id, "rerank", "rerank", attributes={"model": config.reranker_model}) as span:
                fused, rerank_usage = reranker.rerank(query, fused[:config.candidate_k], config.reranker_top_n)
                span.annotate(**rerank_usage)
            rerank_total_ms = round((perf_counter() - stage_started) * 1000, 2)
            latency_breakdown["rerank_provider_ms"] = float(rerank_usage.get("provider_duration_ms") or 0)
            stage_details.append({
                "stage": "rerank", "duration_ms": rerank_total_ms, "count": len(fused),
                "cache_hit": bool(rerank_usage.get("cache_hit")),
                "provider_duration_ms": latency_breakdown["rerank_provider_ms"],
            })
        evaluation_results: list[dict] = []
        per_document: dict[str, int] = {}
        for item in fused:
            document_id = item["document_id"]
            if per_document.get(document_id, 0) >= 2:
                continue
            evaluation_results.append(item)
            per_document[document_id] = per_document.get(document_id, 0) + 1
            if len(evaluation_results) >= max(10, config.top_k):
                break
        results = evaluation_results[:config.top_k]
        final_snapshot = snapshot(results, "rerank_rank" if config.reranker_model else "fusion_rank")
        evaluation_snapshot = snapshot(evaluation_results, "rerank_rank" if config.reranker_model else "fusion_rank")
        duration_ms = round((perf_counter() - started) * 1000, 2)
        latency_breakdown["total_ms"] = duration_ms
        finish_trace(trace_id, duration_ms=round(duration_ms), attributes={
            "space_id": space_id, **asdict(config), "result_count": len(results), "stage_count": len(stage_details),
            "latency_breakdown": latency_breakdown,
        })
        return {
            "trace_id": trace_id,
            "duration_ms": duration_ms,
            "stages": stage_details,
            "stage_results": {
                "bm25": snapshot(lexical, "lexical_rank"),
                "dense": snapshot(dense, "vector_rank"),
                "fusion": fusion_snapshot,
                "rerank": snapshot(fused, "rerank_rank") if config.reranker_model else [],
                "evaluation": evaluation_snapshot,
                "final": final_snapshot,
            },
            "latency_breakdown": latency_breakdown,
            "api_stats": {
                "embedding_requests": int(embedding_usage.get("request_count") or 0),
                "rerank_requests": int(rerank_usage.get("request_count") or 0),
                "embedding_cache_hits": int(bool(embedding_usage.get("cache_hit"))),
                "rerank_cache_hits": int(bool(rerank_usage.get("cache_hit"))),
                "embedding_input_characters": int(embedding_usage.get("input_characters") or 0),
                "rerank_input_characters": int(rerank_usage.get("input_characters") or 0),
            },
            "results": results,
            "config": asdict(config),
        }
    except Exception as error:
        finish_trace(
            trace_id,
            "failed",
            duration_ms=round((perf_counter() - started) * 1000),
            error_code=type(error).__name__,
        )
        raise


def list_index_generations(space_id: str | None = None) -> list[dict]:
    if space_id:
        return rows("SELECT * FROM index_generations WHERE space_id=? ORDER BY created_at DESC", (space_id,))
    return rows("SELECT * FROM index_generations ORDER BY created_at DESC")
