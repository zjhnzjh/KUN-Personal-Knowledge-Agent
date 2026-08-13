from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable, Iterator
from uuid import uuid4

from .database import connect, json_value, now, rows


def _safe_attributes(value: dict[str, Any] | None) -> dict[str, Any]:
    """Keep traces compact and prevent common secret fields from being persisted."""
    safe: dict[str, Any] = {}
    for key, item in (value or {}).items():
        lowered = key.lower()
        if any(token in lowered for token in ("api_key", "authorization", "credential", "password", "secret")):
            safe[key] = "[REDACTED]"
        elif isinstance(item, str):
            safe[key] = item[:500]
        elif isinstance(item, (int, float, bool)) or item is None:
            safe[key] = item
        elif isinstance(item, list):
            safe[key] = item[:30]
        elif isinstance(item, dict):
            safe[key] = _safe_attributes(item)
        else:
            safe[key] = str(item)[:500]
    return safe


def create_trace(trace_type: str, name: str, attributes: dict[str, Any] | None = None) -> str:
    trace_id = uuid4().hex
    with connect() as db:
        db.execute(
            """INSERT INTO infra_traces(id,trace_type,name,status,root_attributes_json,started_at)
               VALUES(?,?,?,?,?,?)""",
            (trace_id, trace_type, name, "running", json_value(_safe_attributes(attributes)), now()),
        )
    return trace_id


def finish_trace(
    trace_id: str,
    status: str = "succeeded",
    *,
    duration_ms: int | None = None,
    error_code: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> None:
    with connect() as db:
        if attributes is not None:
            db.execute(
                """UPDATE infra_traces SET status=?,finished_at=?,duration_ms=?,error_code=?,root_attributes_json=?
                   WHERE id=?""",
                (status, now(), duration_ms, error_code, json_value(_safe_attributes(attributes)), trace_id),
            )
        else:
            db.execute(
                "UPDATE infra_traces SET status=?,finished_at=?,duration_ms=?,error_code=? WHERE id=?",
                (status, now(), duration_ms, error_code, trace_id),
            )


@dataclass
class SpanHandle:
    trace_id: str
    span_id: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def annotate(self, **attributes: Any) -> None:
        self.attributes.update(attributes)


@contextmanager
def trace_span(
    trace_id: str,
    operation: str,
    kind: str,
    *,
    parent_span_id: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[SpanHandle]:
    span_id = uuid4().hex
    started = perf_counter()
    started_at = now()
    handle = SpanHandle(trace_id, span_id, dict(attributes or {}))
    with connect() as db:
        db.execute(
            """INSERT INTO infra_spans(
               id,trace_id,parent_span_id,operation,kind,status,attributes_json,started_at
               ) VALUES(?,?,?,?,?,?,?,?)""",
            (span_id, trace_id, parent_span_id, operation, kind, "running", "{}", started_at),
        )
    try:
        yield handle
    except Exception as error:
        with connect() as db:
            db.execute(
                """UPDATE infra_spans SET status='failed',finished_at=?,duration_ms=?,attributes_json=?,error_code=?
                   WHERE id=?""",
                (
                    now(),
                    round((perf_counter() - started) * 1000, 2),
                    json_value(_safe_attributes(handle.attributes)),
                    type(error).__name__,
                    span_id,
                ),
            )
        raise
    else:
        with connect() as db:
            db.execute(
                """UPDATE infra_spans SET status='succeeded',finished_at=?,duration_ms=?,attributes_json=?
                   WHERE id=?""",
                (
                    now(),
                    round((perf_counter() - started) * 1000, 2),
                    json_value(_safe_attributes(handle.attributes)),
                    span_id,
                ),
            )


def list_traces(limit: int = 50, trace_type: str | None = None, status: str | None = None) -> list[dict]:
    clauses: list[str] = []
    params: list[Any] = []
    if trace_type:
        clauses.append("trace_type=?")
        params.append(trace_type)
    if status:
        clauses.append("status=?")
        params.append(status)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    traces = rows(
        f"SELECT * FROM infra_traces {where} ORDER BY started_at DESC LIMIT ?",
        (*params, max(1, min(limit, 200))),
    )
    for trace in traces:
        trace["attributes"] = json.loads(trace.pop("root_attributes_json") or "{}")
        counts = rows(
            """SELECT COUNT(*) span_count,
               SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) failed_span_count
               FROM infra_spans WHERE trace_id=?""",
            (trace["id"],),
        )[0]
        trace.update(counts)
    return traces


def trace_detail(trace_id: str) -> dict | None:
    matches = rows("SELECT * FROM infra_traces WHERE id=?", (trace_id,))
    if not matches:
        return None
    trace = matches[0]
    trace["attributes"] = json.loads(trace.pop("root_attributes_json") or "{}")
    spans = rows("SELECT * FROM infra_spans WHERE trace_id=? ORDER BY started_at,id", (trace_id,))
    for span in spans:
        span["attributes"] = json.loads(span.pop("attributes_json") or "{}")
    trace["spans"] = spans
    return trace


class JobCancelled(RuntimeError):
    pass


@dataclass
class JobContext:
    job_id: str
    runner: "LocalJobRunner"

    def update(self, *, progress: int | None = None, phase: str | None = None, message: str | None = None) -> None:
        self.runner.update(self.job_id, progress=progress, phase=phase, message=message)

    def check_cancelled(self) -> None:
        states = rows("SELECT status FROM infra_jobs WHERE id=?", (self.job_id,))
        if states and states[0]["status"] == "cancel_requested":
            raise JobCancelled("The job was cancelled by the user.")


JobHandler = Callable[[dict[str, Any], JobContext], dict[str, Any] | None]


class LocalJobRunner:
    """A small persisted runner for this single-user local application.

    SQLite owns job state. Threads only execute claimed jobs, so restart recovery
    and UI state never depend on an in-memory Future object.
    """

    def __init__(self, max_workers: int = 2) -> None:
        self.max_workers = max_workers
        self.worker_id = f"local-{uuid4().hex[:8]}"
        self.handlers: dict[str, JobHandler] = {}
        self.executor: ThreadPoolExecutor | None = None
        self._submitted: set[str] = set()
        self._lock = threading.Lock()
        self._stopping = threading.Event()

    def register(self, job_type: str, handler: JobHandler) -> None:
        self.handlers[job_type] = handler

    def start(self) -> None:
        self._stopping.clear()
        with self._lock:
            if self.executor is None:
                self.executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="kun-infra")
        with connect() as db:
            db.execute(
                """UPDATE infra_jobs SET status='queued',phase='recovered',worker_id=NULL,
                   heartbeat_at=NULL,message='服务重启，任务已恢复排队',updated_at=?
                   WHERE status IN ('running','retry_wait')""",
                (now(),),
            )
            db.execute(
                """UPDATE infra_jobs SET status='cancelled',phase='cancelled',worker_id=NULL,
                   heartbeat_at=NULL,message='任务在服务重启前已请求取消',finished_at=?,updated_at=?
                   WHERE status='cancel_requested'""",
                (now(), now()),
            )
        for item in rows("SELECT id FROM infra_jobs WHERE status='queued' ORDER BY created_at"):
            self._submit(item["id"])

    def shutdown(self) -> None:
        self._stopping.set()
        with self._lock:
            executor = self.executor
            self.executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=False)

    def enqueue(
        self,
        job_type: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str | None = None,
        max_attempts: int = 3,
        message: str = "任务已排队",
    ) -> dict:
        if job_type not in self.handlers:
            raise ValueError(f"Unknown job type: {job_type}")
        if idempotency_key:
            existing = rows("SELECT * FROM infra_jobs WHERE idempotency_key=?", (idempotency_key,))
            if existing:
                return self._decode(existing[0])
        job_id = uuid4().hex
        stamp = now()
        with connect() as db:
            db.execute(
                """INSERT INTO infra_jobs(
                   id,job_type,status,payload_json,idempotency_key,max_attempts,phase,message,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    job_id,
                    job_type,
                    "queued",
                    json_value(payload),
                    idempotency_key,
                    max(1, max_attempts),
                    "queued",
                    message,
                    stamp,
                    stamp,
                ),
            )
        self._submit(job_id)
        return self.get(job_id) or {"id": job_id, "status": "queued"}

    def _submit(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._submitted or self._stopping.is_set():
                return
            if self.executor is None:
                self.executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="kun-infra")
            executor = self.executor
            self._submitted.add(job_id)
        executor.submit(self._execute, job_id)

    def _claim(self, job_id: str) -> dict | None:
        with connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM infra_jobs WHERE id=?", (job_id,)).fetchone()
            if not row or row["status"] != "queued":
                return None
            stamp = now()
            db.execute(
                """UPDATE infra_jobs SET status='running',phase='starting',worker_id=?,attempt=attempt+1,
                   started_at=COALESCE(started_at,?),heartbeat_at=?,updated_at=?,message='任务正在运行'
                   WHERE id=? AND status='queued'""",
                (self.worker_id, stamp, stamp, stamp, job_id),
            )
            claimed = db.execute("SELECT * FROM infra_jobs WHERE id=?", (job_id,)).fetchone()
            return dict(claimed) if claimed else None

    def _heartbeat(self, job_id: str, stopped: threading.Event) -> None:
        while not stopped.wait(5):
            with connect() as db:
                db.execute(
                    "UPDATE infra_jobs SET heartbeat_at=?,updated_at=? WHERE id=? AND status IN ('running','cancel_requested')",
                    (now(), now(), job_id),
                )

    def _execute(self, job_id: str) -> None:
        heartbeat_stop = threading.Event()
        try:
            claimed = self._claim(job_id)
            if not claimed:
                return
            handler = self.handlers.get(claimed["job_type"])
            if not handler:
                raise RuntimeError(f"Handler unavailable: {claimed['job_type']}")
            heartbeat = threading.Thread(target=self._heartbeat, args=(job_id, heartbeat_stop), daemon=True)
            heartbeat.start()
            context = JobContext(job_id, self)
            result = handler(json.loads(claimed["payload_json"]), context) or {}
            context.check_cancelled()
            with connect() as db:
                db.execute(
                    """UPDATE infra_jobs SET status='succeeded',phase='completed',progress=100,
                       result_summary_json=?,message='任务已完成',finished_at=?,heartbeat_at=?,updated_at=? WHERE id=?""",
                    (json_value(_safe_attributes(result)), now(), now(), now(), job_id),
                )
        except JobCancelled:
            with connect() as db:
                db.execute(
                    """UPDATE infra_jobs SET status='cancelled',phase='cancelled',message='任务已取消',
                       finished_at=?,heartbeat_at=?,updated_at=? WHERE id=?""",
                    (now(), now(), now(), job_id),
                )
        except Exception as error:
            state = rows("SELECT attempt,max_attempts FROM infra_jobs WHERE id=?", (job_id,))
            retry = bool(state and state[0]["attempt"] < state[0]["max_attempts"] and not self._stopping.is_set())
            with connect() as db:
                db.execute(
                    """UPDATE infra_jobs SET status=?,phase=?,message=?,error_code=?,finished_at=?,heartbeat_at=?,updated_at=?
                       WHERE id=?""",
                    (
                        "retry_wait" if retry else "failed",
                        "retry_wait" if retry else "failed",
                        "发生可恢复错误，等待重试" if retry else "任务失败",
                        type(error).__name__,
                        None if retry else now(),
                        now(),
                        now(),
                        job_id,
                    ),
                )
            if retry:
                attempt = state[0]["attempt"] if state else 1
                delay = (2, 5, 10)[min(max(attempt - 1, 0), 2)]
                time.sleep(delay)
                with connect() as db:
                    db.execute(
                        "UPDATE infra_jobs SET status='queued',phase='queued',worker_id=NULL,updated_at=? WHERE id=? AND status='retry_wait'",
                        (now(), job_id),
                    )
                # Let the current execution leave `_submitted` in `finally`
                # before the retry is submitted; otherwise a fast retry can be
                # removed from the de-duplication set by the previous attempt.
                retry_timer = threading.Timer(0.05, self._submit, args=(job_id,))
                retry_timer.daemon = True
                retry_timer.start()
                return
        finally:
            heartbeat_stop.set()
            with self._lock:
                self._submitted.discard(job_id)

    def update(
        self,
        job_id: str,
        *,
        progress: int | None = None,
        phase: str | None = None,
        message: str | None = None,
    ) -> None:
        fields = ["updated_at=?", "heartbeat_at=?"]
        values: list[Any] = [now(), now()]
        for column, value in (("progress", progress), ("phase", phase), ("message", message)):
            if value is not None:
                fields.append(f"{column}=?")
                values.append(max(0, min(100, value)) if column == "progress" else value)
        values.append(job_id)
        with connect() as db:
            db.execute(f"UPDATE infra_jobs SET {','.join(fields)} WHERE id=?", tuple(values))

    def cancel(self, job_id: str) -> dict | None:
        with connect() as db:
            row = db.execute("SELECT status FROM infra_jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return None
            status = "cancelled" if row["status"] in {"queued", "retry_wait"} else "cancel_requested"
            finished = now() if status == "cancelled" else None
            db.execute(
                "UPDATE infra_jobs SET status=?,phase=?,message=?,finished_at=?,updated_at=? WHERE id=?",
                (status, status, "正在取消" if status == "cancel_requested" else "任务已取消", finished, now(), job_id),
            )
        return self.get(job_id)

    def retry(self, job_id: str) -> dict | None:
        with connect() as db:
            row = db.execute("SELECT status FROM infra_jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return None
            if row["status"] not in {"failed", "cancelled"}:
                return self.get(job_id)
            db.execute(
                """UPDATE infra_jobs SET status='queued',phase='queued',progress=0,error_code=NULL,
                   finished_at=NULL,worker_id=NULL,message='任务已重新排队',updated_at=? WHERE id=?""",
                (now(), job_id),
            )
        self._submit(job_id)
        return self.get(job_id)

    def get(self, job_id: str) -> dict | None:
        matches = rows("SELECT * FROM infra_jobs WHERE id=?", (job_id,))
        return self._decode(matches[0]) if matches else None

    def list(self, limit: int = 100) -> list[dict]:
        return [self._decode(item) for item in rows(
            "SELECT * FROM infra_jobs ORDER BY created_at DESC LIMIT ?",
            (max(1, min(limit, 500)),),
        )]

    @staticmethod
    def _decode(item: dict) -> dict:
        item["payload"] = json.loads(item.pop("payload_json") or "{}")
        item["result_summary"] = json.loads(item.pop("result_summary_json") or "{}")
        return item


def infra_summary() -> dict[str, Any]:
    job_stats = rows(
        """SELECT COUNT(*) total,
           SUM(CASE WHEN status='queued' THEN 1 ELSE 0 END) queued,
           SUM(CASE WHEN status='running' THEN 1 ELSE 0 END) running,
           SUM(CASE WHEN status='succeeded' THEN 1 ELSE 0 END) succeeded,
           SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) failed
           FROM infra_jobs"""
    )[0]
    trace_stats = rows(
        """SELECT COUNT(*) total,
           SUM(CASE WHEN status='succeeded' THEN 1 ELSE 0 END) succeeded,
           SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) failed,
           AVG(duration_ms) average_ms
           FROM (SELECT * FROM infra_traces ORDER BY started_at DESC LIMIT 50)"""
    )[0]
    recent_latencies = [
        float(item["duration_ms"])
        for item in rows(
            "SELECT duration_ms FROM infra_traces WHERE duration_ms IS NOT NULL ORDER BY started_at DESC LIMIT 50"
        )
    ]
    ordered = sorted(recent_latencies)
    p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)] if ordered else None
    indexes = rows(
        """SELECT id,space_id,status,is_active,provider,model,dimension,strategy,vector_count,index_bytes,
           created_at,activated_at FROM index_generations ORDER BY created_at DESC LIMIT 20"""
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "jobs": job_stats,
        "traces": {**trace_stats, "p95_ms": round(p95, 2) if p95 is not None else None},
        "indexes": indexes,
        "recent_traces": list_traces(8),
    }
