# Architecture

## Design goals

KUN prioritizes local ownership, explicit data boundaries, recoverable workflows, measurable retrieval quality, and a calm consumer-facing experience. Advanced RAG details stay available without turning the primary interface into an operations dashboard.

## Runtime processes

| Process | Responsibility | Network exposure |
| --- | --- | --- |
| React UI | Chat, library, staging confirmation, source preview, settings | None directly |
| Tauri shell | Window, tray, global shortcut, native file open, process lifecycle | None |
| FastAPI | Local API, workflow, parsing, retrieval, task orchestration | Loopback only |
| Worker pool | OCR, parsing, embedding, index rebuild | Outbound provider calls only when enabled |

## Data ownership

The default data root is `%LOCALAPPDATA%/KUN`. It contains copied library files, staging files, exports, indexes, and SQLite metadata. API credentials are not part of the database or export bundle.

## Ingestion state machine

```text
uploaded → parsing → awaiting_confirmation → chunking → embedding → indexing → ready
             │               │                    │          │
             └────────────── failed / retryable ──┴──────────┘
```

Confirmation is a real persistence boundary. Temporary extraction can propose metadata, but it cannot create searchable knowledge before approval.

## Retrieval

1. Query normalization and optional rewrite.
2. Space filter and metadata constraints.
3. BM25 candidates through SQLite FTS5.
4. Dense vector candidates through FAISS.
5. Reciprocal Rank Fusion.
6. Optional reranking.
7. Citation validation and answer generation.

Changing embedding model or dimension creates a new index generation. The old generation remains readable until an atomic swap activates its replacement.

## Memory

- Short-term memory: recent state tied to one conversation.
- Long-term memory: explicit facts/preferences stored as reviewable records.
- Project context: scoped to a knowledge space.
- Sensitive candidates: never auto-committed.

## Skills

Skills declare schemas, access scope, timeout, confirmation policy, and errors. The model may request a Skill; the policy layer decides whether it may run.

Tools are the atomic runtime boundary beneath Skills. The registry validates typed arguments, enforces read/write/network scopes, records redacted execution traces, and refuses unavailable capabilities. See `docs/TOOLS.md`.

Video learning is a staged Skill:

```text
approved source → probe → timestamped transcript → keyframes/OCR
                → proposed metadata → user confirmation → index
```

The first implementation exposes capability detection and media probing. Transcription and keyframe Tools remain explicitly unavailable until their local engines are connected.

## Evaluation

Reports include Recall@5/10, MRR, nDCG@10, latency percentiles, citation-location success, provider cost, dataset version, and machine profile.

## First-release non-goals

- Multi-user accounts or organization RBAC
- Cloud-hosted user document storage
- Enterprise compliance claims
- Automated downloading or redistribution of third-party audio/video
- Bypassing platform access controls, DRM, login, paywalls, or anti-bot restrictions
- Autonomous modification of original files
