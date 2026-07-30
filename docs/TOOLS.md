# Tool System

KUN treats Tools as permissioned, typed, observable runtime capabilities. Skills compose Tools into user-facing workflows; LangGraph owns transitions and recovery.

## Runtime contract

Every Tool declares:

- name and purpose
- JSON input schema
- local read/write scopes
- network scope
- timeout
- whether user confirmation is required
- availability and an honest unavailable reason
- recoverable error codes

`backend/app/tools.py` validates arguments, enforces scopes, redacts sensitive inputs, measures duration, and writes a compact execution record to SQLite. Tool output never grants itself additional permissions.

## Initial catalog

| Tool | Purpose | Current state |
| --- | --- | --- |
| `rag.search` | Hybrid retrieval in one knowledge space | Available |
| `file.search` | Search confirmed local document metadata | Available |
| `document.parse` | Parse an approved staging/library file | Available |
| `memory.search` | Read enabled long-term memory | Available |
| `image.search` | Search confirmed local images by visual meaning, OCR text, and tags | Available |
| `video.probe` | Inspect local video metadata | Available only when `ffprobe` exists |
| `video.transcribe` | Timestamped local transcription | Adapter pending |
| `video.sample_frames` | Extract derived keyframes | Worker pending |
| `web.search` | Search public web sources | Provider pending |
| `web.fetch` | Read an explicitly permitted public HTTPS page with SSRF protection | Available |

Unavailable capabilities remain visible with a reason, but must never display a success checkmark or produce fabricated results.

## Observable UI

The chat UI may expose:

- current workflow node
- Tool name and short purpose
- duration and result count
- retry/degraded-mode status
- sources used by the final answer

It must not expose hidden chain-of-thought, raw secrets, complete private file contents, or provider authorization headers.

## Permission boundary

- Read-only search can run within the user-selected knowledge space.
- File creation, transcript generation, Memory updates, exports, and index writes require the appropriate write scope.
- Any action marked `confirmation_required` needs a confirmation issued by the UI/policy layer; an LLM-generated boolean is not confirmation.
- Web and video adapters must not bypass access controls or redistribute source media.
