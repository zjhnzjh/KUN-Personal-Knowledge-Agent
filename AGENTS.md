# KUN Personal Knowledge Agent — Engineering Instructions

## Product identity

- Product: **KUN Personal Knowledge Agent** (`Knowledge · Understanding · Navigation`).
- Assistant: **坤坤**, an original fictional personal knowledge assistant.
- Personality: confident, warm, focused, concise, patient, and lightly humorous when appropriate.
- Never claim or imply that 坤坤 is a real public figure. Do not imitate a real person's voice, likeness, signature phrases, biography, or endorsements.

## Product boundary

- Windows-first, single-user, local-first personal knowledge software.
- This is not a multi-tenant enterprise system and must not claim enterprise production deployment.
- The supported first-release sources are PDF, DOCX, Markdown, XLSX/XLS, PNG, JPG, and JPEG.
- Video learning is an explicitly experimental source type. Accept only user-approved local MP4/MOV/MKV/WEBM/M4V files or content available through a permitted interface.
- Original files must never be overwritten. Generated files require preview and confirmation.

## Experience rules

- Keep the interface quiet, content-first, and progressively disclose advanced controls.
- Only user messages use chat bubbles. Long user messages may collapse; assistant responses must not be truncated or collapsed automatically.
- Show an immediate real status after send, then stream the answer. Status labels must correspond to actual workflow nodes.
- Every grounded answer must expose its sources. Unsupported claims must be marked as uncertain rather than invented.
- Render assistant answers as concise standard Markdown with real headings, emphasis, lists, tables, and code blocks. Never expose raw Markdown markers as plain text.
- Memory-only answers must identify Memory as their origin and must not attach document citations.
- Citation clicks open a right-side preview and navigate to the exact page, paragraph, sheet/cell range, or image region when available.
- Use a light theme only for the first release.
- Never render a clickable-looking control without a real action. If a capability is not connected, mark it as `待配置`, `不可用`, or `即将支持`.
- A green success check means the corresponding capability was actually verified during the current configuration lifecycle.
- Show observable workflow events and Tool calls, not private chain-of-thought. Events must name the real node/tool and reflect its actual result.

## Data and security rules

- Files, SQLite metadata, FAISS indexes, conversations, and memory live on the user's machine.
- Never commit API keys, credentials, private files, extracted text, indexes, or real user data.
- Store credentials through Windows Credential Manager in the desktop build. Environment variables are allowed only for local development and CI.
- Never print secrets in logs or API responses. Redact authorization headers and provider errors.
- Bind the local API to loopback only. Keep CORS origins explicit.
- Validate file suffix, MIME signature, size, safe filename, and resolved path before processing.
- A document remains in staging until the user confirms its title, summary, tags, sensitivity, and knowledge space.

## RAG integrity

- Embedding generation and vector storage are separate concerns.
- Changing the embedding provider/model/dimension invalidates the corresponding vector index and requires an explicit rebuild.
- Never silently substitute a hash vector and report it as a semantic embedding.
- Record parser version, chunker version, embedding model, dimension, index strategy, and timestamps with every indexed document.
- Hybrid retrieval uses BM25 plus vector retrieval, followed by RRF and optional reranking.
- Evaluation reports must state dataset size, model versions, machine profile, and date. Do not generalize a small benchmark to production accuracy.

## Agent and Skill rules

- LangGraph owns state transitions and checkpointing.
- LLM output never bypasses the permission layer.
- Each Skill declares typed input/output, read/write scope, timeout, confirmation requirement, and recoverable error types.
- Tools are atomic typed capabilities; Skills compose Tools into workflows. Model selection does not grant Tool permission.
- Every Tool declares local read/write scopes, network scope, timeout, confirmation policy, availability, and recoverable errors.
- Validate Tool arguments before execution and redact secrets and private absolute paths from execution logs.
- Persist compact Tool traces: name, status, duration, result count, error code, and source provenance.
- A Tool marked unavailable must fail visibly with a reason. Never fabricate output or show a success indicator.
- File writes, exports, transcript/keyframe creation, index rebuilds, and Memory mutations require policy-layer confirmation when declared.
- Long-term memory is user-visible and reversible. Sensitive personal data is never written automatically.
- Short-term memory is scoped to the current conversation. Stable identity, location, preference, goal, project, and relationship facts may become pending long-term candidates, but never become enabled without user confirmation.
- Model/provider failures must be visible. Degrade to a named capability mode such as `BM25 only`; never pretend the failed capability ran.

## Public web and copyright rules

- Publicly accessible does not mean public domain or unrestricted.
- Web/video content may be summarized and indexed for the user's private study only when access is permitted.
- Preserve available attribution: creator, title, source URL, publication time, access time, and local content hash.
- Do not bypass login, paywall, robots/anti-bot, DRM, download restrictions, or platform APIs.
- Prefer summaries, derived notes, and short necessary quotations. Do not reproduce or export full third-party transcripts unless the user has the rights.
- Never commit third-party original media or full transcripts to the repository/demo dataset without a compatible license.
- Video citations use timestamp ranges and identify speech, OCR, or visual-description provenance.

## Quality gates

- Add tests for parsing, chunking, retrieval, permission boundaries, API behavior, and UI-critical flows.
- Target 100 manually labeled retrieval questions before publishing benchmark claims.
- Release targets: Recall@5 ≥ 0.85, citation-location success ≥ 0.95, retrieval P95 ≤ 1.5 s at 100k chunks on the documented test machine.
- These are gates, not claims, until a reproducible report exists.
