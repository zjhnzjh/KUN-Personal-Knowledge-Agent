# Product specification

## Primary journey

1. Complete four-step onboarding.
2. Create or choose a knowledge space.
3. Upload files or configure a watched folder.
4. Review title, summary, tags, sensitivity, and parse preview.
5. Confirm ingestion and observe real indexing stages.
6. Ask questions and inspect exact citations.
7. Review proposed memories and retrieval traces when desired.

## Interface principles

- Light theme only with a low-saturation violet accent.
- Recent conversations remain visible like ChatGPT.
- Only user messages are bubbles; assistant text is not automatically collapsed.
- Every loading label corresponds to real state.
- Advanced controls live in RAG Lab, settings, and developer mode.

## Acceptance conditions

- Rejecting a staged file leaves no searchable chunks.
- Confirming a staged file copies the original before indexing.
- Duplicate content is detected by SHA-256 within a knowledge space.
- Provider failures are visible and do not silently report vector search.
- Generated artifacts never replace their source file.
