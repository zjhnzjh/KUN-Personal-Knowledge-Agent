---
name: document-skill
description: Parse, summarize, compare, and answer from confirmed PDF, Word, or Markdown documents. Use when a request depends on grounded document evidence or exact source citations.
---

# Document Skill

1. Confirm the knowledge space and requested scope.
2. Parse only confirmed library copies.
3. Search with `rag.search`; prefer the smallest sufficient evidence set.
4. Distinguish document evidence from conversation context and Memory.
5. Answer in concise Markdown and cite only chunks that support actual claims.
6. State that evidence is insufficient when retrieval does not support an answer.

