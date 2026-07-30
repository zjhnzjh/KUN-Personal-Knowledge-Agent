---
name: video-skill
description: Understand user-approved local videos through metadata, timestamped transcription, keyframes, OCR, and grounded retrieval. Use for private video learning, summarization, or finding an exact moment.
---

# Video Skill

## Purpose

Turn a user-approved local video into timestamped personal knowledge. This Skill is for private study and retrieval, not downloading, republishing, or redistributing third-party media.

## Workflow

1. Verify that the source is a confirmed local-library file or a URL explicitly supplied by the user.
2. Record source attribution when available: title, creator, source URL, publication time, access time, and file hash.
3. Call `video.probe` to inspect duration, codecs, and tracks.
4. Call `video.transcribe` only after the user confirms derived-file creation.
5. Call `video.sample_frames` for keyframes; run OCR/image understanding only on those frames.
6. Propose a plain-language title, summary, tags, chapters, and sensitivity level.
7. Wait for user confirmation before indexing transcripts or derived descriptions.
8. Produce citations using exact timestamp ranges and identify whether evidence came from speech, OCR, or visual description.

## Safety and copyright

- Publicly viewable content is not automatically public-domain content.
- Never bypass login, paywall, anti-bot, DRM, or platform access controls.
- Do not copy an entire transcript into an answer when a concise summary and short quotation is sufficient.
- Preserve attribution and source provenance.
- Do not place third-party original videos or full transcripts in the Git repository or demo dataset without a suitable license.
- If a URL cannot be accessed through a permitted interface, ask the user to provide a lawfully obtained local file or subtitle.

## Degraded modes

- Without `ffprobe`: report that media inspection is unavailable.
- Without a transcription engine: metadata-only mode; do not claim speech understanding.
- Without OCR/vision: transcript-only mode; do not claim visual understanding.
