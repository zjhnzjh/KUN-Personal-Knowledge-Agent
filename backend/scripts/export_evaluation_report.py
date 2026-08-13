"""Export the latest stored KUN evaluation run as JSON or Markdown.

Examples:
  python backend/scripts/export_evaluation_report.py --data-dir .kun-data
  python backend/scripts/export_evaluation_report.py --run-id <id> --format json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.database import connect, init_database
from app.evaluation_reporting import build_evaluation_report, load_details, render_markdown


def _select_run(space_id: str, run_id: str | None) -> dict:
    query = "SELECT * FROM evaluation_runs WHERE space_id=?"
    params: tuple[str, ...] = (space_id,)
    if run_id:
        query += " AND id=?"
        params = (space_id, run_id)
    query += " ORDER BY created_at DESC LIMIT 1"
    with connect() as db:
        row = db.execute(query, params).fetchone()
    if not row:
        target = f"run {run_id}" if run_id else f"space {space_id}"
        raise SystemExit(f"No stored evaluation run found for {target}.")
    return dict(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a reproducible KUN RAG evaluation report")
    parser.add_argument("--space-id", default="ai-agent-learning")
    parser.add_argument("--run-id")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.data_dir:
        os.environ["KUN_DATA_DIR"] = str(args.data_dir.resolve())

    init_database()
    settings = get_settings()
    run = _select_run(args.space_id, args.run_id)
    report = build_evaluation_report(
        run,
        load_details(run),
        embedding_model=settings.embedding_model,
        chat_model=settings.deepseek_model,
    )
    content = (
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.format == "json"
        else render_markdown(report)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print(args.output.resolve())
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
