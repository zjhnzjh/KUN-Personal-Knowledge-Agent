from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.credentials import set_secret


def import_deepseek(path: Path) -> bool:
    if not path.is_file():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    profile = data.get("profiles", {}).get("DeepSeek", {})
    key = str(profile.get("api_key", "")).strip()
    if not key:
        return False
    set_secret("deepseek", key)
    return True


def import_dashscope(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        pairs = {}
        for row in csv.reader(stream):
            if len(row) >= 2:
                pairs[row[0].strip()] = row[1].strip()
    key = pairs.get("apiKey", "")
    if not key:
        return False
    set_secret("dashscope", key)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepseek-registry", type=Path, required=True)
    parser.add_argument("--dashscope-csv", type=Path, required=True)
    args = parser.parse_args()
    deepseek = import_deepseek(args.deepseek_registry)
    dashscope = import_dashscope(args.dashscope_csv)
    print(json.dumps({"deepseek_imported": deepseek, "dashscope_imported": dashscope}))


if __name__ == "__main__":
    main()
