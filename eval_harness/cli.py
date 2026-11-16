"""Grade JSONL model responses against the synthetic task specifications."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_harness.grading import grade, summarize
from post_training_lab.data import load


def evaluate_responses(path: Path, split="test", *, require_complete=False):
    tasks = {row["id"]: row for row in load(split)}
    records = []
    seen = set()
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {line_number}: malformed JSON") from exc
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get("response"), str):
            raise ValueError(f"Line {line_number}: expected string id and response fields")
        task_id = row["id"]
        if task_id not in tasks:
            raise ValueError(f"Line {line_number}: unknown {split} task ID {task_id}")
        if task_id in seen:
            raise ValueError(f"Line {line_number}: duplicate response for {task_id}")
        seen.add(task_id)
        task = tasks[task_id]
        records.append({"id": task_id, "domain": task["domain"], "response": row["response"],
                        "grade": grade(task, row["response"])})
    if not records:
        raise ValueError("No model responses supplied")
    missing = sorted(set(tasks) - seen)
    if require_complete and missing:
        raise ValueError(f"Incomplete {split} coverage: {len(missing)} task responses missing")
    return {"split": split, "coverage": {"evaluated": len(records), "available": len(tasks),
                                          "complete": not missing, "missing_ids": missing},
            "statistics": summarize(records), "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses", type=Path)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = evaluate_responses(args.responses, args.split, require_complete=args.require_complete)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
