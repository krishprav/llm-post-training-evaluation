"""Recompute saved grades and validate experiment reports without model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from eval_harness.grading import grade, summarize

ROOT = Path(__file__).resolve().parent


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root=ROOT):
    root = Path(root)
    checks = {}

    def read(path):
        value = json.loads((root / path).read_text())
        require(finite(value), f"Nonfinite values in {path}")
        return value

    spec = read("configs/model.json")
    manifest = read("data/manifest.json")
    datasets = {}
    for split, meta in manifest["splits"].items():
        raw = (root / f"data/{split}.jsonl").read_bytes()
        require(hashlib.sha256(raw).hexdigest() == meta["sha256"], f"Split hash mismatch: {split}")
        rows = [json.loads(line) for line in raw.splitlines()]
        require(len(rows) == meta["rows"], f"Split row count mismatch: {split}")
        datasets[split] = rows
    require([len(datasets[s]) for s in ("train", "validation", "test")] == [144, 36, 36], "Unexpected split sizes")
    tasks = {r["id"]: r for r in datasets["test"]}
    require(len(tasks) == 36, "Duplicate test task IDs")
    checks["split_hashes_match"] = True

    for name, steps in [("sft", 64), ("sft-low-data", 64), ("dpo", 32), ("reward", 40)]:
        report = read(f"reports/{name}_training.json")
        require(report["log_history"][-1]["step"] == steps, f"Incomplete training: {name}")
        require(report["model_id"] == spec["model_id"] and report["model_revision"] == spec["revision"],
                f"Training model revision mismatch: {name}")
        checks[name] = {"optimizer_steps": steps, "metrics_finite": True}

    ppo = read("reports/ppo_training.json")
    require(ppo["rollout_batches"] == 4 and len(ppo["history"]) == 4, "Incomplete PPO rollout history")
    require(sum(len(row["responses"]) for row in ppo["history"]) == 16, "PPO response count mismatch")
    checks["ppo"] = {"rollout_batches": 4, "responses_sampled": 16, "metrics_finite": True}

    verifier_hash = hashlib.sha256((root / "eval_harness/grading.py").read_bytes()).hexdigest()
    for stage in ("base", "sft", "sft-low-data", "dpo", "ppo"):
        report = read(f"reports/{stage}_test.json")
        records = report["records"]
        require(len(records) == 36 and {r["id"] for r in records} == set(tasks), f"Incomplete test coverage: {stage}")
        require(report["split"] == "test" and report["stage"] == stage, f"Evaluation identity mismatch: {stage}")
        require(report["verifier_sha256"] == verifier_hash, f"Stale verifier: {stage}")
        for record in records:
            task = tasks[record["id"]]
            require(record["domain"] == task["domain"], f"Domain mismatch: {record['id']}")
            require(record["grade"] == grade(task, record["response"]), f"Stored grade mismatch: {stage}/{record['id']}")
        for domain, expected in summarize(records).items():
            require(report["statistics"][domain] == expected, f"Aggregate score mismatch: {stage}/{domain}")
        expected_preference = sum(r["preference_logp_margin"] > 0 for r in records) / len(records)
        require(report["statistics"]["preference_accuracy"] == expected_preference, f"Preference score mismatch: {stage}")
        checks[f"{stage}_eval"] = {"unique_test_rows": 36, "grades_recomputed": True, "finite": True}

    reward = read("reports/reward_test.json")
    require(len(reward["records"]) == 36 and {r["id"] for r in reward["records"]} == set(tasks), "Reward test coverage mismatch")
    for record in reward["records"]:
        require(record["correct"] == (record["chosen_score"] > record["rejected_score"]), "Reward pair grade mismatch")
    expected_accuracy = sum(r["correct"] for r in reward["records"]) / 36
    require(reward["test_pairs"] == 36 and reward["accuracy"] == expected_accuracy, "Reward accuracy mismatch")
    checks["reward_eval"] = {"unique_test_pairs": 36, "accuracy": expected_accuracy}

    parity = read("reports/objective_parity.json")
    require(parity["passed"] and parity["tolerance"] == 1e-10, "Objective parity failed")
    for name, metrics in parity["results"].items():
        require(metrics["loss_error"] < 1e-10 and metrics["gradient_max_error"] < 1e-10, f"Objective mismatch: {name}")
    checks["saved_objective_parity"] = True
    return {"passed": True, "scope": "saved_reports_only", "adapter_weights_checked": False, "checks": checks,
            "limitations": ["Regrades saved outputs rather than rerunning inference or training.",
                            "Use audit_evidence.py after reproducing training to inspect local adapter weights."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional destination for this verification result")
    args = parser.parse_args()
    result = verify()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
