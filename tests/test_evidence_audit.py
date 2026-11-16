import json
import shutil

import pytest

from verify_reports import ROOT, verify


@pytest.fixture
def evidence_copy(tmp_path):
    for name in ("reports", "data", "configs", "eval_harness"):
        shutil.copytree(ROOT / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__"))
    return tmp_path


def mutate(root, path, change):
    file = root / path
    report = json.loads(file.read_text())
    change(report)
    file.write_text(json.dumps(report))


def test_saved_reports_can_be_verified_without_adapters():
    result = verify()
    assert result["passed"]
    assert result["scope"] == "saved_reports_only"
    assert result["adapter_weights_checked"] is False


def test_modified_dataset_fails_hash_check(evidence_copy):
    path = evidence_copy / "data/test.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Split hash mismatch"):
        verify(evidence_copy)


def test_inflated_score_is_rejected(evidence_copy):
    mutate(evidence_copy, "reports/sft_test.json", lambda r: r["statistics"]["all"].update(accuracy=1.0))
    with pytest.raises(ValueError, match="Aggregate score mismatch"):
        verify(evidence_copy)


def test_duplicate_output_cannot_replace_missing_task(evidence_copy):
    def replace(report):
        report["records"][-1] = report["records"][0]
    mutate(evidence_copy, "reports/sft_test.json", replace)
    with pytest.raises(ValueError, match="Incomplete test coverage"):
        verify(evidence_copy)


def test_modified_response_is_regraded(evidence_copy):
    path = evidence_copy / "reports/sft_test.json"
    report = json.loads(path.read_text())
    next(row for row in report["records"] if row["grade"]["correct"])["response"] = "invalid output"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Stored grade mismatch"):
        verify(evidence_copy)


def test_nonfinite_metrics_are_rejected(evidence_copy):
    mutate(evidence_copy, "reports/ppo_training.json", lambda r: r["history"][0].update(kl=float("nan")))
    with pytest.raises(ValueError, match="Nonfinite values"):
        verify(evidence_copy)
