import json

import pytest

from eval_harness.cli import evaluate_responses
from post_training_lab.data import load


def responses_file(tmp_path, rows):
    path = tmp_path / "responses.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_partial_evaluation_reports_coverage_and_regrades(tmp_path):
    row = load("test")[0]
    path = responses_file(tmp_path, [{"id": row["id"], "response": row["chosen"], "grade": {"correct": False}}])
    result = evaluate_responses(path)
    assert result["coverage"]["evaluated"] == 1
    assert result["coverage"]["available"] == 36
    assert not result["coverage"]["complete"]
    assert len(result["coverage"]["missing_ids"]) == 35
    assert result["statistics"]["all"]["accuracy"] == 1


def test_full_reference_set_passes_complete_coverage(tmp_path):
    path = responses_file(tmp_path, [{"id": row["id"], "response": row["chosen"]} for row in load("test")])
    result = evaluate_responses(path, require_complete=True)
    assert result["coverage"]["complete"]
    assert result["statistics"]["all"] == {"n": 36, "accuracy": 1.0, "validity": 1.0}


@pytest.mark.parametrize("case", ["duplicate", "unknown", "incomplete", "nonstring", "empty"])
def test_invalid_response_collections_fail(tmp_path, case):
    row = load("test")[0]
    record = {"id": row["id"], "response": row["chosen"]}
    rows = {"duplicate": [record, record], "unknown": [{"id": "unknown", "response": "0"}],
            "incomplete": [record], "nonstring": [{"id": row["id"], "response": 67}], "empty": []}[case]
    path = responses_file(tmp_path, rows)
    with pytest.raises(ValueError):
        evaluate_responses(path, require_complete=(case == "incomplete"))
