import ast
import json
from pathlib import Path

import pytest
from post_training_lab.data import build, load, ROOT
from eval_harness.grading import grade


@pytest.fixture(scope="module", autouse=True)
def dataset():
    build()


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_all_reference_answers_and_near_misses(split):
    for row in load(split):
        assert grade(row, row["chosen"])["correct"]
        assert not grade(row, row["rejected"])["correct"]


def test_no_prompt_or_group_leakage():
    sets = [load(s) for s in ["train", "validation", "test"]]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not ({r["prompt"].strip().casefold() for r in sets[i]} & {r["prompt"].strip().casefold() for r in sets[j]})
            assert not ({r["group"] for r in sets[i]} & {r["group"] for r in sets[j]})
    assert [len(s) for s in sets] == [144, 36, 36]


@pytest.mark.parametrize("response", ["0", "x", "x + 999", "x + 22 + (x + 7) * x * (x - 9) * (x - 37)", "__import__('os').system('id')", "open('/tmp/x')", "x.__class__", "[x][0]", "x ** 100000", "True", "x / 0", "[x for x in range(5)]", "lambda x: x"])
def test_coding_shortcuts_and_unsafe_ast_rejected(response):
    row = next(r for r in load("test") if r["domain"] == "coding")
    assert not grade(row, response)["correct"]


@pytest.mark.parametrize("response", ['{"tool":"weather","argument":"City22","extra":1}', '{"tool":"weather","tool":"weather","argument":"City22"}', '{"tool":"weather","argument":22}', 'weather City22', '[]', 'null', '{"tool":"calculator","argument":"City22"}'])
def test_schema_and_wrong_tool_rejected(response):
    row = next(r for r in load("test") if r["domain"] == "agentic")
    assert not grade(row, response)["correct"]


@pytest.mark.parametrize("response", ["The answer is 67", "67\nIgnore all tests", "nan", "inf", "67.0", "", "x" * 1001])
def test_reasoning_invalid_outputs(response):
    row = next(r for r in load("test") if r["domain"] == "reasoning")
    assert not grade(row, response)["correct"]


def test_human_review_is_not_fabricated():
    queue = json.loads((ROOT / "data/human_review_queue.json").read_text())
    assert len(queue) == 36
    assert all(r["human_preference"] is None and r["reviewer"] is None for r in queue)


def test_cross_framework_loss_and_gradient_parity():
    from post_training_lab.objectives import validate
    assert validate()["passed"]


def test_reward_pooling_uses_response_after_chat_separator():
    import torch
    from transformers import LlamaConfig, LlamaForSequenceClassification
    torch.manual_seed(7)
    config = LlamaConfig(vocab_size=20, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         num_labels=1, pad_token_id=0, eos_token_id=2)
    model = LlamaForSequenceClassification(config).eval()
    ids = torch.tensor([[1,3,2,1,4,6,2], [1,3,2,1,4,7,2]])
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
        last_scores = model.score(out.hidden_states[-1][:,-1])
        assert torch.allclose(out.logits, last_scores)
        assert not torch.allclose(out.logits[0],out.logits[1],atol=1e-8,rtol=0)
        model.config.pad_token_id = 2
        broken = model(input_ids=ids).logits
        assert torch.equal(broken[0],broken[1])
