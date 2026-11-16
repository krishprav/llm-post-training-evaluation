# Verifiable LLM Evaluation Harness

A model-independent grading package for bounded arithmetic, Python-expression generation and simulated tool routing. Expected behavior comes from task payloads, rather than comparing a response to its training answer string.

## What is checked

| Domain | Output contract | Semantic verification |
|---|---|---|
| Arithmetic | Bare integer | Recompute the specified operation from operands |
| Coding | Python arithmetic expression using `x` | Canonical bounded polynomial equivalence plus value checks |
| Tool routing | Exactly two JSON string keys, `tool` and `argument` | Compare the selected tool and argument to the request payload |

Generated expressions are interpreted through an allowlist of AST nodes. They are never executed with `eval`, `exec`, a shell or a subprocess. The grader rejects calls, attributes, comprehensions, excessive bounds, duplicate JSON keys and extra schema fields. Tool routing is a single simulated decision and makes no external calls.

## Grade your own outputs

From the repository root, supply JSONL records containing a test task ID and response string:

```json
{"id":"coding-0-20","response":"22 + x"}
```

```sh
python3 -m eval_harness.cli examples/responses.jsonl --output /tmp/grades.json
python3 -m eval_harness.cli your-responses.jsonl --require-complete --output /tmp/full-grades.json
python3 verify_reports.py
```

These commands use the Python standard library. The CLI reports sample coverage and keeps semantic errors separate from malformed outputs. It rejects unknown IDs, duplicates and non-string responses. `--require-complete` rejects collections missing any of the split's task IDs. A caller-supplied grade is ignored.

## Run model inference and comparisons

With the [full dependencies and model](../README.md#reproduce) available:

```sh
.venv/bin/python -m eval_harness.evaluate base
.venv/bin/python -m eval_harness.evaluate sft
.venv/bin/python make_report.py
```

The full runner evaluates base, SFT, reduced-data SFT, DPO and PPO with identical greedy decoding. Reports include every response, semantic and format grades, domain accuracy, synthetic preference likelihood margins and paired bootstrap intervals.

The 216 tasks share instruction families but hold out exact prompts and parameter groups. This tests a narrow range holdout. It is not an external coding benchmark or a general agent evaluation. See the [data card](../DATA_CARD.md) and [measured results](../reports/RESULTS.md).
