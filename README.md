# LLM Post Training Lab and Verifiable Evaluation Harness

Two research engineering projects that connect open-weight model post-training to independently verifiable evaluation. Built around SmolLM2-135M-Instruct, 216 synthetic tasks, fixed training settings and saved model outputs.

| Project | What it demonstrates | Start here |
|---|---|---|
| LLM Post Training Lab | Completion-only SFT, LoRA, data ablation, DPO, reward modeling and short PPO training | [Training guide](post_training_lab/README.md) |
| Verifiable LLM Evaluation Harness | Payload-based arithmetic grading, bounded Python AST interpretation, strict JSON tool validation and paired comparisons | [Evaluation guide](eval_harness/README.md) |

## Measured results

| Condition | Correct on 36 held-out tasks |
|---|---:|
| Base | 0 / 36 |
| SFT | 22 / 36 (61.1%) |
| SFT with 36 training rows | 20 / 36 (55.6%) |
| DPO from SFT | 20 / 36 (55.6%) |
| PPO from SFT | 22 / 36 (61.1%) |

The reward model ranks 30 of 36 held-out synthetic preference pairs correctly (83.3%). DPO regressed on this small evaluation, and the four-batch PPO run did not improve task accuracy. Strict formatting contributes to the SFT gain, and arithmetic remains at 0%. These are one-seed synthetic-task results, not general capability estimates. See [the full report](reports/RESULTS.md) for domain scores, paired bootstrap intervals and failure analysis.

## Try the evaluation without downloading a model

Only Python 3.11 and the standard library are needed for these commands:

```sh
python3 verify_reports.py
python3 -m eval_harness.cli examples/responses.jsonl --output /tmp/example-grades.json
```

The example deliberately contains a correct answer, a semantic error and an invalid tool call. It reports coverage as 3 of 36 tasks. Supply your own JSONL responses and use `--require-complete` to enforce full split coverage. This regrades saved text, without rerunning inference or training.

## Project 1 — LLM Post Training Lab

An open-weight SmolLM2-135M-Instruct experiment using Python, PyTorch, Hugging Face Transformers, TRL and LoRA/PEFT:

1. Supervised fine-tuning (SFT) with completion-only labels.
2. A controlled data-diversity ablation at equal optimizer steps.
3. Direct preference optimization (DPO) using the frozen SFT policy as reference.
4. Pairwise Bradley-Terry reward modeling.
5. Short on-policy PPO training using the learned reward model and KL control.
6. Independent JAX checks of objective values and gradients against PyTorch.

This is an **RLHF-style synthetic-preference prototype**. The labels come from deterministic task specifications, not human judgments. `data/human_review_queue.json` contains blank review fields for later human adjudication. Do not describe those fields as completed human feedback.

`configs/deepspeed_zero2.json` and `--deepspeed` provide an optional ZeRO-2 CUDA launch path. Distributed GPU execution was not run on the local Apple computer. JAX is used for objective validation, not for training the language model.

## Project 2 — Verifiable Coding Reasoning and Tool Use Evals

A separate evaluation package grades model-generated arithmetic answers, Python expressions and JSON tool calls. The evaluator derives expected behavior from task payloads independently of the chosen response string. It checks semantic correctness and output validity, reports failures by domain, and rejects malformed output and shortcut attempts.

Python expressions run through a restricted AST interpreter supporting only bounded integer arithmetic and the variable `x`; generated Python is never passed to `eval`, `exec`, a shell or a subprocess. Tool calls are simulated single-step routing decisions, not real external actions. This is a small synthetic benchmark, not SWE-bench, HumanEval, or a claim about general agent capability.

## Reproduce

Python 3.11 is recommended. The versions are deliberately pinned to the compatible TRL 0.11.4 APIs for all four trainers; this does not claim use of the newest TRL release.

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m post_training_lab.data
```

Download the Apache-2.0 SmolLM2 checkpoint and resolve the exact revision before training:

```sh
.venv/bin/python download_model.py
.venv/bin/python run_experiments.py
.venv/bin/python make_report.py
```

The full experiment uses the local accelerator when available and writes logs, adapters and run manifests. The reported run used Apple MPS. The test suite requires no model download. `requirements.lock.txt` records the original local environment, while `requirements.txt` declares portable direct dependencies. Run from this directory. Initial checkpoint access and dependency installation require network access. Training and evaluation resolve the pinned checkpoint from your own local Hugging Face cache, with network access disabled for model loading.

After reproducing training, run `.venv/bin/python audit_evidence.py` to inspect local adapter updates and finite metrics. `verify_reports.py` separately checks committed data hashes, recomputes all saved grades and aggregate scores, and validates report consistency without adapters.

For an optional CUDA/DeepSpeed run, install a compatible DeepSpeed build on the target machine, then:

```sh
.venv/bin/python -m post_training_lab.train sft --deepspeed configs/deepspeed_zero2.json
```

That launch path and the Dockerfile are supplied configurations. Check the execution ledger before claiming they were run.

## Inspect the evidence

- `EXPERIMENT_PLAN.md`: settings fixed before training and explicit limitations.
- `reports/RESULTS.md`: measured comparison, ablation and interpretation.
- `reports/*_training.json`: optimizer metrics, timings and stage completion.
- `reports/*_test.json`: individual held-out outputs, independent grades and statistics.
- `reports/objective_parity.json`: PyTorch/JAX loss and gradient agreement.
- `reports/environment.json`: exact model commit, package versions and hardware.
- `reports/run_history.json`: process completion and log paths.
- `data/manifest.json`: split sizes and SHA-256 hashes.
- `tests/test_verifiers.py`: reference, near-miss, leakage, schema and unsafe-expression checks.
- `tests/test_evidence_audit.py`: altered-score, changed-data, duplicate-output and nonfinite-metric rejection.
- `.github/workflows/verify.yml`: model-free report verification and CPU software/objective tests on pushes and pull requests.
- `reports/publication_smoke.json`: a three-task SFT inference check after the portable cache fix, separate from the full experiment.

Adapters and the downloaded base checkpoint remain under `artifacts/` and the Hugging Face cache. They are excluded from the portable source archive. Regenerate them using the pinned commands. No private employer tasks or held-out client datasets are included in this portfolio.

## Provenance and scope

The experiments were prepared and run on 16 November 2026. Git history begins with an import of that existing work, followed by actual reproducibility and publication changes. See [provenance](PROVENANCE.md). Synthetic preferences demonstrate the mechanics of an RLHF-style pipeline, rather than collected human feedback. The source code is MIT licensed. The separately downloaded SmolLM2 model retains its Apache-2.0 license.

## Primary references

- [SmolLM2 checkpoint and model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
- [TRL 0.11.4 SFT](https://huggingface.co/docs/trl/v0.11.4/sft_trainer)
- [TRL 0.11.4 DPO](https://huggingface.co/docs/trl/v0.11.4/dpo_trainer)
- [TRL 0.11.4 reward modeling](https://huggingface.co/docs/trl/v0.11.4/reward_trainer)
- [TRL 0.11.4 PPO](https://huggingface.co/docs/trl/v0.11.4/ppo_trainer)
- [PEFT LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
- [DeepSpeed ZeRO configuration](https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training)
- [JAX automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html)
