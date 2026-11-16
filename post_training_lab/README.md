# LLM Post Training Lab

A compact experiment connecting supervised fine-tuning, preference optimization and independent evaluation on a pinned 135M-parameter open-weight language model.

## Pipeline

1. Generate 144 training, 36 validation and 36 test tasks from nine deterministic families.
2. Fine-tune query/value projections with rank-8 LoRA, completion-only SFT and 64 optimizer steps.
3. Repeat SFT with 36 training rows at equal optimizer steps to test data diversity.
4. Branch from SFT into 32-step DPO with a frozen SFT reference.
5. Train a separate Bradley-Terry reward model for 40 steps.
6. Branch from SFT into four on-policy PPO batches using learned rewards, a value head and KL control.
7. Evaluate all five model conditions on the same held-out task IDs.

The DPO and PPO branches are alternatives. PPO does not initialize from DPO. All preferences are specification-generated synthetic labels.

## Run

From the repository root, follow the [dependency setup](../README.md#reproduce), then:

```sh
.venv/bin/python download_model.py
.venv/bin/python run_experiments.py
.venv/bin/python make_report.py
.venv/bin/python audit_evidence.py
```

To resume completed stages, use `run_experiments.py --resume`. The runner retains prior stage attempts in its ledger. The original ledger records failed implementation attempts as well as successful reruns, with the fixes described in [implementation notes](../IMPLEMENTATION_NOTES.md).

Individual stages are available through `python -m post_training_lab.train {sft,sft-low-data,dpo,reward,ppo}`. They share the hyperparameters recorded in [the experiment plan](../EXPERIMENT_PLAN.md). SFT must finish before DPO or PPO, and reward training must finish before PPO.

## Evidence and limits

SFT reached 61.1% strict task accuracy on 36 synthetic held-out examples. The reward model ranked 83.3% of held-out pairs correctly. DPO reduced task accuracy to 55.6%, and short PPO left it at 61.1%. There is one training seed. This establishes a working pipeline and a bounded experiment, rather than convergence or broad reasoning improvement.

The [full report](../reports/RESULTS.md) includes the failed task categories and paired comparisons. [JAX objective checks](objectives.py) compare values and gradients with PyTorch at 1e-10 tolerance. The DeepSpeed ZeRO-2 configuration is an optional CUDA path that was not executed on the Apple machine. All reported language-model training uses PyTorch.
