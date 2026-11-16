# Implementation and execution notes

The initial Transformers 4.46.3 / TRL 0.11.4 combination exposed a `get_batch_samples` API collision in DPOTrainer. Transformers was pinned to 4.45.2, and all model training and model evaluation stages were rerun under that compatible environment. This was an implementation fix, not a hyperparameter change or test-driven model selection. The process ledger retains failed stage attempts.

Accelerate 0.34.2 does not accept the later `data_seed` TrainingArguments field. The explicit field was removed; training and data construction both retain seed 42. Dataset generation is deterministic. The SFT, DPO and reward-model reports record their actual package versions and completed optimizer steps.

The experiment plan is fixed independently of model output scores. Training progression, model evaluation, objective correctness and software verification are separate evidence categories. `audit_evidence.py` checks adapter updates and finite metrics; it does not demand a favorable capability result.

Training-loss inspection exposed a reward pooling issue: reusing the chat EOS as the classification pad token selected the user-turn boundary, making chosen and rejected scores identical. A distinct reserved token now marks reward padding; assertions check it never occurs in an unpadded fixture. A tiny randomly initialized Llama regression test verifies response-level pooling. Reward training and PPO were rerun after this correctness fix, before their held-out evaluation. The PPO checkpoint directory is explicitly created before serialization.
