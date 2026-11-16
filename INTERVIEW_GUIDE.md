# Explain the projects from the evidence

Read and rerun the code before describing personal proficiency. These portfolio projects are separate from work at Snorkel, Handshake and AfterQuery. The resume dates them November 2026.

## The implementation choices

**SFT:** the pretrained instruct model is adapted to concise task responses. Completion-only labels mask user/prompt tokens so optimization targets the answer. LoRA trains query/value low-rank matrices while freezing the pretrained backbone. The adapter parameter fraction is measured in `sft_training.json`, not estimated from the checkpoint name.

**DPO:** optimize the log-likelihood ratio of a preferred and rejected response relative to a frozen SFT reference. With beta 0.1, the loss is `softplus(-beta * ((logp_chosen - logp_rejected) - (ref_chosen - ref_rejected)))`. The SFT model is the frozen backbone for the fresh DPO adapter. Summed completion log probabilities are used; prompts are not scored as completions.

**Reward modeling:** a sequence-classification head assigns one scalar to a prompt/response. The Bradley-Terry loss is `softplus(-(reward_chosen - reward_rejected))`. Pairwise ranking accuracy is distinct from calibration and from downstream generated-answer accuracy. Synthetic preference correctness does not prove human preference alignment.

**PPO:** sample new responses from the SFT policy, score them with the learned reward model, and update the policy/value head with clipped PPO and KL control. Four batches exercise the integration. They do not prove convergence. Inspect the recorded reward, KL, policy/value losses and actual rollouts before making any performance claim.

**Ablation:** compare 144 versus 36 training examples at the same 64 optimizer steps and initialization. The experiment changes data diversity and repetition; it does not isolate data quality or total token exposure. The test set is never used for model selection.

**JAX:** objectives and gradients are implemented independently from the PyTorch training implementation. Agreement checks cover tie, reversed-preference and extreme-margin cases; PPO test ratios avoid nondifferentiable clipping boundaries. This verifies local objective math, not a full JAX model-training backend.

**DeepSpeed:** ZeRO stage 2 partitions optimizer states and gradients; the supplied file also configures CPU optimizer offload. No distributed CUDA run was performed. Explain the configuration as prepared integration, not demonstrated multi-GPU operations.

## Evaluation judgments

- A code expression must work on multiple hidden `x` values; returning a single hardcoded number fails.
- Tool responses must satisfy the exact two-string-key schema and select the correct tool and argument. Duplicate keys fail.
- Arithmetic responses must be bare integers. A verbose correct explanation can fail the specified output contract; output validity and semantic accuracy are reported separately.
- The synthetic benchmark has 36 test tasks, with 12 in each domain. A single task changes overall accuracy by 2.78 percentage points.
- Pairwise bootstrap intervals quantify item uncertainty for the paired conditions. One training seed cannot establish training stability.
- Learned reward improvement without independent correctness improvement may indicate reward exploitation. Read the failed outputs instead of relying on one scalar.

Use `reports/RESULTS.md` for the actual numbers, including any regressions. Do not quote training acceptance rates as model accuracy, or combine employer task counts with these synthetic portfolio counts.
