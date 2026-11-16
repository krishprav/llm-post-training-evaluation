# Preregistered local portfolio experiment

Recorded before training on 16 November 2026. This is a compact educational experiment, not prior employment, a production deployment, or evidence of frontier-model improvement.

Project 1 implements an RLHF-style workflow: supervised fine-tuning (SFT), LoRA/PEFT, DPO preference optimization, Bradley-Terry reward modeling, and on-policy PPO updates using the learned reward model. All preference labels are generated from deterministic task specifications. There are no collected human preference labels. A review queue is supplied for subsequent human adjudication.

Project 2 evaluates model responses on arithmetic reasoning, Python-expression generation, and simulated tool routing. Coding responses are interpreted by a restricted AST evaluator, never unrestricted Python execution. Agentic evaluation concerns a single simulated tool call, not a general autonomous agent.

## Fixed design

- Base checkpoint: HuggingFaceTB/SmolLM2-135M-Instruct, resolved and pinned to its Hub commit in the run manifest.
- Dataset: 9 task families, each with 16 training, 4 validation, and 4 test instances (144/36/36). Parameter groups and normalized prompts are disjoint across splits. Preferences use equal-purpose near-miss alternatives; they are synthetic rather than human annotations.
- Seed: 42. One training seed, so sampling confidence intervals do not quantify training-seed uncertainty.
- SFT: rank 8 LoRA on attention query/value projections, 64 optimizer steps, batch 4, learning rate 1e-3, maximum 192 tokens, completion-only loss.
- Data ablation: same initialization, rank, optimizer steps, and learning rate; only the first 4 training instances per family are retained (36 rows). This compares data diversity at equal optimizer steps, not equal example exposures.
- DPO: initialize from the SFT model, new rank 8 adapter, beta 0.1, 32 steps, batch 2, learning rate 2e-4. Frozen SFT policy is the reference.
- Reward model: sequence-classification head plus rank 8 LoRA, 40 steps, batch 4, learning rate 5e-4, Bradley-Terry pairwise loss.
- PPO: initialize the SFT policy with a new adapter and value head; 4 rollout batches of 4 responses, PPO epochs 2, mini-batch 2, learning rate 1e-5. Reward is learned-model score, with TRL KL control. This is a short integration run, not convergence evidence.
- Evaluation: fixed greedy decoding, 48 new tokens maximum, all 36 test rows; report domain accuracy, output validity, synthetic preference accuracy, answer-length diagnostics, and paired bootstrap differences. No test-based checkpoint selection or retraining.
- JAX: independently check DPO, pairwise reward, and clipped PPO objectives and gradients against PyTorch using deterministic fixtures. This is objective validation, not JAX LLM training.
- DeepSpeed: provide a ZeRO-2 optimizer-offload configuration and CUDA-only launch path. Apple hardware cannot validate distributed CUDA execution; configuration preparation is not a multi-GPU result.

No benchmark threshold or resume improvement percentage is assumed in advance. Report negative and inconclusive results. Training-stage completion and small-sample evaluation are separate claims.
