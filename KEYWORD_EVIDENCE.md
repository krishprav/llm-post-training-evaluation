# Caudal role and application coverage

| Requirement | Concrete project evidence | Accurate resume scope |
|---|---|---|
| SFT / fine-tuning LLMs | `train.py:sft`, actual adapter and run log | Local SmolLM2-135M fine-tuning |
| RLHF | SFT, reward model and PPO path; blank review queue | RLHF-style prototype with synthetic preferences, not human-label collection |
| Reward modeling | `train.py:reward`, Bradley-Terry objective, test-pair report | Actual compact reward-model training |
| Preference optimization DPO/PPO | Separate branches from SFT; preserved stage logs | Actual short local training runs, not convergence or frontier-scale gains |
| LLM evaluation / benchmarks | `eval_harness`, per-output reports | 36 synthetic test tasks, bounded coding/reasoning/tool routing |
| Python / PyTorch | Model loading, training, inference, graders | Executed |
| Hugging Face / TRL | Transformers, Datasets, SFT/DPO/Reward/PPO trainers | Executed with pinned compatible versions |
| LoRA / PEFT | Rank-8 query/value adapters | Executed, trainable parameter count recorded |
| JAX | Independent losses and gradients | CPU objective validation only |
| DeepSpeed | ZeRO-2 offload config and CUDA-gated launch argument | Configuration supplied; distributed CUDA execution untested |
| Docker | Portable test image definition; prior resume experience | Portfolio image supplied; native tests are the actual new execution evidence |
| Ablation studies | 144-row vs 36-row fixed-step SFT comparison | Data-diversity/repetition ablation, one seed |
| Coding / reasoning / agentic tasks | Independent AST, arithmetic and JSON routing graders | Agentic scope is simulated single tool-call selection |
| Human feedback / reusable datasets | Randomized blank human-review queue, data card, split hashes | Review workflow prepared; actual labels remain synthetic |
| Research notes and reports | Plan, results, environment, failed outputs, interview guide | Generated from actual saved results |

Contact details, GitHub, LinkedIn, degree and graduation year are on the resume. Compensation, joining date, application email recording and file-upload instructions belong in the form and are not technical resume keywords.
