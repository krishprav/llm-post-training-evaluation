# Measured local experiment results

One seed; 36 synthetic held-out tasks; fixed greedy decoding. This is a small portfolio experiment, not a public benchmark or a frontier-model capability claim.

| Condition | Overall correct | Coding | Reasoning | Tool routing | Output valid | Preference accuracy |
|---|---:|---:|---:|---:|---:|---:|
| base | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 94.4% |
| sft | 61.1% | 83.3% | 0.0% | 100.0% | 61.1% | 91.7% |
| sft-low-data | 55.6% | 66.7% | 0.0% | 100.0% | 55.6% | 94.4% |
| dpo | 55.6% | 66.7% | 0.0% | 100.0% | 55.6% | 91.7% |
| ppo | 61.1% | 83.3% | 0.0% | 100.0% | 61.1% | 91.7% |

Preference accuracy compares summed conditional log-likelihoods of the fixed chosen and near-miss responses; it is separate from generated-answer correctness. Exact generated outputs and all failed cases are retained in each JSON report.

## Paired comparisons

- sft minus base: +61.1% accuracy; paired bootstrap 95% interval [+47.2%, +77.8%].
- sft minus sft-low-data: +5.6% accuracy; paired bootstrap 95% interval [+0.0%, +13.9%].
- dpo minus sft: -5.6% accuracy; paired bootstrap 95% interval [-13.9%, +0.0%].
- ppo minus sft: +0.0% accuracy; paired bootstrap 95% interval [+0.0%, +0.0%].

## Reward model and implementation checks

Reward model held-out pair accuracy: 83.3% on 36 synthetic preference pairs.
Independent JAX/PyTorch objective and gradient checks: passed at tolerance 1e-10.

## Interpretation and limits

The data ablation fixes optimizer steps and changes training diversity from 144 rows to 36; the smaller set is therefore repeated more often. It is not a compute-normalized scaling law. DPO and PPO are alternative branches from SFT, not sequential DPO-then-PPO training.

PPO is four rollout batches, sufficient to exercise the generation/reward/value/policy loop but insufficient to establish convergence, safety or sustained reward improvement. Synthetic preference labels are not human judgments. A higher learned reward alone does not demonstrate task correctness.

Confidence intervals resample these 36 test items and do not measure training-seed variability or real-world distribution shift. Operand and argument ranges differ across splits; task families remain shared. No result was used to select a checkpoint or revise training settings.

The base model's 0% score is strict output-contract compliance on these prompts, not proof that it lacks arithmetic or coding knowledge. The observed SFT gain combines formatting and bounded task execution. Arithmetic remains at 0% under the bare-integer contract, and this experiment does not establish improved mathematical reasoning.

DeepSpeed ZeRO-2 configuration was prepared but distributed CUDA execution was not performed. JAX checks the objectives; the actual language-model training uses PyTorch. Docker configuration is supplied separately from the native execution evidence.
