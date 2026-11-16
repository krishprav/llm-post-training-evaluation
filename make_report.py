from pathlib import Path
import json
import hashlib
import numpy as np
from eval_harness.grading import grade, summarize
from post_training_lab.data import load

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
stages = ["base", "sft", "sft-low-data", "dpo", "ppo"]
reports = {s: json.loads((REPORTS / f"{s}_test.json").read_text()) for s in stages}
tasks = {r['id']:r for r in load('test')}
verifier_hash = hashlib.sha256((ROOT / 'eval_harness/grading.py').read_bytes()).hexdigest()
for stage, report in reports.items():
    changes = 0
    for record in report['records']:
        checked = grade(tasks[record['id']], record['response'])
        changes += checked != record['grade']
        record['grade'] = checked
    report['statistics'].update(summarize(report['records']))
    flags = np.array([r['grade']['correct'] for r in report['records']], dtype=float)
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(flags,len(flags),replace=True).mean() for _ in range(2000)])
    report['statistics']['accuracy_bootstrap_95ci'] = np.quantile(boot,[0.025,0.975]).tolist()
    report['verifier_sha256'] = verifier_hash
    report['final_regrade_changes'] = changes
    (REPORTS / f'{stage}_test.json').write_text(json.dumps(report,indent=2))
lines = ["# Measured local experiment results", "", "One seed; 36 synthetic held-out tasks; fixed greedy decoding. This is a small portfolio experiment, not a public benchmark or a frontier-model capability claim.", "",
         "| Condition | Overall correct | Coding | Reasoning | Tool routing | Output valid | Preference accuracy |", "|---|---:|---:|---:|---:|---:|---:|"]
for stage, report in reports.items():
    s = report["statistics"]
    vals = [s["all"]["accuracy"], s["coding"]["accuracy"], s["reasoning"]["accuracy"], s["agentic"]["accuracy"], s["all"]["validity"], s["preference_accuracy"]]
    lines.append("| " + stage + " | " + " | ".join(f"{v:.1%}" for v in vals) + " |")
lines += ["", "Preference accuracy compares summed conditional log-likelihoods of the fixed chosen and near-miss responses; it is separate from generated-answer correctness. Exact generated outputs and all failed cases are retained in each JSON report.", "", "## Paired comparisons", ""]
comparison = {}
for left, right in [("sft", "base"), ("sft", "sft-low-data"), ("dpo", "sft"), ("ppo", "sft")]:
    a = {r["id"]: r for r in reports[left]["records"]}
    b = {r["id"]: r for r in reports[right]["records"]}
    assert set(a) == set(b)
    differences = np.array([int(a[k]["grade"]["correct"]) - int(b[k]["grade"]["correct"]) for k in sorted(a)])
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(differences, len(differences), replace=True).mean() for _ in range(2000)])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    comparison[f"{left}_minus_{right}"] = {"accuracy_delta": differences.mean(), "paired_bootstrap_95ci": [lo, hi]}
    lines.append(f"- {left} minus {right}: {differences.mean():+.1%} accuracy; paired bootstrap 95% interval [{lo:+.1%}, {hi:+.1%}].")
rm = json.loads((REPORTS / "reward_test.json").read_text())
parity = json.loads((REPORTS / "objective_parity.json").read_text())
lines += ["", "## Reward model and implementation checks", "", f"Reward model held-out pair accuracy: {rm['accuracy']:.1%} on {rm['test_pairs']} synthetic preference pairs.",
          f"Independent JAX/PyTorch objective and gradient checks: {'passed' if parity['passed'] else 'failed'} at tolerance {parity['tolerance']}.", "",
          "## Interpretation and limits", "",
          "The data ablation fixes optimizer steps and changes training diversity from 144 rows to 36; the smaller set is therefore repeated more often. It is not a compute-normalized scaling law. DPO and PPO are alternative branches from SFT, not sequential DPO-then-PPO training.", "",
          "PPO is four rollout batches, sufficient to exercise the generation/reward/value/policy loop but insufficient to establish convergence, safety or sustained reward improvement. Synthetic preference labels are not human judgments. A higher learned reward alone does not demonstrate task correctness.", "",
          "Confidence intervals resample these 36 test items and do not measure training-seed variability or real-world distribution shift. Operand and argument ranges differ across splits; task families remain shared. No result was used to select a checkpoint or revise training settings.", "",
          "The base model's 0% score is strict output-contract compliance on these prompts, not proof that it lacks arithmetic or coding knowledge. The observed SFT gain combines formatting and bounded task execution. Arithmetic remains at 0% under the bare-integer contract, and this experiment does not establish improved mathematical reasoning.", "",
          "DeepSpeed ZeRO-2 configuration was prepared but distributed CUDA execution was not performed. JAX checks the objectives; the actual language-model training uses PyTorch. Docker configuration is supplied separately from the native execution evidence.", ""]
(REPORTS / "RESULTS.md").write_text("\n".join(lines))
(REPORTS / "comparisons.json").write_text(json.dumps(comparison, indent=2, default=float))
print("\n".join(lines))
