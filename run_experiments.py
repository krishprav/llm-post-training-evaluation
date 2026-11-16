"""Run the fixed experiment once; preserve a log and timing for every stage."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
stages = [
    ["post_training_lab.data"], ["post_training_lab.objectives"], ["post_training_lab.train", "manifest"],
    ["eval_harness.evaluate", "base"], ["post_training_lab.train", "sft"],
    ["post_training_lab.train", "sft-low-data"], ["post_training_lab.train", "dpo"],
    ["post_training_lab.train", "reward"], ["post_training_lab.train", "ppo"],
    ["eval_harness.evaluate", "sft"], ["eval_harness.evaluate", "sft-low-data"],
    ["eval_harness.evaluate", "dpo"], ["eval_harness.evaluate", "ppo"], ["eval_harness.evaluate", "reward"],
]
parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true', help='Skip already successful stages and preserve prior attempts')
parser.add_argument('--restart-from', default=None, help='Rerun this stage and all later stages after an implementation fix')
args_cli = parser.parse_args()
history_file = ROOT / 'reports/run_history.json'
history = json.loads(history_file.read_text()) if args_cli.resume and history_file.exists() else []
force = False
for args in stages:
    if args_cli.restart_from:
        if args_cli.restart_from in args:
            force = True
        if not force:
            continue
    if args_cli.resume and not force and any(r['stage'] == args and r['exit_code'] == 0 for r in history):
        print(f'SKIP completed {args}', flush=True)
        continue
    label = "_".join(args).replace(".", "-")
    log = ROOT / "reports" / f"{label}.log"
    start = time.monotonic()
    print(f"START {label}", flush=True)
    with log.open("w") as stream:
        result = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    history.append({"stage": args, "exit_code": result.returncode, "elapsed_seconds": time.monotonic() - start, "log": str(log.relative_to(ROOT))})
    (ROOT / "reports/run_history.json").write_text(json.dumps(history, indent=2))
    if result.returncode:
        print(log.read_text()[-6000:], flush=True)
        raise SystemExit(result.returncode)
    print(f"DONE {label}: {history[-1]['elapsed_seconds']:.1f}s", flush=True)
