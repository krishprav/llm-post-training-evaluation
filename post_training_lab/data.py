from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from eval_harness.grading import grade

ROOT = Path(__file__).resolve().parents[1]


def build() -> dict:
    splits = {"train": [], "validation": [], "test": []}
    for domain in ("reasoning", "coding", "agentic"):
        for family in range(3):
            for group in range(24):
                split = "train" if group < 16 else "validation" if group < 20 else "test"
                # Every parameter group is assigned to exactly one split.
                n = group + 2
                op = ("add", "sub", "mul")[family]
                sign = ("+", "-", "*")[family]
                if domain == "reasoning":
                    a, b = n * 2 + 1, n
                    answer = {"add": a + b, "sub": a - b, "mul": a * b}[op]
                    prompt = f"Calculate {a} {sign} {b}. Return only the integer answer."
                    chosen, rejected = str(answer), str(answer + 1)
                    target = {"a": a, "b": b, "op": op}
                elif domain == "coding":
                    action = (f"adds {n} to x", f"subtracts {n} from x", f"multiplies x by {n}")[family]
                    prompt = f"Return only a Python expression that {action}. Use x as the variable."
                    chosen, rejected = f"x {sign} {n}", f"x {sign} {n + 1}"
                    target = {"n": n, "op": op}
                else:
                    tool = ("weather", "calculator", "search")[family]
                    argument = (f"City{n}", f"{n}+{n+1}", f"topic{n}")[family]
                    request = (f"weather in {argument}", f"calculate {argument}", f"search for {argument}")[family]
                    prompt = f'Tools: weather(city), calculator(expression), search(query). User: {request}. Return JSON only with exactly two string keys: "tool" and "argument".'
                    chosen = json.dumps({"tool": tool, "argument": argument}, separators=(",", ":"))
                    rejected = json.dumps({"tool": ("calculator", "search", "weather")[family], "argument": argument}, separators=(",", ":"))
                    target = {"tool": tool, "argument": argument}
                row = {"id": f"{domain}-{family}-{group}", "group": group, "domain": domain,
                       "family": f"{domain}-{family}", "prompt": prompt, "chosen": chosen,
                       "rejected": rejected, "target": target, "label_source": "synthetic_specification"}
                assert grade(row, chosen)["correct"]
                assert not grade(row, rejected)["correct"]
                splits[split].append(row)
    manifest = {"seed": 42, "label_source": "synthetic_specification_not_human_feedback", "splits": {}}
    for split, rows in splits.items():
        raw = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        (ROOT / "data" / f"{split}.jsonl").write_text(raw)
        manifest["splits"][split] = {"rows": len(rows), "sha256": hashlib.sha256(raw.encode()).hexdigest()}
    # A separate blank-label queue supports human review without pretending it happened.
    rng = random.Random(42)
    queue = []
    for r in splits['validation']:
        responses = [r['chosen'], r['rejected']]
        rng.shuffle(responses)
        queue.append({"id": r["id"], "prompt": r["prompt"], "response_a": responses[0], "response_b": responses[1],
                      "human_preference": None, "reviewer": None, "rationale": None})
    (ROOT / "data" / "human_review_queue.json").write_text(json.dumps(queue, indent=2))
    (ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load(split: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / "data" / f"{split}.jsonl").read_text().splitlines()]


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
