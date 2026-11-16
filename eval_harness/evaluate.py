from __future__ import annotations

import argparse
import gc
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from eval_harness.grading import grade, summarize
from post_training_lab.data import ROOT, load
from post_training_lab.train import ART, DEVICE, base, tokenizer, reward_tokenizer, prompt, load_reward


def logprob(model, tok, row, response):
    prefix = tok.encode(prompt(tok, row), add_special_tokens=False)
    suffix = tok.encode(response + tok.eos_token, add_special_tokens=False)
    ids = torch.tensor([prefix + suffix], device=DEVICE)
    with torch.no_grad():
        logp = F.log_softmax(model(input_ids=ids).logits[:, :-1].float(), dim=-1)
        labels = ids[:, 1:]
        values = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return values[:, len(prefix) - 1:].sum().item()


def model_for(stage):
    if stage == "base":
        model = base()
    elif stage == "sft-low-data":
        model = PeftModel.from_pretrained(base(), str(ART / stage))
    elif stage == "sft":
        model = base(ART / "sft-merged")
    else:
        model = PeftModel.from_pretrained(base(ART / "sft-merged"), str(ART / stage))
    model.config.use_cache = True
    return model.to(DEVICE).eval()


def evaluate(stage, split="test"):
    torch.set_num_threads(6)
    start = time.monotonic()
    tok = tokenizer()
    tok.padding_side = "left"
    rows = load(split)
    model = model_for(stage)
    records = []
    for i in range(0, len(rows), 4):
        batch = rows[i:i + 4]
        inputs = tok([prompt(tok, r) for r in batch], padding=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id,
                                     eos_token_id=tok.eos_token_id)
        for row, ids in zip(batch, outputs):
            response = tok.decode(ids[inputs.input_ids.shape[1]:], skip_special_tokens=True)
            delta = logprob(model, tok, row, row["chosen"]) - logprob(model, tok, row, row["rejected"])
            records.append({"id": row["id"], "domain": row["domain"], "response": response,
                            "grade": grade(row, response), "preference_logp_margin": delta,
                            "response_tokens": len(tok.encode(response, add_special_tokens=False))})
        print(f"{stage}: {min(i+4,len(rows))}/{len(rows)}", flush=True)
    stats = summarize(records)
    stats["preference_accuracy"] = sum(r["preference_logp_margin"] > 0 for r in records) / len(records)
    stats["mean_response_tokens"] = float(np.mean([r["response_tokens"] for r in records]))
    flags = np.array([r["grade"]["correct"] for r in records], dtype=float)
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(flags, len(flags), replace=True).mean() for _ in range(2000)])
    stats["accuracy_bootstrap_95ci"] = np.quantile(boot, [0.025, 0.975]).tolist()
    report = {"stage": stage, "split": split, "decoding": {"do_sample": False, "max_new_tokens": 48},
              "statistics": stats, "records": records, "elapsed_seconds": time.monotonic() - start,
              "limitations": ["one training seed", "36 synthetic test tasks", "no public leaderboard comparison", "single tool-call simulation"]}
    (ROOT / f"reports/{stage}_{split}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"stage": stage, "statistics": stats}), flush=True)


def evaluate_reward():
    model = load_reward()
    tok = reward_tokenizer()
    tok.padding_side = "right"
    records = []
    for row in load("test"):
        texts = [prompt(tok, row) + row[k] + tok.eos_token for k in ("chosen", "rejected")]
        inputs = tok(texts, padding=True, truncation=True, max_length=192, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            scores = model(**inputs).logits[:, 0].tolist()
        records.append({"id": row["id"], "domain": row["domain"], "chosen_score": scores[0], "rejected_score": scores[1], "correct": scores[0] > scores[1]})
    report = {"test_pairs": len(records), "accuracy": sum(r["correct"] for r in records) / len(records), "records": records}
    (ROOT / "reports/reward_test.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != "records"}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["base", "sft", "sft-low-data", "dpo", "ppo", "reward"])
    args = ap.parse_args()
    evaluate_reward() if args.stage == "reward" else evaluate(args.stage)
