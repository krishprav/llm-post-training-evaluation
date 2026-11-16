from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, set_seed
from trl import (SFTTrainer, SFTConfig, DPOTrainer, DPOConfig, RewardTrainer, RewardConfig,
                 DataCollatorForCompletionOnlyLM, PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead)
from post_training_lab.data import ROOT, load
from post_training_lab.model_source import model_spec, resolve_model_path

ART = ROOT / "artifacts"
REPORTS = ROOT / "reports"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def info() -> dict:
    return {**model_spec(), "local_path": resolve_model_path()}


def tokenizer():
    tok = AutoTokenizer.from_pretrained(info()["local_path"], local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def reward_tokenizer():
    tok = tokenizer()
    # Chat EOS occurs between user and assistant turns. It cannot also be the
    # classifier padding marker: this Transformers version pools before first pad.
    tok.pad_token = tok.unk_token
    assert tok.pad_token_id is not None and tok.pad_token_id != tok.eos_token_id
    tok.padding_side = 'right'
    return tok


def prompt(tok, row):
    return tok.apply_chat_template([{"role": "user", "content": row["prompt"]}], tokenize=False, add_generation_prompt=True)


def lora(task=TaskType.CAUSAL_LM):
    return LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], task_type=task)


def base(path=None, classification=False):
    cls = AutoModelForSequenceClassification if classification else AutoModelForCausalLM
    kw = {"num_labels": 1} if classification else {}
    model = cls.from_pretrained(str(path or info()["local_path"]), local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager", **kw)
    model.config.pad_token_id = reward_tokenizer().pad_token_id if classification else tokenizer().pad_token_id
    model.config.use_cache = False
    return model


def common(name, steps, batch, lr, deepspeed=None):
    if deepspeed and not torch.cuda.is_available():
        raise RuntimeError("DeepSpeed launch requires a CUDA environment; local MPS evidence does not validate ZeRO execution")
    return dict(output_dir=str(ART / name), max_steps=steps, per_device_train_batch_size=batch,
                learning_rate=lr, logging_steps=max(1, steps // 8), save_strategy="no", report_to=[],
                seed=42, dataloader_pin_memory=False, optim="adamw_torch",
                warmup_ratio=0.0, weight_decay=0.0, max_grad_norm=1.0, disable_tqdm=True,
                use_cpu=DEVICE == "cpu", deepspeed=deepspeed)


def record(name, data):
    data.update({"stage": name, "device": DEVICE, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "model_id": info()["model_id"], "model_revision": info()["revision"],
                 "packages": {p: importlib.metadata.version(p) for p in ('torch','transformers','trl','peft','accelerate')}})
    (REPORTS / f"{name}_training.json").write_text(json.dumps(data, indent=2, default=float))
    print(json.dumps(data, default=float), flush=True)


def sft(name, low_data=False, deepspeed=None):
    set_seed(42)
    start = time.monotonic()
    tok = tokenizer()
    rows = load("train")
    if low_data:
        rows = [r for r in rows if r["group"] < 4]
    dataset = Dataset.from_list([{"text": prompt(tok, r) + r["chosen"] + tok.eos_token} for r in rows])
    response_ids = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(response_template=response_ids, tokenizer=tok)
    config = SFTConfig(**common(name, 64, 4, 1e-3, deepspeed), max_seq_length=192, dataset_text_field="text", packing=False)
    trainer = SFTTrainer(model=base(), tokenizer=tok, train_dataset=dataset, args=config, peft_config=lora(), data_collator=collator)
    trainable, total = trainer.model.get_nb_trainable_parameters()
    result = trainer.train()
    trainer.save_model(str(ART / name))
    tok.save_pretrained(str(ART / name))
    record(name, {"metrics": result.metrics, "train_rows": len(rows), "trainable_parameters": trainable,
                  "total_parameters": total, "trainable_fraction": trainable / total,
                  "log_history": trainer.state.log_history, "elapsed_seconds": time.monotonic() - start})
    if name == "sft":
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(str(ART / "sft-merged"), safe_serialization=True)
        tok.save_pretrained(str(ART / "sft-merged"))


def dpo(deepspeed=None):
    set_seed(42)
    start = time.monotonic()
    tok = tokenizer()
    dataset = Dataset.from_list([{k: prompt(tok, r) if k == "prompt" else r[k] + tok.eos_token for k in ("prompt", "chosen", "rejected")} for r in load("train")])
    config = DPOConfig(**common("dpo", 32, 2, 2e-4, deepspeed), beta=0.1, max_length=192,
                       max_prompt_length=128, precompute_ref_log_probs=True)
    trainer = DPOTrainer(model=base(ART / "sft-merged"), ref_model=None, args=config,
                         tokenizer=tok, train_dataset=dataset, peft_config=lora())
    result = trainer.train()
    trainer.save_model(str(ART / "dpo"))
    record("dpo", {"metrics": result.metrics, "train_rows": len(dataset), "beta": 0.1,
                   "reference": "frozen SFT base with DPO adapter disabled", "log_history": trainer.state.log_history,
                   "elapsed_seconds": time.monotonic() - start})


def reward(deepspeed=None):
    set_seed(42)
    start = time.monotonic()
    tok = reward_tokenizer()
    records = []
    for r in load("train"):
        row = {}
        for kind in ("chosen", "rejected"):
            enc = tok(prompt(tok, r) + r[kind] + tok.eos_token, truncation=True, max_length=192)
            assert tok.pad_token_id not in enc['input_ids'], 'reserved reward pad token occurred in the example'
            row[f"input_ids_{kind}"] = enc["input_ids"]
            row[f"attention_mask_{kind}"] = enc["attention_mask"]
        records.append(row)
    config = RewardConfig(**common("reward", 40, 4, 5e-4, deepspeed), max_length=192)
    trainer = RewardTrainer(model=base(classification=True), tokenizer=tok,
                            train_dataset=Dataset.from_list(records), args=config, peft_config=lora(TaskType.SEQ_CLS))
    result = trainer.train()
    trainer.save_model(str(ART / "reward"))
    record("reward", {"metrics": result.metrics, "train_rows": len(records), "labels": "synthetic specification preferences",
                      "objective": "Bradley-Terry pairwise logistic", "log_history": trainer.state.log_history,
                      "elapsed_seconds": time.monotonic() - start})


def load_reward():
    model = PeftModel.from_pretrained(base(classification=True), str(ART / "reward"))
    return model.to(DEVICE).eval()


def ppo():
    set_seed(42)
    start = time.monotonic()
    tok = tokenizer()
    tok.padding_side = "left"
    config = PPOConfig(model_name=str(ART / "sft-merged"), learning_rate=1e-5, batch_size=4,
                       mini_batch_size=2, ppo_epochs=2, seed=42, log_with=None)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(str(ART / "sft-merged"), peft_config=lora(), torch_dtype=torch.float32)
    model.pretrained_model.config.pad_token_id = tok.pad_token_id
    model.pretrained_model.config.use_cache = True
    trainer = PPOTrainer(config, model, ref_model=None, tokenizer=tok)
    rm = load_reward()
    rm_tok = reward_tokenizer()
    rows = load("train")
    history = []
    for step in range(4):
        # Draw from all domains rather than the first contiguous family.
        batch = [rows[(step * 17 + j * 37) % len(rows)] for j in range(4)]
        query = [torch.tensor(tok.encode(prompt(tok, r)), device=trainer.accelerator.device) for r in batch]
        responses = trainer.generate(query, return_prompt=False, max_new_tokens=24, do_sample=True,
                                     top_k=0, top_p=1.0, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        text = [tok.decode(t, skip_special_tokens=True) for t in responses]
        reward_inputs = rm_tok([prompt(rm_tok, r) + t + rm_tok.eos_token for r, t in zip(batch, text)], padding=True, truncation=True, max_length=192, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            rewards = rm(**reward_inputs).logits[:, 0].clamp(-5, 5)
        scores = [r.detach().to(trainer.accelerator.device) for r in rewards]
        stats = trainer.step(query, responses, scores)
        history.append({"step": step + 1, "mean_learned_reward": rewards.mean().item(),
                        "kl": float(stats["objective/kl"]), "policy_loss": float(stats["ppo/loss/policy"]),
                        "value_loss": float(stats["ppo/loss/value"]), "responses": text})
        print(json.dumps(history[-1]), flush=True)
    (ART / 'ppo').mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(ART / "ppo"))
    record("ppo", {"rollout_batches": 4, "responses_sampled": 16, "ppo_epochs": 2, "history": history,
                   "reward_source": "learned synthetic-preference reward model", "elapsed_seconds": time.monotonic() - start})


def manifest():
    data = {"platform": platform.platform(), "python": platform.python_version(), "device": DEVICE,
            "seed": 42, "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "jax")},
            "model": info(), "plan_sha256": hashlib.sha256((ROOT / "EXPERIMENT_PLAN.md").read_bytes()).hexdigest(),
            "deepspeed_execution": "not run; CUDA hardware unavailable", "human_preference_collection": "not performed"}
    (REPORTS / "environment.json").write_text(json.dumps(data, indent=2))
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sft", "sft-low-data", "dpo", "reward", "ppo", "manifest"])
    ap.add_argument("--deepspeed", default=None)
    args = ap.parse_args()
    ART.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    torch.set_num_threads(6)
    {"sft": lambda: sft("sft", deepspeed=args.deepspeed),
     "sft-low-data": lambda: sft("sft-low-data", low_data=True, deepspeed=args.deepspeed),
     "dpo": lambda: dpo(args.deepspeed), "reward": lambda: reward(args.deepspeed), "ppo": ppo,
     "manifest": manifest}[args.stage]()
