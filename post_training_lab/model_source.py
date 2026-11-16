"""Resolve a pinned checkpoint from each machine's own Hugging Face cache."""
from __future__ import annotations

import json
import re
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

CONFIG = Path(__file__).resolve().parents[1] / "configs/model.json"
MODEL_FILES = ["*.json", "*.safetensors", "*.txt", "*.model"]


def model_spec() -> dict[str, str]:
    config = json.loads(CONFIG.read_text())
    if not config.get("model_id") or not re.fullmatch(r"[0-9a-f]{40}", config.get("revision", "")):
        raise ValueError("Model configuration requires a model ID and a full pinned Hub commit.")
    return {key: config[key] for key in ("model_id", "revision")}


def download_checkpoint() -> str:
    spec = model_spec()
    return snapshot_download(spec["model_id"], revision=spec["revision"], allow_patterns=MODEL_FILES)


def resolve_model_path() -> str:
    spec = model_spec()
    try:
        return snapshot_download(spec["model_id"], revision=spec["revision"],
                                 allow_patterns=MODEL_FILES, local_files_only=True)
    except LocalEntryNotFoundError as exc:
        raise RuntimeError("Pinned checkpoint is not cached. Run python download_model.py first.") from exc
