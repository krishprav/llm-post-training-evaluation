from pathlib import Path
import json
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent
target = ROOT / "configs/model.json"
if target.exists():
    config = json.loads(target.read_text())
    revision = config["revision"]
else:
    revision = HfApi().model_info("HuggingFaceTB/SmolLM2-135M-Instruct").sha
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
path = snapshot_download(model_id, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
target.write_text(json.dumps({"model_id": model_id, "revision": revision, "local_path": path}, indent=2))
print(f"Pinned {model_id} at {revision}")
