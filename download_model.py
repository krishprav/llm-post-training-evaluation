"""Download the pinned public checkpoint without changing tracked configuration."""
from post_training_lab.model_source import download_checkpoint, model_spec


if __name__ == "__main__":
    spec = model_spec()
    path = download_checkpoint()
    print(f"Pinned {spec['model_id']} at {spec['revision']}")
    print(f"Cached locally at {path}")
