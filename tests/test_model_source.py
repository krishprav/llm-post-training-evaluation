import json

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from post_training_lab import model_source


def test_cached_checkpoint_resolution_is_offline_and_revision_pinned(monkeypatch):
    calls = []

    def snapshot(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return "/another-machine/cache/checkpoint"

    monkeypatch.setattr(model_source, "snapshot_download", snapshot)
    assert model_source.resolve_model_path() == "/another-machine/cache/checkpoint"
    model_id, kwargs = calls[0]
    assert model_id == model_source.model_spec()["model_id"]
    assert kwargs["revision"] == model_source.model_spec()["revision"]
    assert kwargs["local_files_only"] is True
    assert "local_path" not in model_source.model_spec()


def test_missing_cache_explains_download_step(monkeypatch):
    def missing(*args, **kwargs):
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(model_source, "snapshot_download", missing)
    with pytest.raises(RuntimeError, match="download_model.py"):
        model_source.resolve_model_path()


def test_download_keeps_tracked_config_unchanged(monkeypatch):
    before = model_source.CONFIG.read_bytes()
    calls = []

    def snapshot(model_id, **kwargs):
        calls.append(kwargs)
        return "/cache/checkpoint"

    monkeypatch.setattr(model_source, "snapshot_download", snapshot)
    assert model_source.download_checkpoint() == "/cache/checkpoint"
    assert calls[0]["revision"] == model_source.model_spec()["revision"]
    assert "local_files_only" not in calls[0]
    assert model_source.CONFIG.read_bytes() == before


def test_mutable_model_revision_is_rejected(tmp_path, monkeypatch):
    config = tmp_path / "model.json"
    config.write_text(json.dumps({"model_id": "example/model", "revision": "main"}))
    monkeypatch.setattr(model_source, "CONFIG", config)
    with pytest.raises(ValueError, match="pinned Hub commit"):
        model_source.model_spec()
