"""Verify completion evidence without assuming training improved model accuracy."""
from pathlib import Path
import hashlib
import json
import math

from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent

def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True

checks = {}
for name, steps in [('sft', 64), ('sft-low-data', 64), ('dpo', 32), ('reward', 40)]:
    report = json.loads((ROOT / f'reports/{name}_training.json').read_text())
    assert finite(report), f'nonfinite training metrics: {name}'
    assert report['log_history'][-1]['step'] == steps
    weights = load_file(str(ROOT / f'artifacts/{name}/adapter_model.safetensors'))
    adapter_magnitude = sum(t.abs().sum().item() for k,t in weights.items() if 'lora_B' in k)
    assert adapter_magnitude > 0, f'unchanged adapter: {name}'
    checks[name] = {'optimizer_steps': steps, 'adapter_lora_B_absolute_sum': adapter_magnitude, 'metrics_finite': True}

ppo = json.loads((ROOT / 'reports/ppo_training.json').read_text())
assert ppo['rollout_batches'] == 4 and len(ppo['history']) == 4 and finite(ppo)
weights = load_file(str(ROOT / 'artifacts/ppo/adapter_model.safetensors'))
magnitude = sum(t.abs().sum().item() for k,t in weights.items() if 'lora_B' in k)
assert magnitude > 0
checks['ppo'] = {'rollout_batches': 4, 'responses_sampled': 16, 'adapter_lora_B_absolute_sum': magnitude, 'metrics_finite': True}

expected = set()
for stage in ['base','sft','sft-low-data','dpo','ppo']:
    report = json.loads((ROOT / f'reports/{stage}_test.json').read_text())
    assert finite(report)
    ids = {r['id'] for r in report['records']}
    assert len(ids) == 36
    expected = expected or ids
    assert ids == expected
    checks[f'{stage}_eval'] = {'unique_test_rows': len(ids), 'finite': True}

parity = json.loads((ROOT / 'reports/objective_parity.json').read_text())
assert parity['passed']
manifest = json.loads((ROOT / 'data/manifest.json').read_text())
for split, data in manifest['splits'].items():
    assert hashlib.sha256((ROOT / f'data/{split}.jsonl').read_bytes()).hexdigest() == data['sha256']
checks['split_hashes_match'] = True
checks['jax_objective_parity'] = True
result = {'passed': True, 'checks': checks,
          'not_claimed': ['human preference collection', 'DeepSpeed CUDA execution', 'multi-GPU training', 'Docker image execution', 'public benchmark or frontier capability gains']}
(ROOT / 'reports/evidence_audit.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
