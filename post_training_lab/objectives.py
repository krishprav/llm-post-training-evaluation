"""Independent JAX/PyTorch objective and gradient checks."""
import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F


def validate() -> dict:
    fixture = np.array([[0.0, 0.0], [1.4, -0.7], [-3.0, 2.0], [50.0, -50.0], [-50.0, 50.0]], dtype=np.float64)
    results = {}
    for name, torch_fn, jax_fn in [
        ("dpo", lambda x: F.softplus(-0.1 * (x[:, 0] - x[:, 1])).mean(),
         lambda x: jnp.logaddexp(0.0, -0.1 * (x[:, 0] - x[:, 1])).mean()),
        ("bradley_terry_reward", lambda x: F.softplus(-(x[:, 0] - x[:, 1])).mean(),
         lambda x: jnp.logaddexp(0.0, -(x[:, 0] - x[:, 1])).mean()),
    ]:
        t = torch.tensor(fixture, requires_grad=True)
        loss = torch_fn(t)
        loss.backward()
        jloss, jgrad = jax.value_and_grad(jax_fn)(jnp.asarray(fixture))
        results[name] = {"loss_error": abs(loss.item() - float(jloss)),
                         "gradient_max_error": float(np.max(np.abs(t.grad.numpy() - np.array(jgrad)))), "fixtures": len(fixture)}
    # Ratios are deliberately away from clipping boundaries, where subgradients differ.
    ratios = np.array([0.6, 0.9, 1.1, 1.4], dtype=np.float64)
    advantages = np.array([1.0, -0.7, 0.5, -1.2], dtype=np.float64)
    t = torch.tensor(ratios, requires_grad=True)
    loss = -torch.minimum(t * torch.tensor(advantages), torch.clamp(t, 0.8, 1.2) * torch.tensor(advantages)).mean()
    loss.backward()
    jloss, jgrad = jax.value_and_grad(lambda x: -jnp.minimum(x * advantages, jnp.clip(x, 0.8, 1.2) * advantages).mean())(ratios)
    results["ppo_clip"] = {"loss_error": abs(loss.item() - float(jloss)),
                           "gradient_max_error": float(np.max(np.abs(t.grad.numpy() - np.array(jgrad)))), "fixtures": len(ratios)}
    passed = all(r["loss_error"] < 1e-10 and r["gradient_max_error"] < 1e-10 for r in results.values())
    return {"passed": passed, "backend": jax.default_backend(), "tolerance": 1e-10, "results": results}


if __name__ == "__main__":
    report = validate()
    (Path(__file__).resolve().parents[1] / "reports/objective_parity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    assert report["passed"]
