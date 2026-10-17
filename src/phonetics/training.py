"""Matched-condition training over a checked experiment bundle."""

import hashlib
import json
from pathlib import Path

import torch

from .data import _sha, create_loaders, load_experiment
from .metrics import batch_statistics, finalize_statistics
from .model import build_encoder
from .reproducibility import seed_everything


def load_initialized_model(experiment: dict, seed: int, device="cpu"):
    if seed not in experiment["initializations"]: raise ValueError("seed has no checked initialization")
    model = build_encoder(experiment["config"], experiment["vocab"])
    model.load_state_dict(experiment["initializations"][seed]["state_dict"], strict=True)
    return model.to(device)


def _move(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}


def _epoch(model, loader, acoustic, condition, lambda_mse, device, optimizer=None):
    model.train(optimizer is not None); totals = {}
    for batch in loader:
        batch = _move(batch, device)
        if optimizer: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            outputs = model(batch["token_ids"], batch["token_offsets"], batch["char_lengths"], return_hidden=True)
            stats = batch_statistics(outputs, batch, acoustic, condition, lambda_mse)
            loss = stats["objective_sum"] / stats["characters"]
            if optimizer: loss.backward(); optimizer.step()
        for key, value in stats.items(): totals[key] = totals.get(key, 0.0) + (value.detach().item() if torch.is_tensor(value) else value)
    return finalize_statistics(totals, condition)


def train_experiment(experiment_path, *, condition: str, seed: int = 42, lambda_mse: float = 1.0,
                     runs_root=None, device=None):
    if condition not in {"ce_only", "mse_only", "combined"}: raise ValueError("invalid condition")
    if condition != "combined" and lambda_mse != 1.0: raise ValueError("lambda_mse only applies to combined")
    if condition == "combined" and (not torch.isfinite(torch.tensor(lambda_mse)) or lambda_mse < 0): raise ValueError("lambda_mse must be finite and nonnegative")
    experiment = load_experiment(experiment_path); config = experiment["config"]; device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_initialized_model(experiment, seed, device)
    seed_everything(seed)
    loaders = create_loaders(experiment, seed, num_workers=0)
    if condition == "mse_only":
        for head in (model.initial_head, model.final_head, model.tone_head):
            for parameter in head.parameters(): parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    suffix = Path(condition) / (f"lambda-{lambda_mse:g}" if condition == "combined" else "") / f"seed-{seed}"
    runs_base = Path(runs_root).resolve() if runs_root else experiment["root"].parent.parent / "runs"
    base = (runs_base / experiment["manifest"]["experiment_id"]).resolve()
    run = (base / suffix).resolve()
    if base not in run.parents or run.exists(): raise FileExistsError(f"unsafe or existing run directory: {run}")
    run.mkdir(parents=True)
    acoustic = experiment["acoustic_targets"].to(device); history=[]; best=float("inf")
    try:
        for epoch in range(config["training"]["epochs"]):
            train_metrics = _epoch(model, loaders["train"], acoustic, condition, lambda_mse, device, optimizer)
            validation_metrics = _epoch(model, loaders["validation"], acoustic, condition, lambda_mse, device)
            history.append({"epoch": epoch + 1, "train": train_metrics, "validation": validation_metrics})
            if validation_metrics["objective"] < best:
                best = validation_metrics["objective"]
                torch.save({"state_dict": model.state_dict(), "seed": seed, "condition": condition, "lambda_mse": lambda_mse,
                            "manifest_sha256": _sha(experiment["root"] / "manifest.json"), "config_sha256": _sha(experiment["root"] / "config.json"),
                            "vocab_sha256": _sha(experiment["root"] / "vocab.json")}, run / "best.pt")
        best_checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=True); model.load_state_dict(best_checkpoint["state_dict"], strict=True)
        test_metrics = _epoch(model, loaders["test"], acoustic, condition, lambda_mse, device)
        actual = {"experiment_id": experiment["manifest"]["experiment_id"], "seed": seed, "condition": condition, "lambda_mse": lambda_mse,
                  "device": str(device), "manifest_sha256": best_checkpoint["manifest_sha256"], "config": config,
                  "code_sha256": {p.name: _sha(p) for p in (Path(__file__), Path(__file__).with_name("metrics.py"),
                                      Path(__file__).with_name("data.py"), Path(__file__).with_name("model.py"),
                                      Path(__file__).with_name("reproducibility.py"))}}
        (run / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        (run / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2) + "\n", encoding="utf-8")
        (run / "actual_config.json").write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:
        raise
    return {"run_dir": run, "history": history, "test_metrics": test_metrics}
