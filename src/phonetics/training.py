"""Sequential seed-specific character training runs for the selected objective."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import statistics
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .artifacts import sha256_file, write_json
from .data import create_loaders
from .experiment import load_experiment, select_seed, training_epoch_limit
from .metrics import batch_objective, finalize_epoch
from .model import build_encoder, shared_encoder_sha256
from .objectives import canonical_model_id, parse_model_id
from .reproducibility import rng_snapshot, seed_everything

CODE_FILES = ("training.py", "model.py", "data.py", "metrics.py", "reproducibility.py", "experiment.py",
              "training_data.py", "objectives.py", "pinyin.py", "artifacts.py", "data_contract.py", "bundles.py", "acoustic.py", "corpus.py", "sound_inventory.py", "architecture.py")


class TrainingStepFailure(FloatingPointError):
    """A non-finite training step with JSON-safe diagnostic context."""

    def __init__(self, message, diagnostic):
        super().__init__(message)
        self.diagnostic = diagnostic


def _nonfinite_status(value):
    if math.isnan(value): return "nan"
    return "positive_infinity" if value > 0 else "negative_infinity"


def _gradient_summary(norm_values, nonfinite_count):
    return {"mean": statistics.fmean(norm_values) if norm_values else 0.0,
            "std": statistics.pstdev(norm_values) if len(norm_values) > 1 else 0.0,
            "max": max(norm_values) if norm_values else 0.0,
            "nonfinite_count": nonfinite_count}


def _failure_diagnostic(event, gradient_norms, nonfinite_count):
    """Retain successful work in the current epoch when a later step fails."""
    return {"nonfinite_count": int(nonfinite_count),
            "gradient_norms": list(gradient_norms), "event": event}


def _jsonable_metrics(values):
    return {key: (float(value.detach().cpu()) if torch.is_tensor(value) else value) for key, value in values.items()}


def _move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _gradient_norm(model):
    squared_norm = torch.zeros((), dtype=torch.float64, device="cpu")
    finite = True
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        if not bool(torch.isfinite(grad).all()):
            finite = False
        squared_norm += grad.to(device="cpu", dtype=torch.float64).square().sum()
    value = float(torch.sqrt(squared_norm).item())
    return value, finite and math.isfinite(value)


def _epoch(model, loader, acoustic_targets, model_id, device, *, optimizer=None, event_writer=None,
           epoch_number=None, step_start=0):
    training = optimizer is not None
    model.train(training)
    totals = {}
    gradient_norms = []
    nonfinite_gradients = 0
    step = step_start
    for batch in loader:
        batch = _move(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(batch)
            stats = batch_objective(outputs, batch, acoustic_targets, model_id)
            objective = stats["objective"]
            if not bool(torch.isfinite(objective)):
                value = float(objective.detach().cpu())
                record = {"event": "nonfinite_loss", "epoch": epoch_number, "step": step,
                          "loss": None, "loss_status": _nonfinite_status(value)}
                if event_writer: event_writer(record)
                raise TrainingStepFailure(f"non-finite loss at epoch {epoch_number}, step {step}",
                                          _failure_diagnostic(record, gradient_norms, nonfinite_gradients))
            if training:
                objective.backward()
                grad_norm, finite_grad = _gradient_norm(model)
                if not finite_grad:
                    nonfinite_gradients += 1
                    record = {"event": "nonfinite_gradient", "epoch": epoch_number, "step": step,
                              "gradient_norm": None, "gradient_norm_status": _nonfinite_status(grad_norm)}
                    if event_writer: event_writer(record)
                    raise TrainingStepFailure(f"non-finite gradient at epoch {epoch_number}, step {step}",
                                              _failure_diagnostic(record, gradient_norms, nonfinite_gradients))
                gradient_norms.append(grad_norm)
                optimizer.step()
            else:
                grad_norm = None
        detached = _jsonable_metrics(stats)
        for key, value in detached.items():
            if key == "objective":
                continue
            totals[key] = totals.get(key, 0) + value
        if event_writer:
            event_writer({"event": "step", "epoch": epoch_number, "step": step,
                          "losses_per_character": {key.removesuffix("_sum"): value / stats["characters"]
                                                     for key, value in detached.items() if key.endswith("_sum")},
                          "objective": detached["objective"], "gradient_norm": grad_norm})
        step += 1
    result = finalize_epoch(totals)
    result["steps"] = step - step_start
    result["gradient_norms"] = gradient_norms
    result["gradient_nonfinite_count"] = nonfinite_gradients
    return result, step


def _state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode()); digest.update(repr(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _test_summary(seed_results):
    metrics = {}
    completed = [row for row in seed_results if "test_metrics" in row]
    keys = sorted(set.intersection(*(set(row["test_metrics"]) for row in completed))) if completed else []
    for key in keys:
        values = [float(row["test_metrics"][key]) for row in completed]
        metrics[key] = {"mean": statistics.fmean(values), "std": statistics.pstdev(values)}
    return metrics


def _write_run_manifest(run_dir, manifest):
    files = {}
    for file in sorted(run_dir.rglob("*")):
        if file.is_file() and file.name != "manifest.json":
            files[file.relative_to(run_dir).as_posix()] = sha256_file(file)
    manifest["artifacts"] = files
    write_json(run_dir / "manifest.json", manifest)


def train_experiment(experiment_path, *, model_id: str, runs_root=None, device: str | None = None,
                     run_id: str | None = None, expected_manifest_sha256: str | None = None):
    """Train the selected model over configured seeds, sequentially, without resume."""
    model_id = canonical_model_id(model_id)
    _, lambda_cosine = parse_model_id(model_id)
    experiment = load_experiment(experiment_path)
    experiment_root = Path(experiment["root"])
    if expected_manifest_sha256 is not None:
        actual_manifest_sha256 = sha256_file(experiment_root / "manifest.json")
        if actual_manifest_sha256 != expected_manifest_sha256:
            raise ValueError("experiment manifest does not match the explicitly selected input pin")
    config = experiment["config"]
    if model_id not in config["model_ids"]:
        raise ValueError("selected model_id is not configured in the experiment")
    epoch_limit = training_epoch_limit(config, model_id)
    seeds = experiment["manifest"]["identity"]["training_seeds"]
    requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    target_device = torch.device(requested_device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"selected device {requested_device!r} requires CUDA")
    experiment_id = experiment["manifest"]["identity"]["experiment_id"]
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be a safe path component")
    default_root = Path(__file__).resolve().parents[2] / "runs"
    output_root = Path(runs_root).resolve() if runs_root else default_root
    run_dir = output_root / experiment_id / model_id / run_id
    if run_dir.exists():
        raise FileExistsError(f"run already exists: {run_dir}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=False, exist_ok=False)
    manifest = {
        "schema_version": 1, "run_type": "training_run", "status": "running", "run_id": run_id,
        "experiment_id": experiment_id, "model_id": model_id, "lambda_cosine": lambda_cosine, "device": str(target_device), "seeds": seeds,
        "selection": "minimum per-seed validation objective; strict improvement keeps earliest tie",
        "experiment_hashes": {name: sha256_file(experiment_root / name) for name in
                               ("manifest.json", "config.json", "labels.json", "corpus.jsonl", "splits.json", "targets.npy")},
        "config": config, "epoch_limit": epoch_limit,
        "code_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in CODE_FILES},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                    "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda},
        "seed_results": [], "summary": None, "artifacts": {},
    }
    write_json(run_dir / "manifest.json", manifest)
    seed_results = []
    statuses = {str(seed): "pending" for seed in seeds}
    global_step = 0
    active_seed = None
    try:
        for seed in seeds:
            active_seed = seed
            norm_values = []
            nonfinite_count = 0
            seed_dir = run_dir / f"seed-{seed}"
            seed_dir.mkdir()
            statuses[str(seed)] = "running"
            event_path = seed_dir / "events.jsonl"
            def log_event(event):
                with event_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            seed_input = select_seed(experiment, seed)
            seed_inputs = json.loads((experiment_root / "seeds" / str(seed) / "input_hashes.json").read_text(encoding="utf-8"))
            seed_everything(seed)
            model = build_encoder(config, seed_input["vocab"], model_id=model_id, seed=seed).to(target_device)
            initial_state = {key: value.detach().cpu().clone() for key, value in model.encoder.state_dict().items()}
            shared_hash = _state_hash(initial_state)
            torch.save({"state_dict": initial_state, "seed": seed, "sha256": shared_hash,
                        "seed_input_hashes": seed_inputs, "experiment_hashes": manifest["experiment_hashes"]},
                       seed_dir / "shared_encoder.pt")
            loaders = create_loaders(seed_input, seed, batch_size=config["training"]["batch_size"], num_workers=0)
            acoustic = torch.as_tensor(experiment["acoustic_targets"], dtype=torch.float32, device=target_device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=0.0)
            best = math.inf
            best_epoch = None
            best_validation = None
            epoch_history = []
            for epoch_num in range(1, epoch_limit + 1):
                train_metrics, global_step = _epoch(model, loaders["train"], acoustic, model_id, target_device,
                                                    optimizer=optimizer, event_writer=log_event,
                                                    epoch_number=epoch_num, step_start=global_step)
                norm_values.extend(train_metrics.pop("gradient_norms"))
                nonfinite_count += train_metrics.pop("gradient_nonfinite_count")
                validation_metrics, _ = _epoch(model, loaders["validation"], acoustic, model_id, target_device,
                                               event_writer=log_event, epoch_number=epoch_num)
                val_objective = float(validation_metrics["objective"])
                epoch_history.append({"epoch": epoch_num, "train": train_metrics, "validation": validation_metrics})
                if val_objective < best:
                    best, best_epoch, best_validation = val_objective, epoch_num, validation_metrics
                    torch.save({"state_dict": model.state_dict(), "model_id": model_id, "lambda_cosine": lambda_cosine, "seed": seed,
                                "best_epoch": best_epoch, "validation_objective": best,
                                "shared_encoder_sha256": shared_hash,
                                "experiment_hashes": manifest["experiment_hashes"],
                                "config_sha256": sha256_file(experiment_root / "config.json"),
                                "vocab_sha256": sha256_file(experiment_root / "seeds" / str(seed) / "vocabulary.json"),
                                "seed_input_hashes": seed_inputs}, seed_dir / "best.pt")
                torch.save({"state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                            "seed": seed, "epoch": epoch_num, "rng_state": rng_snapshot(loaders["train"].generator),
                            "model_id": model_id, "seed_input_hashes": seed_inputs,
                            "experiment_hashes": manifest["experiment_hashes"],
                            "shared_encoder_sha256": shared_hash},
                           seed_dir / "last.pt")
                write_json(seed_dir / "history.json", {"epochs": epoch_history})
                log_event({"event": "epoch", "epoch": epoch_num, "train": train_metrics, "validation": validation_metrics,
                           "best_epoch": best_epoch, "best_validation_objective": best})
            best_checkpoint = torch.load(seed_dir / "best.pt", map_location=target_device, weights_only=True)
            model.load_state_dict(best_checkpoint["state_dict"], strict=True)
            test_metrics, _ = _epoch(model, loaders["test"], acoustic, model_id, target_device,
                                     event_writer=log_event, epoch_number="test")
            test_metrics.pop("gradient_norms"); test_metrics.pop("gradient_nonfinite_count")
            grad_summary = _gradient_summary(norm_values, nonfinite_count)
            result = {"seed": seed, "status": "complete", "best_epoch": best_epoch,
                      "best_validation_objective": best, "best_validation_metrics": best_validation,
                      "test_metrics": test_metrics, "gradient_norm": grad_summary,
                      "shared_encoder_sha256": shared_hash, "seed_input_hashes": seed_inputs}
            seed_results.append(result)
            statuses[str(seed)] = "complete"
            write_json(seed_dir / "summary.json", result)
            manifest["seed_results"] = seed_results
            manifest["seed_status"] = statuses
            write_json(run_dir / "summary.json", {"status": "running", "completed_seeds": len(seed_results),
                        "seed_results": seed_results, "test_metrics": _test_summary(seed_results)})
            _write_run_manifest(run_dir, manifest)
            active_seed = None
        gradient = [row["gradient_norm"] for row in seed_results]
        run_summary = {"seeds_completed": len(seed_results), "test_metrics": _test_summary(seed_results),
                       "gradient_norm": {"mean_of_seed_means": statistics.fmean(x["mean"] for x in gradient),
                                         "max_of_seed_maxima": max(x["max"] for x in gradient),
                                         "nonfinite_count": sum(x["nonfinite_count"] for x in gradient)}}
        manifest.update({"status": "complete", "seed_status": statuses, "seed_results": seed_results, "summary": run_summary})
        write_json(run_dir / "summary.json", {"status": "complete", **run_summary, "seed_results": seed_results})
        _write_run_manifest(run_dir, manifest)
        return {"run_dir": run_dir, "manifest": manifest, "summary": run_summary}
    except Exception as exc:
        failure_diagnostic = getattr(exc, "diagnostic", None)
        if active_seed is not None and statuses.get(str(active_seed)) == "running":
            prior_norms = list(locals().get("norm_values", []))
            prior_nonfinite = int(locals().get("nonfinite_count", 0))
            if failure_diagnostic:
                prior_norms.extend(failure_diagnostic.get("gradient_norms", []))
                prior_nonfinite += int(failure_diagnostic.get("nonfinite_count", 0))
            failed_result = {"seed": active_seed, "status": "failed",
                             "gradient_norm": _gradient_summary(prior_norms, prior_nonfinite)}
            if failure_diagnostic:
                failed_result["failure_event"] = failure_diagnostic["event"]
            seed_results.append(failed_result)
            statuses[str(active_seed)] = "failed"
            failed_dir = run_dir / f"seed-{active_seed}"
            if failed_dir.is_dir():
                write_json(failed_dir / "summary.json", failed_result)
        statuses.update({key: ("failed" if value == "running" else value) for key, value in statuses.items()})
        status = "partial" if seed_results else "failed"
        failure = {"exception": type(exc).__name__, "message": str(exc), "completed_seeds": sum(x["status"] == "complete" for x in seed_results),
                   "seed_status": statuses, "global_step": global_step,
                   "diagnostic": failure_diagnostic}
        write_json(run_dir / "failure.json", failure)
        manifest.update({"status": status, "seed_status": statuses, "seed_results": seed_results, "failure": failure})
        write_json(run_dir / "summary.json", {"status": status, "seed_results": seed_results,
                    "test_metrics": _test_summary(seed_results),
                    "gradient_norm": {"mean_of_seed_means": statistics.fmean(x["gradient_norm"]["mean"] for x in seed_results),
                                      "max_of_seed_maxima": max(x["gradient_norm"]["max"] for x in seed_results),
                                      "nonfinite_count": sum(x["gradient_norm"]["nonfinite_count"] for x in seed_results)}})
        _write_run_manifest(run_dir, manifest)
        raise
