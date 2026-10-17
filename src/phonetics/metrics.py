"""Character-weighted objectives and metrics shared by all conditions."""

import torch
import torch.nn.functional as F

EPS = 1e-12


def normalize_vectors(vectors: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(vectors).all(): raise ValueError("vectors contain non-finite values")
    norms = torch.linalg.vector_norm(vectors.double(), dim=-1)
    if not torch.isfinite(norms).all() or torch.any(norms <= EPS): raise ValueError("vectors have zero, tiny, or overflowed norms")
    return vectors.double() / norms.unsqueeze(-1)


def batch_statistics(outputs, batch, acoustic_targets, condition: str, lambda_mse: float = 1.0):
    initial, final, tone, hidden = outputs
    labels = batch["component_labels"]
    mask = labels[..., 0] != -100
    count = int(mask.sum())
    if count == 0: raise ValueError("batch contains no valid characters")
    if not torch.equal(mask, batch["acoustic_ids"] >= 0): raise ValueError("classification and acoustic masks differ")
    if condition == "combined" and (not torch.isfinite(torch.tensor(lambda_mse)) or lambda_mse < 0): raise ValueError("lambda_mse must be finite and nonnegative")
    ce_sum = None if condition == "mse_only" else sum(F.cross_entropy(logits[mask], labels[..., i][mask], reduction="sum") for i, logits in enumerate((initial, final, tone)))
    predicted = normalize_vectors(hidden[mask]); target = normalize_vectors(acoustic_targets[batch["acoustic_ids"][mask]])
    squared = (predicted - target).square().mean(dim=-1)
    delta = predicted - target; distance = torch.linalg.vector_norm(delta, dim=-1)
    mse_sum = squared.sum(); distance_sum = distance.sum(); squared_distance_sum = distance.square().sum()
    l1_sum = delta.abs().sum(dim=-1).sum(); rmse_sum = torch.sqrt(squared).sum()
    cosine_direct_sum = (predicted * target).sum(dim=-1).sum()
    cosine_from_squared_sum = (1.0 - 0.5 * distance.square()).sum()
    if condition == "ce_only": objective_sum = ce_sum
    elif condition == "mse_only": objective_sum = mse_sum
    elif condition == "combined": objective_sum = ce_sum + float(lambda_mse) * mse_sum
    else: raise ValueError(f"unknown condition: {condition}")
    stats = {"characters": count, "objective_sum": objective_sum, "mse_sum": mse_sum, "distance_sum": distance_sum,
             "squared_distance_sum": squared_distance_sum, "l1_sum": l1_sum, "rmse_sum": rmse_sum,
             "cosine_direct_sum": cosine_direct_sum, "cosine_from_squared_sum": cosine_from_squared_sum}
    if condition != "mse_only":
        stats["ce_sum"] = ce_sum
        stats.update({f"{name}_correct": (logits[mask].argmax(-1) == labels[..., i][mask]).sum()
                      for i, (name, logits) in enumerate(zip(("initial", "final", "tone"), (initial, final, tone)))})
        predictions = torch.stack([x[mask].argmax(-1) for x in (initial, final, tone)], dim=-1)
        stats["exact_correct"] = (predictions == labels[mask]).all(dim=-1).sum()
    return stats


def finalize_statistics(totals: dict, condition: str) -> dict:
    count = totals.get("characters", 0)
    if count <= 0: raise ValueError("loader contains no valid characters")
    result = {"characters": count, "num_characters": count, "objective": totals["objective_sum"] / count,
              "normalized_mse": totals["mse_sum"] / count, "avg_mse": totals["mse_sum"] / count,
              "avg_squared_distance": totals["squared_distance_sum"] / count,
              "avg_distance": totals["distance_sum"] / count, "avg_l1_distance": totals["l1_sum"] / count,
              "avg_rmse": totals["rmse_sum"] / count, "avg_cosine_direct": totals["cosine_direct_sum"] / count,
              "avg_cosine_from_squared_distance": totals["cosine_from_squared_sum"] / count}
    if condition != "mse_only":
        result["ce"] = totals["ce_sum"] / count
        result["exact_accuracy"] = totals["exact_correct"] / count
        for name in ("initial", "final", "tone"): result[f"{name}_accuracy"] = totals[f"{name}_correct"] / count
    return result
