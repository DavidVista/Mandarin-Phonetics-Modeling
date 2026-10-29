"""Character-weighted losses and diagnostic metrics for training objectives."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .training_data import valid_initial_final_pairs
from .objectives import parse_model_id

COSINE_EPSILON = 1e-8


def batch_objective(outputs: dict, batch: dict, acoustic_targets: torch.Tensor, model_id: str) -> dict:
    family, weight = parse_model_id(model_id)
    mask = outputs["mask"]
    count = int(mask.sum().item())
    if count <= 0 or not torch.equal(mask, batch["acoustic_ids"] >= 0):
        raise ValueError("batch target masks are empty or misaligned")
    values = {"characters": count, "objective_sum": outputs["hidden"].new_zeros(())}
    if family == "syllable":
        target = batch["syllable_ids"][mask]
        loss = F.cross_entropy(outputs["syllable_logits"][mask], target, reduction="sum")
        values["syllable_ce_sum"] = loss
        values["objective_sum"] = values["objective_sum"] + loss
        values["syllable_correct"] = int((outputs["syllable_logits"][mask].argmax(-1) == target).sum().item())
    if family in {"components", "components-acoustic"}:
        labels = batch["component_labels"][mask]
        ce_sums = []
        for index, name in enumerate(("initials", "finals", "tones")):
            logits = outputs["component_logits"][name][mask]
            target = labels[:, index]
            loss = F.cross_entropy(logits, target, reduction="sum")
            values[f"{name}_ce_sum"] = loss
            values[f"{name}_correct"] = int((logits.argmax(-1) == target).sum().item())
            ce_sums.append(loss)
        component_sum = sum(ce_sums) / 3.0
        values["component_ce_sum"] = component_sum
        values["objective_sum"] = values["objective_sum"] + component_sum
        predicted = torch.stack([outputs["component_logits"][n][mask].argmax(-1) for n in ("initials", "finals")], dim=-1)
        valid_pairs = torch.tensor(sorted(valid_initial_final_pairs()), dtype=torch.long, device=predicted.device)
        valid = (predicted[:, None, :] == valid_pairs[None, :, :]).all(dim=-1).any(dim=-1)
        values["invalid_if_count"] = int((~valid).sum().item())
        values["joint_ift_correct"] = int((torch.stack([outputs["component_logits"][n][mask].argmax(-1) for n in ("initials", "finals", "tones")], dim=-1) == labels).all(dim=-1).sum().item())
    if family in {"acoustic", "components-acoustic"}:
        target_ids = batch["acoustic_ids"][mask]
        prediction = outputs["acoustic_prediction"][mask]
        targets = acoustic_targets[target_ids]
        cosine_distance = 1.0 - F.cosine_similarity(prediction, targets, dim=-1, eps=COSINE_EPSILON)
        cosine_sum = cosine_distance.sum()
        values["cosine_sum"] = cosine_sum
        values["predicted_norm_below_epsilon"] = int((torch.linalg.vector_norm(prediction.float(), dim=-1) < COSINE_EPSILON).sum().item())
        weighted = (weight if family == "components-acoustic" else 1.0) * cosine_sum
        values["weighted_cosine_sum"] = weighted
        values["objective_sum"] = values["objective_sum"] + weighted
    values["objective"] = values["objective_sum"] / count
    return values


def finalize_epoch(totals: dict) -> dict:
    count = int(totals.get("characters", 0))
    if count <= 0:
        raise ValueError("loader contains no valid characters")
    result = {"characters": count, "objective": totals["objective_sum"] / count}
    for key, value in totals.items():
        if key.endswith("_sum") and key != "objective_sum":
            result[key.removesuffix("_sum")] = value / count
    for key in ("syllable_correct", "initials_correct", "finals_correct", "tones_correct", "joint_ift_correct"):
        if key in totals: result[key.removesuffix("_correct") + "_accuracy"] = totals[key] / count
    if "invalid_if_count" in totals:
        result["invalid_if_rate"] = totals["invalid_if_count"] / count
        result["invalid_if_count"] = totals["invalid_if_count"]
    if "predicted_norm_below_epsilon" in totals:
        result["predicted_norm_below_epsilon"] = totals["predicted_norm_below_epsilon"]
    return result
