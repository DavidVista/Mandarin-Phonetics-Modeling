"""RSA-style cosine analysis and strict paired comparison of encoder runs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .bundle import load_evaluation_bundle
from .representations import validate_matrix

LEVELS = ("contextual", "character_pinyin", "syllable")
BIN_EDGES = tuple(range(0, 121, 10))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distance_bin(distance: float | None) -> tuple[str, str]:
    """Return ``(status, label)`` with an explicit unbounded overflow bin."""
    if distance is None or not isinstance(distance, (int, float)) or not math.isfinite(float(distance)):
        return "invalid_nonfinite", "invalid"
    value = float(distance)
    if value < 0:
        return "invalid_negative", "invalid"
    if value >= 120:
        return "ok", "[120,+inf)"
    lower = int(value // 10) * 10
    return "ok", f"[{lower},{lower + 10})"


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + end - 1) / 2.0 + 1.0
        position = end
    return ranks


def spearman_result(distances: list[float], dissimilarities: list[float]) -> dict[str, Any]:
    if len(distances) != len(dissimilarities):
        raise ValueError("Spearman inputs have different lengths")
    count = len(distances)
    if count < 2:
        return {"rho": None, "valid_pair_count": count, "undefined_reason": "fewer_than_two_pairs"}
    left, right = np.asarray(distances, dtype=np.float64), np.asarray(dissimilarities, dtype=np.float64)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Spearman inputs must be finite after validity filtering")
    left_rank, right_rank = _rankdata(left), _rankdata(right)
    if np.ptp(left_rank) == 0 or np.ptp(right_rank) == 0:
        reason = "constant_distance" if np.ptp(left_rank) == 0 else "constant_hidden_dissimilarity"
        return {"rho": None, "valid_pair_count": count, "undefined_reason": reason}
    rho = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return {"rho": rho, "valid_pair_count": count, "undefined_reason": None}


def cosine_rows(pair_rows: list[dict], object_ids: list[str], matrix: np.ndarray) -> list[dict]:
    values = validate_matrix(object_ids, matrix)
    index = {object_id: position for position, object_id in enumerate(object_ids)}
    result = []
    for pair in pair_rows:
        try:
            left, right = values[index[pair["left_id"]]], values[index[pair["right_id"]]]
        except KeyError as error:
            raise ValueError(f"pair references missing representation {error.args[0]}") from error
        cosine = float(np.dot(left.astype(np.float64), right.astype(np.float64)) /
                       (np.linalg.norm(left.astype(np.float64)) * np.linalg.norm(right.astype(np.float64))))
        cosine = min(1.0, max(-1.0, cosine))
        bin_status, label = distance_bin(pair.get("dimsim_distance"))
        metric_status = pair.get("distance_status", "missing_status")
        analysis_status = "ok" if metric_status == "ok" and bin_status == "ok" else (
            metric_status if metric_status != "ok" else bin_status
        )
        result.append({**pair, "cosine_similarity": cosine, "hidden_dissimilarity": 1.0 - cosine,
                       "analysis_status": analysis_status, "distance_bin": label})
    return result


def summarize_rows(rows: list[dict]) -> dict[str, Any]:
    valid = [row for row in rows if row["analysis_status"] == "ok"]

    def correlation(selected: list[dict]) -> dict[str, Any]:
        return spearman_result(
            [float(row["dimsim_distance"]) for row in selected],
            [float(row["hidden_dissimilarity"]) for row in selected],
        )

    strata = {mask: correlation([row for row in valid if row.get("overlap_mask") == mask])
              for mask in (f"{value:03b}" for value in range(8))}
    strata["no_zero_initial"] = correlation([row for row in valid if row.get("no_zero_initial_diagnostic") is True])
    bins = []
    labels = [f"[{lower},{lower + 10})" for lower in range(0, 120, 10)] + ["[120,+inf)"]
    for label in labels:
        selected = [row for row in valid if row["distance_bin"] == label]
        distances = [float(row["dimsim_distance"]) for row in selected]
        cosines = [float(row["cosine_similarity"]) for row in selected]
        bins.append({"bin": label, "count": len(selected),
                     "distance_min": min(distances) if distances else None,
                     "distance_max": max(distances) if distances else None,
                     "cosine_mean": float(np.mean(cosines)) if cosines else None})
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["analysis_status"]] = status_counts.get(row["analysis_status"], 0) + 1
    return {
        "pair_count": len(rows), "status_counts": dict(sorted(status_counts.items())),
        "overall": correlation(valid), "strata": strata, "bins": bins,
        "uncertainty": {
            "status": "not_estimated_single_seed",
            "interpretation": "fixed-pair descriptive statistic; no population or training-run confidence interval",
            "iid_pair_bootstrap": False, "p_values_reported": False,
        },
    }


def _load_representation_level(root: Path, level: str) -> tuple[list[str], np.ndarray]:
    prefix = "occurrence" if level == "contextual" else level
    ids = json.loads((root / f"{prefix}_ids.json").read_text(encoding="utf-8"))
    matrix = np.load(root / f"{prefix}_hidden.npy", allow_pickle=False)
    return ids, matrix


def _plot_bins(summary: dict, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    populated = [row for row in summary["bins"] if row["count"]]
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot([row["bin"] for row in populated], [row["cosine_mean"] for row in populated], marker="o")
    for index, row in enumerate(populated):
        axis.annotate(f"n={row['count']}", (index, row["cosine_mean"]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)
    axis.set(title=title, xlabel="DIMSIM distance bin", ylabel="Mean raw cosine")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def analyze_representations(
    bundle_path: str | Path, representations_dir: str | Path, *, output_dir: str | Path, run_label: str,
) -> Path:
    bundle, representation_root = load_evaluation_bundle(bundle_path), Path(representations_dir).resolve()
    representation_manifest = json.loads((representation_root / "manifest.json").read_text(encoding="utf-8"))
    required_representation_artifacts = {"occurrence_ids.json", "occurrence_hidden.npy"}
    for level in ("character_pinyin", "syllable"):
        required_representation_artifacts.update({f"{level}_ids.json", f"{level}_counts.json", f"{level}_hidden.npy"})
    representation_artifacts = representation_manifest.get("artifacts", {})
    if not required_representation_artifacts.issubset(representation_artifacts):
        raise ValueError("representation manifest is missing required artifacts")
    for name, digest in representation_artifacts.items():
        artifact = (representation_root / name).resolve()
        if representation_root not in artifact.parents or not artifact.is_file() or _sha256(artifact) != digest:
            raise ValueError(f"representation artifact checksum mismatch: {name}")
    bundle_manifest_sha = _sha256(Path(bundle_path).resolve() / "manifest.json")
    if representation_manifest.get("provenance", {}).get("bundle_manifest_sha256") != bundle_manifest_sha:
        raise ValueError("representations were not extracted from this evaluation bundle")
    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"analysis output already exists: {target}")
    target.mkdir(parents=True)
    (target / "similarities").mkdir()
    (target / "plots").mkdir()
    metrics, table_hashes = {}, {}
    for level in LEVELS:
        ids, matrix = _load_representation_level(representation_root, level)
        if level == "contextual":
            expected_ids = sorted(row["occurrence_id"] for row in bundle["occurrences"])
        else:
            expected_groups = sorted((row for row in bundle["groups"] if row["group_kind"] == level), key=lambda row: row["group_id"])
            expected_ids = [row["group_id"] for row in expected_groups]
            counts = json.loads((representation_root / f"{level}_counts.json").read_text(encoding="utf-8"))
            if counts != [row["occurrence_count"] for row in expected_groups]:
                raise ValueError(f"representation group counts disagree with bundle: {level}")
        if ids != expected_ids:
            raise ValueError(f"representation IDs disagree with complete bundle objects: {level}")
        rows = cosine_rows(bundle["pairs"][level], ids, matrix)
        table = target / "similarities" / f"{level}.jsonl"
        table.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        table_hashes[level] = _sha256(table)
        metrics[level] = summarize_rows(rows)
        _plot_bins(metrics[level], target / "plots" / f"{level}-bin-means.png", f"{run_label}: {level}")
    (target / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    pair_hashes = {level: bundle["manifest"]["artifacts"][f"pairs/{level}.jsonl"] for level in LEVELS}
    manifest = {"schema_version": 1, "run_label": run_label, "bundle_manifest_sha256": bundle_manifest_sha,
                "pair_plan_sha256": pair_hashes, "representation_manifest_sha256": _sha256(representation_root / "manifest.json"),
                "similarity_table_sha256": table_hashes, "metrics_sha256": _sha256(target / "metrics.json")}
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return target


def compare_paired_runs(left_dir: str | Path, right_dir: str | Path, *, output_dir: str | Path) -> Path:
    left_root, right_root = Path(left_dir).resolve(), Path(right_dir).resolve()
    manifests = [json.loads((root / "manifest.json").read_text(encoding="utf-8")) for root in (left_root, right_root)]
    for root, manifest in zip((left_root, right_root), manifests):
        if _sha256(root / "metrics.json") != manifest.get("metrics_sha256"):
            raise ValueError(f"analysis metrics checksum mismatch: {root}")
        for level in LEVELS:
            if _sha256(root / "similarities" / f"{level}.jsonl") != manifest.get("similarity_table_sha256", {}).get(level):
                raise ValueError(f"analysis similarity checksum mismatch: {root} {level}")
    for field in ("bundle_manifest_sha256", "pair_plan_sha256"):
        if manifests[0].get(field) != manifests[1].get(field):
            raise ValueError(f"paired runs disagree on {field}")
    metrics = [json.loads((root / "metrics.json").read_text(encoding="utf-8")) for root in (left_root, right_root)]
    paired_tables = {}
    model_fields = {"cosine_similarity", "hidden_dissimilarity"}
    for level in LEVELS:
        tables = []
        for root in (left_root, right_root):
            tables.append([json.loads(line) for line in (root / "similarities" / f"{level}.jsonl").read_text(encoding="utf-8").splitlines()])
        if [row["pair_id"] for row in tables[0]] != [row["pair_id"] for row in tables[1]]:
            raise ValueError(f"paired runs have different ordered pair IDs for {level}")
        if any({key: value for key, value in left.items() if key not in model_fields} !=
               {key: value for key, value in right.items() if key not in model_fields}
               for left, right in zip(*tables)):
            raise ValueError(f"paired runs disagree on common pair metadata for {level}")
        paired_tables[level] = tables
    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"comparison output already exists: {target}")
    target.mkdir(parents=True)
    (target / "joined").mkdir()
    result = {"left_run": manifests[0]["run_label"], "right_run": manifests[1]["run_label"], "levels": {}}
    joined_hashes = {}
    for level in LEVELS:
        tables = paired_tables[level]
        joined = [{**{key: value for key, value in left.items() if key not in model_fields},
                   "left_cosine_similarity": left["cosine_similarity"], "right_cosine_similarity": right["cosine_similarity"],
                   "left_hidden_dissimilarity": left["hidden_dissimilarity"], "right_hidden_dissimilarity": right["hidden_dissimilarity"]}
                  for left, right in zip(*tables)]
        joined_path = target / "joined" / f"{level}.jsonl"
        joined_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in joined), encoding="utf-8")
        joined_hashes[level] = _sha256(joined_path)
        left_rho, right_rho = metrics[0][level]["overall"]["rho"], metrics[1][level]["overall"]["rho"]
        result["levels"][level] = {"left_rho": left_rho, "right_rho": right_rho,
                                    "delta_rho_right_minus_left": None if left_rho is None or right_rho is None else right_rho - left_rho,
                                    "pair_count": len(tables[0]), "strata": {}}
        for stratum in [f"{value:03b}" for value in range(8)] + ["no_zero_initial"]:
            left_value = metrics[0][level]["strata"][stratum]["rho"]
            right_value = metrics[1][level]["strata"][stratum]["rho"]
            result["levels"][level]["strata"][stratum] = {
                "left_rho": left_value, "right_rho": right_value,
                "delta_rho_right_minus_left": None if left_value is None or right_value is None else right_value - left_value,
            }
    result["uncertainty"] = {"status": "not_estimated", "paired_point_difference_only": True, "iid_pair_bootstrap": False}
    output = target / "comparison.json"
    output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    comparison_manifest = {
        "schema_version": 1,
        "input_analysis_manifest_sha256": {"left": _sha256(left_root / "manifest.json"), "right": _sha256(right_root / "manifest.json")},
        "comparison_sha256": _sha256(output), "joined_sha256": joined_hashes,
    }
    (target / "manifest.json").write_text(json.dumps(comparison_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return target
