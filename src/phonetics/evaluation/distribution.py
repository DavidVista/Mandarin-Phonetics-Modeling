"""Reproducible population analysis for the pinned DIMSIM evaluation bundle."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .dimsim import DimsimDistance

EXPECTED_BUNDLE_MANIFEST_SHA256 = "0a0a6b1ac02de627e1e0fdabb75a46d755d9a7f28f7032810ec4c6bbe0b3e9f6"
EXPECTED_ELIGIBLE_SYLLABLES = 967
EXPECTED_PAIR_COUNT = 467_061
ANALYSIS_SCHEMA_VERSION = 1
MAIN_BIN_EDGES = np.arange(0.0, 121.0, 10.0)
TAIL_THRESHOLDS = (60.0, 90.0, 100.0, 120.0)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def verify_bundle(root: str | Path) -> dict[str, Any]:
    """Verify the exact immutable bundle and every payload hash used here."""
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    actual_manifest_hash = sha256_file(manifest_path)
    if actual_manifest_hash != EXPECTED_BUNDLE_MANIFEST_SHA256:
        raise ValueError(f"unexpected bundle manifest SHA256: {actual_manifest_hash}")
    manifest = _read_json(manifest_path)
    required = {"groups.jsonl", "pairs/syllable.jsonl", "pairs/contextual.jsonl", "pairs/character_pinyin.jsonl"}
    if not required.issubset(manifest.get("artifacts", {})):
        raise ValueError("bundle manifest lacks an analysis input")
    verified = {}
    for relative, expected in manifest["artifacts"].items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"missing or escaping bundle artifact: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"bundle artifact checksum mismatch: {relative}")
        verified[relative] = actual
    if manifest.get("evaluation_id") != "1.0.0" or manifest.get("pairing_policy") != "nonzero_nonzero":
        raise ValueError("unexpected evaluation identity or pairing policy")
    return {"root": root, "manifest": manifest, "manifest_sha256": actual_manifest_hash, "verified": verified}


def bin_label(value: float, *, width: float = 10.0, shift: float = 0.0, overflow: float | None = 120.0) -> str:
    """Return a stable left-closed bin label, optionally with an overflow bin."""
    value = float(value)
    if not math.isfinite(value) or value < 0 or width <= 0:
        raise ValueError("distance must be finite/non-negative and width positive")
    if overflow is not None and value >= overflow:
        return f"[{overflow:g},+inf)"
    lower = math.floor((value - shift) / width) * width + shift
    upper = lower + width
    if lower < 0:
        lower = 0.0
    return f"[{lower:g},{upper:g})"


@dataclass(frozen=True)
class FormulaParts:
    library_initial: str
    library_final: str
    tone: int


def parse_library_syllable(value: str) -> FormulaParts:
    """Parse with DIMSIM's own parser, preserving its internal representation."""
    from dimsim.utils.pinyin import Pinyin

    parsed = Pinyin(value)
    return FormulaParts(str(parsed.consonant), str(parsed.vowel), int(parsed.tone))


def reconstruct_distance(left: str, right: str) -> dict[str, Any]:
    """Reproduce DIMSIM 0.2.2 exactly, including upstream identity comparisons."""
    from dimsim.utils.maps import consonantMap_TwoDCode, vowelMap_TwoDCode
    from dimsim.utils.pinyin import Pinyin
    from dimsim.utils.utils import get_sim_dis_from_hardcod_map

    a, b = Pinyin(left), Pinyin(right)
    ca, cb = consonantMap_TwoDCode[a.consonant], consonantMap_TwoDCode[b.consonant]
    va, vb = vowelMap_TwoDCode[a.vowel], vowelMap_TwoDCode[b.vowel]
    consonant_distance = math.dist(ca, cb)
    vowel_distance = math.dist(va, vb)
    hardcode_distance = float(get_sim_dis_from_hardcod_map(a, b))
    segment_distance = min(consonant_distance + vowel_distance, hardcode_distance)
    tone_distance = abs(a.tone - b.tone) / 10.0
    edit_distance = segment_distance + tone_distance
    # These are the exact comparisons in dimsim.core.model.get_distance.
    initial_difference = int(a.consonant is not b.consonant)
    final_difference = int(not (str(a.vowel) == str(b.vowel)))
    tone_difference = 0.01 if a.tone is not b.tone else 0.0
    difference_count = initial_difference + final_difference + tone_difference
    multiplier = difference_count / 2.1
    return {
        "distance": edit_distance * multiplier,
        "consonant_distance": consonant_distance,
        "vowel_distance": vowel_distance,
        "hardcode_distance": hardcode_distance,
        "hardcode_applied": math.isfinite(hardcode_distance) and hardcode_distance < consonant_distance + vowel_distance,
        "segment_distance": segment_distance,
        "tone_distance": tone_distance,
        "edit_distance": edit_distance,
        "initial_difference": initial_difference,
        "final_difference": final_difference,
        "tone_difference": tone_difference,
        "difference_count": difference_count,
        "multiplier": multiplier,
        "left_library_initial": str(a.consonant),
        "right_library_initial": str(b.consonant),
        "left_library_final": str(a.vowel),
        "right_library_final": str(b.vowel),
        "left_library_tone": int(a.tone),
        "right_library_tone": int(b.tone),
    }


def _rows_to_columns(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    keys = rows[0].keys()
    return {key: np.asarray([row[key] for row in rows]) for key in keys}


def load_syllable_population(root: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load and validate all eligible syllables and their complete unordered pairs."""
    verified = verify_bundle(root)
    root = verified["root"]
    groups = [row for row in _jsonl(root / "groups.jsonl") if row.get("group_kind") == "syllable" and row.get("initial")]
    groups.sort(key=lambda row: row["group_id"])
    pairs = list(_jsonl(root / "pairs/syllable.jsonl"))
    if len(groups) != EXPECTED_ELIGIBLE_SYLLABLES or len(pairs) != EXPECTED_PAIR_COUNT:
        raise ValueError("unexpected eligible syllable or pair population size")
    ids = {row["group_id"] for row in groups}
    endpoints = [(row["left_id"], row["right_id"]) for row in pairs]
    if len(set(endpoints)) != EXPECTED_PAIR_COUNT or any(left >= right or left not in ids or right not in ids for left, right in endpoints):
        raise ValueError("pair population is not the complete canonical set")
    if set(endpoints) != {(groups[i]["group_id"], groups[j]["group_id"]) for i in range(len(groups)) for j in range(i + 1, len(groups))}:
        raise ValueError("pair population is incomplete")
    if any(not row.get("initial") or row.get("pinyin", "").startswith(("y", "w")) for row in groups):
        raise ValueError("eligible set violates the gold nonzero-initial policy")
    return groups, pairs


def compute_population(root: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Recompute public and reconstructed distances for every population pair."""
    verified = verify_bundle(root)
    groups, pairs = load_syllable_population(root)
    by_id = {row["group_id"]: row for row in groups}
    metric = DimsimDistance()
    parsed = {row["pinyin"]: parse_library_syllable(row["pinyin"]) for row in groups}
    parser_mismatch_syllables = sorted(
        row["pinyin"] for row in groups
        if (row["initial"] != parsed[row["pinyin"]].library_initial
            or row["final"] != parsed[row["pinyin"]].library_final
            or int(row["tone"]) != parsed[row["pinyin"]].tone)
    )
    rows: list[dict[str, Any]] = []
    max_bundle_error = max_formula_error = max_reverse_error = 0.0
    max_diagonal = max(abs(metric(row["pinyin"], row["pinyin"])) for row in groups)
    parser_mismatch_count = 0
    hardcode_applied_count = 0
    for pair in pairs:
        left, right = pair["left_pinyin"], pair["right_pinyin"]
        public = metric(left, right)
        reverse = metric(right, left)
        formula = reconstruct_distance(left, right)
        max_bundle_error = max(max_bundle_error, abs(public - float(pair["dimsim_distance"])))
        max_formula_error = max(max_formula_error, abs(public - float(formula["distance"])))
        max_reverse_error = max(max_reverse_error, abs(public - reverse))
        lg, rg = by_id[pair["left_id"]], by_id[pair["right_id"]]
        lp, rp = parsed[left], parsed[right]
        mismatch = (lg["initial"] != lp.library_initial or lg["final"] != lp.library_final or int(lg["tone"]) != lp.tone or
                    rg["initial"] != rp.library_initial or rg["final"] != rp.library_final or int(rg["tone"]) != rp.tone)
        parser_mismatch_count += int(mismatch)
        hardcode_applied_count += int(formula["hardcode_applied"])
        rows.append({
            "left_id": pair["left_id"], "right_id": pair["right_id"], "left_pinyin": left, "right_pinyin": right,
            "distance": public, "bundle_distance": float(pair["dimsim_distance"]), "overlap_mask": pair["overlap_mask"],
            "equal_initial": bool(pair["equal_initial"]), "equal_final": bool(pair["equal_final"]), "equal_tone": bool(pair["equal_tone"]),
            "left_gold_initial": lg["initial"], "right_gold_initial": rg["initial"],
            "left_gold_final": lg["final"], "right_gold_final": rg["final"],
            "left_gold_tone": int(lg["tone"]), "right_gold_tone": int(rg["tone"]),
            "parser_mismatch": mismatch, **formula,
        })
    columns = _rows_to_columns(rows)
    distances = columns["distance"].astype(float)
    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError("DIMSIM returned a nonfinite or negative distance")
    tolerance = 1e-10
    if max(max_bundle_error, max_formula_error, max_reverse_error, max_diagonal) > tolerance:
        raise ValueError("distance validation exceeded numerical tolerance")
    validation = {
        "manifest_sha256": verified["manifest_sha256"], "artifact_sha256": verified["verified"],
        "syllable_count": len(groups), "pair_count": len(pairs), "expected_pair_count": len(groups) * (len(groups) - 1) // 2,
        "finite": True, "nonnegative": True, "max_bundle_absolute_error": max_bundle_error,
        "max_formula_absolute_error": max_formula_error, "max_reverse_absolute_error": max_reverse_error,
        "max_diagonal_absolute_error": max_diagonal, "tolerance": tolerance,
        "parser_mismatch_pair_count": parser_mismatch_count, "hardcode_applied_pair_count": hardcode_applied_count,
        "parser_mismatch_syllables": parser_mismatch_syllables,
        "metric_provenance": metric.provenance,
    }
    return columns, validation


def _quantiles(values: np.ndarray) -> dict[str, float]:
    probs = (0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1)
    return {f"q{int(p * 100):02d}": float(np.quantile(values, p)) for p in probs}


def _counts(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(map(str, values)).items(), key=lambda item: (-item[1], item[0])))


def _group_summary(columns: dict[str, np.ndarray], selector: np.ndarray) -> dict[str, Any]:
    d = columns["distance"].astype(float)[selector]
    n = int(selector.sum())
    masks = columns["overlap_mask"][selector]
    return {
        "count": n, "share": n / len(columns["distance"]), "unique_syllables": len(set(columns["left_pinyin"][selector]) | set(columns["right_pinyin"][selector])),
        "quantiles": _quantiles(d) if n else {}, "overlap_mask_counts": _counts(masks),
        "equal_initial_share": float(columns["equal_initial"][selector].astype(bool).mean()) if n else None,
        "equal_final_share": float(columns["equal_final"][selector].astype(bool).mean()) if n else None,
        "equal_tone_share": float(columns["equal_tone"][selector].astype(bool).mean()) if n else None,
    }


def _component_enrichment(columns: dict[str, np.ndarray], selector: np.ndarray, field: str, top_n: int = 10) -> dict[str, Any]:
    """Report prevalent and enriched unordered component pairs with examples."""
    left = columns[f"left_gold_{field}"]
    right = columns[f"right_gold_{field}"]
    pairs = np.asarray(["|".join(sorted((str(a), str(b)))) for a, b in zip(left, right)])
    global_counts = Counter(pairs.tolist())
    selected_counts = Counter(pairs[selector].tolist())
    total, selected_total = len(pairs), int(selector.sum())
    rows = []
    for pair, count in selected_counts.items():
        expected = global_counts[pair] * selected_total / total
        rows.append({"pair": pair, "count": count, "global_count": global_counts[pair],
                     "expected_count": expected, "enrichment": count / expected if expected else None})
    enriched = sorted((row for row in rows if row["count"] >= 5),
                      key=lambda row: (-float(row["enrichment"]), -row["count"], row["pair"]))[:top_n]
    prevalent = sorted(rows, key=lambda row: (-row["count"], row["pair"]))[:top_n]
    examples = []
    for row in enriched[:5]:
        match = selector & (pairs == row["pair"])
        target = float(np.median(columns["distance"].astype(float)[match]))
        examples.append({"pair": row["pair"], "count": row["count"], "examples": _examples(columns, match, target, 2)})
    return {"prevalent": prevalent, "enriched_min_count_5": enriched, "enriched_examples": examples}


def _modal_composition(columns: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    distances = columns["distance"].astype(float)
    edges = np.arange(0.0, math.ceil(distances.max()) + 1.0, 1.0)
    counts, edges = np.histogram(distances, bins=edges)
    peaks = [i for i in range(1, len(counts) - 1) if counts[i] >= counts[i - 1] and counts[i] > counts[i + 1]]
    peaks = sorted(peaks, key=lambda i: counts[i], reverse=True)[:8]
    result = []
    for index in peaks:
        selected = (distances >= edges[index]) & (distances < edges[index + 1])
        result.append({"left": float(edges[index]), "right": float(edges[index + 1]), "count": int(selected.sum()),
                       "mask_counts": _counts(columns["overlap_mask"][selected]),
                       "initial_pairs": _component_enrichment(columns, selected, "initial", 5)["prevalent"],
                       "final_pairs": _component_enrichment(columns, selected, "final", 5)["prevalent"]})
    return result


def _examples(columns: dict[str, np.ndarray], selector: np.ndarray, target: float, count: int = 5) -> list[dict[str, Any]]:
    indexes = np.flatnonzero(selector)
    ordered = sorted(indexes, key=lambda i: (abs(float(columns["distance"][i]) - target), str(columns["left_pinyin"][i]), str(columns["right_pinyin"][i])))[:count]
    return [{"left": str(columns["left_pinyin"][i]), "right": str(columns["right_pinyin"][i]),
             "distance": float(columns["distance"][i]), "overlap_mask": str(columns["overlap_mask"][i])} for i in ordered]


def _segment_reweighted(columns: dict[str, np.ndarray]) -> dict[str, Any]:
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for left, right, distance in zip(columns["left_pinyin"], columns["right_pinyin"], columns["distance"]):
        a, b = str(left)[:-1], str(right)[:-1]
        buckets[tuple(sorted((a, b)))].append(float(distance))
    means = np.asarray([np.mean(values) for values in buckets.values()])
    return {"unique_segment_pair_count": len(means), "quantiles": _quantiles(means),
            "mean": float(means.mean()), "median": float(np.median(means))}


def _tail_summary(columns: dict[str, np.ndarray]) -> dict[str, Any]:
    distances = columns["distance"].astype(float)
    endpoints = np.concatenate([columns["left_pinyin"], columns["right_pinyin"]])
    result = {}
    for threshold in TAIL_THRESHOLDS:
        selected = distances >= threshold
        syllables = np.concatenate([columns["left_pinyin"][selected], columns["right_pinyin"][selected]])
        counts = Counter(map(str, syllables))
        top = counts.most_common(10)
        original_count = int(selected.sum())
        sensitivity = []
        for syllable, endpoint_count in top:
            remaining = selected & (columns["left_pinyin"] != syllable) & (columns["right_pinyin"] != syllable)
            sensitivity.append({"syllable": syllable, "endpoint_count": endpoint_count, "remaining_pair_count": int(remaining.sum()),
                                "removed_share": (original_count - int(remaining.sum())) / original_count if original_count else 0.0})
        result[f"ge_{int(threshold)}"] = {**_group_summary(columns, selected), "top_endpoints": top, "leave_one_out": sensitivity}
    result["lt_60"] = _group_summary(columns, distances < 60)
    return result


def _histogram_robustness(distances: np.ndarray) -> dict[str, Any]:
    result = {}
    for width in (1.0, 2.0, 5.0, 10.0):
        for shifted in (False, True):
            shift = width / 2 if shifted else 0.0
            lo = shift if shifted else 0.0
            edges = np.arange(lo, math.ceil(distances.max() / width) * width + width, width)
            if shifted:
                edges = np.insert(edges, 0, 0.0)
            counts, edges = np.histogram(distances, bins=edges)
            peaks = np.argsort(counts)[-5:][::-1]
            result[f"width_{width:g}_{'half_shift' if shifted else 'origin'}"] = [
                {"left": float(edges[i]), "right": float(edges[i + 1]), "count": int(counts[i])} for i in peaks
            ]
    return result


def _kde_summary(distances: np.ndarray, points: int = 401) -> dict[str, Any]:
    """Evaluate Gaussian KDE on a fixed grid by deterministic chunks."""
    grid = np.linspace(0.0, max(130.0, float(distances.max())), points)
    std = float(distances.std(ddof=1))
    scott = len(distances) ** (-1 / 5)
    result = {"grid": grid.tolist(), "scott_factor": scott, "population_standard_deviation": std, "curves": {}}
    for factor in (.5, 1.0, 2.0):
        bandwidth = max(np.finfo(float).eps, factor * scott * std)
        density = np.zeros(points)
        for start in range(0, len(distances), 20_000):
            z = (grid[:, None] - distances[start:start + 20_000][None, :]) / bandwidth
            density += np.exp(-.5 * z * z).sum(axis=1)
        density /= len(distances) * bandwidth * math.sqrt(2 * math.pi)
        peaks = [i for i in range(1, points - 1) if density[i] >= density[i - 1] and density[i] > density[i + 1]]
        peaks = sorted(peaks, key=lambda i: density[i], reverse=True)[:8]
        result["curves"][f"{factor:g}x_scott"] = {"bandwidth": bandwidth, "density": density.tolist(),
                                                         "peaks": [{"distance": float(grid[i]), "density": float(density[i])} for i in peaks]}
    return result


def _cluster_summary(columns: dict[str, np.ndarray], syllables: list[str]) -> dict[str, Any]:
    """Average-linkage clustering with precomputed-distance silhouette, k=2..10."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics import silhouette_score

    index = {value: i for i, value in enumerate(syllables)}
    components = {}
    for side in ("left", "right"):
        for pinyin, initial, final, tone in zip(columns[f"{side}_pinyin"], columns[f"{side}_gold_initial"],
                                                columns[f"{side}_gold_final"], columns[f"{side}_gold_tone"]):
            components[str(pinyin)] = (str(initial), str(final), int(tone))
    matrix = np.zeros((len(syllables), len(syllables)), dtype=float)
    for left, right, value in zip(columns["left_pinyin"], columns["right_pinyin"], columns["distance"]):
        i, j = index[str(left)], index[str(right)]
        matrix[i, j] = matrix[j, i] = float(value)
    tree = linkage(squareform(matrix, checks=True), method="average")
    result = {}
    for k in range(2, 11):
        labels = fcluster(tree, k, criterion="maxclust")
        clusters = []
        for label in sorted(set(labels)):
            members = [syllables[i] for i in np.flatnonzero(labels == label)]
            clusters.append({"cluster": int(label), "count": len(members),
                             "initials": Counter(components[x][0] for x in members).most_common(5),
                             "finals": Counter(components[x][1] for x in members).most_common(5),
                             "tones": Counter(components[x][2] for x in members).most_common()})
        result[str(k)] = {"silhouette": float(silhouette_score(matrix, labels, metric="precomputed")), "clusters": clusters}
    return result


def summarize_population(columns: dict[str, np.ndarray], validation: dict[str, Any], *, include_clustering: bool = True) -> dict[str, Any]:
    distances = columns["distance"].astype(float)
    bins = [bin_label(value) for value in distances]
    bin_order = [f"[{x},{x + 10})" for x in range(0, 120, 10)] + ["[120,+inf)"]
    bin_summaries = {}
    for label in bin_order:
        selected = np.asarray(bins) == label
        bin_summaries[label] = {**_group_summary(columns, selected),
                                "initial_pair_enrichment": _component_enrichment(columns, selected, "initial"),
                                "final_pair_enrichment": _component_enrichment(columns, selected, "final")}
    global_mask_counts = _counts(columns["overlap_mask"])
    masks = {mask: _group_summary(columns, columns["overlap_mask"] == mask) for mask in (f"{x:03b}" for x in range(8))}
    tone_only = columns["equal_initial"].astype(bool) & columns["equal_final"].astype(bool) & ~columns["equal_tone"].astype(bool)
    rounded = np.round(distances, 12)
    repeated = Counter(rounded.tolist())
    median = float(np.median(distances))
    mismatch_selector = columns["parser_mismatch"].astype(bool)
    mismatch_endpoints = validation["parser_mismatch_syllables"]
    jqxu = [value for value in mismatch_endpoints if value.startswith(("ju", "qu", "xu"))]
    summary = {
        "schema_version": ANALYSIS_SCHEMA_VERSION, "validation": validation,
        "distribution": {"count": len(distances), "mean": float(distances.mean()), "standard_deviation": float(distances.std()),
                         "quantiles": _quantiles(distances), "median_examples": _examples(columns, np.ones(len(distances), bool), median)},
        "bins": bin_summaries, "overlap_masks": masks, "global_overlap_mask_counts": global_mask_counts,
        "global_overlap_shares": {key: value / len(distances) for key, value in global_mask_counts.items()},
        "tone_only": {**_group_summary(columns, tone_only), "examples": _examples(columns, tone_only, float(np.median(distances[tone_only])))},
        "parser_mismatches": {**_group_summary(columns, mismatch_selector), "endpoint_syllables": mismatch_endpoints,
                              "j_q_x_plus_u_endpoint_syllables": jqxu,
                              "examples": _examples(columns, mismatch_selector, float(np.median(distances[mismatch_selector])) if mismatch_selector.any() else 0)},
        "hardcode_applied": _group_summary(columns, columns["hardcode_applied"].astype(bool)),
        "repeated_distances": {"unique_rounded_12_count": len(repeated), "top": repeated.most_common(20)},
        "histogram_robustness": _histogram_robustness(distances), "kde": _kde_summary(distances),
        "modal_composition_width_1": _modal_composition(columns),
        "segment_pair_reweighted": _segment_reweighted(columns), "tails": _tail_summary(columns),
    }
    if include_clustering:
        syllables = sorted(set(map(str, columns["left_pinyin"])) | set(map(str, columns["right_pinyin"])))
        summary["clustering"] = _cluster_summary(columns, syllables)
    return summary


def historical_comparison(similarity_root: str | Path, expected_bundle_sha256: str = EXPECTED_BUNDLE_MANIFEST_SHA256) -> dict[str, Any]:
    """Summarize hash-validated historical contextual and character-pinyin samples."""
    root = Path(similarity_root).resolve()
    manifest = _read_json(root / "manifest.json")
    if manifest.get("bundle_manifest_sha256") != expected_bundle_sha256:
        raise ValueError("historical analysis references a different bundle")
    result = {"manifest_sha256": sha256_file(root / "manifest.json"), "levels": {}}
    for level in ("contextual", "character_pinyin"):
        table = root / "similarities" / f"{level}.jsonl"
        expected = manifest.get("similarity_table_sha256", {}).get(level)
        if sha256_file(table) != expected:
            raise ValueError(f"historical table checksum mismatch: {level}")
        rows = list(_jsonl(table))
        distances = np.asarray([float(row["dimsim_distance"]) for row in rows])
        result["levels"][level] = {"count": len(rows), "quantiles": _quantiles(distances),
                                    "overlap_mask_counts": _counts(row["overlap_mask"] for row in rows),
                                    "tail_counts": {f"ge_{int(t)}": int((distances >= t).sum()) for t in TAIL_THRESHOLDS}}
    return result


def save_cache(output_dir: str | Path, columns: dict[str, np.ndarray], summary: dict[str, Any], historical: dict[str, Any], *, code_path: str | Path) -> Path:
    """Write ignored numeric tables and a hash-linked manifest atomically enough for reruns."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    table = output / "population.npz"
    np.savez_compressed(table, **columns)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    historical_path = output / "historical.json"
    historical_path.write_text(json.dumps(historical, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "bundle_manifest_sha256": summary["validation"]["manifest_sha256"],
        "analysis_code_sha256": sha256_file(code_path),
        "dimsim_adapter_sha256": sha256_file(Path(code_path).with_name("dimsim.py")),
        "metric_package_file_sha256": summary["validation"]["metric_provenance"]["package_file_sha256"],
        "artifacts": {path.name: sha256_file(path) for path in (table, summary_path, historical_path)},
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def load_cache(output_dir: str | Path, *, code_path: str | Path, bundle_manifest: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Load a cache only when bundle, analysis code, and all outputs still match."""
    output = Path(output_dir).resolve()
    manifest = _read_json(output / "manifest.json")
    if manifest.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise ValueError("unsupported distribution cache schema")
    if sha256_file(bundle_manifest) != manifest.get("bundle_manifest_sha256"):
        raise ValueError("distribution cache bundle hash mismatch")
    if sha256_file(code_path) != manifest.get("analysis_code_sha256"):
        raise ValueError("distribution cache code hash mismatch")
    if sha256_file(Path(code_path).with_name("dimsim.py")) != manifest.get("dimsim_adapter_sha256"):
        raise ValueError("distribution cache DIMSIM adapter hash mismatch")
    for relative, expected in manifest.get("artifacts", {}).items():
        if sha256_file(output / relative) != expected:
            raise ValueError(f"distribution cache artifact hash mismatch: {relative}")
    archive = np.load(output / "population.npz", allow_pickle=False)
    return ({key: archive[key] for key in archive.files}, _read_json(output / "summary.json"), _read_json(output / "historical.json"))
