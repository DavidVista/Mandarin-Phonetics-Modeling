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


def compute_retained_population(root: str | Path, *, expected_manifest_sha256: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Validate and recompute the complete nonzero syllable-pair population."""
    root=Path(root).resolve(); manifest_path=root/"manifest.json"; actual_hash=sha256_file(manifest_path)
    if actual_hash != expected_manifest_sha256: raise ValueError("unexpected evaluation manifest SHA256")
    manifest=_read_json(manifest_path)
    from .geometry_bundle import BUNDLE_TYPE, validate_geometry_bundle
    geometry = manifest.get("bundle_type") == BUNDLE_TYPE
    if geometry:
        manifest = validate_geometry_bundle(root)
        evaluation_id = manifest["identity"]["evaluation_id"]
    elif manifest.get("schema_version") not in {1,2} or manifest.get("pairing_policy") != "nonzero_nonzero" or not isinstance(manifest.get("evaluation_id"),str):
        raise ValueError("unsupported evaluation schema, identity, or pairing policy")
    else:
        evaluation_id = manifest["evaluation_id"]
    artifacts=manifest.get("artifacts")
    if not isinstance(artifacts,dict) or not {"groups.jsonl","pairs/syllable.jsonl"}.issubset(artifacts): raise ValueError("evaluation manifest lacks required payloads")
    verified={}
    for relative,expected in artifacts.items():
        path=(root/relative).resolve()
        if not geometry and (root not in path.parents or not path.is_file() or sha256_file(path)!=expected): raise ValueError(f"invalid evaluation artifact: {relative}")
        verified[relative]=expected
    groups=[]
    for number,row in enumerate(_jsonl(root/"groups.jsonl"),1):
        if row.get("group_kind")!="syllable": continue
        required=("group_id","pinyin","initial","final","tone")
        if any(key not in row for key in required) or not all(isinstance(row[key],str) for key in required): raise ValueError(f"invalid syllable group metadata at row {number}")
        if row["initial"].strip(): groups.append(row)
    groups.sort(key=lambda row:row["group_id"]); by_id={row["group_id"]:row for row in groups}
    if len(groups)<2 or len(by_id)!=len(groups): raise ValueError("eligible syllable population needs at least two distinct groups")
    labels = [row["pinyin"] for row in groups]
    if len(set(labels)) != len(labels):
        raise ValueError("eligible syllable labels are duplicated")
    if geometry and sorted(labels) != manifest["identity"]["inventory"]["eligible"]:
        raise ValueError("syllable groups disagree with the sealed evaluation inventory")
    triples=[(row["initial"],row["final"],row["tone"]) for row in groups]
    if len(triples)!=len(set(triples)): raise ValueError("eligible gold component triples are duplicated")
    parser_mismatches = set()
    for group in groups:
        label = group["pinyin"]
        parsed = parse_library_syllable(label)
        gold = (group["initial"], group["final"], int(group["tone"]))
        if gold != (parsed.library_initial, parsed.library_final, parsed.tone):
            parser_mismatches.add(label)
    pairs=list(_jsonl(root/"pairs/syllable.jsonl")); expected_count=len(groups)*(len(groups)-1)//2
    if len(pairs)!=expected_count: raise ValueError("pair population is incomplete")
    expected_endpoints=((groups[i]["group_id"],groups[j]["group_id"]) for i in range(len(groups)) for j in range(i+1,len(groups)))
    metric=DimsimDistance(); rows=[]; max_bundle=max_formula=max_reverse=0.; tolerance=1e-10
    for number,(pair,(expected_left,expected_right)) in enumerate(zip(pairs,expected_endpoints),1):
        left_id,right_id=pair.get("left_id"),pair.get("right_id")
        if (left_id,right_id)!=(expected_left,expected_right): raise ValueError(f"noncanonical or duplicate pair endpoints at row {number}")
        left_group,right_group=by_id[left_id],by_id[right_id]; left,right=left_group["pinyin"],right_group["pinyin"]
        flags=(left_group["initial"]==right_group["initial"],left_group["final"]==right_group["final"],left_group["tone"]==right_group["tone"]); mask="".join("1" if flag else "0" for flag in flags)
        pair_id=hashlib.sha256(json.dumps([left_id,right_id],ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
        required_values={"left_pinyin":left,"right_pinyin":right,"equal_initial":flags[0],"equal_final":flags[1],"equal_tone":flags[2],"overlap_mask":mask,"pair_id":pair_id,"left_zero_initial":False,"right_zero_initial":False,"no_zero_initial_diagnostic":True,"distance_status":"ok"}
        if any(pair.get(key)!=value or (isinstance(value,bool) and type(pair.get(key)) is not bool) for key,value in required_values.items()): raise ValueError(f"pair labels, flags, mask, or identity mismatch at row {number}")
        bundled=float(pair.get("dimsim_distance",float("nan")))
        if not math.isfinite(bundled) or bundled<0: raise ValueError(f"invalid bundle distance at row {number}")
        public=metric(left,right); reverse=metric(right,left); formula=reconstruct_distance(left,right)
        if not all(math.isfinite(value) and value >= 0 for value in (public, reverse, formula["distance"])):
            raise ValueError(f"invalid recomputed distance at row {number}")
        if formula["hardcode_applied"]: raise ValueError("component decomposition requires no hardcodeMap override")
        max_bundle=max(max_bundle,abs(public-bundled)); max_formula=max(max_formula,abs(public-float(formula["distance"]))); max_reverse=max(max_reverse,abs(public-reverse))
        rows.append({
            **formula,
            "left_id": left_id, "right_id": right_id, "left_pinyin": left, "right_pinyin": right,
            "distance": public, "bundle_distance": bundled, "overlap_mask": mask,
            "equal_initial": flags[0], "equal_final": flags[1], "equal_tone": flags[2],
            "left_gold_initial": left_group["initial"], "right_gold_initial": right_group["initial"],
            "left_gold_final": left_group["final"], "right_gold_final": right_group["final"],
            "left_gold_tone": int(left_group["tone"]), "right_gold_tone": int(right_group["tone"]),
            "parser_mismatch": left in parser_mismatches or right in parser_mismatches,
        })
    diagonal = [metric(label, label) for label in labels]
    if not all(math.isfinite(value) and value >= 0 for value in diagonal):
        raise ValueError("invalid DIMSIM diagonal")
    max_diagonal = max(diagonal)
    if max(max_bundle,max_formula,max_reverse,max_diagonal)>tolerance: raise ValueError("distance validation exceeded numerical tolerance")
    columns=_rows_to_columns(rows); distances=columns["distance"].astype(float)
    if not np.isfinite(distances).all() or np.any(distances<0): raise ValueError("DIMSIM returned a nonfinite or negative distance")
    validation={"manifest_sha256":actual_hash,"evaluation_id":evaluation_id,"artifact_sha256":verified,"syllable_count":len(groups),"pair_count":len(pairs),"expected_pair_count":expected_count,"finite":True,"nonnegative":True,"max_bundle_absolute_error":max_bundle,"max_formula_absolute_error":max_formula,"max_reverse_absolute_error":max_reverse,"max_diagonal_absolute_error":max_diagonal,"tolerance":tolerance,"hardcode_applied_pair_count":0,"metric_provenance":metric.provenance}
    validation["parser_mismatch_syllables"] = sorted(parser_mismatches)
    validation["parser_mismatch_pair_count"] = int(columns["parser_mismatch"].sum())
    if geometry:
        validation["pairing_policy"] = manifest["identity"]["pairing_policy"]
        validation["excluded_pinyin"] = manifest["identity"]["excluded_pinyin"]
    return columns,validation


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
    std = float(distances.std(ddof=1)) if len(distances) > 1 else 0.0
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
        unique_labels = np.unique(labels)
        silhouette = None
        if 1 < len(unique_labels) < len(labels):
            means = np.column_stack([matrix[:, labels == label].mean(axis=1) for label in unique_labels])
            own = np.searchsorted(unique_labels, labels)
            indexes = np.arange(len(labels))
            sizes = np.bincount(own)[own]
            within = np.divide(means[indexes, own] * sizes, sizes - 1, out=np.zeros(len(labels)), where=sizes > 1)
            means[indexes, own] = np.inf
            nearest = means.min(axis=1)
            scale = np.maximum(within, nearest)
            scores = np.divide(nearest - within, scale, out=np.zeros(len(labels)), where=(sizes > 1) & (scale > 0))
            silhouette = float(scores.mean())
        result[str(k)] = {"silhouette": silhouette, "clusters": clusters}
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
        "tone_only": {**_group_summary(columns, tone_only),
                      "examples": _examples(columns, tone_only, float(np.median(distances[tone_only]))) if tone_only.any() else []},
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


def load_verified_population_cache(
    cache_root: str | Path,
    bundle_root: str | Path,
    *,
    expected_cache_manifest_sha256: str,
    expected_bundle_manifest_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load the pinned numeric population without invoking broad legacy summaries.

    Both identities are parameters so a research wrapper cannot silently migrate to
    a newer bundle or cache. Every declared payload in both manifests is verified.
    """
    cache_root, bundle_root = Path(cache_root).resolve(), Path(bundle_root).resolve()
    cache_manifest_path, bundle_manifest_path = cache_root / "manifest.json", bundle_root / "manifest.json"
    if sha256_file(cache_manifest_path) != expected_cache_manifest_sha256:
        raise ValueError("unexpected distribution cache manifest SHA256")
    if sha256_file(bundle_manifest_path) != expected_bundle_manifest_sha256:
        raise ValueError("unexpected evaluation bundle manifest SHA256")
    cache_manifest, bundle_manifest = _read_json(cache_manifest_path), _read_json(bundle_manifest_path)
    for root, manifest, label in ((cache_root, cache_manifest, "cache"), (bundle_root, bundle_manifest, "bundle")):
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError(f"{label} manifest has no artifacts")
        for relative, expected in artifacts.items():
            path = (root / relative).resolve()
            if root not in path.parents or not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"invalid {label} artifact: {relative}")
    if cache_manifest.get("bundle_manifest_sha256") != expected_bundle_manifest_sha256:
        raise ValueError("cache was not built from the selected bundle")
    if bundle_manifest.get("evaluation_id") != "1.0.0" or bundle_manifest.get("pairing_policy") != "nonzero_nonzero":
        raise ValueError("unexpected evaluation schema or pairing policy")
    groups = [row for row in _jsonl(bundle_root / "groups.jsonl") if row.get("group_kind") == "syllable" and row.get("initial")]
    groups.sort(key=lambda row: row["group_id"]); ids = {row["group_id"] for row in groups}
    if len(groups) != EXPECTED_ELIGIBLE_SYLLABLES or len(ids) != len(groups):
        raise ValueError("unexpected or duplicate eligible syllable population")
    endpoints = set(); bundle_pairs = []
    for row in _jsonl(bundle_root / "pairs/syllable.jsonl"):
        left, right = row.get("left_id"), row.get("right_id")
        expected_mask = "".join("1" if row.get(key) is True else "0" for key in ("equal_initial","equal_final","equal_tone"))
        if left not in ids or right not in ids or left >= right or row.get("overlap_mask") != expected_mask or (left,right) in endpoints:
            raise ValueError("invalid pair schema, endpoint, mask, or duplicate")
        endpoints.add((left,right)); bundle_pairs.append((left,right,row["overlap_mask"],float(row["dimsim_distance"])))
    complete = {(groups[i]["group_id"],groups[j]["group_id"]) for i in range(len(groups)) for j in range(i+1,len(groups))}
    if endpoints != complete or len(endpoints) != EXPECTED_PAIR_COUNT:
        raise ValueError("pair population is incomplete")
    cached_summary = _read_json(cache_root / "summary.json"); validation = cached_summary.get("validation",{})
    provenance = validation.get("metric_provenance",{})
    if validation.get("syllable_count") != EXPECTED_ELIGIBLE_SYLLABLES or validation.get("pair_count") != EXPECTED_PAIR_COUNT:
        raise ValueError("cache validation population does not match selected bundle")
    if provenance.get("metric_version") != "0.2.2" or provenance.get("dependencies",{}).get("dimsim") != "0.2.2":
        raise ValueError("cache does not document pinned dimsim==0.2.2")
    archive = np.load(cache_root / "population.npz", allow_pickle=False)
    columns = {key: archive[key] for key in archive.files}
    _validate_cached_population(columns,bundle_pairs,validation,expected_bundle_manifest_sha256)
    return columns, {"cache_manifest": cache_manifest, "bundle_manifest": bundle_manifest}


def _validate_cached_population(columns, bundle_pairs, validation, expected_bundle_manifest_sha256):
    """Reject any cache whose plotted population or validation evidence is stale."""
    required = {"distance", "overlap_mask", "consonant_distance", "vowel_distance", "tone_distance", "multiplier", "hardcode_applied"}
    if required - set(columns):
        raise ValueError(f"population cache lacks columns: {sorted(required - set(columns))}")
    n = len(columns["distance"])
    if n < 1 or any(len(columns[key]) != n for key in required):
        raise ValueError("population cache columns are empty or misaligned")
    if not {"left_id","right_id","bundle_distance"}.issubset(columns) or n != len(bundle_pairs):
        raise ValueError("cache cannot be bound to bundle pair rows")
    cached_endpoints=list(zip(columns["left_id"].astype(str),columns["right_id"].astype(str)))
    if cached_endpoints != [(str(a),str(b)) for a,b,_,_ in bundle_pairs]:
        raise ValueError("cache pair IDs/order do not match selected bundle")
    if not np.array_equal(columns["overlap_mask"].astype(str),np.asarray([m for _,_,m,_ in bundle_pairs])):
        raise ValueError("cache masks do not match selected bundle")
    if not np.allclose(columns["bundle_distance"].astype(float),np.asarray([d for *_,d in bundle_pairs]),rtol=0,atol=0):
        raise ValueError("cache distances do not match selected bundle")
    distances = columns["distance"].astype(float)
    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError("population distances must be finite and non-negative")
    masks = columns["overlap_mask"].astype(str)
    if not set(masks).issubset({f"{i:03b}" for i in range(8)}):
        raise ValueError("invalid exact overlap mask")
    tolerance=validation.get("tolerance")
    mandatory=("max_bundle_absolute_error","max_formula_absolute_error","max_reverse_absolute_error","max_diagonal_absolute_error")
    if not isinstance(tolerance,(int,float)) or not math.isfinite(tolerance) or tolerance<0:
        raise ValueError("cache validation tolerance is missing or invalid")
    if validation.get("manifest_sha256") != expected_bundle_manifest_sha256 or validation.get("expected_pair_count") != EXPECTED_PAIR_COUNT:
        raise ValueError("cache validation bundle identity or expected pair count mismatch")
    if validation.get("finite") is not True or validation.get("nonnegative") is not True:
        raise ValueError("cache validation did not pass finite/nonnegative checks")
    if any(not isinstance(validation.get(key),(int,float)) or not math.isfinite(validation[key]) or validation[key]>tolerance for key in mandatory):
        raise ValueError("cache validation numerical invariant failed")
    actual=np.asarray([d for *_,d in bundle_pairs],float)
    if not np.allclose(distances,actual,rtol=0,atol=tolerance):
        raise ValueError("plotted cache distance differs from selected bundle")


def retained_distribution_data(columns: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Derive lossless plot data for the three approved distribution figures."""
    d = np.asarray(columns["distance"], dtype=float)
    masks = np.asarray(columns["overlap_mask"], dtype=str)
    hardcode = np.asarray(columns["hardcode_applied"], dtype=bool)
    if hardcode.any():
        raise ValueError("component decomposition requires no hardcodeMap override")
    initial = np.asarray(columns["consonant_distance"], float) * np.asarray(columns["multiplier"], float)
    final = np.asarray(columns["vowel_distance"], float) * np.asarray(columns["multiplier"], float)
    tone = np.asarray(columns["tone_distance"], float) * np.asarray(columns["multiplier"], float)
    if not np.allclose(initial + final + tone, d, rtol=1e-12, atol=1e-12):
        raise ValueError("component contributions do not reconstruct DIMSIM distance")
    rows = []
    for low in range(0, 121, 10):
        high = low + 10 if low < 120 else np.inf
        selected = (d >= low) & (d < high)
        count = int(selected.sum())
        means = np.array([initial[selected].mean(), final[selected].mean(), tone[selected].mean()]) if count else np.full(3, np.nan)
        mean_distance = float(d[selected].mean()) if count else np.nan
        shares = means / mean_distance if count and mean_distance != 0 else np.full(3, np.nan)
        rows.append({"bin": f"[{low},{low+10})" if low < 120 else "[120,+inf)", "count": count,
                     "mean_distance": mean_distance if np.isfinite(mean_distance) else None,
                     "mean_initial": float(means[0]) if np.isfinite(means[0]) else None,
                     "mean_final": float(means[1]) if np.isfinite(means[1]) else None,
                     "mean_tone": float(means[2]) if np.isfinite(means[2]) else None,
                     "initial_share": float(shares[0]) if np.isfinite(shares[0]) else None,
                     "final_share": float(shares[1]) if np.isfinite(shares[1]) else None,
                     "tone_share": float(shares[2]) if np.isfinite(shares[2]) else None})
    full_edges = np.r_[np.arange(0.0, 121.0, 10.0), max(120.0, float(d.max())) + 1e-6]
    return {"distance": d, "overlap_mask": masks, "initial_contribution": initial, "final_contribution": final,
            "tone_contribution": tone, "full_histogram_edges": full_edges, "detail_values": d[d < 10]}, rows
