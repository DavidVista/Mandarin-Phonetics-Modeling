"""Compare original DIMSIM distances with the Li et al. human ratings."""
from collections import defaultdict
import csv
import hashlib
import io
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import mean

from ..bundles import seal_bundle, staging_directory, validate_bundle
from ..evaluation.dimsim import DimsimDistance
from ..pinyin import PARSER_PROVENANCE, parse_components, validate_canonical_label


def _read_csv(root, source, name, fields):
    payload = (root / name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != source["files"][name]:
        raise ValueError(f"Checksum mismatch: {name}")
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""))
    if reader.fieldnames != fields:
        raise ValueError(f"Invalid columns: {name}")
    rows = list(reader)
    if any(set(row) != set(fields) or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Invalid row: {name}")
    return rows


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_human_agreement(input_root, output):
    from scipy import __version__ as scipy_version
    from scipy.stats import spearmanr

    root, target = Path(input_root).resolve(), Path(output).resolve()
    if target.is_relative_to(root) or root.is_relative_to(target):
        raise ValueError("Input and output directories must be separate")
    metadata = (root / "source.json").read_bytes()
    source = json.loads(metadata)
    readings = {}
    for row in _read_csv(root, source, "readings.csv", ["character", "pinyin"]):
        char, label = row["character"], row["pinyin"]
        if len(char) != 1 or char in readings:
            raise ValueError(f"Invalid or duplicate character: {char}")
        readings[char] = validate_canonical_label(label)
    nonzero = {label: bool(parse_components(label).initial) for label in set(readings.values())}
    scores = {}
    for row in _read_csv(root, source, "similarity.csv", ["character_a", "character_b", "human_similarity"]):
        a, b = sorted((row["character_a"], row["character_b"]))
        score = float(row["human_similarity"])
        if a == b or a not in readings or b not in readings or (a, b) in scores or not 1 <= score <= 7:
            raise ValueError(f"Invalid or duplicate similarity pair: {a}, {b}")
        scores[a, b] = score
    pairs = list(combinations(sorted(readings), 2))
    if set(pairs) != scores.keys():
        raise ValueError("Human similarities must cover every unordered character pair")

    metric = DimsimDistance()
    identity = {
        "source": source,
        "source_metadata_sha256": hashlib.sha256(metadata).hexdigest(),
        "metric": metric.provenance,
        "parser": PARSER_PROVENANCE,
        "scipy_version": scipy_version,
        "method_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "human_distance": "8 - mean similarity",
        "pinyin_aggregation": "Mean human distance over character pairs with the same unordered pair of distinct tonal readings",
        "slices": {
            "all_character_pairs": "All unordered pairs of distinct characters",
            "nonzero_initial_character_pairs": "Both tonal readings have a nonempty strict Pinyin initial",
            "nonzero_initial_distinct_pinyin_pairs": "Unique unordered pairs of distinct tonal readings in the nonzero-initial character slice",
        },
        "statistic": "Spearman rank correlation",
    }
    if target.exists():
        validate_bundle(target, "dimsim_human_agreement", identity=identity)
        return target
    rows, retained = [], []
    by_reading = defaultdict(list)
    distances = {}
    for a, b in pairs:
        label_a, label_b = readings[a], readings[b]
        key = tuple(sorted((label_a, label_b)))
        if key not in distances:
            distances[key] = metric(*key)
        distance = distances[key]
        if not math.isfinite(distance) or distance < 0:
            raise ValueError(f"Invalid DIMSIM distance: {key}")
        score = scores[a, b]
        row = {"character_a": a, "character_b": b, "pinyin_a": label_a, "pinyin_b": label_b,
               "human_similarity": score, "human_distance": 8 - score, "dimsim_distance": distance,
               "involves_zero_initial": not (nonzero[label_a] and nonzero[label_b]),
               "same_pinyin": label_a == label_b}
        rows.append(row)
        if not row["involves_zero_initial"]:
            retained.append(row)
            if not row["same_pinyin"]:
                by_reading[key].append(row["human_distance"])

    summary = []
    for name, values in (
        ("all_character_pairs", [(row["human_distance"], row["dimsim_distance"]) for row in rows]),
        ("nonzero_initial_character_pairs", [(row["human_distance"], row["dimsim_distance"]) for row in retained]),
        ("nonzero_initial_distinct_pinyin_pairs", [(mean(group), distances[key]) for key, group in by_reading.items()]),
    ):
        if len(values) < 2:
            raise ValueError(f"Insufficient pairs for {name}")
        human, dimsim = zip(*values)
        rho = float(spearmanr(human, dimsim).statistic) if len(set(human)) > 1 and len(set(dimsim)) > 1 else None
        summary.append({"slice": name, "pairs": len(values), "spearman": rho})
    with staging_directory(target) as stage:
        _write_csv(stage / "pairs.csv", rows)
        _write_csv(stage / "summary.csv", summary)
        seal_bundle(stage, "dimsim_human_agreement", identity)
    return target
