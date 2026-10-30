"""Schema-3 immutable contextual-only evaluation bundle implementation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import shutil
from pathlib import Path
from typing import Any

from .anchor_sampling import EDGES, SAMPLER_VERSION as Q_SAMPLER_VERSION, TRIPLET_SCHEMA_VERSION, sample_anchor_q
from .dimsim_sampling import PAIR_SCHEMA_VERSION, SAMPLER_VERSION as DIMSIM_SAMPLER_VERSION, pair_identity, sample_balanced, sample_natural
from .dimsim import DimsimDistance
from ..pinyin import PARSER_PROVENANCE, PINYIN_CONTRACT_VERSION, parse_components, validate_canonical_label

SCHEMA_VERSION = 3
Q_SEED = 20261026
Q_SAMPLER_VERSION = 1
TRIPLET_SCHEMA_VERSION = 1
DIMSIM_SAMPLER_VERSION = 1
PAIR_SCHEMA_VERSION = 1
FILES = {"config.json", "manifest.json", "coverage.json", "sentences.jsonl", "occurrences.jsonl",
         "syllables.jsonl", "pairs/dimsim-natural.jsonl", "pairs/dimsim-balanced.jsonl", "triplets/anchor-q.jsonl"}
SOURCE_HASH_NAMES = {"manifest.json", "config.json", "vocab.json", "samples.jsonl", "splits.json"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _readl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _stable(prefix: str, values: list[str]) -> str:
    return prefix + hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _syllables(occurrences: list[dict]) -> list[dict]:
    members: dict[str, list[str]] = {}
    metadata = {}
    for row in occurrences:
        sid = _stable("syllable:", [row["pinyin"]])
        if row.get("syllable_id") != sid:
            raise ValueError(f"occurrence syllable_id disagrees with canonical Pinyin: {row.get('occurrence_id')}")
        metadata[sid] = {"syllable_id": sid, "pinyin": row["pinyin"], "initial": row["initial"],
                         "final": row["final"], "tone": row["tone"]}
        members.setdefault(sid, []).append(row["occurrence_id"])
    return [{**metadata[sid], "occurrence_ids": sorted(members[sid]), "occurrence_count": len(members[sid])}
            for sid in sorted(metadata)]


def _anchor_q_support(syllables: list[dict]) -> dict[str, tuple[list[str], dict[str, str]]]:
    """Independently derive eligible anchors and exclusions from gold metadata."""
    by_id = {row["syllable_id"]: row for row in syllables}
    ordered = sorted(by_id)

    def overlap(anchor_id: str, candidate_id: str) -> str:
        a, b = by_id[anchor_id], by_id[candidate_id]
        return "".join(str(int(a[key] == b[key])) for key in ("initial", "final", "tone"))

    support = {}
    for component, low_mask, high_mask in EDGES:
        eligible, excluded = [], {}
        for anchor_id in ordered:
            members = by_id[anchor_id]["occurrence_ids"]
            if high_mask == "111" and len(members) < 2:
                excluded[anchor_id] = "high_111_requires_two_occurrences"
                continue
            high = [candidate for candidate in ordered
                    if overlap(anchor_id, candidate) == high_mask and (candidate != anchor_id or high_mask == "111")]
            low = [candidate for candidate in ordered
                   if overlap(anchor_id, candidate) == low_mask and candidate != anchor_id]
            if high and low:
                eligible.append(anchor_id)
            else:
                excluded[anchor_id] = "missing_high_or_low_syllable"
        support[f"{component}:{low_mask}->{high_mask}"] = (eligible, excluded)
    return support


def _anchor_q_coverage(syllables: list[dict], triplets: list[dict], *, seed: int, draws_per_edge: int) -> dict:
    support = _anchor_q_support(syllables)
    by_edge = {edge: [row for row in triplets if row["edge_id"] == edge] for edge in support}
    result = {"sampler_version": Q_SAMPLER_VERSION, "seed": seed, "rng": "numpy.random.PCG64",
              "draws_per_edge": draws_per_edge, "edges": {}}
    unique_triples, unique_anchor_occurrences, unique_anchor_syllables, unique_occurrences = {}, {}, {}, {}
    for edge, (eligible, excluded) in support.items():
        rows = by_edge[edge]
        unique_triples[edge] = len({row["triplet_id"] for row in rows})
        unique_anchor_occurrences[edge] = len({row["anchor_id"] for row in rows})
        unique_anchor_syllables[edge] = len({row["anchor_syllable_id"] for row in rows})
        unique_occurrences[edge] = len({row[field] for row in rows for field in ("anchor_id", "high_id", "low_id")})
        result["edges"][edge] = {
            "eligible_anchor_syllables": len(eligible), "excluded_anchor_syllables": len(excluded),
            "exclusion_reasons": excluded, "draw_count": len(rows),
            "unique_triplets": unique_triples[edge],
            "unique_anchor_occurrences": unique_anchor_occurrences[edge],
            "unique_anchor_syllables": unique_anchor_syllables[edge],
            "unique_occurrences": unique_occurrences[edge],
        }
    result.update({"draw_count": len(triplets), "unique_triplets_by_edge": unique_triples,
                   "unique_anchor_occurrences_by_edge": unique_anchor_occurrences,
                   "unique_anchor_syllables_by_edge": unique_anchor_syllables,
                   "unique_occurrences_by_edge": unique_occurrences})
    return result


def _metric_rows(plans, occurrence_map, scorer, metric_name, mode):
    rows = []
    for item in plans:
        left_id, right_id, pair_id, draw_index = item[:4]
        left, right = occurrence_map[left_id], occurrence_map[right_id]
        status, value = "not_computed", None
        if scorer is not None:
            try:
                value = float(scorer(left["pinyin"], right["pinyin"]))
                if not math.isfinite(value) or value < 0:
                    raise ValueError("DIMSIM distance must be finite and non-negative")
                status = "ok"
            except Exception as error:
                status, value = f"invalid:{type(error).__name__}", None
        mask = "".join(str(int(left[key] == right[key])) for key in ("initial", "final", "tone"))
        row = {"schema_version": PAIR_SCHEMA_VERSION, "sampler_version": DIMSIM_SAMPLER_VERSION,
               "mode": mode, "pair_id": pair_id,
               "left_id": min(left_id, right_id), "right_id": max(left_id, right_id),
               "left_pinyin": occurrence_map[min(left_id, right_id)]["pinyin"],
               "right_pinyin": occurrence_map[max(left_id, right_id)]["pinyin"],
               "dimsim_distance": value, "distance_status": status, "overlap_mask": mask,
               "equal_initial": mask[0] == "1", "equal_final": mask[1] == "1", "equal_tone": mask[2] == "1",
               "left_zero_initial": False, "right_zero_initial": False, "no_zero_initial_diagnostic": True,
               "draw_index": draw_index}
        if mode == "balanced":
            row.update({"draw_id": f"dimsim-balanced:v1:{draw_index:06d}",
                        "first_id": left_id, "second_id": right_id,
                        "first_syllable_id": item[4], "second_syllable_id": item[5]})
        else:
            row["draw_id"] = f"dimsim-natural:v1:{draw_index:06d}"
        rows.append(row)
    return rows


def _validate_provenance(code_provenance):
    if not isinstance(code_provenance, dict) or not code_provenance or any(
        not isinstance(k, str) or not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
        for k, v in code_provenance.items()):
        raise ValueError("explicit preparation code_provenance hashes are required")


def _validate_source_provenance(source_experiment):
    if not isinstance(source_experiment, dict) or set(source_experiment) != {"path", "experiment_id", "hashes"}:
        raise ValueError("source_experiment provenance must contain path, experiment_id, and hashes")
    for field in ("path", "experiment_id"):
        value = source_experiment.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"source_experiment {field} must be a non-empty string")
    hashes = source_experiment.get("hashes")
    if (not isinstance(hashes, dict) or set(hashes) != SOURCE_HASH_NAMES or any(
        not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)
        for value in hashes.values())):
        raise ValueError("source_experiment hashes must contain exactly five valid SHA-256 values")


def _typed_equal(actual, expected) -> bool:
    """Compare JSON contract values without Python bool/int equality aliases."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(_typed_equal(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_typed_equal(left, right) for left, right in zip(actual, expected))
    return actual == expected


def prepare_contextual_bundle(evaluation_id: str, *, experiment_path, output_root, split="test",
                              sentence_limit=None, sentence_seed=42, distance_fn=None,
                              metric_provenance=None, code_provenance=None, protocol_config=None) -> Path:
    from .bundle import (_validate_evaluation_id, _load_source_experiment, _build_occurrences)
    evaluation_id = _validate_evaluation_id(evaluation_id)
    _validate_provenance(code_provenance)
    if not isinstance(protocol_config, dict):
        raise TypeError("protocol_config must be a dictionary")
    defaults = {"dimsim_natural_count": 12000, "dimsim_natural_seed": 42,
                "dimsim_balanced_count": 12000, "dimsim_balanced_seed": 20261026,
                "anchor_q_draws_per_edge": 1000, "anchor_q_seed": Q_SEED}
    unknown = set(protocol_config) - set(defaults)
    if unknown: raise ValueError(f"unknown protocol_config keys: {sorted(unknown)}")
    settings = {**defaults, **protocol_config}
    for key, value in settings.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a non-negative integer")
    for count_key in ("dimsim_natural_count", "dimsim_balanced_count", "anchor_q_draws_per_edge"):
        if settings[count_key] == 0: raise ValueError(f"{count_key} must be positive")
    if distance_fn is None or not callable(distance_fn): raise ValueError("an explicit distance provider is required")
    if not isinstance(metric_provenance, dict) or not isinstance(metric_provenance.get("metric_name"), str):
        raise ValueError("metric_provenance with metric_name is required")
    if metric_provenance["metric_name"] == "dimsim":
        if not isinstance(distance_fn, DimsimDistance) or metric_provenance != distance_fn.provenance:
            raise ValueError("DIMSIM bundles require the pinned DimsimDistance provider and provenance")
    elif metric_provenance["metric_name"] != "synthetic":
        raise ValueError("only DIMSIM and explicitly synthetic metric providers are supported")
    source, root = Path(experiment_path).resolve(), Path(output_root).resolve()
    target = (root / evaluation_id).resolve()
    if target.parent != root or target == source or source in target.parents: raise ValueError("invalid evaluation output path")
    if target.exists(): raise FileExistsError(f"evaluation bundle already exists: {target}")
    samples, vocab, source_manifest, source_hashes = _load_source_experiment(source, split, sentence_limit, sentence_seed)
    occurrences = _build_occurrences(samples, vocab)
    for row in occurrences:
        row["syllable_id"] = _stable("syllable:", [row["pinyin"]])
    syllables = _syllables(occurrences)
    # Always generate Q diagnostics before writing anything; missing support cannot yield a ready bundle.
    triplets, q_coverage = sample_anchor_q(occurrences, syllables, draws_per_edge=settings["anchor_q_draws_per_edge"], seed=settings["anchor_q_seed"])
    plans_natural = sample_natural(occurrences, settings["dimsim_natural_count"], settings["dimsim_natural_seed"])
    plans_balanced = sample_balanced(occurrences, syllables, settings["dimsim_balanced_count"], settings["dimsim_balanced_seed"])
    # Replaying here ensures the in-memory tables are deterministic before any
    # ready bundle is published. Later model runs only consume saved rows.
    if sample_anchor_q(occurrences, syllables, draws_per_edge=settings["anchor_q_draws_per_edge"], seed=settings["anchor_q_seed"])[0] != triplets:
        raise RuntimeError("anchor-Q deterministic preparation replay mismatch")
    if sample_natural(occurrences, settings["dimsim_natural_count"], settings["dimsim_natural_seed"]) != plans_natural:
        raise RuntimeError("natural DIMSIM deterministic preparation replay mismatch")
    if sample_balanced(occurrences, syllables, settings["dimsim_balanced_count"], settings["dimsim_balanced_seed"]) != plans_balanced:
        raise RuntimeError("balanced DIMSIM deterministic preparation replay mismatch")
    occurrence_map = {r["occurrence_id"]: r for r in occurrences}
    natural = _metric_rows(plans_natural, occurrence_map, distance_fn, metric_provenance["metric_name"], "natural")
    balanced = _metric_rows(plans_balanced, occurrence_map, distance_fn, metric_provenance["metric_name"], "balanced")
    selected_sentences = [{"sentence_id": r["sample_id"], "text": r["text"]} for r in sorted(samples, key=lambda x: x["sample_id"])]
    coverage = {"sentence_count": len(selected_sentences), "occurrence_count": len(occurrences),
                "syllable_count": len(syllables), "zero_initial_occurrences_excluded_from_dimsim": sum(not r["initial"] for r in occurrences),
                "dimsim_natural": {"draw_count": len(natural), "unique_pairs": len({r['pair_id'] for r in natural}),
                                   "unique_occurrences": len({r[k] for r in natural for k in ('left_id','right_id')}),
                                   "unique_syllables": len({sid for r in natural for sid in (_stable('syllable:', [r['left_pinyin']]), _stable('syllable:', [r['right_pinyin']]))})},
                "dimsim_balanced": {"draw_count": len(balanced), "unique_pairs": len({r['pair_id'] for r in balanced}),
                                    "unique_occurrences": len({r[k] for r in balanced for k in ('left_id','right_id')}),
                                    "unique_syllables": len({sid for r in balanced for sid in (r['first_syllable_id'], r['second_syllable_id'])}),
                                    "same_pronunciation_draws": sum(r['overlap_mask'] == '111' for r in balanced)},
                "anchor_q": q_coverage}
    package_versions = {}
    for package in ("numpy", "dimsim", "pypinyin"):
        try: package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: package_versions[package] = None
    config = {"schema_version": SCHEMA_VERSION, "evaluation_id": evaluation_id, "split": split,
              "sentence_limit": sentence_limit, "sentence_seed": sentence_seed,
              "population": {"unit": "contextual_occurrence", "zero_initial_excluded_from_dimsim": True,
                             "syllable_identity": "canonical_toned_pinyin", "aggregates": False},
              "sampling": {"dimsim_natural": {"count": settings["dimsim_natural_count"], "without_replacement": True,
                            "rng": "random.Random", "seed": settings["dimsim_natural_seed"],
                            "sampler_version": DIMSIM_SAMPLER_VERSION, "pair_schema_version": PAIR_SCHEMA_VERSION},
                           "dimsim_balanced": {"count": settings["dimsim_balanced_count"], "with_replacement": True,
                             "rng": "numpy.random.PCG64", "seed": settings["dimsim_balanced_seed"], "choice_order": ["first_syllable", "second_syllable", "first_occurrence", "second_occurrence"],
                             "sampler_version": DIMSIM_SAMPLER_VERSION, "pair_schema_version": PAIR_SCHEMA_VERSION},
                           "anchor_q": {"draws_per_edge": settings["anchor_q_draws_per_edge"], "edge_order": [f"{c}:{l}->{h}" for c,l,h in EDGES],
                             "with_replacement": True, "rng": "numpy.random.PCG64", "seed": settings["anchor_q_seed"],
                             "sampler_version": Q_SAMPLER_VERSION, "triplet_schema_version": TRIPLET_SCHEMA_VERSION}},
              "scoring": {"dimsim": metric_provenance, "anchor_q": {"cosine": "stable_float64_scale_before_norm", "tie_tolerance": 1e-12}},
              "runtime_versions": package_versions}
    source_provenance = {"path": str(source), "experiment_id": source_manifest.get("experiment_id"), "hashes": source_hashes}
    _validate_source_provenance(source_provenance)
    target.mkdir(parents=True)
    try:
        (target / "pairs").mkdir(); (target / "triplets").mkdir()
        _jsonl(target / "sentences.jsonl", selected_sentences); _jsonl(target / "occurrences.jsonl", occurrences)
        _jsonl(target / "syllables.jsonl", syllables); _jsonl(target / "pairs/dimsim-natural.jsonl", natural)
        _jsonl(target / "pairs/dimsim-balanced.jsonl", balanced); _jsonl(target / "triplets/anchor-q.jsonl", triplets)
        _json(target / "coverage.json", coverage); _json(target / "config.json", config)
        artifacts = {p.relative_to(target).as_posix(): _sha(p) for p in target.rglob("*") if p.is_file()}
        manifest = {"schema_version": SCHEMA_VERSION, "evaluation_id": evaluation_id,
                    "pinyin_contract": PINYIN_CONTRACT_VERSION, "parser_provenance": PARSER_PROVENANCE,
                    "source_experiment": source_provenance,
                    "metric": metric_provenance, "code_provenance": code_provenance,
                    "counts": {"sentences": len(selected_sentences), "occurrences": len(occurrences), "syllables": len(syllables),
                               "dimsim_natural": len(natural), "dimsim_balanced": len(balanced), "anchor_q": len(triplets)},
                    "artifacts": artifacts}
        _json(target / "manifest.json", manifest)
    except Exception:
        shutil.rmtree(target, ignore_errors=True); raise
    return target


def load_contextual_bundle(root: Path, manifest: dict) -> dict[str, Any]:
    if type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported evaluation bundle schema")
    if manifest.get("pinyin_contract") != PINYIN_CONTRACT_VERSION or manifest.get("parser_provenance") != PARSER_PROVENANCE:
        raise ValueError("unsupported evaluation Pinyin contract")
    _validate_provenance(manifest.get("code_provenance"))
    _validate_source_provenance(manifest.get("source_experiment"))
    config = _read(root / "config.json")
    if config.get("schema_version") != SCHEMA_VERSION: raise ValueError("unsupported evaluation config schema")
    if _validate_id(config.get("evaluation_id")) != _validate_id(manifest.get("evaluation_id")):
        raise ValueError("schema-3 evaluation IDs disagree")
    _validate_config(config, manifest.get("metric"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != FILES - {"manifest.json"}:
        raise ValueError("schema-3 manifest artifact inventory is invalid")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}
    if actual != FILES - {"manifest.json"}: raise ValueError("schema-3 bundle file inventory is invalid")
    for rel, expected in artifacts.items():
        path = (root / rel).resolve()
        if root not in path.parents or not path.is_file() or _sha(path) != expected:
            raise ValueError(f"evaluation artifact checksum mismatch: {rel}")
    sentences, occurrences, syllables = (_readl(root / name) for name in ("sentences.jsonl", "occurrences.jsonl", "syllables.jsonl"))
    sentence_map = {r.get("sentence_id"): r for r in sentences}
    occurrence_map = {r.get("occurrence_id"): r for r in occurrences}
    syllable_map = {r.get("syllable_id"): r for r in syllables}
    if len(sentence_map) != len(sentences) or len(occurrence_map) != len(occurrences) or len(syllable_map) != len(syllables):
        raise ValueError("duplicate or invalid sentence, occurrence, or syllable IDs")
    positions_by_sentence: dict[str, list[int]] = {sid: [] for sid in sentence_map}
    expected_syllables = _syllables(occurrences)
    if expected_syllables != syllables: raise ValueError("syllable metadata or membership disagrees with occurrences")
    for row in occurrences:
        sid = row.get("sample_id"); pos = row.get("character_position")
        if sid not in sentence_map or isinstance(pos, bool) or not isinstance(pos, int) or pos < 0:
            raise ValueError("occurrence sentence alignment is invalid")
        if row.get("occurrence_id") != f"{sid}:{pos}": raise ValueError("occurrence ID does not match sentence position")
        text = sentence_map[sid]["text"]
        if pos >= len(text) or text[pos] != row.get("character"):
            raise ValueError(f"occurrence character alignment mismatch: {row.get('occurrence_id')}")
        parsed = parse_components(validate_canonical_label(row.get("pinyin")))
        if tuple(row.get(k) for k in ("initial", "final", "tone")) != (parsed.initial, parsed.final, parsed.tone):
            raise ValueError(f"occurrence components disagree with Pinyin: {row.get('occurrence_id')}")
        positions_by_sentence[sid].append(pos)
    if any(sorted(pos) != list(range(len(sentence_map[sid]["text"]))) for sid, pos in positions_by_sentence.items()):
        raise ValueError("sentence occurrence positions do not fully cover sentence text")
    occurrence_syllable = {oid: row["syllable_id"] for row in syllables for oid in row["occurrence_ids"]}
    anchor_support = _anchor_q_support(syllables)
    pairs = {"dimsim-natural": _readl(root / "pairs/dimsim-natural.jsonl"),
             "dimsim-balanced": _readl(root / "pairs/dimsim-balanced.jsonl")}
    for mode, rows in pairs.items():
        if any(r.get("mode") != ("natural" if mode.endswith("natural") else "balanced") for r in rows):
            raise ValueError(f"{mode} mode labels are invalid")
        draw_ids = [r.get("draw_id") for r in rows]
        config_mode = "dimsim_" + mode.removeprefix("dimsim-")
        if len(draw_ids) != len(set(draw_ids)) or len(rows) != config["sampling"][config_mode]["count"]:
            raise ValueError(f"{mode} draw count or IDs are invalid")
        if any(type(r.get("draw_index")) is not int or r.get("draw_index") != i for i, r in enumerate(rows)):
            raise ValueError(f"{mode} draw order is invalid")
        if mode.endswith("natural") and len({r.get("pair_id") for r in rows}) != len(rows):
            raise ValueError("natural DIMSIM pairs must be unique")
        for row in rows:
            if (type(row.get("schema_version")) is not int or row.get("schema_version") != PAIR_SCHEMA_VERSION
                    or type(row.get("sampler_version")) is not int or row.get("sampler_version") != DIMSIM_SAMPLER_VERSION):
                raise ValueError(f"{mode} unsupported row or sampler version")
            a, b = row.get("left_id"), row.get("right_id")
            if a not in occurrence_map or b not in occurrence_map or a == b or a >= b:
                raise ValueError(f"{mode} pair has invalid endpoints")
            left, right = occurrence_map[a], occurrence_map[b]
            if not left["initial"] or not right["initial"]: raise ValueError(f"{mode} violates zero-initial exclusion")
            if row.get("pair_id") != pair_identity(a, b): raise ValueError(f"{mode} pair identity mismatch")
            if (row.get("left_pinyin") != left["pinyin"] or row.get("right_pinyin") != right["pinyin"]):
                raise ValueError(f"{mode} Pinyin labels disagree with endpoint metadata")
            expected_draw = f"dimsim-{'natural' if mode.endswith('natural') else 'balanced'}:v1:{row['draw_index']:06d}"
            if row.get("draw_id") != expected_draw: raise ValueError(f"{mode} draw ID mismatch")
            mask = "".join(str(int(left[k] == right[k])) for k in ("initial", "final", "tone"))
            if row.get("overlap_mask") != mask: raise ValueError(f"{mode} overlap mask mismatch")
            if any(row.get("equal_" + key) is not (mask[i] == "1") for i, key in enumerate(("initial", "final", "tone"))):
                raise ValueError(f"{mode} equality flags mismatch")
            if (row.get("left_zero_initial") is not False or row.get("right_zero_initial") is not False
                    or row.get("no_zero_initial_diagnostic") is not True):
                raise ValueError(f"{mode} zero-initial flags mismatch")
            if mode.endswith("balanced"):
                first_id, second_id = row.get("first_id"), row.get("second_id")
                if {first_id, second_id} != {a, b} or first_id == second_id:
                    raise ValueError("balanced DIMSIM orientation does not match pair endpoints")
                if row.get("first_syllable_id") != occurrence_syllable.get(first_id) or row.get("second_syllable_id") != occurrence_syllable.get(second_id):
                    raise ValueError("balanced DIMSIM syllable orientation is invalid")
            value, status = row.get("dimsim_distance"), row.get("distance_status")
            if status == "ok":
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"{mode} ok distance must be finite and non-negative")
            elif value is not None or not isinstance(status, str) or not (status == "not_computed" or status.startswith("invalid:")):
                raise ValueError(f"{mode} distance status/value mismatch")
    triplets = _readl(root / "triplets/anchor-q.jsonl")
    sampling = config.get("sampling", {}).get("anchor_q", {})
    per_edge = sampling.get("draws_per_edge")
    expected_edges = [f"{c}:{l}->{h}" for c,l,h in EDGES]
    if sampling.get("edge_order") != expected_edges or len(triplets) != len(EDGES) * per_edge:
        raise ValueError("anchor-Q edge order or draw count is invalid")
    edge_counts = {edge: 0 for edge in expected_edges}
    draw_ids = set()
    for global_index, row in enumerate(triplets):
        edge = row.get("edge_id");
        expected_global_edge = expected_edges[global_index // per_edge] if per_edge else None
        if edge != expected_global_edge:
            raise ValueError("anchor-Q rows do not follow configured global edge order")
        if (edge not in edge_counts or type(row.get("schema_version")) is not int
                or row.get("schema_version") != TRIPLET_SCHEMA_VERSION
                or type(row.get("sampler_version")) is not int or row.get("sampler_version") != Q_SAMPLER_VERSION):
            raise ValueError("anchor-Q row schema or edge is invalid")
        ids = [row.get(k) for k in ("anchor_id", "high_id", "low_id")]
        if any(x not in occurrence_map for x in ids) or len(set(ids)) != 3:
            raise ValueError("anchor-Q occurrence references must be known and distinct")
        comp, low, high = edge.split(":")[0], edge.split(":")[1].split("->")[0], edge.split("->")[1]
        a, h, l = (occurrence_map[x] for x in ids)
        if occurrence_syllable[ids[0]] not in anchor_support[edge][0]:
            raise ValueError("anchor-Q draw uses an ineligible anchor syllable")
        def mask(x): return "".join(str(int(a[k] == x[k])) for k in ("initial", "final", "tone"))
        if mask(h) != high or mask(l) != low: raise ValueError("anchor-Q gold masks disagree with edge")
        expected_triplet = _stable("triplet:", [edge, *ids])
        if row.get("triplet_id") != expected_triplet or row.get("component") != comp or row.get("low_mask") != low or row.get("high_mask") != high:
            raise ValueError("anchor-Q triplet identity or edge metadata mismatch")
        expected_draw_id = f"anchor-q:v1:{edge}:{row.get('draw_index', -1):06d}"
        if row.get("draw_id") != expected_draw_id:
            raise ValueError("anchor-Q draw identity mismatch")
        if high == "111" and (row.get("anchor_syllable_id") != row.get("high_syllable_id") or
                               row.get("anchor_id") == row.get("high_id")):
            raise ValueError("anchor-Q high-111 identity rule violated")
        for name, oid in (("anchor", ids[0]), ("high", ids[1]), ("low", ids[2])):
            if row.get(name + "_syllable_id") != occurrence_syllable.get(oid):
                raise ValueError("anchor-Q syllable reference mismatch")
        if type(row.get("draw_index")) is not int or row.get("draw_index") != edge_counts[edge]:
            raise ValueError("anchor-Q draw index/order mismatch")
        if row.get("draw_id") in draw_ids: raise ValueError("duplicate anchor-Q draw ID")
        draw_ids.add(row["draw_id"]); edge_counts[edge] += 1
    if any(count != per_edge for count in edge_counts.values()): raise ValueError("anchor-Q edge coverage counts are invalid")
    coverage = _read(root / "coverage.json")
    expected_q_coverage = _anchor_q_coverage(
        syllables, triplets, seed=sampling["seed"], draws_per_edge=per_edge)
    counts = manifest.get("counts", {})
    expected_counts = {"sentences": len(sentences), "occurrences": len(occurrences), "syllables": len(syllables),
                       "dimsim_natural": len(pairs["dimsim-natural"]), "dimsim_balanced": len(pairs["dimsim-balanced"]), "anchor_q": len(triplets)}
    expected_edge_counts = {edge: per_edge for edge in expected_edges}
    if (not _typed_equal(counts, expected_counts) or coverage.get("sentence_count") != len(sentences)
            or coverage.get("occurrence_count") != len(occurrences)
            or coverage.get("anchor_q") != expected_q_coverage
            or coverage.get("anchor_q", {}).get("edges") is None
            or set(coverage.get("anchor_q", {}).get("edges", {})) != set(expected_edges)
            or any(coverage["anchor_q"]["edges"][edge].get("draw_count") != n for edge, n in expected_edge_counts.items())):
        raise ValueError("schema-3 coverage or manifest counts mismatch")
    expected_coverage = {
        "sentence_count": len(sentences), "occurrence_count": len(occurrences), "syllable_count": len(syllables),
        "zero_initial_occurrences_excluded_from_dimsim": sum(not r["initial"] for r in occurrences),
        "dimsim_natural": {"draw_count": len(pairs["dimsim-natural"]),
            "unique_pairs": len({r["pair_id"] for r in pairs["dimsim-natural"]}),
            "unique_occurrences": len({r[k] for r in pairs["dimsim-natural"] for k in ("left_id", "right_id")}),
            "unique_syllables": len({occurrence_syllable[x] for r in pairs["dimsim-natural"] for x in (r["left_id"],r["right_id"])})},
        "dimsim_balanced": {"draw_count": len(pairs["dimsim-balanced"]),
            "unique_pairs": len({r["pair_id"] for r in pairs["dimsim-balanced"]}),
            "unique_occurrences": len({r[k] for r in pairs["dimsim-balanced"] for k in ("left_id", "right_id")}),
            "unique_syllables": len({r[k] for r in pairs["dimsim-balanced"] for k in ("first_syllable_id", "second_syllable_id")}),
            "same_pronunciation_draws": sum(r["overlap_mask"] == "111" for r in pairs["dimsim-balanced"])},
        "anchor_q": expected_q_coverage
    }
    if not _typed_equal(coverage, expected_coverage): raise ValueError("schema-3 coverage diagnostics disagree with saved samples")
    return {"root": root, "manifest": manifest, "config": config, "coverage": coverage,
            "sentences": sentences, "occurrences": occurrences, "syllables": syllables,
            "pairs": pairs, "triplets": triplets}


def _validate_id(value):
    if not isinstance(value, str) or not value or "/" in value or "\\" in value or ".." in value:
        raise ValueError("invalid schema-3 evaluation_id")
    return value


def _validate_config(config, manifest_metric):
    if type(config.get("schema_version")) is not int or config["schema_version"] != SCHEMA_VERSION:
        raise ValueError("schema-3 config version must be the supported integer version")
    expected_population = {"unit": "contextual_occurrence", "zero_initial_excluded_from_dimsim": True,
                           "syllable_identity": "canonical_toned_pinyin", "aggregates": False}
    population = config.get("population")
    if (not isinstance(population, dict) or set(population) != set(expected_population)
            or type(population.get("zero_initial_excluded_from_dimsim")) is not bool
            or type(population.get("aggregates")) is not bool
            or not _typed_equal(population, expected_population)):
        raise ValueError("schema-3 population contract mismatch")
    sampling = config.get("sampling")
    if not isinstance(sampling, dict) or set(sampling) != {"dimsim_natural", "dimsim_balanced", "anchor_q"}:
        raise ValueError("schema-3 sampling contract is invalid")
    expected_keys = {
        "dimsim_natural": {"count", "without_replacement", "rng", "seed", "sampler_version", "pair_schema_version"},
        "dimsim_balanced": {"count", "with_replacement", "rng", "seed", "choice_order", "sampler_version", "pair_schema_version"},
        "anchor_q": {"draws_per_edge", "edge_order", "with_replacement", "rng", "seed", "sampler_version", "triplet_schema_version"},
    }
    specs = (("dimsim_natural", "random.Random", False, DIMSIM_SAMPLER_VERSION, PAIR_SCHEMA_VERSION),
             ("dimsim_balanced", "numpy.random.PCG64", True, DIMSIM_SAMPLER_VERSION, PAIR_SCHEMA_VERSION),
             ("anchor_q", "numpy.random.PCG64", True, Q_SAMPLER_VERSION, TRIPLET_SCHEMA_VERSION))
    for name, rng, replacement, sampler_version, schema_version in specs:
        item = sampling[name]
        if set(item) != expected_keys[name]: raise ValueError(f"schema-3 {name} config fields are invalid")
        count_key = "draws_per_edge" if name == "anchor_q" else "count"
        count = item.get(count_key); seed = item.get("seed")
        if (isinstance(count, bool) or not isinstance(count, int) or count <= 0
                or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                or item.get("rng") != rng
                or type(item.get("sampler_version")) is not int or item["sampler_version"] != sampler_version
                or type(item.get("pair_schema_version" if name != "anchor_q" else "triplet_schema_version")) is not int
                or item["pair_schema_version" if name != "anchor_q" else "triplet_schema_version"] != schema_version
                or (name != "dimsim_natural" and item.get("with_replacement") is not replacement)):
            raise ValueError(f"schema-3 {name} sampling contract is invalid")
    if sampling["dimsim_natural"].get("without_replacement") is not True:
        raise ValueError("natural DIMSIM must sample without replacement")
    if sampling["dimsim_balanced"].get("choice_order") != ["first_syllable", "second_syllable", "first_occurrence", "second_occurrence"]:
        raise ValueError("balanced DIMSIM choice order mismatch")
    score = config.get("scoring", {}).get("anchor_q", {})
    if score != {"cosine": "stable_float64_scale_before_norm", "tie_tolerance": 1e-12}:
        raise ValueError("anchor-Q scoring contract mismatch")
    if config.get("scoring", {}).get("dimsim") != manifest_metric:
        raise ValueError("DIMSIM scoring provenance mismatch")
