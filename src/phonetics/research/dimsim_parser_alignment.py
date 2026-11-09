"""Run the DIMSIM parser audit on a pinned evaluation or experiment bundle."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .artifacts import file_sha256, research_run
from ..bundles import content_hash
from ..evaluation.bundle import load_evaluation_bundle
from ..evaluation.dimsim import DimsimDistance
from ..evaluation.dimsim_pypinyin import PypinyinDimsimDistance
from ..evaluation.dimsim_alignment import (
    build_decompositions, eligible_labels, evaluate_alignment, summarize_statistics,
    write_alignment_research_artifacts,
)


@dataclass(frozen=True)
class DimsimParserAlignmentConfig:
    project_root: Path
    run_id: str
    evaluation_root: Path | None = None
    evaluation_manifest_sha256: str | None = None
    seed: int = 42
    bootstrap_replicates: int = 1000
    confidence: float = .95
    atol: float = 1e-12
    rtol: float = 1e-10
    experiment_root: Path | None = None
    experiment_manifest_sha256: str | None = None

    def resolved(self):
        return {key: str(value.resolve()) if isinstance(value, Path) else value
                for key, value in asdict(self).items()}


def _validate(config):
    if (config.evaluation_root is None) == (config.experiment_root is None):
        raise ValueError("exactly one evaluation or experiment root is required")
    source = "experiment" if config.experiment_root is not None else "evaluation"
    root = getattr(config, f"{source}_root")
    pin = getattr(config, f"{source}_manifest_sha256")
    other = "evaluation" if source == "experiment" else "experiment"
    if getattr(config, f"{other}_manifest_sha256") is not None:
        raise ValueError("manifest pin must match the selected input")
    if not isinstance(pin, str) or re.fullmatch(r"[0-9a-fA-F]{64}", pin) is None:
        raise ValueError(f"{source}_manifest_sha256 is required for the selected input")
    if type(config.seed) is not int or config.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(config.bootstrap_replicates) is not int or config.bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if type(config.confidence) not in (int, float) or not 0 < config.confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
           for value in (config.atol, config.rtol)):
        raise ValueError("tolerances must be finite and nonnegative")
    return Path(root), pin.lower(), source


def _experiment_test_inventory(root, pin):
    from ..experiment import load_common_test

    test_ids, samples = load_common_test(root, expected_manifest_sha256=pin)
    labels = sorted({label for sample in samples for label in sample["labels"]})
    inventory = {"groups": [{"group_kind": "syllable", "pinyin": label} for label in labels]}
    return inventory, {
        "split": "test", "sample_count": len(test_ids), "test_ids_sha256": content_hash(test_ids),
    }


def run_dimsim_parser_alignment(config: DimsimParserAlignmentConfig):
    input_root, pin, source = _validate(config)
    package = Path(__file__).parents[1]
    evaluation = package / "evaluation"
    sources = [Path(__file__), Path(__file__).with_name("artifacts.py"), package / "pinyin.py",
               *(evaluation / name for name in ("dimsim_alignment.py", "dimsim.py", "dimsim_pypinyin.py", "bundle.py"))]
    if source == "experiment":
        sources.extend([package / "experiment.py", package / "bundles.py"])
    table_names = ("decompositions.csv", "coverage.csv", "pairs.csv.gz", "symmetry-summary.csv",
                   "symmetry-exceptions.csv.gz", "statistics-by-syllable.csv",
                   "statistics-by-category.csv", "exclusion-diagnostics.csv")
    expected = {"config.json", *(f"tables/{name}" for name in table_names),
                "diagnostics/bootstrap-diagnostics.json", "diagnostics/validation.json"}
    with research_run(config.project_root, "dimsim-parser-alignment", config.run_id, [input_root],
                      config.resolved(), sources, expected) as (root, manifest):
        actual = file_sha256(input_root / "manifest.json")
        if actual != pin:
            raise ValueError(f"{source} manifest SHA256 mismatch")
        input_provenance = {f"{source}_manifest_sha256": actual}
        if source == "experiment":
            bundle, split_provenance = _experiment_test_inventory(input_root, pin)
            input_provenance.update(split_provenance)
        else:
            bundle = load_evaluation_bundle(input_root)
        decompositions = build_decompositions(bundle)
        alignment = evaluate_alignment(decompositions, atol=config.atol, rtol=config.rtol)
        statistics = summarize_statistics(
            decompositions, alignment, seed=config.seed, bootstrap_replicates=config.bootstrap_replicates,
            confidence=config.confidence, atol=config.atol, rtol=config.rtol,
        )
        validation = {
            **input_provenance, "inventory_count": len(decompositions),
            "eligible_b_nonzero_count": len(eligible_labels(decompositions)),
            "affected_pair_count": len(alignment["affected_pairs"]),
            "symmetry_summary": alignment["symmetry_summary"], "figures": [], "plot_data": [],
            "exclusion_diagnostics": {
                "threshold": 10,
                "direction_policy": ("mean_absolute_delta_over_both_directions"
                                     if statistics["use_both_directions"] else "canonical_direction"),
                "threshold_requires_complete_partner_coverage": True,
            },
        }
        write_alignment_research_artifacts(
            root, decompositions=decompositions, alignment=alignment, statistics=statistics, validation=validation,
        )
        manifest["scientific_provenance"] = {
            **input_provenance, "seed": config.seed, "bootstrap_replicates": config.bootstrap_replicates,
            "confidence": config.confidence, "atol": config.atol, "rtol": config.rtol,
            "scorer_a": DimsimDistance().provenance, "scorer_b": PypinyinDimsimDistance().provenance,
            "affected_union_fraction": statistics["affected_union_fraction"],
            "use_both_directions": statistics["use_both_directions"],
            "figures": [], "plot_data": [],
        }
    return root


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    roots = parser.add_mutually_exclusive_group(required=True)
    roots.add_argument("--evaluation-root", type=Path)
    roots.add_argument("--experiment-root", type=Path)
    parser.add_argument("--evaluation-manifest-sha256")
    parser.add_argument("--experiment-manifest-sha256")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    print(run_dimsim_parser_alignment(DimsimParserAlignmentConfig(**vars(parser.parse_args(argv)))))


if __name__ == "__main__":
    main()
