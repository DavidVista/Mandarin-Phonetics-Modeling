"""Prepare a new evaluation config using sealed inputs from a completed experiment.

Run with python -m phonetics.reassessment --help. This command never runs inference.
"""
import argparse
from pathlib import Path

from .artifacts import sha256_file, write_json
from .encoder_jobs import encoder_inputs, resolve_encoder_config
from .execution import safe_id, training_run_paths
from .pipeline import load_config


def prepare(config_path, output, report_id, *, models=None, pretrained=False):
    config = load_config(config_path)
    safe_id(report_id)
    if config["encoder_geometry"]["id"] == report_id:
        raise ValueError("reassessment requires a new report ID")
    spec = config["encoder_geometry"]
    available = {**training_run_paths(config), **spec["training_runs"]}
    selected = list(models) if models is not None else list(spec["training_runs"])
    if len(selected) != len(set(selected)):
        raise ValueError("evaluation models must be unique")
    # A separately trained Syllable extension shares the same sealed experiment.
    if "syllable" in selected and "syllable" not in available:
        available["syllable"] = str(Path(config["paths"]["runs"]) / config["experiment_id"] /
                                   "syllable" / config["execution"]["run_id"])
    if set(selected) - set(available):
        raise ValueError("unknown evaluation model")
    evaluation = config["evaluation"]
    evaluation["experiment_manifest_sha256"] = sha256_file(Path(evaluation["experiment_root"]) / "manifest.json")
    config["encoder_geometry"] = {**spec, "id": report_id, "diagnostics": True,
        "representations_root": str(Path(spec["representations_root"]).parent / (report_id + "-hidden-states")),
        "training_runs": {m: available[m] for m in selected},
        "contrasts": [p for p in spec.get("contrasts", []) if set(p) <= set(selected)]}
    if pretrained:
        config["encoder_geometry"]["representation_sources"] = {"pretrained-bert": {"type": "pretrained_bert"}}
        config["encoder_geometry"]["contrasts"] += [["pretrained-bert", m] for m in selected]
    config["paths"]["generated"] = str(Path(config["paths"]["generated"]) / report_id)
    config["encoder_geometry"] = resolve_encoder_config(config["encoder_geometry"], str)
    refs, pins = encoder_inputs(config)
    config["encoder_geometry"]["input_manifest_sha256"] = pins
    target = Path(output)
    if target.exists() or target.with_suffix(".inputs.json").exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, config)
    write_json(target.with_suffix(".inputs.json"), {"references": refs, "pins": pins})
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--pretrained-bert", action="store_true")
    args = parser.parse_args()
    print(prepare(args.config, args.output, args.id, models=args.models, pretrained=args.pretrained_bert))


if __name__ == "__main__":
    main()
