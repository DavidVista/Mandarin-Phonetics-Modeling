"""Command-line entry point for reproducible data preparation and training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile


def main(argv=None):
    parser = argparse.ArgumentParser(prog="phonetics")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download-sounds", help="Download and verify the official Mandarin archive")
    download.add_argument("--output", required=True, help="Source bundle directory; existing content must match the source pin")
    download.add_argument("--timeout", type=float, default=60, help="Network read timeout in seconds")
    for stage in ("export-corpus", "prepare-inventory", "prepare-corpus", "prepare-embeddings",
                  "prepare-data", "prepare-experiment", "prepare"):
        command = commands.add_parser(stage, help=f"Run the {stage} stage using a pinned configuration")
        command.add_argument("--config", required=True, help="JSON configuration; paths resolve relative to this file")
        command.add_argument("--allow-inference", action="store_true", help="Explicitly enable acoustic model inference")
    train = commands.add_parser("train", help="Train one objective over the configured seeds")
    train.add_argument("--experiment", required=True)
    train.add_argument("--manifest-sha256", required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--device", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--run-id")
    train.add_argument("--allow-training", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "download-sounds":
            from .source_preparation.download import download_sounds, CHINESE_LESSONS_URL, CHINESE_LESSONS_SHA256
            path = download_sounds(args.output, timeout=args.timeout)
            print(json.dumps({"path": str(path), "url": CHINESE_LESSONS_URL,
                              "sha256": CHINESE_LESSONS_SHA256}, sort_keys=True))
        elif args.command == "train":
            if not args.allow_training:
                raise PermissionError("training requires --allow-training after reviewing inputs and device")
            from .training import train_experiment
            result = train_experiment(Path(args.experiment), model_id=args.model, device=args.device,
                                      runs_root=Path(args.output), run_id=args.run_id,
                                      expected_manifest_sha256=args.manifest_sha256)
            print(json.dumps({"path": str(result["run_dir"]), "status": result["manifest"]["status"]}))
        else:
            from .pipeline import load_config, run_stage
            path = run_stage(load_config(args.config), args.command, allow_inference=args.allow_inference)
            print(json.dumps({"path": str(path), "stage": args.command}, sort_keys=True))
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0
