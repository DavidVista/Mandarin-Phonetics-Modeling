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
    run = commands.add_parser('run', help='Run or resume the configured pipeline; stop at notebook handoffs')
    run.add_argument('--config', required=True)
    run.add_argument('--through', choices=('experiment', 'training', 'analysis', 'visualization'), default='visualization')
    run.add_argument('--allow-inference', action='store_true')
    run.add_argument('--allow-training', action='store_true')
    generate = commands.add_parser('generate-notebooks', help='Generate notebooks and portable input jobs from one config')
    generate.add_argument('--config', required=True)
    generate.add_argument('--stage', choices=('ready', 'acoustic', 'training', 'visualization'), default='ready')
    receive = commands.add_parser('import-result', help='Verify and import the pending notebook result')
    receive.add_argument('--config', required=True)
    receive.add_argument('--result', required=True, help='Extracted result directory containing receipt.json and manifest.json')
    workflow = commands.add_parser('workflow', help='Run one configured report or visualization locally')
    workflow.add_argument('--config', required=True)
    workflow.add_argument('--id', required=True)
    execute = commands.add_parser('execute-job', help='Execute a portable job using its exact SDK')
    execute.add_argument('--job', required=True)
    execute.add_argument('--output', required=True)
    execute.add_argument('--allow-execution', action='store_true')
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
        if args.command in ('run', 'generate-notebooks', 'import-result', 'workflow'):
            from .pipeline import load_config
            config = load_config(args.config)
            if args.command == 'run':
                from .execution import run_pipeline
                result = run_pipeline(config, through=args.through, allow_inference=args.allow_inference,
                                      allow_training=args.allow_training)
            elif args.command == 'generate-notebooks':
                from .notebooks import generate_notebooks
                result = {'jobs': generate_notebooks(config, stage=args.stage)}
            elif args.command == 'import-result':
                from .execution import import_result
                result = {'path': str(import_result(config, args.result)), 'status': 'imported'}
            else:
                from .workflows import run_workflow
                jobs = [w for group in ('analysis', 'visualization') for w in config[group] if w['id'] == args.id]
                if len(jobs) != 1:
                    raise ValueError('select exactly one configured workflow ID')
                result = {'path': str(run_workflow(jobs[0], config['paths']['reports']))}
            print(json.dumps(result, sort_keys=True))
        elif args.command == 'execute-job':
            from .execution import execute_job
            print(json.dumps({'path': str(execute_job(args.job, args.output, allow_execution=args.allow_execution))}))
        elif args.command == "download-sounds":
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
