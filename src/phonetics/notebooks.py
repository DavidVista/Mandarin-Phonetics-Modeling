"""Generate optional notebook clients from configured jobs and current SDK files."""
from __future__ import annotations

import json
from pathlib import Path

from .artifacts import write_json
from .execution import sdk_files, validate_job


def _cell(kind, source, identifier):
    value = {'cell_type': kind, 'id': identifier, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        value.update(execution_count=None, outputs=[])
    return value


def render_job_notebook(job_path, *, install_dependencies=True, _sources=None):
    job_path = Path(job_path)
    from .execution import sdk_hashes
    sources = sdk_files() if _sources is None else _sources
    job = validate_job(job_path, expected_sources=sdk_hashes(sources))
    job_id = job['job_id']
    stage = job['stage']
    requirements = ['numpy==2.1.3', 'pypinyin==0.55.0']
    if stage == 'acoustic':
        requirements += ['soundfile==0.14.0', 'transformers==4.57.6', 'librosa==0.11.0']
    elif stage == 'visualization':
        requirements += ['dimsim==0.2.2', 'matplotlib==3.10.8', 'pandas==3.1.1', 'scipy==1.18.1']
    install = ('import subprocess, sys\n'
               f'REQUIREMENTS = {requirements!r}\n'
               'subprocess.check_call([sys.executable, "-m", "pip", "install", *REQUIREMENTS])\n')
    if not install_dependencies:
        install = f'REQUIREMENTS = {requirements!r}\nprint("Install these dependencies before running:", REQUIREMENTS)\n'
    install += '''from importlib.metadata import version
for requirement in REQUIREMENTS:
    name, expected = requirement.split("==")
    if version(name) != expected:
        raise RuntimeError(f"Required {requirement}")
'''
    bootstrap = '''import hashlib, json, sys, tempfile
from pathlib import Path
if "numpy" in sys.modules and sys.modules["numpy"].__version__ != "2.1.3":
    raise RuntimeError("Restart the kernel after installing NumPy 2.1.3")
if any(name == "phonetics" or name.startswith("phonetics.") for name in sys.modules):
    raise RuntimeError("Restart the kernel before loading the generated SDK")
SDK_ROOT = Path(tempfile.mkdtemp(prefix="phonetics-sdk-", dir=Path.cwd()))
'''
    bootstrap += f'SOURCES = {sources!r}\n'
    bootstrap += f'EXPECTED_SOURCE_SHA256 = {job["sources"]!r}\n'
    bootstrap += '''for name, body in SOURCES.items():
    path = SDK_ROOT / "phonetics" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="\\n")
    if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_SOURCE_SHA256[name]:
        raise RuntimeError(f"SDK checksum mismatch: {name}")
sys.path.insert(0, str(SDK_ROOT))
'''
    select = f'''JOB_ID = {job_id!r}
EXPECTED_INPUT_MANIFEST_SHA256 = {job['input_manifest_sha256']!r}
LOCAL_JOB = Path({str(job_path)!r})
candidates = [LOCAL_JOB] if LOCAL_JOB.is_dir() else []
if Path("/kaggle/input").is_dir():
    for candidate in Path("/kaggle/input").rglob("job.json"):
        try:
            if json.loads(candidate.read_text(encoding="utf-8")).get("job_id") == JOB_ID:
                candidates.append(candidate.parent)
        except (OSError, ValueError):
            continue
candidates = sorted(set(p.resolve() for p in candidates))
if len(candidates) != 1:
    raise RuntimeError(f"Attach exactly one input package for job {{JOB_ID}}; found {{len(candidates)}}")
JOB_PATH = candidates[0]
OUTPUT_PATH = Path.cwd() / "results" / JOB_ID
from phonetics.execution import validate_job
job = validate_job(JOB_PATH)
if job["input_manifest_sha256"] != EXPECTED_INPUT_MANIFEST_SHA256:
    raise RuntimeError("Input manifest pin mismatch")
print("Job:", JOB_ID, "Input SHA256:", EXPECTED_INPUT_MANIFEST_SHA256)
'''
    execute = ''
    if stage == 'training' and job.get('external_input'):
        files = {p.name: p.read_bytes().hex() for p in job_path.iterdir() if p.is_file()}
        reference = job['input_reference']
        select = f'''JOB_ID = {job_id!r}
EXPECTED_INPUT_MANIFEST_SHA256 = {job['input_manifest_sha256']!r}
JOB_PATH = SDK_ROOT / "job"
JOB_PATH.mkdir()
JOB_FILES = {files!r}
for name, body in JOB_FILES.items():
    (JOB_PATH / name).write_bytes(bytes.fromhex(body))
from phonetics.execution import validate_job
job = validate_job(JOB_PATH)
LOCAL_INPUT = Path({reference!r})
candidates = []
if (LOCAL_INPUT / "manifest.json").is_file() and hashlib.sha256((LOCAL_INPUT / "manifest.json").read_bytes()).hexdigest() == EXPECTED_INPUT_MANIFEST_SHA256:
    candidates.append(LOCAL_INPUT.resolve())
if Path("/kaggle/input").is_dir():
    for manifest in Path("/kaggle/input").rglob("manifest.json"):
        if hashlib.sha256(manifest.read_bytes()).hexdigest() == EXPECTED_INPUT_MANIFEST_SHA256:
            candidates.append(manifest.parent.resolve())
candidates = sorted(set(candidates))
if len(candidates) != 1:
    raise RuntimeError(f"Attach exactly one experiment bundle with SHA256 {{EXPECTED_INPUT_MANIFEST_SHA256}}; found {{len(candidates)}}")
INPUT_PATH = candidates[0]
if hashlib.sha256((INPUT_PATH / "manifest.json").read_bytes()).hexdigest() != EXPECTED_INPUT_MANIFEST_SHA256:
    raise RuntimeError("Experiment manifest mismatch")
OUTPUT_PATH = Path.cwd() / "results" / JOB_ID
print("Experiment:", INPUT_PATH, "SHA256:", EXPECTED_INPUT_MANIFEST_SHA256)
'''
    if stage in ('acoustic', 'training'):
        execute += '''from importlib.metadata import version
if version("torch").split("+", 1)[0] != "2.11.0":
    raise RuntimeError("Use Torch 2.11.0; install the CPU/CUDA build appropriate to this environment")
'''
    execute += '''from phonetics.execution import execute_job
result = execute_job(JOB_PATH, OUTPUT_PATH, allow_execution=True, input_path=globals().get("INPUT_PATH"))
import shutil
archive = shutil.make_archive(str(OUTPUT_PATH), "zip", root_dir=result)
print("Result:", result, "Download:", archive)
if job["stage"] == "training":
    print("Run tree:", Path("runs") / job["experiment_id"] / job["parameters"]["model_id"] / job["parameters"]["run_id"])
'''
    title = f'{stage.capitalize()} — {job["parameters"].get("model_id", job_id[:12])}'
    document = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python'}, 'job_id': job_id,
                             'source_sha256': job['sources']},
                'cells': [_cell('markdown', f'# {title}\n\n' + ('Attach the original experiment bundle as an input dataset. The job manifest is embedded; no copied input package is needed. ' if job.get('external_input') else 'Attach the accompanying job directory as an input dataset. ')
                               + 'Run cells in order. The final cell executes the configured stage. '
                               'ML stages require Torch 2.11.0.\n', 'overview'),
                          _cell('code', install, 'dependencies'), _cell('code', bootstrap, 'sdk'),
                          _cell('code', select, 'configuration'), _cell('code', execute, 'run')]}
    target = job_path.parent.parent / 'notebooks' / f'{stage}-{job_id[:16]}.ipynb'
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, document)
    return target


def generate_notebooks(config, *, stage='ready'):
    from .execution import create_job
    if config['schema_version'] != 2:
        raise ValueError('notebook generation requires configuration schema 2')
    if stage == "encoder-geometry":
        return generate_encoder_notebooks(config)
    paths = config['paths']
    result = []
    generation = {}
    if stage in ('ready', 'acoustic') and config['acoustic']['existing_bundle'] is None:
        if Path(paths['inventory']).exists():
            result.append(create_job(config, 'acoustic', paths['inventory']))
        elif stage == 'acoustic':
            raise ValueError('prepare the sound inventory before generating its inference notebook')
    experiment = Path(paths['experiments']) / config['experiment_id']
    if stage in ('ready', 'training'):
        if experiment.exists():
            for model_id in config['experiment']['model_ids']:
                result.append(create_job(config, 'training', experiment, model_id=model_id, _generation=generation))
        elif stage == 'training':
            raise ValueError('prepare the experiment bundle before generating training notebooks')
    if stage in ('ready', 'visualization'):
        for workflow in config['visualization']:
            result.append(create_job(config, 'visualization', None, workflow=workflow))
    return result


def generate_encoder_notebooks(config):
    """Generate two reusable, input-pinned notebook clients without executing them."""
    from .bundles import content_hash
    from .encoder_jobs import encoder_inputs
    from .evaluation.training_summary import TRAINING_MODELS
    from .execution import sdk_hashes

    refs, pins = encoder_inputs(config)
    sources = sdk_files()
    hashes = sdk_hashes(sources)
    spec = config["encoder_geometry"]
    root = Path(config["paths"]["generated"]) / "notebooks"
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for stage, title in (("extract-hidden-states", "Extract hidden states"), ("encoder-geometry", "Run encoder geometry")):
        requirements = ["numpy==2.1.3", "pypinyin==0.55.0"]
        if stage == "encoder-geometry":
            requirements += ["dimsim==0.2.2", "matplotlib==3.10.8", "pandas==3.1.1", "scipy==1.18.1"]
        installation = f"REQUIREMENTS = {requirements!r}\n"
        if config["notebooks"]["install_dependencies"]:
            installation += 'import subprocess, sys\nsubprocess.check_call([sys.executable, "-m", "pip", "install", *REQUIREMENTS])\n'
        installation += """from importlib.metadata import version
for requirement in REQUIREMENTS:
    name, expected = requirement.split("==")
    if version(name) != expected:
        raise RuntimeError(f"Required {requirement}")
"""
        sdk = f"""import hashlib, shutil, sys, tempfile
from pathlib import Path
if any(name == "phonetics" or name.startswith("phonetics.") for name in sys.modules):
    raise RuntimeError("Restart the kernel before loading the generated SDK")
SDK_ROOT = Path(tempfile.mkdtemp(prefix="phonetics-sdk-", dir=Path.cwd()))
SOURCES = {sources!r}
HASHES = {hashes!r}
for name, source in SOURCES.items():
    path = SDK_ROOT / "phonetics" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8", newline="\\n")
    if hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[name]:
        raise RuntimeError(f"SDK checksum mismatch: {{name}}")
sys.path.insert(0, str(SDK_ROOT))
"""
        settings = f"PINS = {pins!r}\nSOURCE_SHA256 = {content_hash(hashes)!r}\n"
        if stage == "extract-hidden-states":
            settings += (
                f"REFERENCES = {refs!r}\nMODELS = {TRAINING_MODELS!r}\nMODEL = MODELS[0]\n"
                f"DEVICE = {spec['device']!r}\nBATCH_SIZE = {spec['batch_size']!r}\n"
            )
            execution = """from phonetics.encoder_jobs import locate_manifest
from phonetics.evaluation.encoder_extraction import extract_hidden_states
evaluation, experiment, training = (locate_manifest(PINS[key], REFERENCES[key]) for key in ("evaluation", "experiment", MODEL))
result = extract_hidden_states(
    evaluation_root=evaluation, evaluation_pin=PINS["evaluation"],
    experiment_root=experiment, experiment_pin=PINS["experiment"],
    training_root=training, training_pin=PINS[MODEL], model=MODEL, device=DEVICE,
    batch_size=BATCH_SIZE, output=Path.cwd() / "results" / MODEL,
    source_sha256=SOURCE_SHA256, allow_execution=True)
"""
            instructions = "Attach the pinned evaluation and experiment bundles and the checkpoint bundle for the selected model. Torch 2.11.0 must be installed before extraction. The final cell starts extraction."
        else:
            report_spec = {key: spec[key] for key in ("id", "representations_root", "tolerance")}
            settings += f"EVALUATION_ROOT = {refs['evaluation']!r}\nSPEC = {report_spec!r}\n"
            settings += """# Set these paths when Arial is supplied as an input dataset.
FONT_FILES = []
# FONT_FILES = ["/kaggle/input/your-fonts/arial.ttf", "/kaggle/input/your-fonts/arialbd.ttf"]
"""
            execution = """from matplotlib import font_manager
from phonetics.visualization.core import require_arial

for filename in FONT_FILES:
    font_file = Path(filename).resolve(strict=True)
    if font_manager.FontProperties(fname=str(font_file)).get_name() != "Arial":
        raise ValueError(f"Expected an Arial font file: {font_file}")
    font_manager.fontManager.addfont(str(font_file))
require_arial()

from phonetics.encoder_jobs import locate_exports, locate_manifest
from phonetics.research.encoder_geometry import run_encoder_geometry
evaluation = locate_manifest(PINS["evaluation"], EVALUATION_ROOT)
result = run_encoder_geometry(evaluation_root=evaluation, pins=PINS, exports=locate_exports(SPEC, PINS),
    output=Path.cwd() / "results" / SPEC["id"], tolerance=SPEC["tolerance"], source_sha256=SOURCE_SHA256)
"""
            instructions = "Attach the pinned evaluation bundle and one verified hidden-state export dataset per model. If Arial is supplied as an input dataset, set FONT_FILES in Configuration."
        execution += 'print(shutil.make_archive(str(result), "zip", root_dir=result))\n'
        document = {
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "source_sha256": hashes, "input_manifest_sha256": pins},
            "cells": [_cell("markdown", f"# {title}\n\n{instructions}\n", "overview"),
                      _cell("code", installation, "dependencies"), _cell("code", sdk, "sdk"),
                      _cell("code", settings, "configuration"), _cell("code", execution, "run")],
        }
        destination = root / f"{stage}-{content_hash({'pins': pins, 'sources': hashes, 'config': spec})[:16]}.ipynb"
        write_json(destination, document)
        results.append({"stage": stage, "notebook": str(destination)})
    return results
