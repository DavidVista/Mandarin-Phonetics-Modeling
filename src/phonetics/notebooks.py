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
