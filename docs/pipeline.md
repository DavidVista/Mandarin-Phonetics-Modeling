# Configuration and local workflows

`configs/experiment.example.json` is the complete editable starting configuration. Copy it before the first run. Its paths are relative to the config directory, not the shell working directory. JSON does not support comments; the sections are described below.

| Section | Meaning |
| --- | --- |
| `paths` | Source, inventory, corpus, acoustic targets, prepared data, experiment parent, caches, training runs, reports, generated jobs and state file |
| `corpus` | Exact source revision, split, text/Pinyin columns and tone convention |
| `model` | Acoustic model/processor identifiers, exact revision, device, precision, extraction layer, audio normalization and pooling |
| `acoustic` | Existing reference directory or null; optional expected manifest SHA256 (null computes and records it automatically) |
| `experiment_id` | Immutable experiment directory name |
| `experiment` | Sentence count, number of seeds, fixed split protocol, encoder, heads, training settings and selected objectives |
| `execution` | Local/notebook mode by eligible stage, explicit training device and run ID |
| `notebooks` | Whether the first code cell installs supplementary dependencies |
| `analysis` | Existing local research workflows; empty by default |
| `visualization` | Existing local or notebook visualization workflows; empty by default |

The current encoder is `experiment.architecture.encoder.type: char_bilstm`. Its embedding dimension, hidden units per direction and number of layers are validated positive integers. Context dimension is derived as twice the hidden dimension. Head settings specify classification hidden dimension and dropout. Acoustic output dimension must match the actual targets. Unknown encoder types, including `bert`, fail before execution. A future BERT adapter must implement character alignment and its own validated config; selecting a name alone cannot supply that implementation.

Supported training changes are `epochs`, `batch_size` and `learning_rate`; all must be positive and finite with integer epochs/batch size. Other optimizer, scheduler, selection and loss settings retain the current protocol and are validated rather than silently ignored. Old flat architecture configs inside existing bundles remain readable without rewriting those bundles.

## Preparation and execution modes

The example imports existing acoustic embeddings. Set `acoustic.existing_bundle` and `acoustic.manifest_sha256` to null to compute new ones. Select `execution.acoustic: notebook` to prepare a portable inference job after the decoded inventory, or `local` and explicitly supply `--allow-inference`. The acoustic device belongs to `model.device`; the trained encoder device belongs to `execution.training_device`.

```sh
phonetics run --config configs/experiment.json --through experiment
phonetics run --config configs/experiment.json --allow-inference --allow-training
```

The last command executes ML stages only when their modes select local computation and their dependencies are installed. It does not install them. `--through` accepts `experiment`, `training`, `analysis`, `visualization`.

Notebook generation never executes an ML stage. Training generation validates the original experiment once per command for all objectives, then writes small immutable reference jobs and notebooks without copying the experiment or writing a second SDK snapshot into each job. Upload the original experiment bundle once as a dataset and attach it to each training notebook. The job manifest and SDK are embedded; the input locator selects exactly one bundle by its pinned manifest SHA256 and training validates its contents. Acoustic and visualization stages retain their input packages. Training generates one job per configured objective, preserving its seed list. The pipeline hands off one job at a time; manual generation can prepare all training notebooks together.

The dependency cell runs before project imports and installs supplementary packages only. Torch 2.11.0 must already be installed with the appropriate CPU/CUDA build for ML jobs. The final cell executes the stage and exports a result ZIP. Download and extract that ZIP:

```sh
phonetics import-result --config configs/experiment.json --result downloaded/result
phonetics run --config configs/experiment.json
```

Import verifies outer and nested inventories, the pending job, original input manifest, source hashes, acoustic extraction parameters or training objective/config/seed completion, and checkpoint/log file hashes. It does not load checkpoint tensors. A wrong or incomplete result leaves the pending state intact. Existing different destination data is never overwritten. Imported acoustic targets continue directly into prepared/experiment assembly.

`paths.state` is generated progress, separate from the editable config. Config/source changes reject stale state; select a new state path and run ID, and new output IDs for stages whose scientific inputs changed. Repeated runs revalidate upstream inputs and completed outputs. State is written atomically. There is no concurrent-writer scheduler; run only one CLI writer for a config/state file.

## Reports

Add a job to `analysis`. Paths select explicit existing artifact directories; their manifest hashes are computed and recorded automatically. The workflows keep their existing input contracts and numerical methods. They do not create new character-level evaluation data.

```json
{
  "id": "distribution-1",
  "workflow": "dimsim-distribution",
  "parameters": {"evaluation_root": "../datasets/evaluations/selected-evaluation"}
}
```

Other available analysis jobs:

```json
{
  "id": "alignment-1",
  "workflow": "dimsim-parser-alignment",
  "parameters": {
    "evaluation_root": "../datasets/evaluations/selected-evaluation",
    "seed": 42, "bootstrap_replicates": 1000, "confidence": 0.95,
    "atol": 1e-12, "rtol": 1e-10
  }
}
```

```json
{
  "id": "graduality-1",
  "workflow": "wav2vec2-graduality",
  "parameters": {
    "base_root": "../datasets/pinyin-embeddings/selected-base",
    "zh_cn_root": "../datasets/pinyin-embeddings/selected-zh-cn",
    "evaluation_root": "../datasets/evaluations/selected-evaluation",
    "bootstrap_repeats": 1000, "bootstrap_seed": 20261024
  }
}
```

Run one report independently of training:

```sh
phonetics workflow --config configs/experiment.json --id distribution-1
```

## Visualization

Add jobs to `visualization`. Both workflows are available locally through `workflow` and in optional generated notebooks. They render existing numerical results and preserve the approved style; the required Arial font must be installed. Scientific evaluation for character experiments is separate. In particular, the current training plot loader expects historical shared-population logs and explicitly rejects independent character experiment logs.

```json
{
  "id": "training-report-1",
  "workflow": "training-visualization",
  "parameters": {
    "training_runs": {
      "syllable": "../runs/selected/syllable/run-1",
      "components": "../runs/selected/components/run-1",
      "acoustic": "../runs/selected/acoustic/run-1",
      "components-acoustic-lambda-1": "../runs/selected/components-acoustic-lambda-1/run-1",
      "components-acoustic-lambda-10": "../runs/selected/components-acoustic-lambda-10/run-1"
    },
    "summary_run_id": "training-summary-1", "mode": "prepare"
  }
}
```

```json
{
  "id": "evaluation-report-1",
  "workflow": "evaluation-visualization",
  "parameters": {
    "anchor_q_output": "../outputs/contextual-anchor-q/selected",
    "dimsim_output": "../outputs/contextual-dimsim/selected",
    "random_control_output": "../outputs/contextual-random-control/selected",
    "bundle_path": "../datasets/evaluations/selected-evaluation",
    "derived_run_ids": {"interpretation": "interpretation-1", "presentation_data": "presentation-1"},
    "derived_mode": "prepare"
  }
}
```

Config-driven workflows publish a verified wrapper under `paths.reports/<id>/`, preserving the original scientific outputs under its `outputs/` tree. Exact matching wrappers can be reused after validation. Separate SDK APIs retain explicit reuse of prepared summaries; the config adapter currently creates complete report outputs with `prepare` mode. Notebook exports carry the same wrapper and original scientific manifests.

## Generated artifacts and validation boundaries

Only source code, config, lock and tests belong in Git. `generated/notebooks/`, `generated/jobs/` and `generated/state/` are ignored. The workspace has one generation script, `meta/tools/generate_notebooks.py`, delegating to the same `phonetics generate-notebooks` command.

Local contract tests do not prove successful ML training, GPU equivalence or execution on Kaggle/macOS/Linux. Those runs require their selected environments and inputs. Source bundles, existing embeddings, current datasets and historical archives are not relocated by this change.
