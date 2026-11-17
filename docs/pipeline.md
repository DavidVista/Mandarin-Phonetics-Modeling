# Configuration and local workflows

Research profiles live in `configs/local/` and `configs/notebook/`. They keep separate experiment bundles, acoustic targets, prepared data, run state, notebooks, training runs, and reports. Source data and processed corpus paths are shared. Resolve paths relative to the selected JSON file. The examples below use the BiLSTM profile; adjust the config path for BERT.

`configs/experiment.mock.json` uses `execution.preparation: mock` to generate a synthetic corpus and target vectors locally. Run it with `phonetics run --config configs/experiment.mock.json --allow-training`. It runs four objectives on CPU for one epoch with one seed and writes a Markdown test-metric report. Its split contains 900 training, 100 validation, and 100 test sentences. This checks training and report plumbing with generated inputs.

| Section | Meaning |
| --- | --- |
| `paths` | Source, inventory, corpus, acoustic targets, prepared data, experiment parent, caches, training runs, reports, generated jobs and state file |
| `corpus` | Exact source revision, split, text/Pinyin columns and tone convention |
| `model` | Acoustic model/processor identifiers, exact revision, device, precision, extraction layer, audio normalization and pooling |
| `acoustic` | Existing reference directory or null; optional expected manifest SHA256 (null computes and records it automatically) |
| `experiment_id` | Immutable experiment directory name |
| `experiment` | Per-seed sentence count, optional common test size, number of seeds, encoder, heads, training settings and selected objectives |
| `evaluation` | Optional pinned character experiment, output root and ID, excluded tone-labelled Pinyin forms and their source audit |
| `execution` | Local/notebook mode by eligible stage, explicit training device and run ID |
| `notebooks` | Whether the first code cell installs supplementary dependencies |
| `analysis` | Local research workflows, empty by default |
| `visualization` | Optional section for archived report configurations |

The default encoder is `experiment.architecture.encoder.type: char_bilstm`. Its embedding dimension, hidden units per direction and number of layers are validated positive integers. Context dimension is derived as twice the hidden dimension. Head settings specify classification hidden dimension and dropout. Acoustic output dimension must match the actual targets.

Supported training changes are `epochs`, `batch_size` and `learning_rate`; all must be positive and finite with integer epochs/batch size. Other optimizer, scheduler, selection and loss settings retain the current protocol and are validated rather than silently ignored. Old flat architecture configs inside existing bundles remain readable without rewriting those bundles.

`experiment.sample_size` sets the total train and validation sentences per seed. Validation uses `sample_size // 10`. The optional `experiment.test_size` sets the exact number of sentences in the shared test and must be a positive integer. If omitted, the test also uses `sample_size // 10`. The eligible corpus must have at least `sample_size + test_size` sentences, using the effective test size.

For a new experiment with 90,000 train and 10,000 validation sentences per seed and a shared test of 100,000 sentences, set `experiment.sample_size` and `experiment.test_size` to `100000` and select a new `experiment_id`.

`experiment.training.epochs` sets the default epoch limit. To override it for individual objectives, add `epochs_by_model` inside `experiment.training`. Keys must appear in `experiment.model_ids` and limits must be positive integers.

```json
"epochs": 20,
"epochs_by_model": {"components": 10, "acoustic": 40}
```

In this example, the other objectives train for 20 epochs. Each run records its effective limit as `epoch_limit` in the training manifest. The selected checkpoint for each seed still minimizes its validation objective. Changing these settings requires a new experiment ID.

## Preparation and execution modes

### Chinese BERT

For a new experiment, replace `experiment.architecture.encoder` with the following object.

```json
{
  "type": "bert",
  "model_id": "google-bert/bert-base-chinese",
  "revision": "8f23c25b06e129b6c986331a13d8d025a92cf0ea",
  "files": {
    "config.json": "88af762cfb15fa9ce183c84dcbfdc9d635196ba635dfe81f265fdd21d67f4c4c",
    "model.safetensors": "3404a1ffd8da507042e8161013ba2a4fc49858b4e3f8fbf5ce5724f94883aec3",
    "tokenizer.json": "55c5edcfb8936895e019b2d8f97dcb547734f25b29f95009155f4c8c15649aae",
    "tokenizer_config.json": "0f6d13e6f4da6f9e24f22ada6bc3be571123d858d7c0c05a8a7cd55a9c23c2e8",
    "vocab.txt": "45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c"
  }
}
```

BERT downloads the required files from the pinned revision and verifies their SHA256 values. Use the `bert` dependency extra for local training. For offline use, set `PHONETICS_BERT_SNAPSHOT` to a directory of verified files, or attach that directory and set `BERT_SNAPSHOT` in the generated notebook. Evaluation and checkpoint restoration need the model config and tokenizer files. Fine-tuning also needs the original weights.

The BERT adapter validates a one-to-one token mapping for every selected character before training. It rejects unalignable sentences and sequences beyond the model limit. Special tokens participate in attention and are excluded from the position-aligned outputs for the heads and evaluation. Unknown target positions are determined by the pinned BERT tokenizer.

BERT supports `Adam` or `AdamW`, nonnegative `weight_decay`, and an optional scheduler such as `{"type": "linear", "warmup_ratio": 0.05}`. Select the learning rate, epoch budget and objectives in the experiment config. Give the BERT experiment a distinct ID and state file. Compare architectures using matching split hashes and explicit common evaluation coverage.

The research profiles generate acoustic embeddings from the decoded inventory and write them to `paths.acoustic`. Both `acoustic.existing_bundle` and `acoustic.manifest_sha256` are null. Select `execution.acoustic: notebook` to prepare a portable inference job, or `local` and supply `--allow-inference` for local inference. The acoustic device belongs to `model.device`; the trained encoder device belongs to `execution.training_device`.

```sh
phonetics run --config configs/local/experiment.bilstm.json --through experiment --allow-inference
phonetics run --config configs/local/experiment.bilstm.json --allow-inference --allow-training
```

The last command executes ML stages only when their modes select local computation and their dependencies are installed. It does not install them. `--through` accepts `experiment`, `training`, `analysis`, `visualization`.

The research profiles use one config from preparation through evaluation and reporting. Set `evaluation.experiment_root` to the experiment's configured output directory and set `experiment_manifest_sha256` to `null` to use the experiment completed by this run. The CLI records the SHA256 of its manifest in run state and checks it before preparing the evaluation bundle. Set an explicit SHA256 to use an existing experiment through the standalone evaluation commands.

When `encoder_geometry.training_runs` is omitted for a run-owned experiment, the CLI derives each path from `paths.runs`, the experiment ID, model ID, and run ID. A `training-report` configured with empty parameters uses the same run paths. The geometry workflow checks the evaluation and training manifest hashes recorded in state before it reads their contents. Repeating the local command reuses validated outputs. Set `encoder_geometry.allow_extraction` to `true` for the orchestrated local run to extract missing hidden states.

The notebook profiles build evaluation locally after the acoustic and training results have been imported. At the geometry step they return `notebook-ready` and the path to the generated encoder notebook. This notebook extracts hidden states and renders geometry together, using SHA256 values from the recorded run state. Open it with the pinned input bundles attached, and adjust `REFERENCES` if needed. Its final output is the report ZIP. Newly extracted hidden states are temporary files.

After an SDK update, `generate-notebooks --stage encoder-geometry --refresh-sdk` uses the existing run's verified input hashes with the current SDK. For training, `generate-notebooks --stage training --refresh-sdk` replaces the pending training job with the new notebook for that model. Use it before executing the replacement, since subsequent imports expect the new job ID. Both commands require the config that matches the recorded run.

`--through training` includes the configured training report. Objective and available validation curves use the actual model and seed lists, the epoch grid of each model, and sample standard deviation for multiple seeds. The objective curves mark selected validation epochs. DIMSIM distributions use the evaluation bundle. Parser alignment uses the complete shared test inventory from the experiment bundle. Reference geometry reuses the training acoustic targets and generates Wav2Vec2 base targets from the same inventory. The notebook profiles request a separate acoustic job for these base targets when needed. DIMSIM agreement with human ratings uses the pinned bundled Li et al. source.

Notebook generation never executes an ML stage. Training generation validates the original experiment once per command for all objectives, then writes small immutable reference jobs and notebooks without copying the experiment or writing a second SDK snapshot into each job. Upload the original experiment bundle once as a dataset and attach it to each training notebook. The job manifest and SDK are embedded; the input locator selects exactly one bundle by its pinned manifest SHA256 and training validates its contents. Acoustic and visualization stages retain their input packages. Training generates one job per configured objective, preserving its seed list. The pipeline hands off one job at a time; manual generation can prepare all training notebooks together.

The dependency cell runs before project imports and installs supplementary packages only. Torch 2.11.0 must already be installed with the appropriate CPU/CUDA build for ML jobs. The final cell executes the stage and exports a result ZIP. Training archives preserve `runs/<experiment_id>/<model_id>/<run_id>/`, and the cell prints that path. For manual transfer, merge the contents of the archive's `runs/` directory into `paths.runs`. Keep the archive root with its receipt and manifest when importing a pending job. The importer accepts both the run tree and the legacy `payload/` layout. After an SDK upgrade, import a pending training result with the original config, state and job directory. The importer verifies the job against its recorded SDK and updates the state to the current SDK after successful import.

```sh
phonetics import-result --config configs/notebook/experiment.bilstm.json --result downloaded/result
phonetics run --config configs/notebook/experiment.bilstm.json
```

Import verifies outer and nested inventories, the pending job, original input manifest, source hashes, acoustic extraction parameters or training objective/config/seed completion, and checkpoint/log file hashes. It does not load checkpoint tensors. A wrong or incomplete result leaves the pending state intact. Existing different destination data is never overwritten. Imported acoustic targets continue directly into prepared/experiment assembly.

`paths.state` is generated progress, separate from the editable config. Config/source changes reject stale state; select a new state path and run ID, and new output IDs for stages whose scientific inputs changed. Repeated runs revalidate upstream inputs and completed outputs. State is written atomically. There is no concurrent-writer scheduler; run only one CLI writer for a config/state file.

## Syllable geometry evaluation bundle

After preparing the experiment, add an `evaluation` section with `id`,
`experiment_root`, `experiment_manifest_sha256`, `output_root`,
`excluded_pinyin` and `exclusion_source` (the audit run ID and manifest SHA256).
Configure report jobs once their pinned inputs are available.

```sh
phonetics prepare-evaluation --config configs/experiment.json
```

This standalone stage uses the pinned experiment's common test split. It keeps
whole test sentences and records every position with a strict nonempty
initial. Forms in `excluded_pinyin` remain in the bundle and are marked for
exclusion from the primary report. The bundle contains observed and eligible
inventories, coverage counts including zero-initial positions, and all unique
unordered nonzero-initial pairs with their DIMSIM A distances. The manifest binds the
experiment, test IDs, exclusions, source audit, optional confirming audit
(`exclusion_confirmation`), and artifact hashes. Identical
inputs can reuse a verified bundle. Use a new evaluation ID when changing the
input or exclusions.

## Reports

Add a job to `analysis`. Paths select explicit existing artifact directories; their manifest hashes are computed and recorded automatically. The workflows keep their existing input contracts and numerical methods. They do not create new character-level evaluation data.

```json
{
  "id": "distribution-1",
  "workflow": "dimsim-distribution",
  "parameters": {"evaluation_root": "../datasets/evaluations/selected-evaluation"}
}
```

The DIMSIM distribution job requires an evaluation bundle with a complete
nonzero-initial syllable-pair table. It publishes the retained figures and
tables, plus `tables/descriptive_summary.json` with quantiles, overlap masks,
peak and tail diagnostics, segment-pair reweighting and clustering.

The standalone human agreement command uses the local transcription of
Li et al. (2022), Appendix S1, Tables S2 and S3. The CSVs were recovered from
the existing pilot transcription, without an independent comparison to the
publisher appendix during packaging. The command verifies the CSV hashes in
`source.json`, transforms mean similarity into distance as `8 - similarity`,
and evaluates original DIMSIM (A). Install the project's `analysis` extra
for DIMSIM and SciPy.

```sh
phonetics dimsim-human-agreement --input datasets/dimsim-human-agreement/li-2022-v1 --output outputs/dimsim-human-agreement/li-2022-v1
```

The output contains `pairs.csv`, `summary.csv` and a manifest. The summary gives
Spearman correlations for all character pairs, character pairs with two
nonzero initials, and unique pairs of distinct tonal readings in that
nonzero-initial subset. The last slice averages human distances over character
pairs sharing the same reading pair.

Other available analysis jobs:

```json
{
  "id": "alignment-1",
  "workflow": "dimsim-parser-alignment",
  "parameters": {
    "experiment_root": "../datasets/experiments/local/bilstm",
    "seed": 42, "bootstrap_replicates": 1000, "confidence": 0.95,
    "atol": 1e-12, "rtol": 1e-10
  }
}
```

```json
{
  "id": "reference-geometry-1",
  "workflow": "reference-geometry",
  "parameters": {
    "base_root": "../datasets/pinyin-embeddings/selected-base-none",
    "base_manifest_sha256": "<SHA256 of the selected base manifest.json>",
    "zh_cn_root": "../datasets/pinyin-embeddings/selected-zh-cn",
    "evaluation_root": "../datasets/evaluations/selected-geometry-evaluation",
    "tolerance": 1e-12
  }
}
```

Before running the report on a fresh checkout, obtain the pinned `acoustic_embeddings` bundle with `padding: none` and `pooling: mean_all` and place it at `base_root`. To use a replacement, prepare it separately in the same bundle format using the chosen model revision, audio inventory and extraction settings. Select a new bundle path and report ID, then set `base_manifest_sha256` to the replacement manifest's actual SHA256. The report consumes prepared embeddings and checks the base bundle before calculation.

The reference geometry workflow uses the common inventory, records missing
labels, verifies matching source audio for both checkpoints and writes one
self-contained HTML page, numeric matrices, map aggregates, CSV tables, and
three distance maps in PNG/EPS with figure metadata under the report ID.
The static maps share component order, and both cosine maps share color limits.
The page synchronizes the heatmaps and
their zoom, and uses saved aggregates for filter-dependent tables and anchor
histograms. The neighborhood view includes top-10 overlap, Spearman and
near-versus-far comparisons. The wav2vec2 comparison is exploratory.
The historical graduality protocol remains available through its research
module and archived results.

Run one report independently of training:

```sh
phonetics workflow --config configs/experiment.json --id distribution-1
```

The parser audit can also use the unique observed readings from the common
test split of a character experiment. Supply its exact manifest SHA256:

```sh
python -m phonetics.research.dimsim_parser_alignment --project-root . --run-id parser-test-1 --experiment-root datasets/experiments/char-zh-cn-v1 --experiment-manifest-sha256 <sha256>
```

The audit scores both parsers on the selected inventory and publishes
`tables/exclusion-diagnostics.csv` with the mean absolute distance change and
partner coverage for each eligible syllable. The historical threshold flag
requires full partner coverage. Use a new run ID for each audit.

## Encoder geometry

The optional `encoder_geometry` section selects one training run per model in
the pinned experiment, the
hidden-state output root, an extraction device and batch size, and the report
ID and tie tolerance. Its input is the sealed evaluation bundle selected by
`evaluation`. Set `encoder_geometry.device` explicitly and review the input
dataset versions before extraction.

The pinned experiment defines model order and seed IDs. Set optional
`encoder_geometry.contrasts` to a list of ordered pairs of distinct model IDs,
for example `[["components", "components-acoustic-lambda-3"]]`. The report
subtracts the first model from the second within each seed. When omitted, it
uses the original contrasts whose models are present.

```sh
phonetics generate-notebooks --config configs/experiment.json --stage encoder-geometry
```

This creates one notebook with pinned evaluation, experiment, and training
manifests and an embedded SDK and plot style. Attach these bundles and any
compatible hidden-state exports. The notebook verifies the selected checkpoint
SHA256 and epoch per model and seed. Set `encoder_geometry.allow_extraction`
to `true` to generate the notebook with `ALLOW_EXTRACTION = True`. The
default is `false`, which limits the notebook to existing exports. The CLI
uses its own `--allow-execution` flag. Extraction requires Torch 2.11.0.
The notebook writes a report ZIP. It removes the hidden states it extracted when the temporary directory closes.

To import a saved export for local use:

```sh
phonetics import-hidden-states --config configs/experiment.json --model syllable --result PATH
```

The local workflow uses the same implementation and supports verified exports
without Torch. Add `--allow-execution` after approval to extract missing
exports. The standalone `extract-hidden-states` command can still produce one
model's export independently.

```sh
phonetics workflow --config configs/experiment.json --id ENCODER_GEOMETRY_ID
```

If Arial is supplied as a Kaggle input dataset, set `FONT_FILES` in the
notebook configuration. The final cell registers the files with Matplotlib
before invoking the workflow. The local workflow requires Arial to be installed.

The primary report excludes the marked forms and positions whose target is UNK.
It uses the shared syllable inventory across seeds and identical positions
across models within each seed. All retained occurrences contribute to mean
cosine distances, with equal weight per character within a reading and equal
weight per occurrence within each character. Full Match uses distinct
occurrences. A diagonal is unavailable in the shared comparison when any seed
has fewer than two eligible occurrences for that reading.

Spearman uses distinct syllable pairs. DIMSIM Agreement excludes reference
ties and assigns half credit to a model tie, giving
`(wins + 0.5 * ties) / (wins + losses + ties)` with a neutral level of 50%.
Differences of at most the configured tolerance in absolute value count as
ties. Same mask agreement is also reported by overlap mask. The component
baseline is the number of mismatched initial, final and tone labels. Paired
differences subtract the first model from the second within each seed.

Set `"diagnostics": true` in the `encoder_geometry` configuration and choose a
new report ID to include optional diagnostic CSV tables. The notebook carries
this setting in `SPEC`. The report links the tables and records slice inventories
and DIMSIM B provenance. The UNK comparison retains a common syllable inventory
and Full Match support. The high-support slice requires more than 10 occurrences
and at least three characters in every seed. The Full Match table compares the
two estimators on identical anchors and one-component candidates. The primary
estimator conditions the character-balanced occurrence weights on distinct
occurrences. The alternative gives equal weight to unordered pairs of distinct
characters and averages over their occurrence pairs. The expanded
slice includes the marked readings when supported in every seed. DIMSIM A and B
are compared on the same strictly ordered triples of distinct syllables.

Keep the generated report directory together when opening `encoder_geometry.html`.
It loads the selected seed view locally and displays DIMSIM with two selected
models. The across-seeds view averages matrices and displays seedwise metric
means and sample SD. The static distance maps show mean distances for each model.
The residual histograms in `figures/` are exported in PNG and EPS. They average
seedwise bin proportions and annotate the lowest and highest 5% per distinct
pair mask on the primary inventory, including ties at the cutoffs.

## Visualization

Archived configs can still use these `visualization` workflows through `workflow` and generated notebooks. They render saved numerical results. The required Arial font must be installed. The archived training visualization verifies matching split and vocabulary hashes and uses its original model and seed inventory.

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

Config-driven workflows publish a verified wrapper under `paths.reports/<id>/`. Reference geometry writes its report directly there. Legacy adapters preserve their original scientific outputs under its `outputs/` tree. Exact matching wrappers can be reused after validation. Separate SDK APIs retain explicit reuse of prepared summaries; the config adapter currently creates complete report outputs with `prepare` mode. Notebook exports carry the same wrapper and original scientific manifests.

## Generated artifacts and validation boundaries

Only source code, config, lock and tests belong in Git. `generated/notebooks/`, `generated/jobs/` and `generated/state/` are ignored. The workspace has one generation script, `meta/tools/generate_notebooks.py`, delegating to the same `phonetics generate-notebooks` command.

Local contract tests do not prove successful ML training, GPU equivalence or execution on Kaggle/macOS/Linux. Those runs require their selected environments and inputs. Source bundles, existing embeddings, current datasets and historical archives are not relocated by this change.
