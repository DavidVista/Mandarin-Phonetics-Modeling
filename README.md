# Mandarin Phonetics Modeling

> Shaping and probing the **hidden representation space** of Mandarin encoders through **acoustic alignment**.

## Key Findings

- **Pinyin prediction alone does not yield a phonetically organized space.** Hidden-space similarity is driven mainly by discrete pinyin categories and, for the BiLSTM, by visual Wubi overlap.
- **Acoustic alignment reshapes the geometry.** Adding a continuous acoustic target raises the explanatory power of acoustic similarity from $R^2 = 0.268$ (pinyin encoder) to $0.611$ (pinyin-aligned) and $0.805$ (aligned), after PCA.
- **Alignment compresses the manifold.** The number of principal components needed to explain 90% of the variance drops from **497 → 332 → 67** across the three objectives.
- **The geometry shows up behaviorally.** As alignment strengthens, decoders recover the original lexical item instead of copying a perturbed homophone (recovery $0.018 \rightarrow 0.654$ for LSTM).
- **The effect is architecture-independent.** The same trend reproduces with BERT encoders despite their different inductive bias, confirming it comes from the alignment objective rather than the architecture.

## Motivation

> In many representation learning tasks, the objective is not only to distinguish between classes but to organize the **hidden representation space** according to a desired notion of similarity, such as phonetic, semantic, or visual relatedness. We study this problem through Mandarin phonetic representation learning by comparing three encoder objectives: a **pinyin encoder** trained for phonetic prediction, a **pinyin-aligned encoder** jointly optimized for pinyin prediction and **acoustic alignment**, and an **aligned encoder** trained solely to match continuous acoustic embeddings. Across both BiLSTM and BERT architectures, we investigate three complementary strategies for shaping hidden representations: **a shared sub-character input basis** (Wubi decomposition), **a continuous acoustic target space** derived from wav2vec2 speech embeddings, and **dimensionality reduction** to reveal latent structure. To explain the resulting representations, we introduce a probing framework that combines interpretable regression analysis with a homophone reconstruction benchmark, allowing us to disentangle the contributions of visual, phonetic, acoustic, and semantic factors to hidden-space organization.

## Training Pipeline

![](images/pipeline.png)

### Pipeline objectives

The three encoders differ only in how the encoder is supervised, controlled by the **acoustic alignment** term (MSE against wav2vec2 embeddings):

$$
\mathcal{L}_{\text{pinyin}}=\text{CE}_{\text{dec}} + \mathbb{\lambda}_{\text{enc}} \Sigma_{\text{CE}}
$$

$$
\mathcal{L}_{\text{pinyin-aligned}}=\text{CE}_{\text{dec}} + \mathbb{\lambda}_{\text{enc}} (\Sigma_{\text{CE}} + \mathbb{\lambda}_{\text{MSE}} \text{MSE})
$$

$$
\mathcal{L}_{\text{aligned}}=\text{CE}_{\text{dec}} + \mathbb{\lambda}_{\text{enc}} \text{MSE}
$$

## Evaluation Method

![](images/evaluation.png)

## Results

The results follow one storyline: **acoustic alignment** first perturbs the optimization, then reshapes the **hidden representation space**, and finally changes how decoders behave when probed.

### 1. Optimization dynamics

Introducing **acoustic alignment** into the encoder objective measurably perturbs decoder training, costing **-1.64% after the first epoch**. This is the first sign that the alignment pressure is competing with the pinyin objective rather than simply riding along with it.

![](images/decoder.png)

### 2. Hidden geometry

Regression analysis explains hidden-space similarity in terms of visual, phonetic, acoustic, and semantic factors. As **acoustic alignment** strengthens, acoustic similarity becomes the dominant explanatory factor and the **hidden representation space** collapses into a progressively lower-dimensional manifold:

| Objective | Acoustic $R^2$ (post-PCA) | PCs for 90% variance |
| :-------- | ------------------------: | -------------------: |
| Pinyin encoder | 0.268 | 497 |
| Pinyin-aligned encoder | 0.611 | 332 |
| Aligned encoder | **0.805** | **67** |

![](images/regression.png)

Full coefficient tables, EDA, and SHAP interpretability results are available in this [multi-page diagram](https://viewer.diagrams.net/?tags=%7B%7D&lightbox=1&highlight=0000ff&edit=_blank&layers=1&nav=1&title=Regression.drawio&dark=auto#Uhttps%3A%2F%2Fdrive.google.com%2Fuc%3Fid%3D1W-LC8AjJy-AxZTbzUfHF3dNrhw-hKX1Z%26export%3Ddownload).

### 3. Probing

The geometric change is confirmed behaviorally. Each perturbed homophone bigram is passed through an encoder and reconstructed by its decoder; a high **recovery** rate means the hidden state discarded character identity and kept only phonetic information, while a high **copy** rate means it preserved character identity. The representative LSTM target-position outcome:

| Metric       | Pinyin Encoder | Pinyin-Aligned Encoder | Aligned Encoder |
| :----------- | -------------: | ---------------------: | --------------: |
| **Recovery** |          0.018 |                  0.042 |       **0.654** |
| **Copy**     |          0.768 |                  0.725 |       **0.077** |

As alignment strengthens, recovery rises and copying collapses — hidden states progressively fold character-specific information into shared phonetic representations. The same qualitative trend reproduces with BERT encoders (recovery $0.081 \rightarrow 0.434$), confirming the effect is driven by the objective, not the architecture. Comprehensive phrase-level and BERT tables are reported in the paper.

## Conclusion

We find that optimizing an encoder only for pinyin prediction does not naturally produce a **hidden representation space** organized by phonetic similarity. Instead, hidden-state similarity is primarily explained by discrete pinyin categories and, for the BiLSTM encoder, by visual Wubi overlap. Introducing explicit **acoustic alignment** with continuous embeddings fundamentally changes this geometry: the **pinyin-aligned encoder** substantially increases the explanatory power of acoustic similarity, while the **aligned encoder** makes it the dominant factor, increasing the regression fit from **$R^2=0.268$** for the pinyin encoder to **$0.611$** and **$0.805$**, respectively, after PCA. Alignment also concentrates the representations into progressively lower-dimensional manifolds, reducing the number of principal components required to explain 90% of the variance from **497** to **332** and finally **67**.

These structural changes are reflected behaviorally in the homophone probing benchmark. As alignment strengthens, decoders become increasingly likely to recover the original lexical item rather than copying the perturbed homophone, indicating that hidden states progressively collapse character-specific information into shared phonetic representations. The same qualitative trend is reproduced with BERT encoders, despite their substantially different inductive bias, demonstrating that the effect arises from the alignment objective rather than the encoder architecture. Overall, the study shows that explicit continuous supervision can reshape neural representations from discrete symbolic categories toward graded similarity spaces, providing a practical framework for both learning and interpreting structured latent representations beyond Mandarin phonetics.

## Local setup and pipeline

Use Python 3.12 and uv. Create the environment before activating it. The CPU preparation and analysis environment excludes Torch:

```sh
uv sync --frozen --extra sources --extra audio --extra analysis --extra test
```

Activate on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate on Linux/macOS:

```sh
source .venv/bin/activate
```

Copy `configs/experiment.example.json` to `configs/experiment.json`, review it once, then run:

```sh
phonetics run --config configs/experiment.json --through experiment
phonetics run --config configs/experiment.json
```

The first command stops after experiment preparation. The second continues through the configured training and report stages. Paths resolve relative to the config file. Source and model revisions, encoder and head parameters, training settings, devices, execution modes and analysis jobs are defined in one config. Acoustic targets can be imported from a selected bundle or computed from the configured model. Input hashes are computed automatically and recorded in bundles, jobs and run state; an explicit expected acoustic manifest hash can also be supplied.

Preparation order: download sounds → decode inventory → export and clean corpus → import or infer acoustic targets → assemble prepared data → assemble experiment. Each stage also has an individual CLI command. Computing acoustic targets locally requires the `inference` dependency extra and `--allow-inference`; local training requires `training` and `--allow-training`. Install the Torch 2.11.0 CPU/CUDA build for the selected environment separately. Torch is loaded for model execution.

Each seed samples 100000 train+validation sentences by default: 90000 train and 10000 validation, with one additional shared 10000-sentence test. Five seeds build independent train-only character vocabularies. Test sentences are removed before seed sampling. The default encoder uses 64-dimensional character embeddings and a bidirectional sentence LSTM with 64 units per direction. Data splits and vocabularies are deterministic; GPU outputs need not match bitwise between systems.

The source stage downloads the pinned official Chinese Lessons archive and records attribution and license provenance. Stage results are reused after validation of their inputs, parameters and content. Changed inputs or parameters require distinct output IDs.

### Notebook handoff and resumption

`execution.acoustic`, `execution.training` and `execution.visualization` choose `local` or `notebook`. Analysis reports run locally. Acoustic execution mode controls computation of new targets; selecting an acoustic input bundle uses the import path. Training device and run ID are explicit.

At a notebook stage, `run` prints `status: waiting` and the notebook and job paths. Training generation validates the experiment once for all objectives and creates small reference jobs. Upload the original experiment bundle as a dataset and attach it to the training notebooks; their job manifests and SDK are embedded. Acoustic and visualization jobs use accompanying input packages. Acoustic notebooks are generated after inventory preparation; training notebooks after experiment preparation. Running the final cell executes the configured stage. ML notebooks require Torch 2.11.0, installed separately with the CPU/CUDA build appropriate to the selected device.

Download and extract the result ZIP. A training archive contains `runs/<experiment_id>/<model_id>/<run_id>/`. For manual transfer, merge the contents of the archive's `runs/` into the configured runs directory. To verify and register a pending job, pass the extracted archive root to the importer:

```sh
phonetics import-result --config configs/experiment.json --result path/to/extracted-result
phonetics run --config configs/experiment.json
```

The CLI checks the pending job identity, source snapshot, input pins, result inventory, parameters and completed seeds before publication. Checkpoint bytes are authenticated by hashes; result import does not deserialize them. Each objective has a separate notebook job; resume advances to the next configured objective.

Generate optional notebooks separately with the same generator:

```sh
phonetics generate-notebooks --config configs/experiment.json --stage training
```

Stages are `acoustic`, `training`, `visualization` or `ready` (all whose inputs are available). Generated notebooks, execution jobs and run state are written to the configured paths. The default `generated/` directory is ignored by Git.

See [configuration and workflow examples](docs/pipeline.md) for report and visualization parameters, state handling and input contracts. Report workflows consume evaluation bundles, numerical evaluation outputs and training logs according to their declared schemas. Input schemas and manifest pins are validated before report publication. Visualizations use the shared style protocol.

## Artifacts and reproducibility

Data bundles record their configuration, input identities and artifact hashes. Experiment bundles bind the shared test set, per-seed training and validation splits, character vocabularies and acoustic targets. Training manifests bind the experiment, objective, seeds, source code, runtime, logs and checkpoints. Report outputs retain their input pins and scientific provenance.

Storage locations and output identifiers are selected through the configuration. Published bundles and runs are immutable; new scientific inputs or parameters receive new output identifiers. Generated data and execution outputs are stored separately from versioned source code.
