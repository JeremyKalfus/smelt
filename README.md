# Smelt

Smelt is a smell-recognition research repo built on top of SMELLNET-BASE. It started as a benchmark-faithful reproduction effort for SCENTFORMER-style baselines and then grew into an enhanced-setting moonshot system that uses all 12 sensor channels, grouped file-level validation, validation-locked file aggregation, and heterogeneous ensembling.

This README is the repo summary: what was reproduced, what was tried, what failed, what worked, and which numbers are exploratory versus final-definitive.

## TL;DR

- We built a benchmark-faithful `exact-upstream` track and reproduced the sensor-only SMELLNET-BASE baselines closely enough to trust the data path.
- We built a `research-extension` track for fused/raw/diff variants, stronger backbones, and GC-MS pretrain → fine-tune experiments. Some ideas helped, several did not.
- The biggest gains came from the `moonshot-enhanced-setting` track:
  - use all 12 channels instead of the benchmark-retained 6
  - keep diff-only input
  - use grouped file-level validation
  - lock checkpointing and file-level decisions on validation only
  - use a heterogeneous ensemble chosen on validation only
- `m05` is the defensible post-audit moonshot protocol:
  - grouped 5-fold CV over `offline_training`
  - CV / OOF-only search
  - no candidate-level official-test metrics during search
  - freeze the final bank, aggregator behavior, and epoch budgets before refit
  - refit on full official training, then evaluate the official test at the very end
- Current final-definitive post-audit result (`m05`):
  - `88.0` file-level Top-1
  - `100.0` file-level Top-5
  - `84.6667` file-level macro-F1
- Strongest tracked exploratory result (`m04`):
  - `94.0` file-level Top-1
  - `100.0` file-level Top-5
  - `92.0` file-level macro-F1

## What this repo is trying to do

Smelt has two explicit goals:

1. Benchmark-faithful comparison
   - stay as close as possible to the public SmellNet / SCENTFORMER setup
   - use this for apples-to-apples comparison

2. Best possible smell-detection system on SMELLNET-BASE
   - use any reasonable modeling choice that improves results
   - be explicit when the setting is no longer strictly benchmark-faithful

The second track is where the largest gains came from.

## Evaluation tracks

### 1) Exact-upstream / benchmark-faithful

This track keeps the public base benchmark setup intact:

- official `offline_training` vs `offline_testing` split
- 6 retained channels
- baseline subtraction
- `g=25` temporal differencing
- `window_size=100`
- `stride=50`
- train-only standardization
- window-level classification metrics as the main benchmark-comparable result

### 2) Research-extension

This track explored architectural and supervision changes without changing the core base task:

- raw vs diff vs fused inputs
- Inception-style models
- GC-MS pretrain → fine-tune experiments
- stronger backbones

### 3) Moonshot enhanced-setting

This track explicitly aims for best detector performance, not strict benchmark parity:

- all 12 channels
- diff-only windows
- grouped file-level validation
- file-level aggregation
- validation-locked checkpoint selection
- validation-locked ensemble selection

## Dataset and preprocessing summary

### Exact-upstream base split

- 50 classes
- 250 training CSVs
- 50 test CSVs
- public split audited from the raw Hugging Face snapshot and matched to the upstream file contract

### Exact-upstream preprocessing contract

- retained channels: `['NO2', 'C2H5OH', 'VOC', 'CO', 'Alcohol', 'LPG']`
- retained channel count: `6`
- differencing period: `25`
- window size: `100`
- stride: `50`
- exact-upstream window counts:
  - `2512` train windows
  - `502` test windows

### Legacy grouped-holdout setup (`m01c`-`m04`)

The pre-`m05` locked moonshot path used one validation file per class from the training split:

- `2013` train windows
- `499` validation windows
- `502` test windows

### `m05` grouped 5-fold CV setup

The final post-audit `m05` protocol replaces the tiny deterministic grouped holdout with explicit grouped folds:

- `5` folds over the official `offline_training` split
- each fold holds out exactly `1` training CSV per class as validation
- each fold contains `200` train CSVs and `50` validation CSVs
- every official training CSV serves as validation exactly once
- the official `offline_testing` split is untouched during CV selection and evaluated exactly once after final refit

## SmellNet paper reference points

For SMELLNET-BASE, the paper anchors we used were:

- SCENTFORMER / Transformer, sensor-only, `w=100`, `p=25`: `56.1` Top-1, `87.4` Top-5, `55.5` macro-F1
- SCENTFORMER / Transformer, cross-modal GC-MS, `w=100`, `p=25`: `63.3` Top-1, `86.1` Top-5, `61.7` macro-F1
- CNN, sensor-only, `w=100`, `p=25`: `52.7` Top-1, `85.6` Top-5, `50.5` macro-F1
- CNN, cross-modal GC-MS, `w=100`, `p=25`: `58.9` Top-1, `88.4` Top-5, `57.0` macro-F1

These are baseline anchors, not claims that this repo exactly reproduces every upstream implementation detail outside the explicitly documented `exact-upstream` path.

## Results: exact-upstream / benchmark-faithful

### Transformer baseline (`t07`)

- Top-1: `54.1833`
- Top-5: `88.4462`
- Macro precision: `57.3991`
- Macro recall: `54.2141`
- Macro F1: `50.4290`

Interpretation:
- close to the paper's sensor-only Transformer baseline
- good enough to trust the benchmark-faithful path

### CNN baseline (`t08`)

- Top-1: `55.5777`
- Top-5: `87.0518`
- Macro precision: `62.9565`
- Macro recall: `55.5273`
- Macro F1: `53.8937`

Interpretation:
- slightly better than the reproduced Transformer baseline
- close enough to the benchmark-faithful band needed for later work

## Results: research-extension experiments

### Inception-style fused supervised (`t10`)

- view: `fused_raw_diff`
- feature count: `12`
- Top-1: `40.4382`
- Top-5: `79.0837`
- Macro precision: `43.8060`
- Macro recall: `40.1343`
- Macro F1: `38.3780`

Interpretation:
- underperformed badly
- fused raw+diff was not a good mainline path in this form

### View-isolation diagnostics (`t10b`)

#### Raw-aligned only Inception

- Top-1: `45.4183`
- Top-5: `78.4861`
- Macro precision: `43.7461`
- Macro recall: `45.5667`
- Macro F1: `40.8153`

#### Diff-only Inception

- Top-1: `51.3944`
- Top-5: `87.8486`
- Macro precision: `54.6318`
- Macro recall: `51.4485`
- Macro F1: `49.3695`

Interpretation:
- diff-only was clearly better than raw or fused
- this became the mainline candidate for later non-benchmark-faithful experiments

### GC-MS pretrain and fine-tune failure (`t11`)

#### GC-MS pretraining

- Top-1: `26.8924`
- Top-5: `63.9442`
- Macro precision: `21.9851`
- Macro recall: `27.0040`
- Macro F1: `22.6244`

#### Fine-tune from GC-MS pretraining

- Top-1: `44.6215`
- Top-5: `78.6853`
- Macro precision: `43.3507`
- Macro recall: `44.2505`
- Macro F1: `41.1715`

#### Fair baseline used for comparison

- diff-only Inception baseline: `51.3944` Top-1, `49.3695` macro-F1

#### Delta

- Top-1 delta: `-6.7729`
- Macro-F1 delta: `-8.1980`

Interpretation:
- this was a negative result
- the tested GC-MS pretrain → fine-tune path hurt performance

## Moonshot enhanced-setting results

This is the main story of the repo.

### `m01`: first all-12-channel moonshot CNN

Window-level:

- Top-1: `67.1315`
- Top-5: `94.6215`
- Macro precision: `71.0805`
- Macro recall: `66.8889`
- Macro F1: `66.1748`

File-level on the same run:

- mean logits: `80.0` Top-1, `98.0` Top-5, `75.3333` macro-F1
- mean probabilities: `82.0` Top-1, `98.0` Top-5, `78.0` macro-F1
- majority vote: `78.0` Top-1, `96.0` Top-5, `73.0` macro-F1

Interpretation:
- all 12 channels plus file aggregation was the first major jump

### `m01b`: anti-cheat + channel ablations

#### Anti-cheat checks

- eval-only replay: pass
- independent recomputation from saved predictions: pass
- grouped split leakage audit: pass
- shuffled-label control: pass

Shuffled-label control collapse:

- window Top-1: `1.1952`
- window Top-5: `8.5657`
- file Top-1: `2.0`
- best file Top-1 across aggregators: `4.0`

Interpretation:
- strong evidence that the moonshot win was not leakage or evaluator cheating

#### Channel ablations

##### benchmark6 control

Window-level:

- Top-1: `59.1633`
- Top-5: `89.6414`
- Macro precision: `67.1005`
- Macro recall: `59.0444`
- Macro F1: `58.2981`

Best file-level:

- Top-1: `80.0`
- Top-5: `98.0`
- Macro F1: `75.8`

##### extra6-only control

Window-level:

- Top-1: `42.0319`
- Top-5: `79.4821`
- Macro precision: `39.9384`
- Macro recall: `42.1515`
- Macro F1: `38.2309`

Best file-level:

- Top-1: `52.0`
- Top-5: `76.0`
- Macro F1: `45.4970`

##### second-seed all12 run

Window-level:

- Top-1: `68.5259`
- Top-5: `92.6295`
- Macro precision: `69.7807`
- Macro recall: `68.6626`
- Macro F1: `67.1110`

Best file-level:

- Top-1: `88.0`
- Top-5: `98.0`
- best macro-F1 at mean probabilities: `85.3333`

Interpretation from `m01b`:

- all 12 channels helped materially over benchmark6
- the extra 6 channels carried real signal on their own
- aggregation was a major driver of file-level performance
- file-level peak was not stable enough yet to be the final headline

### `m01c`: locked grouped-holdout moonshot baseline

This is the first defensible grouped-holdout moonshot summary because aggregator selection and checkpoint selection were both locked on validation only.

Protocol:

- model family: `cnn`
- channel set: `all12`
- view mode: `diff_all12`
- `g=25`, `window_size=100`, `stride=50`
- grouped validation: 1 validation file per class
- standardization: train-only
- locked aggregator rule: best validation file Top-1, tie-break validation file macro-F1
- checkpoint rule: best validation file Top-1 under the locked aggregator

Locked summary across 3 fresh seeds:

- Window Top-1: `68.1939 ± 0.4695`
- Window Top-5: `93.7583 ± 0.8023`
- Window Macro-F1: `66.9686 ± 0.2075`
- Locked file Top-1: `84.6667 ± 3.3993`
- Locked file Top-5: `97.3333 ± 2.4944`
- Locked file Macro-F1: `81.0000 ± 3.7810`

Per-seed locked file-level results:

- seed42: `80.0` Top-1, `75.6667` macro-F1
- seed7: `88.0` Top-1, `84.0` macro-F1
- seed13: `86.0` Top-1, `83.3333` macro-F1

Interpretation:
- this is the strongest honest enhanced-setting single-family baseline in the repo

### `m02`: deep temporal ResNet

Architecture summary:

- model family: `deep_temporal_resnet`
- block type: `se_basic_residual_block`
- stage depths: `[3, 4, 6, 3]`
- stage widths: `[64, 128, 256, 384]`
- parameter count: `8,981,970`
- device: `mps`
- batch size: `16`
- gradient accumulation: `4`
- effective batch size: `64`

Results:

- Window Top-1: `62.7490`
- Window Top-5: `93.0279`
- Window Macro-F1: `61.3758`
- Locked file Top-1: `86.0`
- Locked file Top-5: `98.0`
- Locked file Macro-F1: `81.3333`

Interpretation:
- slightly improved file-level accuracy over the `m01c` mean
- but hurt window-level performance badly
- not the right main path for another big jump

### `m03`: learned file-level model vs locked ensemble

#### Primary frozen encoder selection

Selected seed:

- `m01c` seed42
- locked primary aggregator on validation: `majority_vote`
- validation file Top-1: `90.0`

#### Locked cross-seed ensemble

Selected method:

- `vote`

Final file-level metrics:

- Top-1: `90.0`
- Top-5: `94.0`
- Macro precision: `86.0`
- Macro recall: `90.0`
- Macro F1: `87.3333`

#### Learned AttentionDeepSets file-level head

Per-seed:

- seed11: `82.0` Top-1, `76.6667` macro-F1
- seed23: `80.0` Top-1, `75.0` macro-F1
- seed37: `84.0` Top-1, `79.0` macro-F1

Mean ± std:

- Top-1: `82.0 ± 1.6330`
- Top-5: `97.3333 ± 0.9428`
- Macro-F1: `76.8889 ± 1.6405`

Interpretation:
- the ensemble was the real win
- the learned frozen file-level head was stable but weaker than the locked ensemble

### `m04`: heterogeneous moonshot ensemble bank

This is the current best tracked exploratory result, but it is not the final-definitive protocol.

Model bank summary:

- reused locked CNN seed13: `68.5259` window Top-1, `86.0` locked file Top-1
- reused locked CNN seed42: `67.5299` window Top-1, `80.0` locked file Top-1
- reused locked CNN seed7: `68.5259` window Top-1, `88.0` locked file Top-1
- new CNN seed101: `64.3426` window Top-1, `82.0` locked file Top-1
- new CNN seed202: `69.9203` window Top-1, `84.0` locked file Top-1
- reused `m02` deep temporal ResNet: `62.7490` window Top-1, `86.0` locked file Top-1
- new deep temporal ResNet seed7: `58.5657` window Top-1, `80.0` locked file Top-1
- new H-Inception seed17: `41.0359` window Top-1, `60.0` locked file Top-1
- new H-Inception seed29: `48.6056` window Top-1, `74.0` locked file Top-1
- new Patch Transformer seed19: `62.7490` window Top-1, `80.0` locked file Top-1

Selected final ensemble:

- method: `diversity_greedy_probabilities`
- selected members:
  - locked CNN seed13
  - deep temporal ResNet (`m02`)
  - CNN seed101
  - H-Inception seed29
  - Patch Transformer seed19

Final file-level metrics:

- Top-1: `94.0`
- Top-5: `100.0`
- Macro precision: `91.0`
- Macro recall: `94.0`
- Macro F1: `92.0`

Interpretation:
- this is the strongest tracked exploratory result in the repo
- diversity-aware selection beat naive averaging
- the best ensemble was multi-family + multi-seed
- `m04` should still be treated as exploratory because candidate-level official-test
  metrics existed during search/output

### `m05`: post-audit grouped-cv protocol

`m05` is the defensible moonshot protocol after the leakage/selection audit.

Protocol:

- keep the strongest feature setting: `all12`, diff-only, `g=25`, `window_size=100`, `stride=50`
- keep the strongest current bank scope unless a code-level issue forces a change
- build explicit grouped 5-fold CV over the official training split
- use CV / OOF file-level evidence only for:
  - model-bank ranking
  - locked aggregator choice per member
  - ensemble member selection
  - ensemble method selection
  - frozen epoch-budget selection for full-train refit
- do not compute candidate-level official-test metrics during search
- refit only the frozen selected members on the full official training split
- evaluate the official test only after finalization

Selected CV / OOF ensemble:

- method: `diversity_greedy_probabilities`
- selected members:
  - `m01c_cnn_all12_diff_locked_seed42`
  - `m01c_cnn_all12_diff_locked_seed7`
  - `m04_deep_temporal_resnet_all12_diff_seed7`
  - `m04_hinception_all12_diff_seed29`
- weights: `[0.25, 0.25, 0.25, 0.25]`
- CV / OOF file metrics used for selection:
  - Top-1: `88.8`
  - Top-5: `98.4`
  - Macro precision: `89.3190`
  - Macro recall: `88.8`
  - Macro F1: `87.9878`

Final official-test metrics after full-train refit:

- Top-1: `88.0`
- Top-5: `100.0`
- Macro precision: `83.0`
- Macro recall: `88.0`
- Macro F1: `84.6667`

Interpretation:
- this is the protocol to use for final claims after the audit
- it landed below `m04` by `6.0` Top-1 points and about `7.33` macro-F1 points
- the most likely reason is the protocol tightening itself: grouped 5-fold CV and strict search-time test isolation removed the old opportunity to peek through candidate-level official-test outputs

## What actually mattered

### Things that helped a lot

- all 12 channels instead of the benchmark-retained 6
- diff-only view instead of fused raw+diff
- file-level aggregation
- validation-locked model and aggregator selection
- heterogeneous ensembling

### Things that did not help

- the first fused raw+diff Inception path
- the tested GC-MS pretrain → fine-tune path
- making only the per-window model deeper without improving the file-level decision layer

## Anti-cheat / validity checks

We explicitly ran and preserved:

- eval-only replay from checkpoints
- independent recomputation from saved predictions
- split leakage audits
- exact duplicate-content audits across train folds, validation folds, and official test
- boundary checks for window generation
- train-only standardization checks
- shuffled-label controls
- validation-only aggregator selection
- validation-only checkpoint selection
- CV / OOF-only ensemble and bank selection for `m05`
- repeated exact-upstream regression checks

Representative anti-cheat numbers:

- moonshot shuffled-label all12 run:
  - `1.1952` window Top-1
  - `2.0` file Top-1
  - best file Top-1 across aggregators: `4.0`

These collapses are what we wanted to see.

## Repository layout

- `src/smelt/`: core package
- `src/smelt/datasets/`: dataset contracts and file-aware data paths
- `src/smelt/preprocessing/`: baseline subtraction, differencing, windowing, standardization
- `src/smelt/models/`: benchmark baselines, moonshot CNNs, temporal ResNet, transformer-like models
- `src/smelt/training/`: runners, replay/export tools, ensemble selection
- `src/smelt/evaluation/`: metrics, file-level aggregation, diagnostics, export helpers
- `configs/exact-upstream/`: benchmark-faithful configs
- `configs/research-extension/`: research-only configs
- `configs/moonshot-enhanced/`: locked moonshot configs
- `results/tables/`: structured summaries used throughout the project
- `results/figdata/`: long-form plotting tables
- `results/runs/`: per-run artifacts
- `results/embeddings/`: exported frozen feature bundles
- `artifacts/methods/`: regression-smoke and audit artifacts

## Install and validate

Requirements:

- Python `>=3.10`
- a local or cached SMELLNET-BASE snapshot

Install:

```bash
python -m pip install -e ".[dev]"
```

Repo validation contract:

```bash
ruff check .
ruff format --check .
pytest -q
python -m compileall src tests scripts
```

## Data root

Most data-backed commands expect `SMELT_DATA_ROOT` to point at the SMELLNET-BASE data directory.

Example:

```bash
export SMELT_DATA_ROOT="$HOME/.cache/huggingface/hub/datasets--DeweiFeng--smell-net/snapshots/<snapshot-id>/data"
```

## Useful entry points

- exact-upstream regression smoke:

```bash
python scripts/smelt_preprocess_smoke.py \
  --data-root "$SMELT_DATA_ROOT" \
  --diff-period 25 \
  --window-size 100 \
  --stride 50
```

- moonshot run:

```bash
python -m smelt.training.run_moonshot \
  --config configs/moonshot-enhanced/m01c_cnn_all12_diff_locked_seed42.yaml
```

- heterogeneous ensemble search:

```bash
python -m smelt.training.run_m04_ensemble ...
```

- post-audit grouped-cv protocol:

```bash
python -m smelt.training.run_m05
```

- verification-only export pass:

```bash
python scripts/smelt_verification_export.py \
  --run-root results/runs \
  --table-root results/tables \
  --file-level-root results/file-level-eval \
  --class-vocab-manifest-path artifacts/manifests/base_class_vocab.json \
  --category-map-path configs/exact-upstream/category_map.json \
  --exact-regression-artifact-path artifacts/methods/verification_exact_upstream_regression_smoke.json
```

## Where to look first

If you want the current headline artifacts first:

- final-definitive `m05` official-test summary:
  - [results/tables/m05_final_test.json](/Users/jeremykalfus/CodingProjects/smelt/results/tables/m05_final_test.json)
- final-definitive `m05` scorecard versus earlier protocols:
  - [results/tables/m05_scorecard.json](/Users/jeremykalfus/CodingProjects/smelt/results/tables/m05_scorecard.json)
- `m05` CV / OOF ensemble-selection details:
  - [results/tables/m05_cv_ensemble_selection.json](/Users/jeremykalfus/CodingProjects/smelt/results/tables/m05_cv_ensemble_selection.json)
- current exploratory `m04` comparison:
  - [results/tables/m04_final_comparison.json](/Users/jeremykalfus/CodingProjects/smelt/results/tables/m04_final_comparison.json)
- locked moonshot baseline summary:
  - [results/tables/m01c_seed_summary.json](/Users/jeremykalfus/CodingProjects/smelt/results/tables/m01c_seed_summary.json)

If you want the verification and paper-ready exports first:

- verification inventory:
  - [results/tables/verification_inventory.json](/Users/jeremykalfus/CodingProjects/smelt/results/tables/verification_inventory.json)
- exact-upstream verification:
  - [results/tables/verification_exact_upstream.json](/Users/jeremykalfus/CodingProjects/smelt/results/tables/verification_exact_upstream.json)
- moonshot protocol verification:
  - [results/tables/verification_moonshot_protocol.json](/Users/jeremykalfus/CodingProjects/smelt/results/tables/verification_moonshot_protocol.json)
- leakage and selection audit:
  - [results/tables/verification_leakage_selection_audit.json](/Users/jeremykalfus/CodingProjects/smelt/results/tables/verification_leakage_selection_audit.json)
- bootstrap confidence intervals:
  - [results/tables/verification_bootstrap_ci.json](/Users/jeremykalfus/CodingProjects/smelt/results/tables/verification_bootstrap_ci.json)
- paper table inputs:
  - [results/tables/paper_baseline_table.csv](/Users/jeremykalfus/CodingProjects/smelt/results/tables/paper_baseline_table.csv)
  - [results/tables/paper_ablation_table.csv](/Users/jeremykalfus/CodingProjects/smelt/results/tables/paper_ablation_table.csv)
  - [results/tables/paper_main_results_table.csv](/Users/jeremykalfus/CodingProjects/smelt/results/tables/paper_main_results_table.csv)
  - [results/tables/paper_diversity_table.csv](/Users/jeremykalfus/CodingProjects/smelt/results/tables/paper_diversity_table.csv)

## Bottom line

If you care about benchmark-faithful comparison, use the `exact-upstream` track.

If you care about final-definitive moonshot claims, use `m05`:

- `88.0` file-level Top-1
- `100.0` file-level Top-5
- `84.6667` file-level macro-F1

If you care about the strongest tracked exploratory number in this repo, the current `m04`
moonshot heterogeneous ensemble is:

- `94.0` file-level Top-1
- `100.0` file-level Top-5
- `92.0` file-level macro-F1
