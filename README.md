# Smelt

Smelt is a smell-recognition research repo built on top of SMELLNET-BASE. It started as a benchmark-faithful reproduction effort for SCENTFORMER-style baselines and grew into a protocol-hardened enhanced detector (grouped file-level CV, validation-locked file aggregation, heterogeneous ensembling), plus a set of dataset forensics: channel-quality audits, channel-attribution probes, cross-release verification, and an online-split generalization evaluation.

This README is the repo summary: what was reproduced, what was tried, what failed, what worked, and which numbers are primary versus extension versus disqualified.

## TL;DR

- We built a benchmark-faithful `exact-upstream` track and reproduced the sensor-only SMELLNET-BASE baselines closely enough to trust the data path (numbers match the ICLR camera-ready Table 2).
- **Primary result (`m05b`, post-audit, benchmark-retained 6 channels — apples-to-apples with the public benchmark):**
  - `78.0` file-level Top-1, `94.0` Top-5, `73.47` macro-F1 on the official test split
  - vs `66.0` for the reproduced CNN baseline evaluated file-level the same way
  - paired on the same 50 test files: `+12.0` [95% CI `+4.0, +22.0`], exact McNemar `p = 0.031`, 6–0 discordance
  - selection calibration: CV/OOF predicted `77.2`, final test gave `78.0`
- **Research extension (`m05`, same protocol on all 12 raw channels):**
  - `88.0` file-level Top-1 / `100.0` Top-5 / `84.67` macro-F1
  - the gain over `m05b` is **not statistically significant** (`+10.0` [CI `−2.0, +24.0`], `p = 0.227`; same-machine rerun `+8.0`, `p = 0.344`)
  - and disappears entirely on the online real-time split (both ensembles: identical `30.4` overall)
  - channel attribution: the extra signal is almost entirely the BME680 `Gas_Resistance` channel — the single most informative channel in the dataset (see channel probes below)
- The protocol for both: grouped 5-fold CV over `offline_training`, CV/OOF-only search, no candidate-level official-test metrics before finalization, frozen refit, single terminal test evaluation.
- Historical exploratory numbers (`m03` 90.0, `m04` 94.0) are **disqualified as headlines**: their searches had official-test contact. The score staircase 94 → 88 → 78 is leakage and flattering comparisons being squeezed out, not models getting worse.

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

These are baseline anchors, not claims that this repo exactly reproduces every upstream implementation detail outside the explicitly documented `exact-upstream` path. The sensor-only anchors were verified against Table 2 of the ICLR 2026 camera-ready (arXiv 2506.00239); the paper's appendix also documents the channel-dropping rationale (suspected sensor malfunction) that our channel-quality audit confirms quantitatively.

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
- this is the protocol to use for 12-channel claims after the audit
- it landed below `m04` by `6.0` Top-1 points and about `7.33` macro-F1 points
- the most likely reason is the protocol tightening itself: grouped 5-fold CV and strict search-time test isolation removed the old opportunity to peek through candidate-level official-test outputs
- cross-platform reproduction: rerunning the full protocol on different hardware/backend (AMD ROCm/Windows vs the original Apple Silicon run) gives `86.0` Top-1 / `82.33` macro-F1 (`results/tables/rocm_repro/`) — a ~2-point backend/seed variance band at n=50

### `m05b`: benchmark6 post-audit protocol (primary apples-to-apples)

`m05b` mirrors the `m05` protocol exactly with one change: the bank is restricted to the
benchmark-retained 6 channels. This is the primary result, because it is the only enhanced
number comparable to the public benchmark setting.

Selected CV / OOF ensemble:

- method: `weighted_probabilities_all` over all 10 bank members (with weaker per-model signal
  at 6 channels, breadth beat the diversity-greedy subset selection that won under all12)
- CV / OOF selection metrics: `77.2` Top-1, `76.57` macro-F1

Final official-test metrics after full-train refit:

- Top-1: `78.0` (95% bootstrap CI `[66.0, 88.0]`)
- Top-5: `94.0`
- Macro-F1: `73.47`

Paired against the reproduced file-level CNN baseline (same channels, same 50 files):

- baseline: `66.0` → `m05b`: `78.0`, delta `+12.0` [CI `+4.0, +22.0`]
- exact McNemar `p = 0.031`; the detector never misses a file the baseline gets right (6–0)

Paired against `m05` (all12):

- `+10.0` [CI `−2.0, +24.0`], `p = 0.227` (original m05); `+8.0`, `p = 0.344` (same-machine rerun)
- **not significant at n=50** — no channel-superiority claim is defensible

### Channel forensics and probes

`scripts/smelt_channel_quality_audit.py` (raw 12-column data quality):

- `Benzene` is dead: `32.4%` of raw values are the uint32 overflow sentinel `4294967295`;
  `222/250` training files are complete flatlines
- one BME680 unit was frozen for a nine-class recording block (matches App. Table 16 of the
  upstream paper); `Humidity`/`Gas_Resistance` flatline in ~20% of training files
- the *retained* `Alcohol` channel also flatlines in `26/250` training and `5/50` test files

`scripts/smelt_channel_signal_probe.py` (per-channel class information, nearest-centroid,
chance = 2.0):

- `Gas_Resistance` alone: `30.0` — the most informative single channel in the dataset
  (best retained channel is `NO2` at `22.0`)
- benchmark6 `54.0` → benchmark6 + `Gas_Resistance` `70.0` → all12 `72.0`: nearly the entire
  12-channel advantage is that one legitimately chemical sensor
- recommendation: if a future release restores one channel, restore `Gas_Resistance`

`scripts/smelt_env_barcode_probe.py` (session-leakage test):

- raw absolute environmental levels leak some class identity (ENV4: `32.0`), but the
  day-held-out split defeats most of it (`Pressure` alone: `8.0`) and the pipeline's
  differencing reduces ENV4 to `22.0` — environmental leakage cannot explain the 12-channel gap

### Online (real-time) split evaluation

`scripts/smelt_online_eval.py` evaluates the frozen refit ensembles once, 50-way, with zero
selection contact, on the real-time recordings from the arXiv version of the dataset
(`online_nuts`: 10 files, `online_spices`: 13 files; present in the archived HF revision only):

- `m05b` (6ch): `30.4` overall — `10.0` nuts, `46.2` spices
- `m05` (12ch): **identical** `30.4` / `10.0` / `46.2` — the offline all12 advantage does not
  survive distribution shift
- arXiv-v1-reported upstream models: `10.7` nuts, `25.4` spices (metric granularity may
  differ; treat as context, not head-to-head)

## What actually mattered

### Things that helped a lot

- file-level aggregation (this alone lifts the plain CNN baseline from `55.6` window-level to
  `66.0` file-level — always compare like with like)
- validation-locked model and aggregator selection
- heterogeneous ensembling
- diff-only view instead of fused raw+diff
- of the extra channels, effectively one: `Gas_Resistance` (the all12-vs-benchmark6 delta is
  not significant and vanishes online; see channel probes)

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

## Data root and dataset revisions

Most data-backed commands expect `SMELT_DATA_ROOT` to point at the SMELLNET-BASE data directory.

Important: the public `DeweiFeng/smell-net` dataset was reorganized on 2026-04-13. The current
release (`base_data/training|testing`) contains **only the 6 benchmark channels**. The original
12-column tree (`data/offline_training|offline_testing`), plus the online split and GC-MS
extras, lives at archived revision `71bcae740b88`:

```bash
python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='DeweiFeng/smell-net', repo_type='dataset', revision='71bcae740b88', allow_patterns=['data/offline_training/*','data/offline_testing/*','data/online_nuts/*','data/online_spices/*']))"
export SMELT_DATA_ROOT="<printed-snapshot-path>/data"
```

We verified all 300 base recordings are content-identical on the 6 retained channels across the
two releases, so `m05b` and the exact-upstream track reproduce from either revision; the all12
`m05` extension requires the archived revision.

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

- post-audit grouped-cv protocol, benchmark-retained 6 channels (primary):

```bash
python -m smelt.training.run_m05b
```

- post-audit grouped-cv protocol, all 12 channels (research extension):

```bash
python -m smelt.training.run_m05
```

- channel forensics and probes:

```bash
python scripts/smelt_channel_quality_audit.py --data-root "$SMELT_DATA_ROOT"
python scripts/smelt_channel_signal_probe.py --data-root "$SMELT_DATA_ROOT"
python scripts/smelt_env_barcode_probe.py --data-root "$SMELT_DATA_ROOT"
```

- paired protocol comparison (bootstrap CIs + exact McNemar from saved predictions):

```bash
python scripts/smelt_m05b_analysis.py
```

- online-split evaluation of a frozen protocol run:

```bash
python scripts/smelt_online_eval.py --run-dir <protocol-run-dir> --protocol-id m05b --data-root "$SMELT_DATA_ROOT"
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

- collaboration mini-writeup (full narrative with all numbers):
  - [docs/research-extension/miniwriteup_dewei.md](docs/research-extension/miniwriteup_dewei.md)
- primary `m05b` official-test summary:
  - [results/tables/m05b_final_test.json](results/tables/m05b_final_test.json)
- headline paired test (m05b vs file-level baseline, same channels):
  - [results/tables/m05b_vs_exact_upstream_baseline_comparison.json](results/tables/m05b_vs_exact_upstream_baseline_comparison.json)
- channel comparison (m05b vs m05, non-significant):
  - [results/tables/m05b_vs_m05_channel_comparison.json](results/tables/m05b_vs_m05_channel_comparison.json)
- protocol scorecard (m01c → m03 → m04 → m05 → m05b with hygiene notes):
  - [results/tables/m05b_scorecard.csv](results/tables/m05b_scorecard.csv)
- channel forensics:
  - [results/tables/channel_quality_audit.csv](results/tables/channel_quality_audit.csv)
  - [results/tables/channel_signal_probe.csv](results/tables/channel_signal_probe.csv)
  - [results/tables/env_barcode_probe.csv](results/tables/env_barcode_probe.csv)
- online-split generalization:
  - [results/tables/m05b_online_eval_summary.json](results/tables/m05b_online_eval_summary.json)
- 12-channel extension and same-machine reproduction:
  - [results/tables/m05_final_test.json](results/tables/m05_final_test.json)
  - [results/tables/rocm_repro/m05_final_test.json](results/tables/rocm_repro/m05_final_test.json)

If you want the verification and paper-ready exports first:

- verification inventory:
  - [results/tables/verification_inventory.json](results/tables/verification_inventory.json)
- exact-upstream verification:
  - [results/tables/verification_exact_upstream.json](results/tables/verification_exact_upstream.json)
- moonshot protocol verification:
  - [results/tables/verification_moonshot_protocol.json](results/tables/verification_moonshot_protocol.json)
- leakage and selection audit:
  - [results/tables/verification_leakage_selection_audit.json](results/tables/verification_leakage_selection_audit.json)
- bootstrap confidence intervals:
  - [results/tables/verification_bootstrap_ci.json](results/tables/verification_bootstrap_ci.json)
- paper table inputs:
  - [results/tables/paper_baseline_table.csv](results/tables/paper_baseline_table.csv)
  - [results/tables/paper_ablation_table.csv](results/tables/paper_ablation_table.csv)
  - [results/tables/paper_main_results_table.csv](results/tables/paper_main_results_table.csv)
  - [results/tables/paper_diversity_table.csv](results/tables/paper_diversity_table.csv)

## Bottom line

If you care about benchmark-faithful comparison, use the `exact-upstream` track for
window-level numbers and `m05b` for the enhanced detector:

- `m05b`: `78.0` file-level Top-1 [CI `66.0, 88.0`], `94.0` Top-5, `73.47` macro-F1
- `+12.0` over the file-level baseline on identical channels/files/metric, `p = 0.031`

If you care about the 12-channel research extension, use `m05` (`88.0` / `100.0` / `84.67`),
with two caveats stated up front: the gain over `m05b` is not statistically significant at
n=50, and it requires the archived 12-column dataset revision.

Historical exploratory numbers (`m03` 90.0, `m04` 94.0) are retained in the tables for
transparency but are not claimable: their searches had official-test contact.
