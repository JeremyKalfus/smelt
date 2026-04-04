# PLANS.md

This file refines `docs/specs/smellnetplan.md`. It does not replace it.

Execution policy:

- one ticket == one commit
- do not start a later ticket while the current ticket has a broken validation command
- stop and fix on the first benchmark-compatibility or leakage failure
- no silent fallbacks

Current active moonshot ticket:

- `m03`: learned file-level model + locked seed ensemble
  - keep the locked `m01c` window encoder protocol fixed
  - reuse frozen `m01c` seed runs via eval-only embedding/logit export
  - select the primary frozen encoder and ensemble method from validation only
  - train only the file-level head in this ticket

## track separation

All operative artifacts must stay in one of three explicit tracks:

- `exact-upstream`
  - configs: `configs/exact-upstream/`
  - docs: `docs/exact-upstream/`
  - result tables: `results/exact-upstream/`
- `research-extension`
  - configs: `configs/research-extension/`
  - docs: `docs/research-extension/`
  - result tables: `results/research-extension/`
- `moonshot-enhanced-setting`
  - configs: `configs/moonshot-enhanced/`
  - docs: `docs/research-extension/`
  - result tables: `results/tables/` and `results/figdata/` with explicit moonshot naming

Do not mix these names in config files, docs, run names, or table exports.

## upstream benchmark contract

Verified against the official upstream repo at `.reference/smellnet_upstream/` plus the hosted dataset tree it references.

- `SMELLNET-BASE` data layout is fixed, file-based, and closed-world:
  - `data/offline_training/<class>/*.csv`
  - `data/offline_testing/<class>/*.csv`
  - 50 single-ingredient classes
  - 5 training csvs per class, 1 testing csv per class
- raw sensor csv schema is 12 columns:
  - `NO2`, `C2H5OH`, `VOC`, `CO`, `Alcohol`, `LPG`, `Benzene`, `Temperature`, `Pressure`, `Humidity`, `Gas_Resistance`, `Altitude`
- the benchmark path keeps 6 channels by dropping:
  - `Benzene`, `Temperature`, `Pressure`, `Humidity`, `Gas_Resistance`, `Altitude`
- train/test split logic is directory-driven only. there is no random split in `run.py`.
- preprocessing order is:
  1. subtract first row per csv
  2. drop filtered columns
  3. optional temporal differencing `df.diff(periods=g).iloc[g:]`
  4. sliding windows of shape `(N, T, C)`
  5. train-only standardization on flattened train windows
- sliding windows are generated per file with:
  - `for start in range(0, len(df) - window_size + 1, stride)`
  - default `stride = window_size // 2`
- upstream contrastive mode is not "pretrain then supervised fine-tune"
  - it is closed-world gc-ms retrieval against a full gc-ms gallery
  - this is benchmark-critical and must be reproduced before any new training regime
- upstream gc-ms contrastive mode is a retrieval-style benchmark path, not equivalent to the planned `research-extension` pretrain-then-finetune supervised classifier
- every config name, run name, and result table name must make that distinction obvious
- upstream metrics are:
  - `acc@1`, `acc@5`
  - `precision_macro`, `recall_macro`, `f1_macro`
  - `confusion_matrix`
  - optional per-category `acc@1` and `acc@5`
- upstream `--real-test-dir` is loaded but unused in `run.py`
- shipped data does not expose proven public day ids in the benchmark split. use only a file-heldout robustness proxy unless public day/session metadata is later proven.

## conflict handling

Research plan intent:

- gc-ms contrastive pretraining + supervised fine-tuning

Upstream compatibility reality:

- official gc-ms mode in `run.py` is retrieval against the full gc-ms bank, not supervised fine-tuning

Resolution:

- `smelt` will implement two distinct paths
- `exact-upstream/upstream_retrieval`: exact-compatible benchmark sanity path
- `research-extension/pretrain_finetune`: explicit experimental deviation, documented as such in configs and reports

## safe mirror vs reimplement

Safe to mirror semantically:

- per-file baseline subtraction
- filtered-column drop
- differencing-before-windowing
- stride and window semantics
- train-only standardization
- metric names and report structure

Too entangled or unsafe to mirror directly:

- implicit gc-ms row-order alignment
- unused but required `real-test-dir` handling
- all mixture pipeline codepaths in the first implementation pass
- collection/preprocessing logic that depends on column position or `state.txt`

## hard stop rules

- stop if train/test file hashes overlap
- stop if class vocab differs across split manifest and gc-ms anchor table
- stop if `artifacts/manifests/gcms_class_map.json` is missing for any gc-ms run
- stop if the gc-ms class map is not bijective for the classes in scope
- stop if there are missing classes, duplicate anchors, or ambiguous gc-ms mappings
- stop if standardization touches official test data during fitting
- stop if split parity is wrong
- stop if preprocessing parity is wrong
- stop if metric/report parity is wrong
- stop if the training/eval pipeline is unstable
- stop if any ticket changes benchmark semantics without a named config-level deviation
- stop if a validation command fails

## baseline gate

Do not proceed to any `research-extension` ticket until the `exact-upstream` sanity path is verified end to end.

Verification means:

- split parity passes
- preprocessing parity passes
- metric/report parity passes
- the exact-upstream training/eval path runs stably

## shared validation commands

These are the exact command contracts the implementation tickets will make runnable:

- `python -m pip install -e .[dev]`
- `ruff check .`
- `ruff format --check .`
- `pytest -q`
- `python -m compileall src tests scripts`
- `python -m smelt.datasets.audit_base --data-root "$SMELT_DATA_ROOT" --emit artifacts/manifests/smellnet_base.json`
- `python -m smelt.datasets.audit_base --data-root "$SMELT_DATA_ROOT" --emit artifacts/manifests/gcms_class_map.json --verify-gcms-bijection`
- `python -m smelt.training.run --config configs/exact-upstream/e0_transformer_upstream.yaml`
- `python -m smelt.training.run --config configs/exact-upstream/f1_cnn_upstream.yaml`
- `python -m smelt.training.run --config configs/research-extension/e1_fit_gcms.yaml`
- `python -m smelt.evaluation.compare --inputs results/exact-upstream/runs --out results/exact-upstream/tables/base_comparison.csv`

## experiment registry and order

- `e0`: `exact-upstream` transformer sanity run, `w=100`, `g=25`, gc-ms retrieval on
- `e0b`: `exact-upstream` transformer sanity run, `w=100`, `g=25`, gc-ms retrieval off
- `f1`: `exact-upstream` hardened cnn fallback on the same benchmark path
- `e1`: `research-extension` fit-gcms main model, fused raw + diff input
- `a1`: `research-extension` remove gc-ms pretraining from `e1`
- `a2`: `research-extension` diff-only input
- `a3`: `research-extension` raw-only input
- `a4`: `research-extension` remove fft/hardening from the best `e1` variant if hardening is used
- `e2`: `research-extension` fit-gcms + approved hardening knobs
- `f2`: optional two-model ensemble only if single-model runs stall below target
- `r1`: file-heldout robustness proxy on best main model
- `r2`: file-heldout robustness proxy on baseline/fallback
- `e3`: `research-extension` full-budget rerun of the chosen best single model with a second seed

## success and pivot thresholds

- compatibility gate:
  - `e0` must preserve the upstream trend that `g=25` materially beats `g=0`
  - `e0` should land within 5 acc@1 points of the upstream repo's best `w=100`, `g=25`, transformer retrieval result in the same evaluation mode
  - if not, stop and audit split, preprocessing, class vocab, and metric parity
- main-model gate:
  - `e1` must beat `e0` or `f1` by at least 1.0 acc@1, or show a clear robustness win
  - if not, pivot effort to `f1` plus ablations
- gc-ms gate:
  - if gc-ms pretraining adds <= 0.5 acc@1 and no robustness gain, keep it as an ablation result, not the default
- ensemble gate:
  - use `f2` only after single-model tickets are stable and still below the target
- paper-claim gate:
  - only claim "beat SCENTFORMER" when the evaluation mode matches the benchmark-compatible base setting

## ordered tickets

### t01 repo bootstrap

- depends on: phase 0 approval
- scope: add `src/smelt/`, `tests/`, `configs/`, `scripts/`, package metadata, `ruff`, `pytest`, and minimal cli wiring
- acceptance:
  - editable install works
  - `import smelt` works
  - lint/test/smoke commands exist
- validate:
  - `python -m pip install -e .[dev]`
  - `ruff check .`
  - `pytest -q`
  - `python -m compileall src tests scripts`
- blockers: missing python toolchain or resolver failures
- compatibility checkpoint: no benchmark logic yet
- leakage checkpoint: not applicable

### t02 data manifest and split auditor

- depends on: `t01`
- scope: implement an exact-upstream base-split auditor that verifies structure, class counts, file counts, csv readability, and schema consistency before loaders or preprocessing land
- acceptance:
  - emits a base manifest json with resolved root, split paths, class vocab, per-class file counts, per-file metadata, schema summary, and contract pass/fail state
  - confirms 50 classes plus 5 train csvs/class and 1 test csv/class in `--strict-upstream` mode
  - fails loudly on missing split dirs, class vocab mismatch, strict-count mismatch, unreadable csvs, or schema drift
  - does not implement gc-ms mapping logic in this ticket
- validate:
  - `python -m smelt.datasets.audit_base --data-root "$SMELT_DATA_ROOT" --emit artifacts/manifests/smellnet_base.json --strict-upstream`
  - `pytest -q`
- blockers: dataset missing or malformed
- compatibility checkpoint: split contract matches upstream exactly
- leakage checkpoint: not applicable in `t02` beyond split-structure verification

### t03 sensor loader and schema validator

- depends on: `t02`
- scope: implement deterministic loading for offline train/test splits with explicit raw 12-column validation and preserved split/class/file identity
- acceptance:
  - loader preserves file-level split
  - loader preserves the raw 12-column sensor schema by default
  - loader may expose validated benchmark-channel selection only as an explicit opt-in view, not as preprocessing
  - file ordering is deterministic
- validate:
  - `pytest -q`

## moonshot locked protocol

After `m01c`, the operative `moonshot-enhanced-setting` mainline is locked to:

- `all12`
- `diff-only`
- grouped-by-file validation inside the training split
- train-only standardization
- validation-locked file-level aggregator selection
- validation-only checkpoint selection under the locked aggregator

`m02` may change the backbone and the minimum recipe support needed to run it well, but it must not change this protocol.
  - `python -m smelt.datasets.audit_base --data-root "$SMELT_DATA_ROOT" --emit artifacts/manifests/smellnet_base.json --strict-upstream`
- blockers: inconsistent csv headers
- compatibility checkpoint: raw 12-column order and split/file identities match upstream
- leakage checkpoint: loader never mixes split roots

### t04 preprocessing parity: baseline subtraction, differencing, windowing, standardization

- depends on: `t03`
- scope: implement exact-compatible preprocess primitives and parity tests against upstream semantics
- acceptance:
  - subtract-first-row parity
  - `g=25` differencing parity
  - per-file windowing with `stride = window // 2` default
  - train-only scaler fit path
- validate:
  - `pytest -q tests/preprocessing/test_baseline_subtraction.py tests/preprocessing/test_differencing.py tests/preprocessing/test_windowing.py tests/preprocessing/test_standardization.py`
- blockers: parity mismatch against audited upstream behavior
- compatibility checkpoint: preprocess outputs match documented upstream contract
- leakage checkpoint: scaler fit uses train windows only

### t05 metric and report parity

- depends on: `t04`
- scope: implement classification metrics and gc-ms retrieval metrics with upstream-compatible field names
- acceptance:
  - report includes `acc@1`, `acc@5`, macro precision/recall/f1, confusion matrix, per-category summary
  - retrieval evaluator supports closed-world gc-ms gallery scoring
  - result schemas and filenames make `exact-upstream` retrieval vs `research-extension` pretrain-finetune distinction explicit
- validate:
  - `pytest -q tests/evaluation/test_classification_metrics.py tests/evaluation/test_retrieval_metrics.py`
- blockers: metric mismatch or naming drift
- compatibility checkpoint: result schema is upstream-compatible
- leakage checkpoint: not applicable

### t06 gc-ms anchors, class vocab, and explicit mapping artifacts

- depends on: `t02`, `t05`
- scope: load gc-ms anchors with explicit class-name mapping instead of implicit row order, while preserving closed-world benchmark compatibility
- acceptance:
  - anchor table, class vocab, and split manifest align exactly
  - saves `artifacts/manifests/gcms_class_map.json`
  - code fails loudly on any missing or extra class
  - code fails loudly on duplicate anchors or ambiguous mappings
  - mapping is bijective for the classes in scope
- validate:
  - `pytest -q tests/datasets/test_gcms_mapping.py`
  - `python -m smelt.datasets.audit_base --data-root "$SMELT_DATA_ROOT" --emit artifacts/manifests/smellnet_base.json`
  - `python -m smelt.datasets.audit_base --data-root "$SMELT_DATA_ROOT" --emit artifacts/manifests/gcms_class_map.json --verify-gcms-bijection`
- blockers: class-name mismatch between sensor split and gc-ms csv
- compatibility checkpoint: closed-world gallery matches the 50-class benchmark
- leakage checkpoint: mapping is explicit and auditable

### t07 exact upstream-compatible transformer sanity baseline

- depends on: `t04`, `t05`, `t06`
- scope: reproduce the official transformer baseline path for `w=100`, `g in {0,25}`, retrieval on/off
- acceptance:
  - config-driven run works end to end
  - `g=25` beats `g=0`
  - result tables include upstream-compatible metrics
  - result names make retrieval mode explicit
- validate:
  - `python -m smelt.training.run --config configs/exact-upstream/e0_transformer_upstream.yaml`
  - `python -m smelt.training.run --config configs/exact-upstream/e0b_transformer_upstream_no_gcms.yaml`
  - `python -m smelt.evaluation.compare --inputs results/exact-upstream/runs --out results/exact-upstream/tables/e0_transformer.csv`
- blockers: parity gap > 5 acc@1 from the audited upstream reference mode, or unstable training/eval pipeline
- compatibility checkpoint: exact-compatible mode confirmed
- leakage checkpoint: no validation/test reuse

### t08 exact upstream-compatible cnn fallback

- depends on: `t07`
- scope: add the benchmark-compatible cnn fallback on the same pipeline
- acceptance:
  - cnn runs on exact-compatible retrieval path
  - fallback result is reproducible and comparable to `e0`
- validate:
  - `python -m smelt.training.run --config configs/exact-upstream/f1_cnn_upstream.yaml`
  - `python -m smelt.evaluation.compare --inputs results/exact-upstream/runs --out results/exact-upstream/tables/f1_cnn.csv`
- blockers: cnn path diverges from shared data/eval contract
- compatibility checkpoint: same preprocessing and metrics as `e0`
- leakage checkpoint: inherited from previous tickets

### t09 fused raw + diff dataset path

- depends on: `t04`, `t06`
- scope: add a `research-extension`-only input path that aligns raw and `g=25` diff windows and concatenates them by channel
- acceptance:
  - raw+diff fusion is explicit and tested
  - `a2` diff-only and `a3` raw-only reuse the same data contract
- validate:
  - `pytest -q tests/datasets/test_fused_raw_diff.py`
- blockers: shape mismatch or ambiguous alignment
- compatibility checkpoint: marked as a documented deviation from upstream input semantics
- leakage checkpoint: fusion happens within split only

### t10 inception-style backbone

- depends on: `t09`
- scope: implement the `research-extension` fit-gcms backbone with `forward_features()` and shape checks
- acceptance:
  - backbone trains on a smoke batch
  - feature extractor output shape is stable
- validate:
  - `pytest -q tests/models/test_inception_backbone.py`
  - `python -m smelt.training.smoke --config configs/research-extension/e1_fit_gcms.yaml`
- blockers: unstable shapes or failing smoke step
- compatibility checkpoint: model is a new research path, not the exact-compatible baseline
- leakage checkpoint: not applicable

### t11 gc-ms pretrain + supervised fine-tune path

- depends on: `t10`, `t06`
- scope: implement the `research-extension` path requested by the plan: gc-ms contrastive pretraining followed by supervised fine-tuning
- acceptance:
  - pretrain and fine-tune stages are separate, resumable, and logged
  - `a1` no-pretrain ablation is supported by config only
- validate:
  - `python -m smelt.training.run --config configs/research-extension/e1_fit_gcms.yaml`
  - `python -m smelt.training.run --config configs/research-extension/a1_fit_no_pretrain.yaml`
- blockers: pretrain artifacts cannot be loaded cleanly into fine-tuning
- compatibility checkpoint: path is clearly labeled as a research deviation from upstream retrieval mode
- leakage checkpoint: no test labels or anchors are used for tuning beyond the closed-world benchmark contract

### t12 hardening, ablations, and file-heldout robustness proxy

- depends on: `t11`, `t08`
- scope: add fft/noise/feature-dropout switches, run `a4`, and define the closest valid file-heldout robustness proxy
- acceptance:
  - hardening knobs are binary and isolated
  - robustness protocol is documented and reproducible
  - `r1` and `r2` can be run without changing the official test split
- validate:
  - `python -m smelt.training.run --config configs/research-extension/e2_fit_gcms_hardened.yaml`
  - `python -m smelt.training.run --config configs/research-extension/a4_no_hardening.yaml`
  - `python -m smelt.evaluation.robustness --config configs/research-extension/r1_best_model.yaml`
  - `python -m smelt.evaluation.robustness --config configs/exact-upstream/r2_baseline.yaml`
- blockers: no defensible file-heldout grouping or hardening causes benchmark drift
- compatibility checkpoint: official split metrics remain the primary report
- leakage checkpoint: robustness grouping never reuses official test data for tuning
