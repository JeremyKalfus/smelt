# AGENTS.md

## purpose

`smelt` is a focused research repo for benchmark-compatible `SMELLNET-BASE` experiments. v0 is for reproducible data handling, one exact-compatible sanity path, one ambitious main model, one fallback model, disciplined ablations, and paper-ready outputs.

## repo layout

- `.reference/`: read-only upstream references and external audits, never the main code path
- `docs/specs/smellnetplan.md`: seed research direction from the user
- `docs/exact-upstream/`: benchmark-compatible notes, parity reports, and sanity-run docs
- `docs/research-extension/`: research-only notes, deviation docs, and ablation docs
- `PLANS.md`: ordered execution plan for long multi-step work; update it before major work starts
- `configs/exact-upstream/`: exact-compatible configs only
- `configs/research-extension/`: research-only configs only
- `scripts/`: thin entry scripts only
- `src/smelt/`: main package
- `src/smelt/datasets/`: dataset contracts, manifests, loaders
- `src/smelt/preprocessing/`: baseline subtraction, differencing, windowing, normalization
- `src/smelt/models/`: benchmark baselines and research models
- `src/smelt/training/`: training loops, losses, checkpointing
- `src/smelt/evaluation/`: benchmark metrics, reports, exports
- `src/smelt/utils/`: shared helpers only
- `tests/`: unit and smoke tests
- `experiments/`: tracked experiment manifests and run notes
- `results/exact-upstream/`: benchmark-compatible summaries and tables
- `results/research-extension/`: research-only summaries and tables

## build, test, and lint commands

Use these commands as the default repo contract once the tooling ticket lands:

- install: `python -m pip install -e .[dev]`
- lint: `ruff check .`
- format check: `ruff format --check .`
- tests: `pytest -q`
- smoke import/bytecode check: `python -m compileall src tests scripts`

If one of these commands is not available yet, fix the repo tooling first instead of silently swapping to an ad hoc flow.

## coding conventions

- write clear python with small, reviewable diffs
- keep scripts config-driven and thin
- add assertions and shape checks in critical paths
- prefer deterministic ordering for file discovery, class vocab, and manifests
- make benchmark assumptions explicit in code and config
- keep code comments lowercase and human-like
- keep commit messages lowercase, short, and human-like

## smelt v0 constraints

- benchmark target is `SMELLNET-BASE` first
- preserve an exact-compatible upstream sanity path before new model work
- main upside path is fused raw + `p=25` differencing with a strong cnn-style backbone plus gc-ms supervision
- keep one hardened cnn fallback path alive
- keep mixture codepaths out of the first implementation pass
- no dashboards, guis, notebooks-as-pipeline, or broad platform abstractions
- no silent benchmark changes

## benchmark compatibility rules

- preserve the official fixed file split: `offline_training` vs `offline_testing`
- split by recording file before any window generation
- keep exact-compatible and research-only configs, docs, and result tables in separate tracks
- default benchmark preprocessing mirrors upstream semantics:
  - subtract the first row per csv
  - drop `Benzene`, `Temperature`, `Pressure`, `Humidity`, `Gas_Resistance`, `Altitude`
  - apply differencing before windowing when enabled
  - default stride is `window_size // 2`
  - fit normalization on train windows only
- mirror upstream metric names and the closed-world gc-ms retrieval baseline before introducing new experimental modes
- make every gc-ms result name explicit about whether it is `exact-upstream` retrieval or `research-extension` pretrain/fine-tune
- if a research path deviates from upstream semantics, isolate it in config, name it clearly, and document the deviation in results

## leakage-prevention rules

- never random-split windows
- never fit scalers, thresholds, calibration, or early stopping on the official test split
- never rely on implicit gc-ms row order in new code; create explicit class vocab and anchor mappings
- require an auditable gc-ms map artifact such as `artifacts/manifests/gcms_class_map.json`
- fail hard on missing classes, duplicate anchors, ambiguous mappings, or any non-bijective class-anchor map in scope
- audit duplicate files and split overlap before training
- stop immediately on any split overlap, schema drift, missing class-anchor match, or metric/report mismatch

## workflow rules

- use `PLANS.md` for long multi-step work
- one ticket should be small enough to finish and verify in one loop
- keep changes small and test after each ticket
- finish and verify the exact-upstream sanity path before starting any research-extension ticket
- always use subagents liberally and keep context clean
- always check and self-test before moving on
- no silent fallbacks, ever. if something fails validation, stop and tell the user
- use the `find-skills` skill if you get stuck

## definition of done

A ticket is done only when:

- code, config, and tests are updated together
- the ticket's exact validation commands pass
- benchmark compatibility status is stated explicitly
- leakage audit checkpoint passes or is marked not applicable with a reason
- outputs and artifacts are reproducible from tracked commands
- there is no hidden fallback behavior or unresolved validation failure
