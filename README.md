# smelt

`smelt` is a focused Python research repo for `SMELLNET-BASE`.

Current status:

- bootstrap only
- no dataset logic yet
- no model logic yet
- no benchmark logic yet

Tracks:

- `exact-upstream`: benchmark-compatible configs, docs, and result tables
- `research-extension`: explicit deviations and research-only artifacts

Planned layout:

- `src/smelt/`: package code
- `configs/exact-upstream/`: exact-compatible configs
- `configs/research-extension/`: research-only configs
- `docs/specs/`: preserved source planning docs
- `docs/exact-upstream/`: benchmark parity notes
- `docs/research-extension/`: research notes
- `results/exact-upstream/`: benchmark-compatible outputs
- `results/research-extension/`: research outputs

Bootstrap commands:

```bash
python -m pip install -e .[dev]
ruff check .
ruff format --check .
pytest -q
python -m compileall src tests scripts
python -c "import smelt; print('ok')"
```
