"""evaluate frozen refit ensembles on the SmellNet online (real-time) split.

the online split (`data/online_nuts`, `data/online_spices` in the pre-2026-04-13
dataset revision) was recorded under real-time conditions and is the paper's
generalization benchmark: upstream models collapse there. our frozen m05b / m05
ensembles have never seen these files, and no selection decision is made on
them — each ensemble is evaluated exactly once with its locked members,
aggregators, method and weights from the refit plan.

classification stays 50-way: models predict over the full class vocabulary even
though the online recordings only cover a subset of classes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smelt.datasets import load_base_sensor_dataset
from smelt.datasets.base_loader import load_sensor_file
from smelt.evaluation import (
    build_file_score_bundle,
    load_category_mapping,
    write_dict_rows_csv,
    write_json,
)
from smelt.training.m04 import compute_file_result_from_score_bundle, evaluate_score_ensemble
from smelt.training.run import resolve_device
from smelt.training.run_moonshot import (
    build_moonshot_model,
    evaluate_moonshot_checkpoint,
    load_moonshot_run_config,
    prepare_moonshot_tensors_from_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="m05b/m05 protocol run dir")
    parser.add_argument("--protocol-id", required=True, help="e.g. m05b or m05")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--online-dirs",
        nargs="+",
        default=["online_nuts", "online_spices"],
        help="directories under data-root holding online recordings",
    )
    parser.add_argument("--category-map", type=Path, default=None)
    parser.add_argument("--table-root", type=Path, default=Path("results/tables"))
    return parser.parse_args()


def sanitize_online_csv(csv_path: Path) -> tuple[Path, int]:
    """forward-fill empty cells (dropped 1 Hz serial reads) into a temp copy.

    the online recordings contain a handful of empty cells (43 across ~171k in
    the full split). the strict loader rejects them by design; we fill each
    empty cell with the previous row's value and count every fill.
    """
    import csv as csv_module
    import tempfile

    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv_module.reader(handle))
    header, data = rows[0], rows[1:]
    fills = 0
    # pass 1: back-fill leading gaps from the first valid value in the column
    for j in range(len(header)):
        if data and data[0][j].strip() == "":
            replacement = next(
                (row[j] for row in data if row[j].strip() != ""),
                None,
            )
            if replacement is None:
                raise SystemExit(f"column {header[j]} entirely empty in {csv_path}")
            data[0][j] = replacement
            fills += 1
    # pass 2: forward-fill everything else
    previous = data[0]
    for row in data[1:]:
        for j, value in enumerate(row):
            if value.strip() == "":
                row[j] = previous[j]
                fills += 1
        previous = row
    if fills == 0:
        return csv_path, 0
    temp = Path(tempfile.mkdtemp()) / csv_path.name
    with open(temp, "w", newline="", encoding="utf-8") as handle:
        writer = csv_module.writer(handle)
        writer.writerow(header)
        writer.writerows(data)
    return temp, fills


def load_online_records(data_root: Path, online_dirs: list[str]) -> tuple[list, int]:
    records = []
    total_fills = 0
    for online_dir in online_dirs:
        split_dir = data_root / online_dir
        if not split_dir.is_dir():
            raise SystemExit(f"missing online split directory: {split_dir}")
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for csv_path in sorted(class_dir.glob("*.csv")):
                clean_path, fills = sanitize_online_csv(csv_path)
                total_fills += fills
                record = load_sensor_file(clean_path, clean_path.parent, online_dir, class_dir.name)
                # rebuild identity: single shared split name (the windowing layer
                # requires it); the true origin stays in relative_path
                record = type(record)(
                    split="online",
                    class_name=record.class_name,
                    relative_path=f"{online_dir}/{class_dir.name}/{csv_path.name}",
                    absolute_path=str(csv_path.resolve()),
                    column_names=record.column_names,
                    rows=record.rows,
                )
                records.append(record)
    return records, total_fills


def main() -> int:
    args = parse_args()
    import torch

    refit_plan_path = args.run_dir / f"{args.protocol_id}_refit_plan.json"
    refit_plan = json.loads(refit_plan_path.read_text(encoding="utf-8"))
    selection_path = args.run_dir / f"{args.protocol_id}_cv_ensemble_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_method = str(selection["selected_method"])
    mode = "logits" if selected_method == "mean_logits_all" else "probabilities"

    dataset = load_base_sensor_dataset(args.data_root)
    online_list, total_fills = load_online_records(args.data_root, args.online_dirs)
    online_records = tuple(online_list)
    known_classes = set(dataset.class_vocab)
    for record in online_records:
        if record.class_name not in known_classes:
            raise SystemExit(f"online class {record.class_name!r} not in base vocab")
    print(
        f"online recordings: {len(online_records)} across {args.online_dirs} "
        f"(forward-filled cells: {total_fills})"
    )

    category_map = args.category_map or Path("configs/exact-upstream/category_map.json")
    category_mapping = load_category_mapping(category_map)

    bundles = []
    weights = []
    member_rows = []
    for member in refit_plan["members"]:
        member_id = member["member_id"]
        config = load_moonshot_run_config(Path(member["config_path"]))
        refit_dirs = sorted((args.run_dir / "full_train_refit").glob(f"{member_id}_*full_refit-*"))
        if not refit_dirs:
            raise SystemExit(f"no refit run dir for member {member_id}")
        checkpoint_path = refit_dirs[-1] / "checkpoint_final.pt"
        if not checkpoint_path.is_file():
            raise SystemExit(f"missing checkpoint: {checkpoint_path}")

        prepared = prepare_moonshot_tensors_from_records(
            class_names=tuple(sorted(dataset.class_vocab)),
            resolved_data_root=dataset.resolved_data_root,
            train_records=dataset.train_records,
            validation_records=(),
            test_records=online_records,
            config=config,
            validation_files_per_class=0,
            view_manifest_updates={"protocol": args.protocol_id, "split_strategy": "online_eval"},
        )
        model, _summary = build_moonshot_model(
            config=config,
            input_dim=prepared.train_windows.shape[2],
            num_classes=len(prepared.class_names),
        )
        device = resolve_device(config.device)
        model = model.to(device)
        from smelt.training.run import build_dataloader
        from smelt.training.run_moonshot import require_split_array, require_standardized_split

        loader = build_dataloader(
            require_split_array(prepared.test_windows, split_name="test"),
            require_split_array(prepared.test_labels, split_name="test"),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        _ckpt, _evaluation, bundle = evaluate_moonshot_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            data_loader=loader,
            windows=require_standardized_split(
                prepared.standardized_test_split, split_name="test"
            ).windows,
            class_names=prepared.class_names,
            category_mapping=category_mapping,
            device=device,
        )
        aggregator = str(member["selected_aggregator"])
        file_bundle = build_file_score_bundle(bundle=bundle, aggregator=aggregator)
        bundles.append(file_bundle)
        weights.append(float(member["selected_weight"]))
        member_result = compute_file_result_from_score_bundle(
            file_bundle,
            category_mapping=category_mapping,
            aggregator_name=aggregator,
        )
        member_rows.append(
            {
                "member_id": member_id,
                "aggregator": aggregator,
                "online_file_acc@1": str(member_result.metrics.acc_at_1),
                "online_file_acc@5": str(member_result.metrics.acc_at_5),
                "online_file_macro_f1": str(member_result.metrics.f1_macro),
            }
        )
        print(
            f"  {member_id}: file acc@1 {member_result.metrics.acc_at_1:.1f} "
            f"acc@5 {member_result.metrics.acc_at_5:.1f}"
        )

    final_result = evaluate_score_ensemble(
        bundles=tuple(bundles),
        category_mapping=category_mapping,
        mode=mode,
        method_name=selected_method,
        weights=tuple(weights),
    )

    # per-file rows and per-online-split accuracy
    per_file = []
    split_totals: dict[str, list[int]] = {}
    reference = final_result.rows
    for row in reference:
        online_split = row.relative_path.split("/")[0]
        correct = int(row.true_class == row.predicted_class)
        split_totals.setdefault(online_split, []).append(correct)
        per_file.append(
            {
                "online_split": online_split,
                "relative_path": row.relative_path,
                "true_class": row.true_class,
                "predicted_class": row.predicted_class,
                "correct": str(correct),
                "top5": json.dumps(list(row.top5_classes)),
            }
        )
    summary_rows = [
        {
            "protocol_id": args.protocol_id,
            "scope": "all_online",
            "n_files": str(len(reference)),
            "file_acc@1": f"{final_result.metrics.acc_at_1:.2f}",
            "file_acc@5": f"{final_result.metrics.acc_at_5:.2f}",
            "file_macro_f1": f"{final_result.metrics.f1_macro:.2f}",
        }
    ]
    for online_split, outcomes in sorted(split_totals.items()):
        summary_rows.append(
            {
                "protocol_id": args.protocol_id,
                "scope": online_split,
                "n_files": str(len(outcomes)),
                "file_acc@1": f"{100.0 * sum(outcomes) / len(outcomes):.2f}",
                "file_acc@5": "",
                "file_macro_f1": "",
            }
        )

    prefix = f"{args.protocol_id}_online_eval"
    args.table_root.mkdir(parents=True, exist_ok=True)
    write_dict_rows_csv(args.table_root / f"{prefix}_summary.csv", summary_rows)
    write_dict_rows_csv(args.table_root / f"{prefix}_per_file.csv", per_file)
    write_dict_rows_csv(args.table_root / f"{prefix}_members.csv", member_rows)
    write_json(
        args.table_root / f"{prefix}_summary.json",
        {
            "protocol_id": args.protocol_id,
            "selected_method": selected_method,
            "classification": "50-way over the full base vocabulary",
            "selection_contamination": "none: members, aggregators, method and weights frozen "
            "before any online file was read",
            "forward_filled_cells": total_fills,
            "rows": summary_rows,
            "torch_version": torch.__version__,
        },
    )
    for row in summary_rows:
        print(f"{row['scope']}: n={row['n_files']} acc@1={row['file_acc@1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
