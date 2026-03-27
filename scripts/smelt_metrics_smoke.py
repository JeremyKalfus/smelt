"""run a deterministic exact-upstream metric/report smoke example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smelt.evaluation import (
    compute_classification_metrics,
    export_classification_report,
    load_category_mapping,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--category-map",
        type=Path,
        default=Path("configs/exact-upstream/category_map.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    category_mapping = load_category_mapping(args.category_map)
    class_names = ("almond", "allspice", "angelica", "apple", "avocado")
    metrics = compute_classification_metrics(
        class_names=class_names,
        true_labels=[0, 1, 2, 3, 4],
        predicted_labels=[0, 0, 2, 3, 3],
        topk_indices=[
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [2, 1, 0, 3, 4],
            [3, 2, 1, 0, 4],
            [3, 4, 2, 1, 0],
        ],
        category_mapping={class_name: category_mapping[class_name] for class_name in class_names},
    )
    report_paths = export_classification_report(
        output_root=args.output_root,
        run_name="metric_smoke",
        metrics=metrics,
        methods_summary={
            "retained_columns": ["NO2", "C2H5OH", "VOC", "CO", "Alcohol", "LPG"],
            "differencing_period": 25,
            "window_size": 100,
            "stride": 50,
        },
    )
    print(f"acc@1: {metrics.acc_at_1}")
    print(f"acc@5: {metrics.acc_at_5}")
    print(f"precision_macro: {metrics.precision_macro}")
    print(f"recall_macro: {metrics.recall_macro}")
    print(f"f1_macro: {metrics.f1_macro}")
    print(f"per_category: {[row.to_dict() for row in metrics.per_category]}")
    print(json.dumps(report_paths.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
