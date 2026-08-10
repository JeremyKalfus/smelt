"""compare the m05b (benchmark6) and m05 (all12) final official-test results.

produces the channel-attribution evidence for the write-up:

* file-level metrics for each protocol
* bootstrap confidence intervals over the 50 official test files
* a paired exact mcnemar test on the identical test files
* per-class agreement / disagreement breakdown
* a top-5 view, since top-1 alone is noisy at n=50

this script reads only saved per-file predictions. it trains nothing and it
never touches the official test split beyond the single evaluation each
protocol already performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from smelt.evaluation import write_dict_rows_csv, write_json

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 17


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("C:/smelt_runs"))
    parser.add_argument(
        "--m05-predictions",
        type=Path,
        default=Path(
            "results/runs/m05_grouped_cv_refit-20260420-153535-fbfed23f/final_test/"
            "diversity_greedy_probabilities/per_file_predictions.csv"
        ),
    )
    parser.add_argument("--table-root", type=Path, default=Path("results/tables"))
    parser.add_argument(
        "--output-prefix",
        default="m05b_vs_m05",
        help="table filename prefix, e.g. m05b_vs_baseline",
    )
    parser.add_argument(
        "--reference-label",
        default="m05",
        help="protocol_id recorded for the reference arm",
    )
    return parser.parse_args()


def load_predictions(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            # relative_path is the stable key; absolute_path differs per machine
            key = row["relative_path"]
            rows[key] = {
                "true": row["true_class"],
                "pred": row["predicted_class"],
                "top5": json.loads(row["top5_classes"]),
            }
    return rows


def find_latest_m05b_predictions(run_root: Path) -> Path | None:
    candidates = sorted(
        run_root.glob("m05b_grouped_cv_refit-*/final_test/*/per_file_predictions.csv")
    )
    return candidates[-1] if candidates else None


def macro_f1(records: list[dict[str, object]]) -> float:
    classes = sorted({str(r["true"]) for r in records})
    f1s = []
    for cls in classes:
        tp = sum(1 for r in records if r["true"] == cls and r["pred"] == cls)
        fp = sum(1 for r in records if r["true"] != cls and r["pred"] == cls)
        fn = sum(1 for r in records if r["true"] == cls and r["pred"] != cls)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return 100.0 * float(np.mean(f1s))


def bootstrap_ci(correct: np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = correct.size
    draws = rng.integers(0, n, size=(resamples, n))
    means = correct[draws].mean(axis=1) * 100.0
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def exact_mcnemar(b: int, c: int) -> float:
    """two-sided exact mcnemar p-value from the discordant pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return float(min(1.0, 2.0 * tail))


def main() -> int:
    args = parse_args()
    m05b_path = find_latest_m05b_predictions(args.run_root)
    if m05b_path is None:
        print("no m05b per-file predictions found yet; nothing to compare")
        return 1
    print(f"m05b predictions: {m05b_path}")
    print(f"m05  predictions: {args.m05_predictions}")

    m05b = load_predictions(m05b_path)
    m05 = load_predictions(args.m05_predictions)
    shared = sorted(set(m05b) & set(m05))
    if not shared:
        raise SystemExit("m05 and m05b prediction files share no relative_path keys")
    print(f"paired official-test files: {len(shared)} (m05b={len(m05b)}, m05={len(m05)})")

    b_records = [m05b[k] for k in shared]
    a_records = [m05[k] for k in shared]
    b_correct = np.asarray([r["true"] == r["pred"] for r in b_records], dtype=bool)
    a_correct = np.asarray([r["true"] == r["pred"] for r in a_records], dtype=bool)
    b_top5 = np.asarray([r["true"] in r["top5"] for r in b_records], dtype=bool)
    a_top5 = np.asarray([r["true"] in r["top5"] for r in a_records], dtype=bool)

    # discordant pairs: m05 right / m05b wrong, and the reverse
    only_m05 = int(np.sum(a_correct & ~b_correct))
    only_m05b = int(np.sum(b_correct & ~a_correct))
    p_value = exact_mcnemar(only_m05, only_m05b)

    b_lo, b_hi = bootstrap_ci(b_correct)
    a_lo, a_hi = bootstrap_ci(a_correct)
    delta = (a_correct.mean() - b_correct.mean()) * 100.0
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(shared), size=(BOOTSTRAP_RESAMPLES, len(shared)))
    delta_samples = (a_correct[draws].mean(axis=1) - b_correct[draws].mean(axis=1)) * 100.0
    delta_lo = float(np.percentile(delta_samples, 2.5))
    delta_hi = float(np.percentile(delta_samples, 97.5))

    rows = [
        {
            "protocol_id": "m05b",
            "channel_set": "benchmark6",
            "channel_count": "6",
            "file_acc@1": f"{100.0 * b_correct.mean():.2f}",
            "file_acc@1_ci_lower": f"{b_lo:.2f}",
            "file_acc@1_ci_upper": f"{b_hi:.2f}",
            "file_acc@5": f"{100.0 * b_top5.mean():.2f}",
            "file_macro_f1": f"{macro_f1(b_records):.2f}",
            "n_files": str(len(shared)),
        },
        {
            "protocol_id": args.reference_label,
            "channel_set": "reference",
            "channel_count": "reference",
            "file_acc@1": f"{100.0 * a_correct.mean():.2f}",
            "file_acc@1_ci_lower": f"{a_lo:.2f}",
            "file_acc@1_ci_upper": f"{a_hi:.2f}",
            "file_acc@5": f"{100.0 * a_top5.mean():.2f}",
            "file_macro_f1": f"{macro_f1(a_records):.2f}",
            "n_files": str(len(shared)),
        },
    ]

    comparison = {
        "n_paired_files": len(shared),
        "m05_all12_acc@1": float(100.0 * a_correct.mean()),
        "m05b_benchmark6_acc@1": float(100.0 * b_correct.mean()),
        "delta_acc@1_all12_minus_benchmark6": float(delta),
        "delta_acc@1_ci_lower": delta_lo,
        "delta_acc@1_ci_upper": delta_hi,
        "discordant_only_all12_correct": only_m05,
        "discordant_only_benchmark6_correct": only_m05b,
        "mcnemar_exact_two_sided_p": p_value,
        "both_correct": int(np.sum(a_correct & b_correct)),
        "both_wrong": int(np.sum(~a_correct & ~b_correct)),
        "interpretation": (
            "delta is the extra-channel contribution under an identical selection protocol; "
            "the mcnemar p-value tests whether the discordant pattern is distinguishable from "
            "chance on the 50 official test files"
        ),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }

    per_class = []
    for cls in sorted({str(r["true"]) for r in b_records}):
        idx = [i for i, r in enumerate(b_records) if r["true"] == cls]
        per_class.append(
            {
                "class_name": cls,
                "n_files": str(len(idx)),
                "benchmark6_correct": str(int(sum(b_correct[i] for i in idx))),
                "all12_correct": str(int(sum(a_correct[i] for i in idx))),
                "benchmark6_predicted": ";".join(str(b_records[i]["pred"]) for i in idx),
                "all12_predicted": ";".join(str(a_records[i]["pred"]) for i in idx),
            }
        )

    args.table_root.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    write_dict_rows_csv(args.table_root / f"{prefix}_comparison.csv", rows)
    write_json(
        args.table_root / f"{prefix}_comparison.json",
        {"rows": rows, "paired_comparison": comparison},
    )
    write_dict_rows_csv(args.table_root / f"{prefix}_per_class.csv", per_class)

    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
