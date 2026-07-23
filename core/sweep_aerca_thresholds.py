#!/usr/bin/env python3
"""
Sweep sparsified AERCA baselines over quantile thresholds.

Thresholds:
  0.05, 0.10, ..., 0.95 by default.

What it does:
  For each synthetic test suite:
    1. Load time_series_noisy.csv and ground_truth_graph.json.
    2. Split time series exactly like the project pipeline.
    3. Run AERCA once to get the dense discovery matrix.
    4. For every threshold, build a sparsified AERCA graph.
    5. Verify that graph using masked AERCA.
    6. Compute F1, Precision, Recall, BIC, MSE, and edge count.
  Then it aggregates results across all suites per threshold.

Important:
  The threshold here is a QUANTILE threshold passed to build_baseline_graph(..., q_threshold=t),
  not an absolute edge-weight cutoff. Academic landmine defused. Mostly.

Run from your project root, where these exist:
  data/synth_data/test_*/

Example:
  python -m core.sweep_aerca_thresholds \
    --data-dir data/synth_data \
    --workers 20 \
    --chunk-size 500 \
    --bic-lambda 65 \
    --target-edges 3.82 \
    --out-prefix output/aerca_threshold_sweep
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import glob
import json
import os
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.paths import DATA_DIR, OUTPUT_DIR


@dataclass
class ThresholdSuiteResult:
    suite: str
    threshold: float
    precision: float
    recall: float
    f1: float
    bic: float
    mse: float
    edges: int
    cutoff_value: float
    status: str = "ok"
    error: str = ""


def make_thresholds(start: float, stop: float, step: float) -> list[float]:
    """Create stable decimal thresholds: 0.05, 0.10, ..., 0.95."""
    vals: list[float] = []
    x = start
    # Add a tiny epsilon to keep floating point accumulation stable.
    while x <= stop + 1e-12:
        vals.append(round(x, 4))
        x += step
    return vals


def make_chunks(csv_path: Path, variables: list[str], chunk_size: int) -> tuple[np.ndarray, int, int]:
    """
    Load synthetic CSV and split it like run_synthetic_search.py, dropping incomplete chunks.

    Returns:
      xs: shape [num_chunks, chunk_size, num_variables]
      effective_samples: num_chunks * chunk_size
      raw_rows: number of rows in CSV before chunk dropping
    """
    df = pd.read_csv(csv_path)

    missing = [v for v in variables if v not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing behavior columns: {missing}")

    data_array = df[variables].to_numpy(dtype=np.float32) / 100.0
    raw_rows = len(data_array)

    n_full_chunks = raw_rows // chunk_size
    if n_full_chunks < 1:
        raise ValueError(f"Not enough rows for chunk_size={chunk_size}: got {raw_rows} rows")

    xs = np.array(
        [
            data_array[i * chunk_size : (i + 1) * chunk_size]
            for i in range(n_full_chunks)
        ],
        dtype=np.float32,
    )

    effective_samples = n_full_chunks * chunk_size
    return xs, effective_samples, raw_rows


def process_suite_sweep(
    suite_dir_str: str,
    thresholds: list[float],
    chunk_size: int,
    bic_lambda: float,
    force_cpu: bool,
    quiet: bool,
) -> list[ThresholdSuiteResult]:
    """
    Worker function.

    Runs dense AERCA once per suite, then evaluates all thresholds.
    This avoids doing the expensive dense extraction 19 times per suite.
    """
    suite_dir = Path(suite_dir_str)

    try:
        if force_cpu:
            # Must happen before torch import.
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # Limit per-process numerical-library threads when running many workers.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        os.environ.setdefault("WANDB_SILENT", "true")
        os.environ.setdefault("TQDM_DISABLE", "1")

        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        from core.aerca_verifier import extract_aerca_matrices, build_baseline_graph, run_masked_aerca
        from utils.helpers import calculate_bic, evaluate_dag

        gt_path = suite_dir / "ground_truth_graph.json"
        csv_path = suite_dir / "time_series_noisy.csv"

        if not gt_path.exists():
            raise FileNotFoundError(f"Missing {gt_path}")
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing {csv_path}")

        with gt_path.open("r", encoding="utf-8") as f:
            cfg: dict[str, Any] = json.load(f)

        variables = cfg["behaviors"]
        xs, effective_samples, raw_rows = make_chunks(csv_path, variables, chunk_size)

        if quiet:
            import contextlib

            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                est_matrix, signed_matrix, dense_weights = extract_aerca_matrices(xs, variables)
        else:
            est_matrix, signed_matrix, dense_weights = extract_aerca_matrices(xs, variables)

        results: list[ThresholdSuiteResult] = []

        for threshold in thresholds:
            try:
                if quiet:
                    import contextlib

                    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                        initial_edges = build_baseline_graph(
                            est_matrix,
                            signed_matrix,
                            variables,
                            q_threshold=threshold,
                        )
                        mse, num_edges, _weights, updated_edges = run_masked_aerca(
                            xs,
                            initial_edges,
                            variables,
                            init_weights=dense_weights,
                        )
                else:
                    initial_edges = build_baseline_graph(
                        est_matrix,
                        signed_matrix,
                        variables,
                        q_threshold=threshold,
                    )
                    mse, num_edges, _weights, updated_edges = run_masked_aerca(
                        xs,
                        initial_edges,
                        variables,
                        init_weights=dense_weights,
                    )

                bic = calculate_bic(
                    mse=mse,
                    num_edges=num_edges,
                    num_samples=effective_samples,
                    params_per_edge=bic_lambda,
                )
                precision, recall, f1, _gt_edges, _pred_edges = evaluate_dag(
                    updated_edges,
                    str(gt_path),
                    variables,
                )
                cutoff_value = float(np.quantile(est_matrix, threshold))

                results.append(
                    ThresholdSuiteResult(
                        suite=suite_dir.name,
                        threshold=float(threshold),
                        precision=float(precision),
                        recall=float(recall),
                        f1=float(f1),
                        bic=float(bic),
                        mse=float(mse),
                        edges=int(num_edges),
                        cutoff_value=cutoff_value,
                    )
                )

            except Exception as e:
                results.append(
                    ThresholdSuiteResult(
                        suite=suite_dir.name,
                        threshold=float(threshold),
                        precision=0.0,
                        recall=0.0,
                        f1=0.0,
                        bic=float("nan"),
                        mse=float("nan"),
                        edges=0,
                        cutoff_value=float("nan"),
                        status="error",
                        error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                    )
                )

        return results

    except Exception as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return [
            ThresholdSuiteResult(
                suite=suite_dir.name,
                threshold=float(t),
                precision=0.0,
                recall=0.0,
                f1=0.0,
                bic=float("nan"),
                mse=float("nan"),
                edges=0,
                cutoff_value=float("nan"),
                status="error",
                error=err,
            )
            for t in thresholds
        ]


def aggregate(results: list[ThresholdSuiteResult], thresholds: list[float], target_edges: float) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []

    for threshold in thresholds:
        rows = [r for r in results if abs(r.threshold - threshold) < 1e-9]
        ok = [r for r in rows if r.status == "ok"]

        if not ok:
            summaries.append(
                {
                    "threshold": threshold,
                    "n_success": 0,
                    "n_failed": len(rows),
                    "f1_mean": float("nan"),
                    "f1_std": float("nan"),
                    "precision_mean": float("nan"),
                    "recall_mean": float("nan"),
                    "bic_mean": float("nan"),
                    "mse_mean": float("nan"),
                    "edges_mean": float("nan"),
                    "edges_std": float("nan"),
                    "edge_distance_to_target": float("nan"),
                }
            )
            continue

        def vals(field: str) -> list[float]:
            return [float(getattr(r, field)) for r in ok]

        def mean(field: str) -> float:
            return float(np.mean(vals(field)))

        def std(field: str) -> float:
            v = vals(field)
            return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0

        edges_mean = mean("edges")
        summaries.append(
            {
                "threshold": threshold,
                "n_success": len(ok),
                "n_failed": len(rows) - len(ok),
                "f1_mean": mean("f1"),
                "f1_std": std("f1"),
                "precision_mean": mean("precision"),
                "recall_mean": mean("recall"),
                "bic_mean": mean("bic"),
                "mse_mean": mean("mse"),
                "edges_mean": edges_mean,
                "edges_std": std("edges"),
                "edge_distance_to_target": abs(edges_mean - target_edges),
            }
        )

    return summaries


def write_outputs(
    results: list[ThresholdSuiteResult],
    summaries: list[dict[str, Any]],
    out_prefix: str,
) -> None:
    per_suite_csv = Path(f"{out_prefix}_per_suite.csv")
    summary_csv = Path(f"{out_prefix}_summary.csv")
    summary_json = Path(f"{out_prefix}_summary.json")
    latex_path = Path(f"{out_prefix}_latex_rows.tex")

    with per_suite_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in sorted(results, key=lambda x: (x.threshold, x.suite)):
            writer.writerow(asdict(r))

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(summaries[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    with latex_path.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated AERCA threshold sweep rows\n")
        f.write("% Columns: Method & F1 & Std. F1 & Prec. & Recall & BIC & MSE & Edges \\\\\n")
        for s in summaries:
            if s["n_success"] == 0:
                continue
            f.write(
                f"AERCA $q={s['threshold']:.2f}$ "
                f"& {s['f1_mean']:.4f} "
                f"& {s['f1_std']:.4f} "
                f"& {s['precision_mean']:.4f} "
                f"& {s['recall_mean']:.4f} "
                f"& {s['bic_mean']:.4f} "
                f"& {s['mse_mean']:.4f} "
                f"& {s['edges_mean']:.2f} \\\\\n"
            )

    print(f"\nSaved per-suite results: {per_suite_csv}")
    print(f"Saved threshold summary: {summary_csv}")
    print(f"Saved summary JSON:      {summary_json}")
    print(f"Saved LaTeX rows:        {latex_path}")


def print_summary_table(summaries: list[dict[str, Any]], target_edges: float) -> None:
    print("\n" + "=" * 108)
    print("AERCA THRESHOLD SWEEP SUMMARY")
    print("=" * 108)
    print(
        f"{'thr':>5} | {'F1':>7} | {'Std':>7} | {'Prec':>7} | {'Recall':>7} | "
        f"{'BIC':>10} | {'MSE':>8} | {'Edges':>7} | {'|E-target|':>10}"
    )
    print("-" * 108)
    for s in summaries:
        if s["n_success"] == 0:
            print(f"{s['threshold']:>5.2f} | ERROR")
            continue
        print(
            f"{s['threshold']:>5.2f} | "
            f"{s['f1_mean']:>7.4f} | "
            f"{s['f1_std']:>7.4f} | "
            f"{s['precision_mean']:>7.4f} | "
            f"{s['recall_mean']:>7.4f} | "
            f"{s['bic_mean']:>10.4f} | "
            f"{s['mse_mean']:>8.4f} | "
            f"{s['edges_mean']:>7.2f} | "
            f"{s['edge_distance_to_target']:>10.2f}"
        )

    ok = [s for s in summaries if s["n_success"] > 0 and not np.isnan(s["f1_mean"])]
    if not ok:
        print("\nNo successful threshold summaries were produced.")
        return

    best_f1 = max(ok, key=lambda s: (s["f1_mean"], -s["edge_distance_to_target"]))
    closest_edges = min(ok, key=lambda s: (s["edge_distance_to_target"], -s["f1_mean"]))

    # Good practical choice: closest to target edges, but among near ties choose better F1.
    edge_tolerance = 0.5
    near_target = [s for s in ok if s["edge_distance_to_target"] <= edge_tolerance]
    best_near_target = max(near_target, key=lambda s: s["f1_mean"]) if near_target else None

    print("\n" + "*" * 108)
    print("SELECTION HELPERS")
    print("*" * 108)
    print(
        "Best F1: "
        f"q={best_f1['threshold']:.2f}, "
        f"F1={best_f1['f1_mean']:.4f}, "
        f"Std={best_f1['f1_std']:.4f}, "
        f"Prec={best_f1['precision_mean']:.4f}, "
        f"Recall={best_f1['recall_mean']:.4f}, "
        f"Edges={best_f1['edges_mean']:.2f}"
    )
    print(
        f"Closest edges to target {target_edges:.2f}: "
        f"q={closest_edges['threshold']:.2f}, "
        f"F1={closest_edges['f1_mean']:.4f}, "
        f"Std={closest_edges['f1_std']:.4f}, "
        f"Prec={closest_edges['precision_mean']:.4f}, "
        f"Recall={closest_edges['recall_mean']:.4f}, "
        f"Edges={closest_edges['edges_mean']:.2f}"
    )
    if best_near_target is not None:
        print(
            f"Best F1 within ±{edge_tolerance:.2f} edges of target: "
            f"q={best_near_target['threshold']:.2f}, "
            f"F1={best_near_target['f1_mean']:.4f}, "
            f"Std={best_near_target['f1_std']:.4f}, "
            f"Prec={best_near_target['precision_mean']:.4f}, "
            f"Recall={best_near_target['recall_mean']:.4f}, "
            f"Edges={best_near_target['edges_mean']:.2f}"
        )
    else:
        print(f"No threshold was within ±{edge_tolerance:.2f} edges of target.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR / "synth_data"), help="Folder containing test_*/ suites")
    parser.add_argument("--workers", type=int, default=20, help="Parallel worker processes")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size, matching run_synthetic_search.py unless you changed it")
    parser.add_argument("--bic-lambda", type=float, default=65.0, help="BIC params_per_edge for synthetic data")
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-stop", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--target-edges", type=float, default=3.82, help="Average edge count of LLM Beam, for density-matched selection")
    parser.add_argument("--out-prefix", default=str(OUTPUT_DIR / "aerca_threshold_sweep"))
    parser.add_argument("--no-force-cpu", action="store_true", help="Allow CUDA. Use a low worker count to avoid GPU memory exhaustion.")
    parser.add_argument("--verbose", action="store_true", help="Do not mute AERCA worker prints")
    args = parser.parse_args()

    thresholds = make_thresholds(args.threshold_start, args.threshold_stop, args.threshold_step)

    test_dirs = sorted(glob.glob(str(Path(args.data_dir) / "test_*")))
    if not test_dirs:
        print(f"No test suites found under {args.data_dir}/test_*", file=sys.stderr)
        return 2

    print("=" * 88)
    print("Running AERCA threshold sweep")
    print(f"Suites:       {len(test_dirs)}")
    print(f"Thresholds:   {thresholds}")
    print(f"Workers:      {args.workers}")
    print(f"Chunk size:   {args.chunk_size}")
    print(f"BIC lambda:   {args.bic_lambda}")
    print(f"Target edges: {args.target_edges}")
    print(f"Force CPU:    {not args.no_force_cpu}")
    print("=" * 88)

    all_results: list[ThresholdSuiteResult] = []

    with futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                process_suite_sweep,
                d,
                thresholds,
                args.chunk_size,
                args.bic_lambda,
                not args.no_force_cpu,
                not args.verbose,
            ): d
            for d in test_dirs
        }

        done = 0
        for fut in futures.as_completed(futs):
            suite_results = fut.result()
            all_results.extend(suite_results)
            done += 1

            ok = [r for r in suite_results if r.status == "ok"]
            if ok:
                best = max(ok, key=lambda r: r.f1)
                closest = min(ok, key=lambda r: abs(r.edges - args.target_edges))
                print(
                    f"[{done:03d}/{len(test_dirs):03d}] {Path(futs[fut]).name}: "
                    f"best q={best.threshold:.2f} F1={best.f1:.4f}; "
                    f"closest E q={closest.threshold:.2f} E={closest.edges}"
                )
            else:
                err = suite_results[0].error.splitlines()[0] if suite_results else "Unknown error"
                print(f"[{done:03d}/{len(test_dirs):03d}] {Path(futs[fut]).name}: ERROR — {err}")

    summaries = aggregate(all_results, thresholds, args.target_edges)
    write_outputs(all_results, summaries, args.out_prefix)
    print_summary_table(summaries, args.target_edges)

    failures = [r for r in all_results if r.status != "ok"]
    if failures:
        print(f"\nWarning: {len(failures)} suite-threshold runs failed. Check error column in per-suite CSV.")

    print("\nReminder: q is a quantile threshold, not absolute edge weight. Yes, that matters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
