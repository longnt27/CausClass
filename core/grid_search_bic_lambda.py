#!/usr/bin/env python3
"""
Grid search bic_penalty_lambda on the first N synthetic test suites.

Example:
    python -m core.grid_search_bic_lambda \
        --test_limit 10 \
        --lambdas 0,5,10,15,20,25,30,35,45,55,65 \
        --workers 10 \
        --output_dir output/lambda_grid_runs/first10

Notes:
- One process handles one synthetic test suite and sweeps all lambdas for that test.
  That avoids recomputing the expensive AERCA baseline for every lambda.
- If you only have one GPU and hit CUDA OOM, lower --workers to 1 or 2.
"""

import argparse
import copy
import concurrent.futures
import datetime as dt
import glob
import json
import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Reduce third-party library console output.
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pandas as pd

from core.aerca_verifier import get_initial_graph_from_aerca, run_masked_aerca
from utils.helpers import (
    apply_edit,
    calculate_bic,
    evaluate_dag,
    hash_graph,
    load_env_file,
)
from core.graph_edit_agent import LLMGraphAgent
from utils.paths import DATA_DIR, OUTPUT_DIR


def parse_lambdas(value: str) -> List[float]:
    vals = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise argparse.ArgumentTypeError("At least one lambda is required.")
    return vals


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_synthetic_suite(target_dir: str, chunk_size: int) -> Tuple[List[str], str, np.ndarray, str]:
    gt_file_path = os.path.join(target_dir, "ground_truth_graph.json")
    csv_file_path = os.path.join(target_dir, "time_series_noisy.csv")

    if not os.path.exists(gt_file_path):
        raise FileNotFoundError("Missing ground_truth_graph.json in %s" % target_dir)
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError("Missing time_series_noisy.csv in %s" % target_dir)

    with open(gt_file_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    variables = config["behaviors"]
    real_world_context = config.get("scenario_context", "Classroom dynamics study.")

    df = pd.read_csv(csv_file_path)
    data_array = df.values.astype(float) / 100.0

    n_chunks = len(data_array) // chunk_size
    if n_chunks <= 0:
        raise ValueError(
            "%s has only %d rows, smaller than chunk_size=%d" %
            (csv_file_path, len(data_array), chunk_size)
        )

    xs = np.array([
        data_array[i * chunk_size : (i + 1) * chunk_size]
        for i in range(n_chunks)
    ])

    return variables, real_world_context, xs, gt_file_path


def graph_edge_pairs(edges: Sequence[Dict[str, Any]]) -> List[str]:
    return sorted(["%s->%s" % (e.get("source"), e.get("target")) for e in edges])


def run_beam_for_lambda(
    *,
    lambda_value: float,
    xs: np.ndarray,
    variables: List[str],
    gt_file_path: str,
    agent: LLMGraphAgent,
    auto_initial_graph: List[Dict[str, Any]],
    dense_weights: Any,
    raw_discovery_matrix: np.ndarray,
    base_mse: float,
    base_edges: List[Dict[str, Any]],
    base_weights: Any,
    base_precision: float,
    base_recall: float,
    base_f1: float,
    chunk_size: int,
    beam_width: int,
    max_iterations: int,
    max_age: int,
    min_bic_delta_add: float,
    delete_guardrail_threshold: float,
    max_edits_per_llm_call: int,
) -> Dict[str, Any]:
    """Run the existing LLM-guided beam search for one lambda value."""
    num_samples = int(len(xs) * chunk_size)
    base_bic = calculate_bic(
        base_mse,
        len(base_edges),
        num_samples=num_samples,
        params_per_edge=lambda_value,
    )

    global_tabu = {hash_graph(base_edges)}
    beam = [{
        "graph": copy.deepcopy(base_edges),
        "mse": base_mse,
        "bic": base_bic,
        "weights": base_weights,
        "age": 0,
        "local_tabu": [],
        "f1": base_f1,
        "p": base_precision,
        "r": base_recall,
    }]

    log = {
        "lambda": lambda_value,
        "baseline": {
            "precision": base_precision,
            "recall": base_recall,
            "f1": base_f1,
            "bic": base_bic,
            "mse": base_mse,
            "edge_count": len(base_edges),
            "graph": base_edges,
        },
        "iterations": [],
        "final": None,
    }

    evaluated_candidates = 0
    accepted_candidates = 0

    for iteration in range(1, max_iterations + 1):
        iter_log: Dict[str, Any] = {
            "iteration": iteration,
            "proposals": [],
            "survivors": [],
        }
        candidates = []

        for tree in beam:
            tree["has_valid_offspring"] = False
            status = "OPTIMIZING" if tree["age"] == 0 else "STUCK (Age: %s)" % tree["age"]

            proposed_edits = agent.propose_edits(
                tree["graph"],
                tree["local_tabu"],
                status,
                raw_discovery_matrix,
            )[:max_edits_per_llm_call]

            for edit in proposed_edits:
                action = edit.get("action")
                src = edit.get("source")
                tgt = edit.get("target")
                reasoning = edit.get("reasoning", "N/A")

                proposal_record = {
                    "parent_hash": hash_graph(tree["graph"]),
                    "action": action,
                    "source": src,
                    "target": tgt,
                    "reasoning": reasoning,
                    "status": "",
                    "metrics": None,
                    "bic_improvement": 0.0,
                }

                if action not in {"add", "delete"}:
                    proposal_record["status"] = "INVALID_ACTION"
                    iter_log["proposals"].append(proposal_record)
                    continue

                if action == "delete":
                    try:
                        i = variables.index(src)
                        j = variables.index(tgt)
                        original_signal = abs(float(raw_discovery_matrix[i, j]))
                        if original_signal > delete_guardrail_threshold:
                            proposal_record["status"] = "BLOCKED_BY_GUARDRAIL"
                            iter_log["proposals"].append(proposal_record)
                            continue
                    except ValueError:
                        pass

                is_forward = any(e["source"] == src and e["target"] == tgt for e in tree["graph"])
                is_backward = any(e["source"] == tgt and e["target"] == src for e in tree["graph"])

                if action == "add" and (is_forward or is_backward):
                    proposal_record["status"] = "INVALID_TOPOLOGY_ALREADY_EXISTS"
                    iter_log["proposals"].append(proposal_record)
                    continue
                if action == "delete" and not is_forward:
                    proposal_record["status"] = "INVALID_TOPOLOGY_DOES_NOT_EXIST"
                    iter_log["proposals"].append(proposal_record)
                    continue

                new_graph = apply_edit(tree["graph"], edit)
                h_graph = hash_graph(new_graph)
                if h_graph in global_tabu:
                    proposal_record["status"] = "SKIPPED_GLOBAL_TABU"
                    iter_log["proposals"].append(proposal_record)
                    continue

                global_tabu.add(h_graph)
                proposal_record["status"] = "PENDING_EVALUATION"
                candidates.append({
                    "graph": new_graph,
                    "parent": tree,
                    "edit": edit,
                    "log_ref": proposal_record,
                })
                iter_log["proposals"].append(proposal_record)

        if not candidates:
            for tree in beam:
                tree["age"] += 1
        else:
            next_generation = []
            for c in candidates:
                evaluated_candidates += 1

                c_mse, c_num_e, c_weights, c_updated_graph = run_masked_aerca(
                    xs,
                    c["graph"],
                    variables,
                    init_weights=dense_weights,
                )
                c_bic = calculate_bic(
                    c_mse,
                    c_num_e,
                    num_samples,
                    params_per_edge=lambda_value,
                )
                c_p, c_r, c_f1, _, _ = evaluate_dag(c_updated_graph, gt_file_path, variables)
                bic_improvement = c["parent"]["bic"] - c_bic

                c["log_ref"]["metrics"] = {
                    "bic": c_bic,
                    "mse": c_mse,
                    "f1": c_f1,
                    "precision": c_p,
                    "recall": c_r,
                    "edge_count": c_num_e,
                }
                c["log_ref"]["bic_improvement"] = bic_improvement

                threshold_delta = min_bic_delta_add if c["edit"].get("action") == "add" else 0.0

                if bic_improvement >= threshold_delta:
                    accepted_candidates += 1
                    c["log_ref"]["status"] = "ACCEPTED"
                    next_generation.append({
                        "graph": c_updated_graph,
                        "mse": c_mse,
                        "bic": c_bic,
                        "weights": c_weights,
                        "age": 0,
                        "local_tabu": copy.deepcopy(c["parent"]["local_tabu"]),
                        "f1": c_f1,
                        "p": c_p,
                        "r": c_r,
                    })
                    c["parent"]["has_valid_offspring"] = True
                else:
                    c["log_ref"]["status"] = "REJECTED"
                    e_act = str(c["edit"].get("action", "?")).upper()
                    e_src = c["edit"].get("source", "?")
                    e_tgt = c["edit"].get("target", "?")
                    c["parent"]["local_tabu"].append("FAILED %s: %s->%s" % (e_act, e_src, e_tgt))

            for tree in beam:
                if not tree.get("has_valid_offspring", False):
                    tree["age"] += 1

            combined = [t for t in (beam + next_generation) if t["age"] < max_age]
            if not combined:
                log["iterations"].append(iter_log)
                break

            combined.sort(key=lambda x: x["bic"])
            beam = combined[:beam_width]

        for survivor in beam:
            iter_log["survivors"].append({
                "hash": hash_graph(survivor["graph"]),
                "f1": survivor["f1"],
                "precision": survivor["p"],
                "recall": survivor["r"],
                "bic": survivor["bic"],
                "mse": survivor["mse"],
                "edge_count": len(survivor["graph"]),
                "age": survivor["age"],
            })

        log["iterations"].append(iter_log)

    best_final = beam[0]
    log["final"] = {
        "precision": best_final["p"],
        "recall": best_final["r"],
        "f1": best_final["f1"],
        "bic": best_final["bic"],
        "mse": best_final["mse"],
        "edge_count": len(best_final["graph"]),
        "graph": best_final["graph"],
        "graph_pairs": graph_edge_pairs(best_final["graph"]),
        "evaluated_candidates": evaluated_candidates,
        "accepted_candidates": accepted_candidates,
    }

    return log


def process_one_test_suite(
    target_dir: str,
    lambdas: List[float],
    output_dir: str,
    api_key: str,
    chunk_size: int,
    default_threshold: float,
    beam_width: int,
    max_iterations: int,
    max_age: int,
    min_bic_delta_add: float,
    delete_guardrail_threshold: float,
    max_edits_per_llm_call: int,
) -> List[Dict[str, Any]]:
    suite_name = os.path.basename(target_dir.rstrip("/"))
    suite_out = Path(output_dir) / suite_name
    suite_out.mkdir(parents=True, exist_ok=True)

    print("[%s] Loading synthetic suite..." % suite_name, flush=True)
    variables, context, xs, gt_file_path = load_synthetic_suite(target_dir, chunk_size)

    print("[%s] Phase 0 AERCA baseline once for all lambdas..." % suite_name, flush=True)
    auto_initial_graph, dense_weights, raw_discovery_matrix = get_initial_graph_from_aerca(
        xs,
        variables,
        default_threshold=default_threshold,
    )

    base_mse, _, base_weights, base_edges = run_masked_aerca(
        xs,
        auto_initial_graph,
        variables,
        init_weights=dense_weights,
    )
    base_p, base_r, base_f1, _, _ = evaluate_dag(base_edges, gt_file_path, variables)

    agent = LLMGraphAgent(api_key, variables, context)
    rows = []

    for lambda_value in lambdas:
        print("[%s] Lambda %.4g started..." % (suite_name, lambda_value), flush=True)
        started = dt.datetime.now()
        try:
            log = run_beam_for_lambda(
                lambda_value=lambda_value,
                xs=xs,
                variables=variables,
                gt_file_path=gt_file_path,
                agent=agent,
                auto_initial_graph=auto_initial_graph,
                dense_weights=dense_weights,
                raw_discovery_matrix=raw_discovery_matrix,
                base_mse=base_mse,
                base_edges=base_edges,
                base_weights=base_weights,
                base_precision=base_p,
                base_recall=base_r,
                base_f1=base_f1,
                chunk_size=chunk_size,
                beam_width=beam_width,
                max_iterations=max_iterations,
                max_age=max_age,
                min_bic_delta_add=min_bic_delta_add,
                delete_guardrail_threshold=delete_guardrail_threshold,
                max_edits_per_llm_call=max_edits_per_llm_call,
            )
            elapsed_sec = (dt.datetime.now() - started).total_seconds()
            log["metadata"] = {
                "suite": suite_name,
                "target_dir": target_dir,
                "elapsed_sec": elapsed_sec,
                "variables": variables,
                "chunk_size": chunk_size,
                "default_threshold": default_threshold,
                "beam_width": beam_width,
                "max_iterations": max_iterations,
                "max_age": max_age,
                "min_bic_delta_add": min_bic_delta_add,
                "delete_guardrail_threshold": delete_guardrail_threshold,
                "max_edits_per_llm_call": max_edits_per_llm_call,
            }

            log_path = suite_out / ("lambda_%s_log.json" % str(lambda_value).replace(".", "p"))
            json_dump(log_path, log)

            final = log["final"] or {}
            baseline = log["baseline"] or {}
            row = {
                "suite": suite_name,
                "target_dir": target_dir,
                "lambda": lambda_value,
                "status": "ok",
                "elapsed_sec": elapsed_sec,
                "base_precision": baseline.get("precision"),
                "base_recall": baseline.get("recall"),
                "base_f1": baseline.get("f1"),
                "base_bic": baseline.get("bic"),
                "base_mse": baseline.get("mse"),
                "base_edges": baseline.get("edge_count"),
                "final_precision": final.get("precision"),
                "final_recall": final.get("recall"),
                "final_f1": final.get("f1"),
                "final_bic": final.get("bic"),
                "final_mse": final.get("mse"),
                "final_edges": final.get("edge_count"),
                "evaluated_candidates": final.get("evaluated_candidates"),
                "accepted_candidates": final.get("accepted_candidates"),
                "log_path": str(log_path),
            }
            rows.append(row)
            print(
                "[%s] Lambda %.4g done: F1=%.4f, P=%.4f, R=%.4f, edges=%s, BIC=%.3f" % (
                    suite_name,
                    lambda_value,
                    float(row["final_f1"] or 0.0),
                    float(row["final_precision"] or 0.0),
                    float(row["final_recall"] or 0.0),
                    row["final_edges"],
                    float(row["final_bic"] or 0.0),
                ),
                flush=True,
            )
        except Exception as e:
            elapsed_sec = (dt.datetime.now() - started).total_seconds()
            err = {
                "suite": suite_name,
                "lambda": lambda_value,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            err_path = suite_out / ("lambda_%s_error.json" % str(lambda_value).replace(".", "p"))
            json_dump(err_path, err)
            rows.append({
                "suite": suite_name,
                "target_dir": target_dir,
                "lambda": lambda_value,
                "status": "error",
                "elapsed_sec": elapsed_sec,
                "error": str(e),
                "log_path": str(err_path),
            })
            print("[%s] Lambda %.4g ERROR: %s" % (suite_name, lambda_value, e), flush=True)

    pd.DataFrame(rows).to_csv(suite_out / "suite_summary.csv", index=False)
    return rows


def summarize_results(rows: List[Dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "all_results.csv", index=False)

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    summary = (
        ok.groupby("lambda")
        .agg(
            suites=("suite", "count"),
            mean_final_f1=("final_f1", "mean"),
            std_final_f1=("final_f1", "std"),
            mean_final_precision=("final_precision", "mean"),
            mean_final_recall=("final_recall", "mean"),
            mean_final_bic=("final_bic", "mean"),
            mean_final_mse=("final_mse", "mean"),
            mean_final_edges=("final_edges", "mean"),
            mean_base_f1=("base_f1", "mean"),
            mean_base_edges=("base_edges", "mean"),
            total_evaluated_candidates=("evaluated_candidates", "sum"),
            total_accepted_candidates=("accepted_candidates", "sum"),
            total_elapsed_sec=("elapsed_sec", "sum"),
        )
        .reset_index()
    )

    summary["mean_f1_gain_vs_base"] = summary["mean_final_f1"] - summary["mean_base_f1"]
    summary = summary.sort_values(
        ["mean_final_f1", "mean_final_edges", "mean_final_bic"],
        ascending=[False, True, True],
    )
    summary.to_csv(output_dir / "lambda_summary.csv", index=False)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Grid search bic_penalty_lambda on synthetic CausClass tests.")
    parser.add_argument("--test_glob", default=str(DATA_DIR / "synth_data" / "test_*"), help="Glob for synthetic test dirs.")
    parser.add_argument("--test_limit", type=int, default=10, help="Use the first N sorted test dirs.")
    parser.add_argument("--lambdas", type=parse_lambdas, default=parse_lambdas("0,5,10,15,20,25,30,35,45,55,65"))
    parser.add_argument("--workers", type=int, default=10, help="Parallel test-suite workers. Lower this if CUDA OOMs.")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR / "lambda_grid_runs" / "latest"))
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--default_threshold", type=float, default=0.4, help="Initial verifier threshold, same as run_synthetic_search.py.")
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--max_iterations", type=int, default=30)
    parser.add_argument("--max_age", type=int, default=4)
    parser.add_argument("--min_bic_delta_add", type=float, default=-0.15, help="Same add leniency as run_synthetic_search.py.")
    parser.add_argument("--delete_guardrail_threshold", type=float, default=0.75)
    parser.add_argument("--max_edits_per_llm_call", type=int, default=3)
    parser.add_argument("--debug_one", action="store_true", help="Run only one suite and one worker. Sanity check mode.")
    args = parser.parse_args()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    load_env_file()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("[CRITICAL] DEEPSEEK_API_KEY is missing from environment variables.")
        return 1

    test_dirs = sorted(glob.glob(args.test_glob))[: args.test_limit]
    if args.debug_one:
        test_dirs = test_dirs[:1]
        args.workers = 1

    if not test_dirs:
        print("[CRITICAL] No synthetic test dirs matched: %s" % args.test_glob)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["lambdas"] = list(args.lambdas)
    config["test_dirs"] = test_dirs
    config["started_at"] = dt.datetime.now().isoformat()
    json_dump(output_dir / "grid_config.json", config)

    print("=" * 80)
    print("Lambda grid search")
    print("Tests   : %d" % len(test_dirs))
    print("Lambdas : %s" % ", ".join([str(x) for x in args.lambdas]))
    print("Workers : %d" % args.workers)
    print("Output  : %s" % output_dir)
    print("=" * 80)
    print("One worker sweeps all lambdas for one test. Reduce --workers if CUDA memory is limited.")

    all_rows: List[Dict[str, Any]] = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_one_test_suite,
                target_dir,
                args.lambdas,
                str(output_dir),
                api_key,
                args.chunk_size,
                args.default_threshold,
                args.beam_width,
                args.max_iterations,
                args.max_age,
                args.min_bic_delta_add,
                args.delete_guardrail_threshold,
                args.max_edits_per_llm_call,
            ): target_dir
            for target_dir in test_dirs
        }

        for future in concurrent.futures.as_completed(futures):
            target_dir = futures[future]
            try:
                rows = future.result()
                all_rows.extend(rows)
                summarize_results(all_rows, output_dir)
                print("[DONE] %s" % target_dir, flush=True)
            except Exception as e:
                print("[FATAL] %s failed outside lambda loop: %s" % (target_dir, e), flush=True)
                traceback.print_exc()
                all_rows.append({
                    "suite": os.path.basename(target_dir.rstrip("/")),
                    "target_dir": target_dir,
                    "lambda": None,
                    "status": "fatal",
                    "error": str(e),
                })
                summarize_results(all_rows, output_dir)

    summary = summarize_results(all_rows, output_dir)
    if summary.empty:
        print("\n[ERROR] No successful lambda runs. Check *_error.json files.")
        return 1

    print("\n" + "*" * 80)
    print("GRID SEARCH SUMMARY")
    print("*" * 80)
    cols = [
        "lambda",
        "suites",
        "mean_final_f1",
        "std_final_f1",
        "mean_f1_gain_vs_base",
        "mean_final_precision",
        "mean_final_recall",
        "mean_final_edges",
        "mean_final_bic",
    ]
    printable = summary[cols].copy()
    with pd.option_context("display.max_rows", 200, "display.width", 180):
        print(printable.to_string(index=False))

    best = summary.iloc[0]
    print("\nRecommended lambda by mean synthetic F1: %.4g" % best["lambda"])
    print("Mean F1: %.4f | Mean edges: %.2f | Mean BIC: %.3f" % (
        best["mean_final_f1"],
        best["mean_final_edges"],
        best["mean_final_bic"],
    ))
    print("Saved:")
    print("- %s" % (output_dir / "all_results.csv"))
    print("- %s" % (output_dir / "lambda_summary.csv"))
    print("- per-suite/per-lambda logs under %s" % output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
