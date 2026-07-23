
import os
import sys

# ==========================================
# SILENCE ALL THIRD-PARTY LIBRARIES
# ==========================================
os.environ["WANDB_SILENT"] = "true"
os.environ["TQDM_DISABLE"] = "1"

import argparse
import concurrent.futures
import copy
import datetime
import glob
import json
import multiprocessing as mp
import random
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.helpers import load_env_file, calculate_bic, evaluate_dag, hash_graph, apply_edit
from core.graph_edit_agent import LLMGraphAgent
from core.aerca_verifier import get_initial_graph_from_aerca, run_masked_aerca
from utils.paths import DATA_DIR, OUTPUT_DIR


Edge = Dict[str, Any]
Edit = Dict[str, Any]
Tree = Dict[str, Any]


def count_samples(xs: np.ndarray, fallback_chunk_size: int) -> int:
    if xs is None or len(xs) == 0:
        return fallback_chunk_size
    return int(len(xs) * xs.shape[1]) if len(xs.shape) >= 2 else int(len(xs) * fallback_chunk_size)


def metrics_record(method: str, target_dir: str, p: float, r: float, f1: float,
                   bic: float, mse: float, edge_count: int, graph: List[Edge],
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rec = {
        "test_suite": target_dir,
        "method": method,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "bic": float(bic),
        "mse": float(mse),
        "edge_count": int(edge_count),
        "graph": graph,
    }
    if extra:
        rec.update(extra)
    return rec


def edge_exists(graph: List[Edge], src: str, tgt: str) -> bool:
    return any(e.get("source") == src and e.get("target") == tgt for e in graph)


def reverse_edge_exists(graph: List[Edge], src: str, tgt: str) -> bool:
    return any(e.get("source") == tgt and e.get("target") == src for e in graph)


def add_failed_tabu(local_tabu: List[str], edit: Edit) -> None:
    action = str(edit.get("action", "?")).upper()
    src = str(edit.get("source", "?"))
    tgt = str(edit.get("target", "?"))
    item = f"FAILED {action}: {src}->{tgt}"
    if item not in local_tabu:
        local_tabu.append(item)


def generate_random_edits(current_edges: List[Edge], variables: List[str],
                          rng: random.Random, num_edits: int = 3) -> List[Edit]:
    """Random graph-edit baseline. No semantic guidance, same AERCA verifier."""
    edits: List[Edit] = []
    used = set()
    existing = {(e.get("source"), e.get("target")) for e in current_edges}
    existing_undirected = set(existing) | {(b, a) for (a, b) in existing}
    all_pairs = [(s, t) for s in variables for t in variables if s != t]
    add_pairs = [(s, t) for (s, t) in all_pairs if (s, t) not in existing_undirected]
    delete_pairs = list(existing)

    for _ in range(num_edits * 10):
        if len(edits) >= num_edits:
            break
        if add_pairs and delete_pairs:
            action = "add" if rng.random() < 0.5 else "delete"
        elif add_pairs:
            action = "add"
        elif delete_pairs:
            action = "delete"
        else:
            break
        src, tgt = rng.choice(add_pairs if action == "add" else delete_pairs)
        key = (action, src, tgt)
        if key in used:
            continue
        used.add(key)
        edits.append({
            "action": action,
            "source": src,
            "target": tgt,
            "reasoning": "Random edit baseline: no semantic guidance.",
        })
    return edits


def is_strong_delete_blocked(edit: Edit, variables: List[str], raw_discovery_matrix: np.ndarray,
                             guardrail_delete_threshold: float) -> bool:
    if edit.get("action") != "delete":
        return False
    try:
        i = variables.index(edit.get("source"))
        j = variables.index(edit.get("target"))
        return abs(float(raw_discovery_matrix[i, j])) > guardrail_delete_threshold
    except Exception:
        return False


def validate_edit_topology(edit: Edit, graph: List[Edge], variables: List[str]) -> Tuple[bool, str]:
    action = edit.get("action")
    src = edit.get("source")
    tgt = edit.get("target")
    if action not in {"add", "delete"}:
        return False, "INVALID_ACTION"
    if src not in variables or tgt not in variables or src == tgt:
        return False, "INVALID_NODES"
    is_forward = edge_exists(graph, src, tgt)
    is_backward = reverse_edge_exists(graph, src, tgt)
    if action == "add" and (is_forward or is_backward):
        return False, "INVALID_TOPOLOGY_ALREADY_EXISTS_OR_REVERSE_EXISTS"
    if action == "delete" and not is_forward:
        return False, "INVALID_TOPOLOGY_DOES_NOT_EXIST"
    return True, "OK"


def evaluate_one_candidate(xs: np.ndarray, variables: List[str], gt_file_path: str,
                           parent: Tree, edit: Edit, dense_weights: Any,
                           raw_discovery_matrix: np.ndarray, global_tabu: set,
                           num_samples: int, bic_penalty_lambda: float,
                           min_bic_delta: float, guardrail_delete_threshold: float) -> Tuple[Optional[Tree], Dict[str, Any]]:
    action = edit.get("action")
    src = edit.get("source")
    tgt = edit.get("target")
    log_ref: Dict[str, Any] = {
        "parent_hash": hash_graph(parent["graph"]),
        "action": action,
        "source": src,
        "target": tgt,
        "reasoning": edit.get("reasoning", "N/A"),
        "status": "",
        "metrics": None,
        "improvement": 0.0,
    }

    ok, reason = validate_edit_topology(edit, parent["graph"], variables)
    if not ok:
        log_ref["status"] = reason
        return None, log_ref

    if is_strong_delete_blocked(edit, variables, raw_discovery_matrix, guardrail_delete_threshold):
        log_ref["status"] = "BLOCKED_BY_RAW_SIGNAL_GUARDRAIL"
        return None, log_ref

    new_graph = apply_edit(parent["graph"], edit)
    h_graph = hash_graph(new_graph)
    if h_graph in global_tabu:
        log_ref["status"] = "SKIPPED_GLOBAL_TABU"
        return None, log_ref
    global_tabu.add(h_graph)

    c_mse, c_num_e, c_weights, c_updated_graph = run_masked_aerca(
        xs, new_graph, variables, init_weights=dense_weights
    )
    c_bic = calculate_bic(c_mse, c_num_e, num_samples, bic_penalty_lambda)
    c_p, c_r, c_f1, _, _ = evaluate_dag(c_updated_graph, gt_file_path, variables)
    bic_improvement = float(parent["bic"] - c_bic)

    log_ref["metrics"] = {
        "bic": float(c_bic),
        "mse": float(c_mse),
        "f1": float(c_f1),
        "precision": float(c_p),
        "recall": float(c_r),
        "edge_count": int(c_num_e),
    }
    log_ref["improvement"] = bic_improvement

    threshold_delta = min_bic_delta if action == "add" else 0.0
    if bic_improvement >= threshold_delta:
        log_ref["status"] = "ACCEPTED"
        child: Tree = {
            "graph": c_updated_graph,
            "mse": float(c_mse),
            "bic": float(c_bic),
            "weights": c_weights,
            "age": 0,
            "local_tabu": copy.deepcopy(parent.get("local_tabu", [])),
            "f1": float(c_f1),
            "p": float(c_p),
            "r": float(c_r),
        }
        return child, log_ref

    log_ref["status"] = "REJECTED"
    return None, log_ref


def run_graph_search(method_name: str, xs: np.ndarray, variables: List[str], gt_file_path: str,
                     start_tree: Tree, dense_weights: Any, raw_discovery_matrix: np.ndarray,
                     num_samples: int, bic_penalty_lambda: float,
                     proposer: Callable[[List[Edge], List[str], str], List[Edit]],
                     search_mode: str = "beam", beam_width: int = 5, max_iterations: int = 30,
                     max_age: int = 4, min_bic_delta: float = -0.15,
                     guardrail_delete_threshold: float = 0.75) -> Tuple[Tree, Dict[str, Any]]:
    assert search_mode in {"beam", "greedy"}
    print(f"[{method_name}] Starting {search_mode} search...")

    global_tabu = {hash_graph(start_tree["graph"])}
    beam: List[Tree] = [copy.deepcopy(start_tree)]
    method_log: Dict[str, Any] = {"method": method_name, "search_mode": search_mode, "iterations": []}

    for iteration in range(1, max_iterations + 1):
        iter_log: Dict[str, Any] = {"iteration": iteration, "proposals_evaluated": [], "survivors": []}
        candidates: List[Tree] = []
        active_trees = beam if search_mode == "beam" else [beam[0]]

        for tree in active_trees:
            tree["has_valid_offspring"] = False
            status = "OPTIMIZING" if tree.get("age", 0) == 0 else f"STUCK (Age: {tree.get('age', 0)})"
            proposed_edits = proposer(tree["graph"], tree.get("local_tabu", []), status)

            for edit in proposed_edits:
                child, proposal_log = evaluate_one_candidate(
                    xs=xs,
                    variables=variables,
                    gt_file_path=gt_file_path,
                    parent=tree,
                    edit=edit,
                    dense_weights=dense_weights,
                    raw_discovery_matrix=raw_discovery_matrix,
                    global_tabu=global_tabu,
                    num_samples=num_samples,
                    bic_penalty_lambda=bic_penalty_lambda,
                    min_bic_delta=min_bic_delta,
                    guardrail_delete_threshold=guardrail_delete_threshold,
                )
                iter_log["proposals_evaluated"].append(proposal_log)
                if child is not None:
                    child["parent_hash"] = hash_graph(tree["graph"])
                    child["edit"] = edit
                    candidates.append(child)
                    tree["has_valid_offspring"] = True
                elif proposal_log.get("status") == "REJECTED":
                    add_failed_tabu(tree.setdefault("local_tabu", []), edit)

        if not candidates:
            for tree in beam:
                tree["age"] = int(tree.get("age", 0)) + 1
        else:
            if search_mode == "greedy":
                candidates.sort(key=lambda x: x["bic"])
                beam = [candidates[0]]
            else:
                for tree in beam:
                    if not tree.get("has_valid_offspring", False):
                        tree["age"] = int(tree.get("age", 0)) + 1
                combined = [t for t in (beam + candidates) if int(t.get("age", 0)) < max_age]
                if not combined:
                    print(f"[{method_name}] Stopped: no surviving trees.")
                    break
                combined.sort(key=lambda x: x["bic"])
                beam = combined[:beam_width]

        if search_mode == "greedy" and int(beam[0].get("age", 0)) >= max_age:
            print(f"[{method_name}] Stopped: greedy state exceeded max_age={max_age}.")
            break

        for survivor in beam:
            iter_log["survivors"].append({
                "hash": hash_graph(survivor["graph"]),
                "f1": float(survivor["f1"]),
                "precision": float(survivor["p"]),
                "recall": float(survivor["r"]),
                "bic": float(survivor["bic"]),
                "mse": float(survivor["mse"]),
                "edge_count": len(survivor["graph"]),
                "age": int(survivor.get("age", 0)),
            })
        method_log["iterations"].append(iter_log)

        best = beam[0]
        print(
            f"[{method_name}] Iter {iteration:02d}: best F1={best['f1']:.4f}, "
            f"BIC={best['bic']:.4f}, MSE={best['mse']:.6f}, edges={len(best['graph'])}, beam={len(beam)}"
        )

        if all(int(t.get("age", 0)) >= max_age for t in beam):
            print(f"[{method_name}] Stopped: all states exceeded max_age={max_age}.")
            break

    best_final = beam[0]
    method_log["final"] = {
        "f1": float(best_final["f1"]),
        "precision": float(best_final["p"]),
        "recall": float(best_final["r"]),
        "bic": float(best_final["bic"]),
        "mse": float(best_final["mse"]),
        "edge_count": len(best_final["graph"]),
        "graph": best_final["graph"],
    }
    return best_final, method_log


def process_test_suite(target_dir: str, batch_time_str: str, api_key: str,
                       config_args: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    gt_file_path = os.path.join(target_dir, "ground_truth_graph.json")
    csv_file_path = os.path.join(target_dir, "time_series_noisy.csv")

    if not os.path.exists(gt_file_path) or not os.path.exists(csv_file_path):
        print(f"[SKIP] Missing files in {target_dir}")
        return None

    suite_name = os.path.basename(target_dir.rstrip(os.sep))
    out_dir = os.path.join(config_args["output_root"], suite_name)
    os.makedirs(out_dir, exist_ok=True)
    log_output_path = os.path.join(out_dir, "ablation_search_log.json")

    json_log: Dict[str, Any] = {
        "metadata": {
            "test_suite": target_dir,
            "timestamp": datetime.datetime.now().isoformat(),
            "batch": batch_time_str,
            "config": config_args,
        },
        "baseline": {},
        "methods": {},
    }

    try:
        with open(gt_file_path, "r", encoding="utf-8") as f:
            suite_config = json.load(f)
        variables = suite_config["behaviors"]
        real_world_context = suite_config.get("scenario_context", "Classroom dynamics study.")

        df = pd.read_csv(csv_file_path)
        data_array = df.values / 100.0
        chunk_size = int(config_args["chunk_size"])
        xs = np.array([
            data_array[i * chunk_size: (i + 1) * chunk_size]
            for i in range(len(data_array) // chunk_size)
        ])
        if len(xs) == 0:
            raise ValueError(f"No complete chunks found in {csv_file_path}. Try smaller --chunk_size.")

        num_samples = count_samples(xs, chunk_size)
        agent = LLMGraphAgent(api_key, variables, real_world_context)

        print("\n" + "=" * 90)
        print(f"[SUITE] {target_dir} | variables={variables} | chunks={len(xs)} | samples={num_samples}")
        print("=" * 90)

        auto_initial_graph, dense_weights, raw_discovery_matrix = get_initial_graph_from_aerca(
            xs,
            variables,
            default_threshold=float(config_args["baseline_threshold"]),
        )

        base_mse, base_num_e, base_weights, base_edges = run_masked_aerca(
            xs,
            auto_initial_graph,
            variables,
            init_weights=dense_weights,
        )
        base_bic = calculate_bic(
            base_mse,
            base_num_e,
            num_samples=num_samples,
            params_per_edge=float(config_args["bic_penalty_lambda"]),
        )
        base_p, base_r, base_f1, _, _ = evaluate_dag(base_edges, gt_file_path, variables)

        start_tree: Tree = {
            "graph": base_edges,
            "mse": float(base_mse),
            "bic": float(base_bic),
            "weights": base_weights,
            "age": 0,
            "local_tabu": [],
            "f1": float(base_f1),
            "p": float(base_p),
            "r": float(base_r),
        }

        json_log["baseline"] = {
            "f1": float(base_f1),
            "precision": float(base_p),
            "recall": float(base_r),
            "bic": float(base_bic),
            "mse": float(base_mse),
            "edge_count": len(base_edges),
            "graph": base_edges,
        }

        records: List[Dict[str, Any]] = [
            metrics_record("aerca_baseline", target_dir, base_p, base_r, base_f1,
                           base_bic, base_mse, len(base_edges), base_edges)
        ]

        def llm_proposer(graph: List[Edge], local_tabu: List[str], status: str) -> List[Edit]:
            return agent.propose_edits(graph, local_tabu, status, raw_discovery_matrix)

        rng = random.Random(int(config_args["random_seed"]) + abs(hash(target_dir)) % 100000)

        def random_proposer(graph: List[Edge], local_tabu: List[str], status: str) -> List[Edit]:
            return generate_random_edits(
                graph,
                variables,
                rng=rng,
                num_edits=int(config_args["random_edits_per_state"]),
            )

        if not config_args.get("skip_random", False):
            random_best, random_log = run_graph_search(
                method_name="random_beam", xs=xs, variables=variables, gt_file_path=gt_file_path,
                start_tree=start_tree, dense_weights=dense_weights, raw_discovery_matrix=raw_discovery_matrix,
                num_samples=num_samples, bic_penalty_lambda=float(config_args["bic_penalty_lambda"]),
                proposer=random_proposer, search_mode="beam", beam_width=int(config_args["beam_width"]),
                max_iterations=int(config_args["max_iterations"]), max_age=int(config_args["max_age"]),
                min_bic_delta=float(config_args["min_bic_delta"]),
                guardrail_delete_threshold=float(config_args["guardrail_delete_threshold"]),
            )
            json_log["methods"]["random_beam"] = random_log
            records.append(metrics_record("random_beam", target_dir, random_best["p"], random_best["r"],
                                          random_best["f1"], random_best["bic"], random_best["mse"],
                                          len(random_best["graph"]), random_best["graph"]))

        if not config_args.get("skip_llm_greedy", False):
            greedy_best, greedy_log = run_graph_search(
                method_name="llm_greedy", xs=xs, variables=variables, gt_file_path=gt_file_path,
                start_tree=start_tree, dense_weights=dense_weights, raw_discovery_matrix=raw_discovery_matrix,
                num_samples=num_samples, bic_penalty_lambda=float(config_args["bic_penalty_lambda"]),
                proposer=llm_proposer, search_mode="greedy", beam_width=1,
                max_iterations=int(config_args["max_iterations"]), max_age=int(config_args["max_age"]),
                min_bic_delta=float(config_args["min_bic_delta"]),
                guardrail_delete_threshold=float(config_args["guardrail_delete_threshold"]),
            )
            json_log["methods"]["llm_greedy"] = greedy_log
            records.append(metrics_record("llm_greedy", target_dir, greedy_best["p"], greedy_best["r"],
                                          greedy_best["f1"], greedy_best["bic"], greedy_best["mse"],
                                          len(greedy_best["graph"]), greedy_best["graph"]))

        if not config_args.get("skip_llm_beam", False):
            beam_best, beam_log = run_graph_search(
                method_name="llm_beam", xs=xs, variables=variables, gt_file_path=gt_file_path,
                start_tree=start_tree, dense_weights=dense_weights, raw_discovery_matrix=raw_discovery_matrix,
                num_samples=num_samples, bic_penalty_lambda=float(config_args["bic_penalty_lambda"]),
                proposer=llm_proposer, search_mode="beam", beam_width=int(config_args["beam_width"]),
                max_iterations=int(config_args["max_iterations"]), max_age=int(config_args["max_age"]),
                min_bic_delta=float(config_args["min_bic_delta"]),
                guardrail_delete_threshold=float(config_args["guardrail_delete_threshold"]),
            )
            json_log["methods"]["llm_beam"] = beam_log
            records.append(metrics_record("llm_beam", target_dir, beam_best["p"], beam_best["r"],
                                          beam_best["f1"], beam_best["bic"], beam_best["mse"],
                                          len(beam_best["graph"]), beam_best["graph"]))

        with open(log_output_path, "w", encoding="utf-8") as f:
            json.dump(json_log, f, indent=4, ensure_ascii=False)
        print(f"[DONE] {target_dir} | log={log_output_path}")
        return records

    except Exception as e:
        print(f"\n[FATAL ERROR] Execution failed for {target_dir}: {str(e)}")
        traceback.print_exc()
        json_log["CRASH_LOG"] = str(e)
        with open(log_output_path, "w", encoding="utf-8") as f:
            json.dump(json_log, f, indent=4, ensure_ascii=False)
        return None


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    summary = (
        df.groupby("method")
        .agg(
            suites=("test_suite", "nunique"),
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_bic=("bic", "mean"),
            mean_mse=("mse", "mean"),
            mean_edges=("edge_count", "mean"),
        )
        .reset_index()
    )
    preferred_order = {"aerca_baseline": 0, "random_beam": 1, "llm_greedy": 2, "llm_beam": 3}
    summary["_order"] = summary["method"].map(lambda x: preferred_order.get(x, 999))
    return summary.sort_values(["_order", "method"]).drop(columns=["_order"])


def print_summary(summary: pd.DataFrame) -> None:
    print("\n" + "*" * 90)
    print("ABLATION STUDY SUMMARY".center(90))
    print("*" * 90)
    if summary.empty:
        print("No successful results.")
        return
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("*" * 90 + "\n")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="CausClass synthetic ablation runner: AERCA baseline vs random search vs LLM greedy vs LLM beam."
    )
    parser.add_argument("--workers", type=int, default=1, help="Concurrent test suites. Use carefully with GPU/API limits.")
    parser.add_argument("--debug", type=int, default=0, help="Set to 1 to run only one suite.")
    parser.add_argument("--test_limit", type=int, default=0, help="Limit number of data/synth_data/test_* suites. 0 = all.")
    parser.add_argument("--output_root", type=str, default=str(OUTPUT_DIR / "ablation_runs"), help="Where CSV summaries and JSON logs are saved.")

    parser.add_argument("--chunk_size", type=int, default=500, help="Chunk size for verifier input, matching run_synthetic_search.py.")
    parser.add_argument("--baseline_threshold", type=float, default=0.4, help="AERCA initial graph quantile threshold.")
    parser.add_argument("--bic_penalty_lambda", type=float, default=65.0, help="Edge penalty multiplier in BIC-like objective.")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width for random_beam and llm_beam.")
    parser.add_argument("--max_iterations", type=int, default=30, help="Maximum graph search iterations.")
    parser.add_argument("--max_age", type=int, default=4, help="Stop stale search branches after this many stuck iterations.")
    parser.add_argument("--min_bic_delta", type=float, default=-0.15, help="Allowed BIC degradation for add edits, matching original code.")
    parser.add_argument("--guardrail_delete_threshold", type=float, default=0.75, help="Block deletion if raw AERCA signal is above this threshold.")

    parser.add_argument("--random_seed", type=int, default=42, help="Random baseline seed.")
    parser.add_argument("--random_edits_per_state", type=int, default=3, help="Number of random edit proposals per graph state.")

    parser.add_argument("--skip_random", action="store_true", help="Skip random search baseline.")
    parser.add_argument("--skip_llm_greedy", action="store_true", help="Skip LLM greedy baseline.")
    parser.add_argument("--skip_llm_beam", action="store_true", help="Skip LLM beam main method.")

    args = parser.parse_args()

    load_env_file()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("[CRITICAL] DEEPSEEK_API_KEY is missing from environment variables.")
        sys.exit(1)

    os.makedirs(args.output_root, exist_ok=True)
    test_dirs = sorted(glob.glob(str(DATA_DIR / "synth_data" / "test_*")))
    if args.debug == 1:
        test_dirs = test_dirs[:1]
        args.workers = 1
    elif args.test_limit and args.test_limit > 0:
        test_dirs = test_dirs[:args.test_limit]

    if not test_dirs:
        print(f"[ERROR] No synthetic test directories found under {DATA_DIR / 'synth_data'}.")
        sys.exit(1)

    config_args = vars(args).copy()
    batch_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    print("\n" + "=" * 90)
    print(f"[SYSTEM] Running ablation: {len(test_dirs)} suites | workers={args.workers}")
    print("[SYSTEM] Methods: AERCA baseline, random_beam, llm_greedy, llm_beam")
    print(f"[SYSTEM] Output root: {args.output_root}")
    print("=" * 90 + "\n")

    all_records: List[Dict[str, Any]] = []
    if args.workers == 1:
        for d in test_dirs:
            result = process_test_suite(d, batch_time_str, api_key, config_args)
            if result:
                all_records.extend(result)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_test_suite, d, batch_time_str, api_key, config_args): d
                for d in test_dirs
            }
            for future in concurrent.futures.as_completed(futures):
                d = futures[future]
                try:
                    result = future.result()
                    if result:
                        all_records.extend(result)
                        print(f"[SUCCESS] {d}")
                    else:
                        print(f"[FAILED] {d}")
                except Exception as e:
                    print(f"[FAILED] {d}: {e}")

    if not all_records:
        print("\n[ERROR] No test suites were successfully processed.")
        sys.exit(1)

    df_all = pd.DataFrame(all_records)
    df_csv = df_all.drop(columns=["graph"], errors="ignore")
    summary = summarize_results(df_csv)

    all_results_path = os.path.join(args.output_root, "ablation_all_results.csv")
    summary_path = os.path.join(args.output_root, "ablation_summary.csv")
    graphs_path = os.path.join(args.output_root, "ablation_graphs.json")

    df_csv.to_csv(all_results_path, index=False)
    summary.to_csv(summary_path, index=False)
    with open(graphs_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    print_summary(summary)
    print("Saved:")
    print(f"- {all_results_path}")
    print(f"- {summary_path}")
    print(f"- {graphs_path}")
    print(f"- per-suite logs under {args.output_root}/test_*/ablation_search_log.json")
