"""
Strict Causal Discovery Optimization - Grid Search Pipeline (Production Grade)
---------------------------------------------------------------------------
This script performs a multiprocessing grid search to find the optimal hyperparameters
for the LLM-guided Tabu Beam Search causal discovery framework. 

FIXES INCLUDED:
- Numeric aggregation safety (numeric_only=True).
- Reserved keyword handling (renaming lambda to lambda_val).
- Pre-output data persistence (CSV saved before terminal reporting).
"""

import os
import sys

# Disable Python warnings globally.
os.environ["PYTHONWARNINGS"] = "ignore"

import json
import copy
import glob
import itertools
import pandas as pd
import numpy as np
import concurrent.futures
import multiprocessing as mp
from tqdm import tqdm

from utils.helpers import load_env_file, calculate_bic, evaluate_dag, hash_graph, apply_edit
from core.graph_edit_agent import LLMGraphAgent
from core.aerca_verifier import get_initial_graph_from_aerca, run_masked_aerca
from utils.paths import DATA_DIR, OUTPUT_DIR

# =============================================================================
# WORKER FUNCTION
# Executes a complete causal discovery pipeline for a single hyperparameter set.
# =============================================================================
def evaluate_combo(task_args):
    TARGET_DIR, threshold, penalty_lambda, min_delta, API_KEY = task_args

    # SILENT MODE: Redirect stdout and stderr strictly within the child process
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    try:
        # 1. Path Initialization
        gt_file_path = os.path.join(TARGET_DIR, "ground_truth_graph.json")
        csv_file_path = os.path.join(TARGET_DIR, "time_series_noisy.csv")
        
        if not os.path.exists(gt_file_path) or not os.path.exists(csv_file_path): 
            return None
            
        with open(gt_file_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            
        variables = config["behaviors"]
        real_world_context = config.get("scenario_context", "Classroom dynamics study.")

        # 2. Data Loading & Normalization
        df = pd.read_csv(csv_file_path)
        
        data_array = df.values / 100.0
        chunk_size = 500
        xs = np.array([data_array[i * chunk_size : (i + 1) * chunk_size] for i in range(len(data_array) // chunk_size)])

        # 3. Phase 0: Baseline Discovery
        auto_initial_graph, dense_weights, raw_discovery_matrix = get_initial_graph_from_aerca(
            xs, variables, default_threshold=threshold
        )
        
        agent = LLMGraphAgent(API_KEY, variables, real_world_context)
        
        # Evaluate baseline configuration
        base_mse, num_e, base_weights, base_edges = run_masked_aerca(
            xs, auto_initial_graph, variables, init_weights=dense_weights
        )
        
        base_bic = calculate_bic(base_mse, num_e, num_samples=len(xs)*chunk_size, params_per_edge=penalty_lambda)
        base_p, base_r, base_f1, _, _ = evaluate_dag(base_edges, gt_file_path, variables)
        
        # Initialize Beam Search structures
        global_tabu = {hash_graph(base_edges)}
        beam = [{
            "graph": base_edges, "mse": base_mse, "bic": base_bic, "weights": base_weights,
            "age": 0, "local_tabu": [], "f1": base_f1, "p": base_p, "r": base_r
        }]

        # 4. Phase 1: LLM-Guided Tabu Beam Search
        max_iterations = 20
        for iteration in range(1, max_iterations + 1):
            candidates = []
            for b_idx, tree in enumerate(beam):
                tree["has_valid_offspring"] = False
                status = "OPTIMIZING" if tree['age'] == 0 else f"STUCK (Age: {tree['age']})"
                proposed_edits = agent.propose_edits(tree["graph"], tree["local_tabu"], status, raw_discovery_matrix)
                
                for edit in proposed_edits:
                    action, src, tgt = edit.get("action"), edit.get("source"), edit.get("target")

                    if action == "delete":
                        try:
                            i = variables.index(src); j = variables.index(tgt)
                            # Preserve strong verifier signals during delete proposals.
                            if abs(raw_discovery_matrix[i, j]) > 0.75: continue
                        except ValueError: pass

                    is_forward = any(e['source'] == src and e['target'] == tgt for e in tree["graph"])
                    is_backward = any(e['source'] == tgt and e['target'] == src for e in tree["graph"])
                    if action == "add" and (is_forward or is_backward): continue 
                    if action == "delete" and not is_forward: continue 

                    new_graph = apply_edit(tree["graph"], edit)
                    if hash_graph(new_graph) not in global_tabu:
                        global_tabu.add(hash_graph(new_graph))
                        candidates.append({"graph": new_graph, "parent": tree, "edit": edit})

            if not candidates:
                for tree in beam: tree['age'] += 1
            else:
                next_gen = []
                for c in candidates:
                    c_mse, c_num_e, c_weights, c_updated = run_masked_aerca(xs, c["graph"], variables, init_weights=dense_weights)
                    c_bic = calculate_bic(c_mse, c_num_e, len(xs)*chunk_size, penalty_lambda)
                    c_p, c_r, c_f1, _, _ = evaluate_dag(c_updated, gt_file_path, variables)
                    
                    bic_improvement = c["parent"]["bic"] - c_bic
                    
                    threshold_delta = min_delta if c['edit']['action'] == 'add' else -0.05 

                    if bic_improvement >= threshold_delta:
                        next_gen.append({
                            "graph": c_updated, "mse": c_mse, "bic": c_bic, "weights": c_weights, 
                            "age": 0, "local_tabu": copy.deepcopy(c["parent"]["local_tabu"]), 
                            "f1": c_f1, "p": c_p, "r": c_r
                        })
                        c["parent"]["has_valid_offspring"] = True
                    else:
                        c["parent"]["local_tabu"].append(f"FAILED {c['edit']['action'].upper()}: {c['edit']['source']}->{c['edit']['target']}")

                for tree in beam:
                    if not tree.get("has_valid_offspring", False): tree["age"] += 1
                combined = [t for t in (beam + next_gen) if t["age"] < 3]
                if not combined: break
                combined.sort(key=lambda x: x["bic"])
                beam = combined[:3]

        # 5. Cleanup
        model_name = "masked_aerca"
        checkpoint_path = os.path.join(OUTPUT_DIR, "saved_models", f'{model_name}_{os.getpid()}.pt')
        if os.path.exists(checkpoint_path): os.remove(checkpoint_path)

        best_final = beam[0]
        return {
            "test_suite": TARGET_DIR,
            "threshold": threshold,
            "lambda": penalty_lambda,
            "min_delta": min_delta,
            "base_p": base_p, "base_r": base_r, "base_f1": base_f1,
            "final_p": best_final["p"], "final_r": best_final["r"], "final_f1": best_final["f1"]
        }

    except Exception as e:
        import traceback
        print(f"\n[CRASH] Test: {TARGET_DIR} | Thresh: {threshold} | Lỗi: {str(e)}")
        print(traceback.format_exc())
        return None

# =============================================================================
# MAIN EXECUTION THREAD
# =============================================================================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    load_env_file()
    API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    if not API_KEY:
        print("[CRITICAL ERROR] DEEPSEEK_API_KEY missing.")
        sys.exit(1)

    test_dirs = sorted(glob.glob(str(DATA_DIR / "synth_data" / "test_*")))[:3]
    
    thresholds = [0.40, 0.45, 0.50, 0.55] 
    lambdas = [1.0, 1.2, 1.5, 1.8]        
    min_deltas = [0.05, 0.10, 0.15]

    tasks = []
    for th, lam, min_d in itertools.product(thresholds, lambdas, min_deltas):
        for test_dir in test_dirs:
            tasks.append((test_dir, th, lam, min_d, API_KEY))

    print("\n" + "="*80)
    print(f" [SYSTEM] GRID SEARCH INITIATED: {len(tasks)} JOBS | 20 WORKERS")
    print("="*80 + "\n")

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(evaluate_combo, task) for task in tasks]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Grid Search Progress"):
            res = future.result()
            if res is not None: results.append(res)

    if not results:
        print("\n[CRITICAL ERROR] No results generated."); sys.exit(1)

    df_res = pd.DataFrame(results).rename(columns={'lambda': 'lambda_val'})
    grouped = df_res.groupby(['threshold', 'lambda_val', 'min_delta']).mean(numeric_only=True).reset_index()
    grouped['net_f1_improve'] = grouped['final_f1'] - grouped['base_f1']
    sorted_df = grouped.sort_values(by='final_f1', ascending=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "grid_search_results.csv"
    sorted_df.to_csv(results_path, index=False)
    print(f"\n[SYSTEM] Results successfully persisted to '{results_path}'.")

    print("\n" + "*"*80)
    print("* TOP 10 OPTIMIZED HYPERPARAMETER COMBINATIONS".center(78) + " *")
    print("*"*80)
    print(f"{'Rank':<5} | {'Thresh':<6} | {'Lambda':<6} | {'MinDelta':<8} || {'Base F1':<8} | {'Final F1':<8} | {'Net Imp':<8}")
    print("-" * 80)
    for i, row in enumerate(sorted_df.head(10).itertuples(), 1):
        print(f"{i:<5} | {row.threshold:<6.2f} | {row.lambda_val:<6.2f} | {row.min_delta:<8.2f} || {row.base_f1:<8.4f} | {row.final_f1:<8.4f} | {row.net_f1_improve:+.4f}")
    print("*"*80 + "\n")
