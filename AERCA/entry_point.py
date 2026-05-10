import os
import sys

# ==========================================
# SILENCE ALL THIRD-PARTY LIBRARIES
# ==========================================
os.environ["WANDB_SILENT"] = "true"      # Mute WandB init spam
os.environ["TQDM_DISABLE"] = "1"         # Kill all progress bars globally

import json
import copy
import datetime
import glob
import argparse
import traceback
import numpy as np
import pandas as pd
import wandb
import concurrent.futures
import multiprocessing as mp

from helper import load_env_file, calculate_bic, evaluate_dag, hash_graph, apply_edit
from llm_agent import LLMGraphAgent
from engine import get_initial_graph_from_aerca, run_masked_aerca

# ==========================================
# WORKER FUNCTION (Isolated Process Memory)
# ==========================================
def process_test_suite(TARGET_DIR, batch_time_str, API_KEY):
    gt_file_path = os.path.join(TARGET_DIR, "ground_truth_graph.json")
    csv_file_path = os.path.join(TARGET_DIR, "time_series_noisy.csv")
    
    if not os.path.exists(gt_file_path) or not os.path.exists(csv_file_path): 
        return None
        
    with open(gt_file_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    variables = config["behaviors"]
    real_world_context = config.get("scenario_context", "Classroom dynamics study.")

    df = pd.read_csv(csv_file_path)
    data_array = df.values / 100.0  
    chunk_size = 500
    xs = np.array([data_array[i * chunk_size : (i + 1) * chunk_size] for i in range(len(data_array) // chunk_size)])

    run_name = f"Test_{TARGET_DIR.split('_')[-1]}"
    wandb_group_name = f"Strict_Causal_Batch_{batch_time_str}"
    
    hyperparams = {
        "test_suite": TARGET_DIR,
        "bic_penalty_lambda": 65, 
        "beam_width": 5,
        "max_iterations": 30,
        "max_age": 4,
        "min_bic_delta": -0.15,          
        "aerca_lr": 0.01               
    }
    
    run = wandb.init(project="causclass-var-optimization", group=wandb_group_name, name=run_name, config=hyperparams, reinit=True)
    llm_table = wandb.Table(columns=["Epoch", "Action", "Source", "Target", "Reasoning", "Status", "Imp", "F1"])

    try:
        # PHASE 0
        auto_initial_graph, dense_weights, raw_discovery_matrix = get_initial_graph_from_aerca(xs, variables, default_threshold=0.4)
        agent = LLMGraphAgent(API_KEY, variables, real_world_context)
        
        base_mse, num_e, base_weights, base_edges = run_masked_aerca(
            xs, auto_initial_graph, variables, init_weights=dense_weights
        )
        
        base_bic = calculate_bic(base_mse, num_e, num_samples=len(xs)*chunk_size, params_per_edge=hyperparams["bic_penalty_lambda"])
        base_p, base_r, base_f1, _, _ = evaluate_dag(base_edges, gt_file_path, variables)
        
        run.log({
            "Baseline/BIC": base_bic, 
            "Baseline/MSE": base_mse,
            "Baseline/F1": base_f1, 
            "Baseline/Precision": base_p, 
            "Baseline/Recall": base_r
        }, step=0)

        global_tabu = {hash_graph(base_edges)}
        beam = [{
            "graph": base_edges, "mse": base_mse, "bic": base_bic, "weights": base_weights,
            "age": 0, "local_tabu": [], "f1": base_f1, "p": base_p, "r": base_r
        }]

        # PHASE 1
        for iteration in range(1, hyperparams["max_iterations"] + 1):
            candidates = []
            
            for b_idx, tree in enumerate(beam):
                tree["has_valid_offspring"] = False
                status = "OPTIMIZING" if tree['age'] == 0 else f"STUCK (Age: {tree['age']})"
                
                proposed_edits = agent.propose_edits(tree["graph"], tree["local_tabu"], status, raw_discovery_matrix)
                
                for edit in proposed_edits:
                    action, src, tgt = edit.get("action"), edit.get("source"), edit.get("target")

                    if action == "delete":
                        try:
                            i = variables.index(src)
                            j = variables.index(tgt)
                            original_signal = abs(raw_discovery_matrix[i, j])
                            
                            # [ĐÃ SỬA] Guardrail V2 nâng lên 0.85. 
                            # Chỉ những tín hiệu toán học tuyệt đối mạnh mới được phép cản LLM.
                            if original_signal > 0.75:
                                llm_table.add_data(iteration, action.upper(), src, tgt, "N/A", "BLOCKED (GUARDRAIL V2)", 0.0, tree["f1"])
                                continue
                        except ValueError:
                            pass

                    is_forward = any(e['source'] == src and e['target'] == tgt for e in tree["graph"])
                    is_backward = any(e['source'] == tgt and e['target'] == src for e in tree["graph"])
                    
                    if action == "add" and (is_forward or is_backward): continue 
                    if action == "delete" and not is_forward: continue 

                    new_graph = apply_edit(tree["graph"], edit)
                    h_graph = hash_graph(new_graph)
                    
                    if h_graph not in global_tabu:
                        global_tabu.add(h_graph)
                        candidates.append({"graph": new_graph, "parent": tree, "edit": edit})

            if not candidates:
                for tree in beam: tree['age'] += 1
            else:
                next_generation = []
                for c in candidates:
                    c_mse, c_num_e, c_weights, c_updated_graph = run_masked_aerca(
                        xs, c["graph"], variables, init_weights=dense_weights
                    )
                    c_bic = calculate_bic(c_mse, c_num_e, len(xs)*chunk_size, hyperparams["bic_penalty_lambda"])
                    c_p, c_r, c_f1, _, _ = evaluate_dag(c_updated_graph, gt_file_path, variables)
                    
                    bic_improvement = c["parent"]["bic"] - c_bic
                    
                    e_act = c['edit']['action'].upper()
                    e_src = c['edit']['source']
                    e_tgt = c['edit']['target']
                    e_reason = c['edit'].get('reasoning', 'No reasoning provided.')
                    
                    # [ĐÃ SỬA] THƯỢNG PHƯƠNG BẢO KIẾM CHO DELETE (-0.05)
                    # ADD thì cần dương 0.1, DELETE thì cho phép âm nhẹ
                    threshold_delta = hyperparams["min_bic_delta"] if c['edit']['action'] == 'add' else 0.00 
                    
                    if bic_improvement >= threshold_delta:
                        status_str = "ACCEPTED"
                        next_generation.append({
                            "graph": c_updated_graph, "mse": c_mse, "bic": c_bic, "weights": c_weights,
                            "age": 0, "local_tabu": copy.deepcopy(c["parent"]["local_tabu"]),
                            "f1": c_f1, "p": c_p, "r": c_r
                        })
                        c["parent"]["has_valid_offspring"] = True
                    else:
                        status_str = "REJECTED"
                        c["parent"]["local_tabu"].append(f"FAILED {e_act}: {e_src}->{e_tgt}")

                    llm_table.add_data(iteration, e_act, e_src, e_tgt, e_reason, status_str, bic_improvement, c_f1)

                for tree in beam:
                    if not tree.get("has_valid_offspring", False): tree["age"] += 1

                combined = [t for t in (beam + next_generation) if t["age"] < hyperparams["max_age"]]
                if not combined: break
                combined.sort(key=lambda x: x["bic"])
                beam = combined[:hyperparams["beam_width"]]

                best_current = beam[0] 
        
                wandb.log({
                    "Iteration": iteration,
                    
                    # Group F1, Precision, Recall
                    "F1_Score/Optimized": best_current["f1"],
                    "F1_Score/Baseline": base_f1,
                    "Precision/Optimized": best_current["p"],
                    "Precision/Baseline": base_p,
                    "Recall/Optimized": best_current["r"],
                    "Recall/Baseline": base_r,
                    
                    # Group Toán học (BIC & MSE)
                    "Optimization/BIC_Optimized": best_current["bic"],
                    "Optimization/BIC_Baseline": base_bic,
                    "Optimization/MSE_Optimized": best_current["mse"],
                    "Optimization/MSE_Baseline": base_mse,
                    
                    # Group Kích thước đồ thị
                    "Graph/Edge_Count_Optimized": len(best_current["graph"]),
                    "Graph/Edge_Count_Baseline": len(base_edges)
                })


        best_final = beam[0]
        
        run.log({
            "Final/BIC": best_final["bic"], 
            "Final/F1": best_final["f1"], 
            "Final/Precision": best_final["p"], 
            "Final/Recall": best_final["r"],
            "LLM_Reasoning_Log": llm_table
        })

        run.finish()
        
        return {
            "test_suite": TARGET_DIR,
            "base_p": base_p, "base_r": base_r, "base_f1": base_f1,
            "final_p": best_final["p"], "final_r": best_final["r"], "final_f1": best_final["f1"]
        }

    except Exception as e:
        # Expose fatal errors safely
        print(f"\n[FATAL ERROR] Execution failed for {TARGET_DIR}: {str(e)}")
        traceback.print_exc()
        run.finish(exit_code=1)
        return None

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # REQUIRED for PyTorch multi-processing to avoid CUDA/Lazy context race conditions
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Strict Causal Discovery - Multiprocessing Engine")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent processes.")
    parser.add_argument("--debug", type=int, default=0, help="Set to 1 for sequential execution.")
    args = parser.parse_args()

    load_env_file()
    API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    if not API_KEY:
        print("[CRITICAL] DEEPSEEK_API_KEY is missing from environment variables.")
        sys.exit(1)

    test_dirs = sorted(glob.glob("synth_data/test_*"))
    if args.debug == 1: 
        test_dirs = test_dirs[:1]
        args.workers = 1 

    batch_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    print("\n" + "="*80)
    print(f" [SYSTEM] Initiating multiprocessing: {len(test_dirs)} suites with {args.workers} workers.")
    print(f" [SYSTEM] Progress bars and warnings are muted. Please wait for the final report...")
    print("="*80 + "\n")

    results = []
    
    # Switched to ProcessPoolExecutor to ensure isolated PyTorch memory spaces
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_test_suite, d, batch_time_str, API_KEY): d for d in test_dirs}
        
        # Iterating without tqdm to keep terminal completely clean
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)
                print(f"  [SUCCESS] {res['test_suite']} completed.")

    if results:
        df_res = pd.DataFrame(results)
        
        print("\n" + "*"*60)
        print("* SYSTEM EVALUATION REPORT - BATCH SUMMARY".center(58) + " *")
        print("*"*60)
        print(f"  Total Test Suites Processed : {len(df_res)}")
        print(f"  --------------------------------------------------------")
        print(f"  Metric       | Baseline (Phase 0) | Final (LLM + TBS)")
        print(f"  --------------------------------------------------------")
        print(f"  Avg Precision| {df_res['base_p'].mean():.4f}           | {df_res['final_p'].mean():.4f}")
        print(f"  Avg Recall   | {df_res['base_r'].mean():.4f}           | {df_res['final_r'].mean():.4f}")
        print(f"  Avg F1-Score | {df_res['base_f1'].mean():.4f}           | {df_res['final_f1'].mean():.4f}")
        print(f"  --------------------------------------------------------")
        net_f1 = df_res['final_f1'].mean() - df_res['base_f1'].mean()
        print(f"  Net F1 Improvement: {net_f1:+.4f}")
        print("*"*60 + "\n")
        print(" [SYSTEM] Execution complete. View WandB dashboard for detailed reasoning logs.")
    else:
        print("\n[ERROR] No test suites were successfully processed. Review the exception logs above.")
