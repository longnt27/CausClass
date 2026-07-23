"""
Phase 2: Neuro-Symbolic Causal Discovery Pipeline
Project: CausClass
Description: Executes AERCA combined with an LLM-guided Tabu Search on a single 
             real-world multivariate time-series dataset. Output is a valid Directed Acyclic Graph (DAG).
"""

import os
import sys
import json
import copy
import datetime
import argparse
import traceback
import numpy as np
import pandas as pd
import wandb

# ==========================================
# SILENCE ALL THIRD-PARTY LIBRARIES
# ==========================================
os.environ["WANDB_SILENT"] = "true"      # Mute WandB initialization spam
os.environ["TQDM_DISABLE"] = "1"         # Disable globally injected progress bars

from utils.helpers import load_env_file, calculate_bic, hash_graph, apply_edit
from core.graph_edit_agent import LLMGraphAgent
from core.aerca_verifier import get_initial_graph_from_aerca, run_masked_aerca
from utils.paths import OUTPUT_DIR

def run_causal_discovery(input_csv, output_json, api_key):
    print("[INFO] ========================================================")
    print(f"[INFO] INITIATING PHASE 2: CAUSAL DISCOVERY PIPELINE")
    print(f"[INFO] Target Dataset: {input_csv}")
    print("[INFO] ========================================================")

    if not os.path.exists(input_csv):
        print(f"[FATAL ERROR] Input data file not found: {input_csv}")
        return

    # Statically define behavioral variables (Strict mapping to Phase 1 output)
    variables = ["Talk", "Read", "Phone", "Hand", "Lean", "Stand"]
    real_world_context = "Classroom physical interaction dynamics. Behaviors represent the percentage of students engaging in the specific action."

    # Data Preprocessing
    df = pd.read_csv(input_csv)
    
    # [CRITICAL STEP] Isolate mathematical matrix by removing the temporal index
    if "time_bin_sec" in df.columns:
        df = df.drop(columns=["time_bin_sec"])
        
    # Normalize distribution to [0, 1] probability space for AERCA optimization
    data_array = df[variables].values / 100.0  
    
    # Treat the entire session as a single continuous time-series chunk
    xs = np.array([data_array])

    print(f"[INFO] Data loaded successfully. Matrix shape: {data_array.shape} (Time Steps x Variables)")

    # System Hyperparameters (Optimized for Sparsity and Stability)
    hyperparams = {
        "bic_penalty_lambda": 65, 
        "beam_width": 5,
        "max_iterations": 30,
        "max_age": 4,
        "min_bic_delta": -0.15,          
        "aerca_lr": 0.01               
    }
    
    run_name = f"Inference_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run = wandb.init(project="causclass-real-inference", name=run_name, config=hyperparams, reinit=True)
    
    # Audit log to document LLM reasoning for the thesis appendix
    llm_reasoning_log = []

    try:
        print("[PROCESS] Phase 0: Executing pure AERCA (Mathematical Baseline)...")
        auto_initial_graph, dense_weights, raw_discovery_matrix = get_initial_graph_from_aerca(xs, variables, default_threshold=0.4)
        agent = LLMGraphAgent(api_key, variables, real_world_context)
        
        base_mse, num_e, base_weights, base_edges = run_masked_aerca(
            xs, auto_initial_graph, variables, init_weights=dense_weights
        )
        
        base_bic = calculate_bic(base_mse, num_e, num_samples=data_array.shape[0], params_per_edge=hyperparams["bic_penalty_lambda"])
        
        run.log({"Baseline/BIC": base_bic, "Baseline/MSE": base_mse}, step=0)
        print(f"[INFO] Baseline established. Initial graph contains {len(base_edges)} edges. BIC Score: {base_bic:.2f}")

        global_tabu = {hash_graph(base_edges)}
        beam = [{
            "graph": base_edges, "mse": base_mse, "bic": base_bic, "weights": base_weights,
            "age": 0, "local_tabu": []
        }]

        print("[PROCESS] Phase 1: Running LLM-guided tabu search with guardrails...")
        
        for iteration in range(1, hyperparams["max_iterations"] + 1):
            candidates = []
            
            for b_idx, tree in enumerate(beam):
                tree["has_valid_offspring"] = False
                status = "OPTIMIZING" if tree['age'] == 0 else f"STUCK (Age: {tree['age']})"
                
                # Conceal MSE and Weights; provide only Topology and raw matrix to prevent numerical hallucination
                proposed_edits = agent.propose_edits(tree["graph"], tree["local_tabu"], status, raw_discovery_matrix)
                
                for edit in proposed_edits:
                    action, src, tgt = edit.get("action"), edit.get("source"), edit.get("target")
                    reasoning = edit.get("reasoning", "No reasoning provided.")

                    if action == "delete":
                        try:
                            i = variables.index(src)
                            j = variables.index(tgt)
                            original_signal = abs(raw_discovery_matrix[i, j])
                            
                            # Guardrail V2: Absolute mathematical signals override semantic hallucination
                            if original_signal > 0.75:
                                log_entry = {
                                    "iteration": iteration, "action": action.upper(),
                                    "edge": f"{src}->{tgt}", "reasoning": reasoning,
                                    "status": "BLOCKED (GUARDRAIL V2)", "bic_improvement": 0.0
                                }
                                llm_reasoning_log.append(log_entry)
                                continue
                        except ValueError:
                            pass

                    # Prevent redundant operations
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
                    c_bic = calculate_bic(c_mse, c_num_e, data_array.shape[0], hyperparams["bic_penalty_lambda"])
                    
                    bic_improvement = c["parent"]["bic"] - c_bic
                    
                    e_act = c['edit']['action'].upper()
                    e_src = c['edit']['source']
                    e_tgt = c['edit']['target']
                    e_reason = c['edit'].get('reasoning', '')
                    
                    # Acceptance Threshold Configuration
                    threshold_delta = hyperparams["min_bic_delta"] if c['edit']['action'] == 'add' else 0.00 
                    
                    if bic_improvement >= threshold_delta:
                        status_str = "ACCEPTED"
                        next_generation.append({
                            "graph": c_updated_graph, "mse": c_mse, "bic": c_bic, "weights": c_weights,
                            "age": 0, "local_tabu": copy.deepcopy(c["parent"]["local_tabu"])
                        })
                        c["parent"]["has_valid_offspring"] = True
                    else:
                        status_str = "REJECTED (MSE Spike Detected)"
                        c["parent"]["local_tabu"].append(f"FAILED {e_act}: {e_src}->{e_tgt}")

                    log_entry = {
                        "iteration": iteration, "action": e_act,
                        "edge": f"{e_src}->{e_tgt}", "reasoning": e_reason,
                        "status": status_str, "bic_improvement": round(bic_improvement, 4)
                    }
                    llm_reasoning_log.append(log_entry)

                for tree in beam:
                    if not tree.get("has_valid_offspring", False): tree["age"] += 1

                combined = [t for t in (beam + next_generation) if t["age"] < hyperparams["max_age"]]
                if not combined: break
                combined.sort(key=lambda x: x["bic"])
                beam = combined[:hyperparams["beam_width"]]

                best_current = beam[0] 
        
                wandb.log({
                    "Iteration": iteration,
                    "Optimization/BIC_Optimized": best_current["bic"],
                    "Optimization/MSE_Optimized": best_current["mse"],
                    "Graph/Edge_Count": len(best_current["graph"])
                })
                
                print(f"   [Iter {iteration:02d}] Best Graph Edges: {len(best_current['graph'])} | Current BIC: {best_current['bic']:.2f}")

        best_final = beam[0]
        run.log({"Final/BIC": best_final["bic"], "Final/Edge_Count": len(best_final["graph"])})
        run.finish()
        
        # ==========================================
        # EXPORT FINAL GRAPH AND AUDIT LOG
        # ==========================================
        final_output = {
            "metadata": {
                "source_file": input_csv,
                "timestamp": datetime.datetime.now().isoformat(),
                "final_bic": round(best_final["bic"], 4),
                "final_mse": round(best_final["mse"], 6),
                "edge_count": len(best_final["graph"])
            },
            "causal_graph": best_final["graph"],
            "llm_reasoning_history": llm_reasoning_log
        }
        
        output_dir = os.path.dirname(os.path.abspath(output_json))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
            
        print("\n" + "="*80)
        print("[SUCCESS] CAUSAL DISCOVERY PIPELINE COMPLETED SUCCESSFULLY!")
        print(f" - Final Graph Topology: Converged with {len(best_final['graph'])} edges.")
        print(f" - Output File (Topology & Reasoning Log): {output_json}")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n[FATAL ERROR] Runtime exception during inference pipeline: {str(e)}")
        traceback.print_exc()
        run.finish(exit_code=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Phase 2: CausClass on Real-world Multivariate Data")
    parser.add_argument("--input", type=str, required=True, help="Absolute path to the Phase 1 CSV output")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR / "final_causal_graph.json"), help="Destination path for the JSON output")
    args = parser.parse_args()

    load_env_file()
    API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    if not API_KEY:
        print("[CRITICAL ERROR] Missing DEEPSEEK_API_KEY environment variable. Terminating.")
        sys.exit(1)

    run_causal_discovery(args.input, args.output, API_KEY)
