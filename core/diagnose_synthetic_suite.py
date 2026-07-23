import os
import json
import glob
import pandas as pd
import numpy as np

from utils.helpers import load_env_file, evaluate_dag
from core.graph_edit_agent import LLMGraphAgent
from core.aerca_verifier import get_initial_graph_from_aerca, run_masked_aerca
from utils.paths import DATA_DIR

load_env_file()
API_KEY = os.environ.get("DEEPSEEK_API_KEY")

test_dirs = sorted(glob.glob(str(DATA_DIR / "synth_data" / "test_*")))
if not test_dirs:
    print(f"[ERROR] No synthetic data found under {DATA_DIR / 'synth_data'}.")
    exit(1)
if not API_KEY:
    print("[ERROR] DEEPSEEK_API_KEY is missing from the environment.")
    exit(1)

TARGET_DIR = test_dirs[0]
threshold = 0.6
penalty_lambda = 0.20

print("="*80)
print(f"🎯 QUICK DIAGNOSTIC TEST: {TARGET_DIR}")
print(f"⚙️ Config: Threshold={threshold}, Lambda={penalty_lambda}")
print("="*80)

# 1. Load Data
gt_file_path = os.path.join(TARGET_DIR, "ground_truth_graph.json")
csv_file_path = os.path.join(TARGET_DIR, "time_series_noisy.csv")

with open(gt_file_path, "r", encoding="utf-8") as f:
    config = json.load(f)
variables = config["behaviors"]
context = config.get("scenario_context", "Classroom dynamics study.")

df = pd.read_csv(csv_file_path)
data_array = df.values / 100.0  
chunk_size = 500
xs = np.array([data_array[i * chunk_size : (i + 1) * chunk_size] for i in range(len(data_array) // chunk_size)])

# 2. Phase 0: run the verifier baseline.
print("\n[Phase 0] Running verifier baseline on the time series...")
auto_initial_graph, dense_weights, raw_discovery_matrix = get_initial_graph_from_aerca(xs, variables, default_threshold=threshold)

base_mse, num_e, base_weights, base_edges = run_masked_aerca(xs, auto_initial_graph, variables, init_weights=dense_weights)
base_p, base_r, base_f1, _, _ = evaluate_dag(base_edges, gt_file_path, variables)

print("-" * 50)
print(f"📊 BASELINE RESULTS (PHASE 0):")
print(f"   - Precision: {base_p:.4f}")
print(f"   - Recall   : {base_r:.4f}")
print(f"   👉 BASE F1 SCORE          : {base_f1:.4f}")
print("-" * 50)

if base_f1 > 0.6:
    print("[OK] Baseline F1 is above the diagnostic threshold.")
else:
    print("[WARN] Baseline F1 is below the diagnostic threshold.")

# 3. Phase 1: request graph-edit proposals from the LLM agent.
print("\n[Phase 1] Requesting graph-edit proposals from the LLM agent...")
agent = LLMGraphAgent(API_KEY, variables, context)
edits = agent.propose_edits(base_edges, [], "OPTIMIZING", raw_discovery_matrix)

print(f"\nLLM proposals ({len(edits)}):")
print(json.dumps(edits, indent=2, ensure_ascii=False))
print("\n" + "="*80)
