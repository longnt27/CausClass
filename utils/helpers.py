import os
import sys
import json
import datetime
import numpy as np
import copy

from utils.paths import ENV_FILE, OUTPUT_DIR

class DualLogger(object):
    """Logs standard output to both terminal and a dedicated file."""
    def __init__(self, log_dir=None):
        self.terminal = sys.stdout
        if log_dir is None:
            log_dir = OUTPUT_DIR / "logs"
        os.makedirs(log_dir, exist_ok=True)
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = open(f"{log_dir}/aerca_run_{time_str}.log", "a", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def load_env_file():
    """Loads environment variables from .env file."""
    env_path = ENV_FILE
    if os.path.exists(env_path):
       with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
               if '=' in line and not line.startswith('#'):
                    key, val = line.strip().split('=', 1)
                    os.environ[key.strip()] = val.strip()

def calculate_bic(mse, num_edges, num_samples, params_per_edge=80):
    """Calculates the Bayesian Information Criterion (BIC)."""
    return num_samples * np.log(mse + 1e-8) + params_per_edge * num_edges * np.log(num_samples)

def evaluate_dag(pred_edges, gt_file_path, variables):
    with open(gt_file_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    gt_edges = set()

    # Newer synthetic suites store edges directly as a JSON array.
    if "edges_logic_from_llm" in gt_data:
        for edge in gt_data["edges_logic_from_llm"]:
            gt_edges.add((edge["source"], edge["target"]))
            
    # Older suites store the ground-truth graph as an adjacency matrix.
    elif "matrices" in gt_data:
        gt_adj = np.array(gt_data['matrices']['A_GroundTruth_Adjacency'])
        for i, src in enumerate(variables):
            for j, tgt in enumerate(variables):
                if gt_adj[i, j] == 1:
                    gt_edges.add((src, tgt))

    # Convert predicted edge dictionaries into comparable edge tuples.
    pred_set = set()
    for pred in pred_edges:
        pred_set.add((pred['source'], pred['target']))

    # Compute set-based classification counts.
    tp = len(pred_set.intersection(gt_edges))
    fp = len(pred_set - gt_edges)
    fn = len(gt_edges - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Preserve the legacy return signature for downstream scripts.
    return precision, recall, f1, gt_edges, pred_edges

def hash_graph(edge_list):
    """Generates a deterministic string hash for a graph to track visited states."""
    if not edge_list:
        return "EMPTY_GRAPH"
        
    edges = [f"{e['source']}->{e['target']}" for e in edge_list]
    edges.sort() 
    
    return "|".join(edges)

def apply_edit(current_edges, edit):
    """Returns a deep-copied graph with the LLM proposed edit applied."""
    new_edges = copy.deepcopy(current_edges)
    
    if "action" not in edit or "source" not in edit or "target" not in edit:
        return new_edges
        
    base_edge = {"source": edit["source"], "target": edit["target"]}
    
    if edit["action"] == "add":
        if not any(e["source"] == base_edge["source"] and e["target"] == base_edge["target"] for e in new_edges):
            new_edges.append(base_edge)
            
    elif edit["action"] == "delete":
        new_edges = [e for e in new_edges if not (e["source"] == base_edge["source"] and e["target"] == base_edge["target"])]
        
    return new_edges
