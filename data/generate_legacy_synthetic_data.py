import os
import json
import numpy as np
import pandas as pd
import random
import argparse
import time
import atexit
from pathlib import Path
from scipy.special import softmax
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from utils.paths import DATA_DIR, ENV_FILE

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
load_dotenv(ENV_FILE, override=True)
GEMINI_MODEL = os.getenv("GEMINI_SYNTHETIC_MODEL", "gemini-3-flash-preview")

BEHAVIORS = ['Talk', 'Read', 'Phone', 'Hand', 'Lean', 'Stand']
NUM_VARS = len(BEHAVIORS)
BEH_IDX = {b: i for i, b in enumerate(BEHAVIORS)}

# Edge strengths are kept small enough for stable dynamics.
STRENGTH_MAP = {
    'high': 2.0,   
    'medium': 1.2,
    'low': 0.6    
}

# ==========================================
# 1. AUTOMATED TEST DIRECTORY MANAGEMENT
# ==========================================
_CLIENT = None
_CLIENT_API_KEY = None


def close_client():
    global _CLIENT, _CLIENT_API_KEY
    if _CLIENT is not None:
        _CLIENT.close()
        _CLIENT = None
        _CLIENT_API_KEY = None


atexit.register(close_client)


def get_client():
    global _CLIENT, _CLIENT_API_KEY
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("[Initialization Error] GEMINI_API_KEY is missing. Please verify your .env file.")
    if _CLIENT is None or _CLIENT_API_KEY != api_key:
        close_client()
        _CLIENT = genai.Client(api_key=api_key)
        _CLIENT_API_KEY = api_key
    return _CLIENT


def generate_json_with_gemini(prompt, system_instruction, max_output_tokens=2000):
    response = get_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=f"{system_instruction}\n\n{prompt}\n\nReturn only a valid JSON object.",
        config={
            "max_output_tokens": max_output_tokens,
            "response_mime_type": "application/json",
        },
    )
    raw_content = response.text.strip()
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError:
        start_idx = raw_content.find("{")
        end_idx = raw_content.rfind("}")
        if start_idx == -1 or end_idx == -1:
            raise
        return json.loads(raw_content[start_idx:end_idx + 1])


def is_retryable_error(error):
    if isinstance(error, genai_errors.APIError):
        return error.code not in {400, 401, 403, 404}
    return True


def get_next_test_folder(base_dir=None):
    if base_dir is None:
        base_dir = DATA_DIR / "synth_data"
    os.makedirs(base_dir, exist_ok=True)
    existing_tests = [d for d in os.listdir(base_dir) if d.startswith("test_")]
    if not existing_tests:
        next_idx = 1
    else:
        indices = [int(d.split("_")[1]) for d in existing_tests if d.split("_")[1].isdigit()]
        next_idx = max(indices) + 1 if indices else 1
        
    new_dir = os.path.join(base_dir, f"test_{next_idx:02d}")
    os.makedirs(new_dir, exist_ok=True) 
    return new_dir

def generate_dynamic_context():
    times = ["early morning", "mid-morning", "post-lunch period", "late afternoon"]
    subjects = ["Calculus lecture", "History reading", "interactive discussion"]
    events = ["stressful exam coming up", "teacher is strict", "teacher has back turned"]
    return f"Time: {random.choice(times)}. Subject: {random.choice(subjects)}. Event: {random.choice(events)}."

# ==========================================
# 2. GEMINI GRAPH GENERATION WITH STRICT PROMPT & RETRY LOGIC
# ==========================================
def generate_valid_graph_from_llm(min_edges=7, max_edges=15, max_attempts=10):
    context = generate_dynamic_context()
    
    prompt = f"""You are an Expert Data Scientist modeling a classroom causal graph (DAG) for STUDENT behaviors.
Context: {context}

AVAILABLE BEHAVIORS AND THEIR STRICT DEFINITIONS:
- 'Talk': A student talking to peers.
- 'Read': A student reading the textbook.
- 'Phone': A student secretly using a smartphone.
- 'Hand': A student RAISING THEIR HAND to ask a question (DO NOT interpret this as holding an object).
- 'Lean': A student slouching/leaning on the desk due to fatigue/boredom.
- 'Stand': A student standing up from their chair.

STRICT CAUSAL RULES (CRITICAL):
1. Output MUST be a STRICT Directed Acyclic Graph (DAG). NO feedback loops.
2. NO TRANSITIVE SHORTCUTS. 
3. CAUSALITY, NOT CORRELATION. 
4. DO NOT create self-loops (A->A).
5. All behaviors apply to the STUDENTS, not the teacher.
6. Output format: JSON object with a single key "edges" containing a list of edges.
Each edge MUST have: "source", "target", "type" ('positive' or 'negative'), "strength" ('medium', 'high'), and "reasoning".
7. THINK IN PEER INFLUENCE & CONTAGION, NOT INDIVIDUAL SEQUENCES.
   - WRONG: "Raising hand causes standing." (Individual action sequence).
   - RIGHT: "Widespread talking creates a noisy environment, causing others to stop reading." (Macro contagion).

Generate EXACTLY 4 to 6 highly logical, unambiguous causal edges. Keep the graph SPARSE.
"""

    attempt = 1
    while attempt <= max_attempts:
        try:
            if attempt > 1: print(f"      [LLM] Retrying generation (Attempt {attempt})...")

            parsed_data = generate_json_with_gemini(
                prompt,
                system_instruction="You are a strict classroom DAG generator for synthetic data. Return only a valid JSON object.",
                max_output_tokens=2000,
            )
            raw_edges = parsed_data.get("edges", [])
            
            # Validate the generated graph before using it.
            sanitized_edges = sanitize_llm_graph(raw_edges)
            
            # Ensure the graph remains dense enough for the benchmark.
            if len(sanitized_edges) < min_edges:
                print(f"      [POLICY] Graph too sparse after sanitize (Only {len(sanitized_edges)} edges). Rejecting and Retrying...")
                attempt += 1
                continue
                
            return sanitized_edges, context
            
        except Exception as e:
            if not is_retryable_error(e):
                raise RuntimeError(f"Gemini graph generation failed with a non-retryable API error: {e}") from e
            if attempt >= max_attempts:
                raise RuntimeError(f"Gemini graph generation failed after {max_attempts} attempts.") from e
            print(f"      [API Error] {e}. Resting 3s...")
            time.sleep(3)
            attempt += 1
    raise RuntimeError(f"Gemini graph generation failed after {max_attempts} attempts.")

# ==========================================
# 3. GRAPH SANITIZATION
# ==========================================
def has_cycle(graph_dict):
    """Detect cycles with depth-first search."""
    visited = set()
    rec_stack = set()
    
    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        for neighbor in graph_dict.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor): return True
            elif neighbor in rec_stack:
                return True
        rec_stack.remove(node)
        return False

    for node in graph_dict:
        if node not in visited:
            if dfs(node): return True
        visited.clear() 
    return False

def sanitize_llm_graph(raw_edges):
    valid_edges = []
    seen_pairs = set()
    
    # Step 1: filter unknown nodes, self-loops, duplicates, and reverse pairs.
    for e in raw_edges:
        src, tgt = e.get("source"), e.get("target")
        if src not in BEHAVIORS or tgt not in BEHAVIORS: continue
        if src == tgt: continue 
        
        if (src, tgt) in seen_pairs or (tgt, src) in seen_pairs:
            continue
            
        seen_pairs.add((src, tgt))
        valid_edges.append(e)
        
    # Step 2: remove edges that would introduce cycles.
    final_edges = []
    adj_list = {b: [] for b in BEHAVIORS}
    
    for e in valid_edges:
        src, tgt = e["source"], e["target"]
        adj_list[src].append(tgt)
        
        if has_cycle(adj_list):
            adj_list[src].remove(tgt) 
            print(f"      [POLICY] Detected & Removed Cycle-inducing edge: {src} -> {tgt}")
        else:
            final_edges.append(e)
            
    return final_edges

# ==========================================
# 4. VAR MATRIX PARSER
# ==========================================
def build_matrices(edges_json):
    W = np.zeros((NUM_VARS, NUM_VARS))
    A = np.zeros((NUM_VARS, NUM_VARS), dtype=int)
    for edge in edges_json:
        src, tgt = edge.get("source"), edge.get("target")
        e_type, strength = edge.get("type"), edge.get("strength", "low")
        
        if src not in BEH_IDX or tgt not in BEH_IDX or src == tgt: continue
            
        i, j = BEH_IDX[src], BEH_IDX[tgt]
        val = STRENGTH_MAP.get(strength, 0.5)
        
        if e_type == "negative": W[i, j] = -val
        elif e_type == "positive": W[i, j] = val
        A[i, j] = 1
    return W, A

# ==========================================
# 5. CORE VAR(1) SIMULATION
# ==========================================
def run_simulation(W, num_steps=1000, num_students=100):
    T = 1.0 
    sim_vars = NUM_VARS + 1 
    
    W_sim = np.zeros((sim_vars, sim_vars))
    W_sim[:NUM_VARS, :NUM_VARS] = W
    
    baseline = np.zeros(sim_vars)
    baseline[NUM_VARS] = 1.0
    
    S = np.random.randint(0, sim_vars, size=num_students)
    X_observed = np.zeros((num_steps, NUM_VARS)) 
    
    for t in range(1, num_steps):
        # Peer influence from the current classroom state.
        current_pct = np.bincount(S, minlength=sim_vars) / num_students
        peer_influence = np.dot(current_pct, W_sim)
        
        # Individual inertia keeps each simulated student near their current state.
        S_onehot = np.eye(sim_vars)[S]
        individual_inertia = S_onehot * 3.5
        
        # Combine individual inertia, peer influence, baseline, and noise.
        M_individuals = individual_inertia + peer_influence + baseline + np.random.normal(0, 0.3, size=(num_students, sim_vars))
        
        P = softmax(M_individuals / T, axis=1)
        cum_P = np.cumsum(P, axis=1)
        random_draws = np.random.rand(num_students, 1)
        S = np.argmax(random_draws < cum_P, axis=1)
        
        counts = np.bincount(S, minlength=sim_vars)
        X_observed[t] = (counts[:NUM_VARS] / num_students) * 100.0

    # Add light observation noise.
    eta = np.random.uniform(0.99, 1.0, size=(num_steps, 1)) 
    X_noisy = (X_observed * eta) + np.random.normal(0, 0.1, size=(num_steps, NUM_VARS))
    
    return pd.DataFrame(np.clip(X_noisy, 0, 100.0), columns=BEHAVIORS)

# ==========================================
# 6. TEST SUITE ORCHESTRATION
# ==========================================
def generate_test_suite(num_tests=5, output_dir=None):
    print(f"[SYSTEM] Generating {num_tests} causal test cases with {GEMINI_MODEL}...")
    for i in range(num_tests):
        folder = get_next_test_folder(output_dir)
        try:
            sanitized_edges, context = generate_valid_graph_from_llm(min_edges=4, max_edges=10)
            
            W, A = build_matrices(sanitized_edges)
            df = run_simulation(W, num_steps=1000, num_students=100)
            
            graph_data = {
                "scenario_context": context,
                "behaviors": BEHAVIORS,
                "edges_logic_from_llm": sanitized_edges, 
                "matrices": {
                    "W_GroundTruth_Weights": W.tolist(),
                    "A_GroundTruth_Adjacency": A.tolist()
                }
            }
            
            with open(os.path.join(folder, "ground_truth_graph.json"), "w", encoding="utf-8") as f:
                json.dump(graph_data, f, indent=4)
                
            df.to_csv(os.path.join(folder, "time_series_noisy.csv"), index=False)
            print(f"   -> [SUCCESS] Test {folder} generated. Retained {len(sanitized_edges)} valid edges.")
        except Exception as e:
            print(f"   -> [FATAL ERROR] Pipeline crashed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=5)
    parser.add_argument("--output-dir", default=str(DATA_DIR / "synth_data"))
    args = parser.parse_args()
    generate_test_suite(num_tests=args.num, output_dir=Path(args.output_dir))
