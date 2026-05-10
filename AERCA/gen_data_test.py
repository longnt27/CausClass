import os
import json
import numpy as np
import pandas as pd
import random
import argparse
import time
from dotenv import load_dotenv
from openai import OpenAI

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not API_KEY:
    raise ValueError("[Initialization Error] DEEPSEEK_API_KEY is missing. Please verify your .env file.")

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

BEHAVIORS = ['Talk', 'Read', 'Phone', 'Hand', 'Lean', 'Stand']
NUM_VARS = len(BEHAVIORS)
BEH_IDX = {b: i for i, b in enumerate(BEHAVIORS)}

STRENGTH_MAP = {
    'high': 0.3,   
    'medium': 0.15
}

# ==========================================
# 1. DIRECTORY MANAGEMENT
# ==========================================
def get_next_test_folder(base_dir="synth_data"):
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
    times = ["early morning", "mid-morning", "post-lunch food coma", "late afternoon"]
    subjects = ["Calculus lecture", "boring History reading", "interactive discussion"]
    events = ["stressful exam coming up", "teacher is strict", "teacher has back turned"]
    return f"Time: {random.choice(times)}. Subject: {random.choice(subjects)}. Event: {random.choice(events)}."

# ==========================================
# 2. ALIGNED LLM GRAPH GENERATOR (THE VACCINE)
# ==========================================
def generate_valid_graph_from_llm(min_edges=4, max_edges=6):
    context = generate_dynamic_context()
    
    # [SIÊU PROMPT] Ép nó tư duy vật lý, cấm dùng biến ẩn tâm lý!
    prompt = f"""You are an Expert Data Scientist modeling a MACROSCOPIC classroom causal graph (DAG) for a Time-Series VAR(1) model.
Context: {context}

AVAILABLE BEHAVIORS:
- 'Talk': Widespread student-to-student chatting (Creates a noisy environment).
- 'Read': A high percentage of the class focusing on reading textbooks.
- 'Phone': Epidemic of covert smartphone usage.
- 'Hand': Multiple students raising hands to ask questions.
- 'Lean': Widespread slouching or leaning on desks.
- 'Stand': Groups of students standing up.

CRITICAL RULES FOR ALIGNMENT (YOUR GRAPH WILL BE REVIEWED BY A RUTHLESS PRUNER):
1. BEHAVIORS MUST PHYSICALLY CAUSE BEHAVIORS. Do not invent butterfly-effect causality.
   - ACCEPTABLE: "Widespread talking creates noise, reducing the percentage of students reading." (Talk -> Read, negative)
   - ACCEPTABLE: "High phone usage isolates students, reducing peer-to-peer talking." (Phone -> Talk, negative)
2. NO LATENT CONFOUNDERS (CRITICAL!): Do NOT create an edge between two behaviors if they are merely symptoms of a shared internal state (like 'boredom', 'disengagement', 'fatigue', or 'low energy'). 
   - UNACCEPTABLE (BULLSHIT): "Phone usage signals disengagement, reducing hand-raising." (Reject: Phone doesn't physically stop hands, boredom stops both. NO EDGE).
   - UNACCEPTABLE (BULLSHIT): "Leaning shows low energy, reducing hand-raising." (Reject: NO EDGE).
3. THINK IN PEER INFLUENCE & CONTAGION, NOT INDIVIDUAL SEQUENCES.
4. Output MUST be a STRICT Directed Acyclic Graph (DAG). NO feedback loops.
5. Output format: JSON object with a single key "edges" containing a list of edges.
Each edge MUST have: "source", "target", "type" ('positive' or 'negative'), "strength" ('medium', 'high' ONLY), and "reasoning".

Generate EXACTLY {min_edges} to {max_edges} bulletproof, highly logical MACRO-LEVEL causal edges that represent DIRECT PHYSICAL/ENVIRONMENTAL INFLUENCE.
"""

    attempt = 1
    while True:
        try:
            if attempt > 1: print(f"      [LLM] Retrying generation (Attempt {attempt})...")
            
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a strict, hyper-logical DAG generator. Return only valid JSON object."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}, 
                temperature=0.2, # Ép nó lạnh lùng tối đa, cấm bay bổng
                max_tokens=2000 
            )
            
            # Đã vứt bỏ trò cắt chuỗi ngu học. Dùng JSON thuần túy.
            raw_content = response.choices[0].message.content.strip()
            parsed_data = json.loads(raw_content)
            raw_edges = parsed_data.get("edges", [])
            
            sanitized_edges = sanitize_llm_graph(raw_edges)
            
            if len(sanitized_edges) < min_edges:
                print(f"      [POLICY] Graph too sparse after sanitize ({len(sanitized_edges)} edges). Retrying...")
                attempt += 1
                continue 
                
            return sanitized_edges, context
            
        except Exception as e:
            print(f"      [API Error] {e}. Resting 3s...")
            time.sleep(3)
            attempt += 1

# ==========================================
# 3. MATHEMATICAL BOUNCER
# ==========================================
def has_cycle(graph_dict):
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
    
    for e in raw_edges:
        src, tgt = e.get("source"), e.get("target")
        if src not in BEHAVIORS or tgt not in BEHAVIORS: continue
        if src == tgt: continue 
        
        if (src, tgt) in seen_pairs or (tgt, src) in seen_pairs:
            continue
            
        seen_pairs.add((src, tgt))
        valid_edges.append(e)
        
    final_edges = []
    adj_list = {b: [] for b in BEHAVIORS}
    
    for e in valid_edges:
        src, tgt = e["source"], e["target"]
        adj_list[src].append(tgt)
        
        if has_cycle(adj_list):
            adj_list[src].remove(tgt) 
        else:
            final_edges.append(e)
            
    return final_edges

# ==========================================
# 4. MATRICES BUILDER
# ==========================================
def build_matrices(edges_json):
    W = np.zeros((NUM_VARS, NUM_VARS))
    A = np.zeros((NUM_VARS, NUM_VARS), dtype=int)
    for edge in edges_json:
        src, tgt = edge.get("source"), edge.get("target") 
        e_type, strength = edge.get("type"), edge.get("strength", "low")
        
        if src not in BEH_IDX or tgt not in BEH_IDX or src == tgt: continue
            
        i, j = BEH_IDX[src], BEH_IDX[tgt]
        val = STRENGTH_MAP.get(strength, 0.4)
        
        if e_type == "negative": W[i, j] = -val
        elif e_type == "positive": W[i, j] = val
        A[i, j] = 1
    return W, A

# ==========================================
# 5. PURE VAR(1) MACRO ENGINE
# ==========================================
def run_simulation(W, num_steps=1000):
    X = np.zeros((num_steps, NUM_VARS))
    
    W_var = np.copy(W)
    np.fill_diagonal(W_var, 0.5) # Dropped slightly to 0.5 so cross-edges pop more
    
    # NEW: Heterogeneous Noise. 
    # Assign a unique, random volatility (standard deviation) to each variable.
    # E.g., 'Talk' might get 2.2, while 'Read' gets 0.8. 
    noise_scales = np.random.uniform(0.5, 2.5, size=NUM_VARS)
    
    X[0] = np.zeros(NUM_VARS)
    
    for t in range(1, num_steps):
        # Apply the unique noise scales instead of a flat 1.5
        noise = np.random.normal(0, noise_scales)
        X[t] = np.dot(X[t-1], W_var) + noise
        
    X = X[200:] # Burn-in
    
    X_shifted = X + 50.0
    X_final = np.clip(X_shifted, 0.0, 100.0)
        
    return pd.DataFrame(X_final, columns=BEHAVIORS)

# ==========================================
# 6. TEST SUITE ORCHESTRATION
# ==========================================
def generate_test_suite(num_tests=5):
    print(f"[SYSTEM] Generating {num_tests} PURE VAR(1) Test Cases with Aligned Logic...")
    for i in range(num_tests):
        folder = get_next_test_folder()
        try:
            sanitized_edges, context = generate_valid_graph_from_llm(min_edges=2, max_edges=4)
            
            W, A = build_matrices(sanitized_edges)
            df = run_simulation(W, num_steps=1000)
            
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
            print(f"   -> [SUCCESS] Test {folder} generated. Retained {len(sanitized_edges)} bulletproof edges.")
        except Exception as e:
            print(f"   -> [FATAL ERROR] Pipeline crashed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=10)
    args = parser.parse_args()
    
    if os.path.exists("synth_data"):
        import shutil
        shutil.rmtree("synth_data")
        
    generate_test_suite(num_tests=args.num)
