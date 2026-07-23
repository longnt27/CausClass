import os
import json
import numpy as np
import pandas as pd
import torch
from core.models.aerca import AERCA
from utils.paths import DATA_DIR

def load_and_prepare_data(csv_file, json_file, chunk_size=500):
    """Load the CSV, split it into chunks, and build the ground-truth DAG."""
    print(f"[*] Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    data_array = df.values
    
    # Split the long time series into fixed-size chunks.
    num_chunks = len(data_array) // chunk_size
    xs = np.array([data_array[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)])
    print(f"[*] AERCA input shape: {xs.shape} (Samples x TimeSteps x Vars)")
    
    # Build the ground-truth causal graph from JSON.
    with open(json_file, 'r') as f:
        config = json.load(f)
        
    var2idx = {v: i for i, v in enumerate(config['variables'])}
    n_vars = len(var2idx)
    
    # Ground-truth adjacency matrix for discovery evaluation.
    causal_struct = np.zeros((n_vars, n_vars))
    all_edges = config['micro_edges']
    for edge in all_edges:
        src = var2idx[edge['source']]
        tgt = var2idx[edge['target']]
        causal_struct[src, tgt] = 1.0
        
    return xs, causal_struct, n_vars

if __name__ == "__main__":
    # --- 1. SETTINGS ---
    DATA_FILE = DATA_DIR / 'raw_exclusive_4s.csv'
    CONFIG_FILE = DATA_DIR / 'classroom_dag.json'
    EPOCHS = 200
    WINDOW_K = 1

    if not DATA_FILE.exists() or not CONFIG_FILE.exists():
        raise SystemExit(
            "Legacy classroom test inputs are missing. Expected "
            f"{DATA_FILE} and {CONFIG_FILE}."
        )
    
    # --- 2. Data preparation ---
    xs, gt_dag, num_vars = load_and_prepare_data(DATA_FILE, CONFIG_FILE, chunk_size=500)
    
    test_idx = int(0.8 * len(xs))
    xs_train_val = xs
    xs_test = xs[test_idx:]
    
    # --- 3. Initialize AERCA ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Initializing AERCA model on device: {device.type.upper()}")
    
    model = AERCA(
        num_vars=num_vars,
        hidden_layer_size=100,
        num_hidden_layers=3,
        device=device,
        window_size=WINDOW_K,
        stride=1,
        epochs=EPOCHS,
        lr=0.005,
        data_name='classroom',
        causal_quantile=0.96
    )
    
    # --- 4. TRAIN & TEST ---
    print("\n==============================================")
    print("              STARTING AERCA TRAINING         ")
    print("==============================================")
    model._training(xs_train_val)
    
    print("\n==============================================")
    print("            EVALUATING CAUSAL DISCOVERY       ")
    print("==============================================")
    print("\n==============================================")
    print("             EXTRACTING AERCA WEIGHT MATRIX   ")
    print("==============================================")
    
    # Load the best checkpoint.
    checkpoint_path = os.path.join(model.save_dir, f'{model.model_name}_{os.getpid()}.pt')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        # Run a forward pass on the first test chunk.
        _, _, _, encoder_coeffs, _, _, _, _ = model._testing_step(xs_test[0], add_u=False)
        
        # Paper heuristic: max-median aggregation of coefficient weights.
        est_matrix = torch.max(torch.median(torch.abs(encoder_coeffs), dim=0)[0], dim=0).values.cpu().numpy()

    print("\n[+] Raw weight matrix:")
    print(np.round(est_matrix, 2))

    # Remove self-history from the discovery matrix before thresholding.
    np.fill_diagonal(est_matrix, 0.0)
    
    print("\n[+] Weight matrix after removing the diagonal:")
    print(np.round(est_matrix, 2))

    q_threshold = np.quantile(est_matrix, 0.85)
    pred_dag = (est_matrix >= q_threshold).astype(float)

    print("\n[+] Ground truth:")
    print(gt_dag)
    print("\n[+] AERCA predicted graph after thresholding:")
    print(pred_dag)

    from sklearn.metrics import f1_score, roc_auc_score
    gt_offdiag = gt_dag.flatten()
    pred_offdiag = pred_dag.flatten()
    
    f1 = f1_score(gt_offdiag, pred_offdiag)
    auroc = roc_auc_score(gt_offdiag, est_matrix.flatten())
    
    print(f"\n==============================================")
    print(f"[*] FINAL CUSTOM EVALUATION:")
    print(f"    - AUROC:    {auroc:.4f}")
    print(f"    - F1-Score: {f1:.4f}")
    print(f"==============================================")
    print("\n[+] Legacy classroom test completed.")
