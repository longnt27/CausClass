import torch
import torch.nn as nn
import numpy as np
import copy
from core.models.aerca import AERCA

# ==========================================
# PHASE 0: BASELINE DISCOVERY (COMPUTE INTENSIVE)
# ==========================================
def extract_aerca_matrices(data_chunks, variables):
    """
    [PHASE 0 - STEP 1] RAW MATRIX EXTRACTION
    Trains the AERCA Neural Network to discover the underlying Vector Autoregression (VAR) dynamics.
    This is a computationally heavy operation (GPU intensive) and should be executed only ONCE per dataset.
    
    Returns:
        est_matrix: Absolute median coefficients (used for edge detection via thresholding).
        signed_matrix: Mean directional coefficients (used to determine positive/negative relationships).
        dense_weights: The full state_dict of the trained model for Warm-Starting future phases.
    """
    print("\n[AERCA Engine] Initializing Neural Network for Raw Matrix Extraction...")
    n_vars = len(variables)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Enforce strict determinism for reproducible scientific results
    torch.manual_seed(42)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(42)
    
    model = AERCA(num_vars=n_vars, hidden_layer_size=64, num_hidden_layers=2,
                  device=device, window_size=1, stride=1, epochs=150, lr=0.005,
                  data_name='init', causal_quantile=0.0)
    
    model._training(data_chunks)
    model.eval()
    
    with torch.no_grad():
        _, _, _, encoder_coeffs, _, _, _, _ = model._testing_step(data_chunks[0], add_u=False)
        # est_matrix defines the STRUCTURE (Skeleton)
        est_matrix = torch.max(torch.median(torch.abs(encoder_coeffs), dim=0)[0], dim=0).values.cpu().numpy()
        # signed_matrix defines the BEHAVIORAL NATURE (Positive/Negative correlation)
        signed_matrix = torch.mean(torch.mean(encoder_coeffs, dim=0), dim=0).cpu().numpy()

    # Prevent self-loops in the initial discovery phase
    np.fill_diagonal(est_matrix, 0.0)
    
    # Return dense weights for warm-starting later graph-search evaluations.
    return est_matrix, signed_matrix, copy.deepcopy(model.state_dict())

def build_baseline_graph(est_matrix, signed_matrix, variables, q_threshold):
    """
    [PHASE 0 - STEP 2] THRESHOLD PRUNING (LIGHTWEIGHT)
    Applies a statistical percentile cutoff to the raw matrix to form the initial directed graph.
    This operation is O(1) in terms of compute and can be iterated thousands of times for Grid Search.
    """
    n_vars = len(variables)
    cutoff_val = np.quantile(est_matrix, q_threshold)
    pred_dag = (est_matrix >= cutoff_val).astype(int)
    
    initial_edges = []
    for i in range(n_vars):
        for j in range(n_vars):
            if pred_dag[i, j] == 1:
                initial_edges.append({
                    "source": variables[i], 
                    "target": variables[j],
                    "weight": float(signed_matrix[i, j])
                })
                
    return initial_edges

def get_initial_graph_from_aerca(data_chunks, variables, default_threshold=0.75):
    """
    [BACKWARD COMPATIBILITY WRAPPER]
    Combines Step 1 and Step 2 for legacy scripts that expect a single call.
    """
    print(f"[AERCA Engine] Executing Phase 0 with a default threshold of {default_threshold}...")
    
    # Preserve dense model weights for masked verifier runs.
    est_matrix, signed_matrix, dense_weights = extract_aerca_matrices(data_chunks, variables)
    initial_edges = build_baseline_graph(est_matrix, signed_matrix, variables, q_threshold=default_threshold)
    
    print(f"[AERCA Engine] Baseline Discovery complete. Found {len(initial_edges)} initial edges.")
    
    # Return all discovery artifacts for the graph-search runners.
    return initial_edges, dense_weights, est_matrix

# ==========================================
# PHASE 1 & 2: MASKED VALIDATION
# ==========================================
def create_mask_matrix(edge_list, var2idx, n_vars):
    """Generates a binary adjacency mask to forcefully block backpropagation on rejected edges."""
    mask = torch.zeros((n_vars, n_vars))
    for edge in edge_list:
        if edge['source'] in var2idx and edge['target'] in var2idx:
            mask[var2idx[edge['source']], var2idx[edge['target']]] = 1.0
            
    # Self-loops (Inertia) are inherently allowed in VAR models
    mask.fill_diagonal_(1.0)
    return mask

def run_masked_aerca(data_chunks, edge_list, variables, init_weights=None):
    """
    Validates a proposed graph structure by forcing the AERCA model to predict 
    the multivariate time-series using ONLY the allowed edges.
    Calculates the Validation Mean Squared Error (MSE) to feed the BIC evaluation.
    """
    import copy
    import torch
    import torch.nn as nn
    from core.models.aerca import AERCA

    n_vars = len(variables)
    var2idx = {v: i for i, v in enumerate(variables)}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Keep seed handling deterministic for reproducible verifier scores.
    torch.manual_seed(42)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(42)
        
    # Use fewer epochs for lightweight masked fine-tuning.
    model = AERCA(num_vars=n_vars, hidden_layer_size=64, num_hidden_layers=2,
                  device=device, window_size=1, stride=1, epochs=15, lr=0.005,
                  data_name='masked', causal_quantile=0.0)
    
    # Warm Start: Load previous neural state to accelerate convergence
    if init_weights is not None:
        model.load_state_dict(init_weights)
        
    # Apply the adjacency mask inside the SENNGC forward pass.
    mask_tensor = create_mask_matrix(edge_list, var2idx, n_vars).to(device)
    model.causal_mask = mask_tensor 
    
    # =================================================================
    # 4. Train/validation split
    # =================================================================
    n_chunks = len(data_chunks)
    train_size = max(1, int(n_chunks * 0.8))
    train_chunks = data_chunks[:train_size]
    val_chunks = data_chunks[train_size:]
    
    if len(val_chunks) == 0: 
        val_chunks = data_chunks
        
    # Train only on the training split while enforcing the graph mask.
    model._training(train_chunks)
    
    model.eval()
    total_mse = 0.0
    
    with torch.no_grad():
        # Extract signed weights from the first chunk for sign-aware edge reporting.
        _, _, _, encoder_coeffs, _, _, _, _ = model._testing_step(data_chunks[0], add_u=False)
        signed_matrix = torch.mean(torch.mean(encoder_coeffs, dim=0), dim=0).cpu().numpy()

        # Reapply the mask after extraction to remove any residual unmasked weights.
        signed_matrix = signed_matrix * mask_tensor.cpu().numpy()
        
        mse_loss_fn = nn.MSELoss()
        
        # =================================================================
        # 5. Evaluate MSE on the held-out validation split
        # =================================================================
        for chunk in val_chunks: 
            nexts_hat, nexts, _, _, _, _, _ = model.forward(chunk, add_u=False)
            loss = mse_loss_fn(nexts_hat, nexts)
            total_mse += loss.item()
            
    # Calculate the true validation mean squared error
    avg_mse = total_mse / len(val_chunks)
    
    # Inject learned weights back into the edge structure for Sign-Aware evaluation
    updated_edges = []
    for edge in edge_list:
        i, j = var2idx[edge['source']], var2idx[edge['target']]
        updated_edge = copy.deepcopy(edge)
        updated_edge['weight'] = float(signed_matrix[i, j])
        updated_edges.append(updated_edge)
        
    return avg_mse, len(updated_edges), copy.deepcopy(model.state_dict()), updated_edges
