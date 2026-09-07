"""AERCA discovery and masked graph scoring, research protocol v2.

Public matrices are [source, target]. SENNGC coefficients are [target, source].
See docs/reproducibility.md before comparing to historical report results.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from core.models.aerca import AERCA
from utils.reproducibility import seed_everything, split_verifier_data


@dataclass(frozen=True)
class VerifierConfig:
    seed: int = 42
    dense_epochs: int = 150
    masked_epochs: int = 15
    hidden_layer_size: int = 64
    num_hidden_layers: int = 2
    lr: float = 0.005
    device: str = "auto"

    def __post_init__(self):
        for value in (
            self.dense_epochs,
            self.masked_epochs,
            self.hidden_layer_size,
            self.num_hidden_layers,
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("epoch and layer sizes must be positive integers")
        if not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")


def _model(variables, config, *, masked=False):
    if len(set(variables)) != len(variables):
        raise ValueError("variable names must be unique")
    seed_everything(config.seed)
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return AERCA(
        num_vars=len(variables),
        hidden_layer_size=config.hidden_layer_size,
        num_hidden_layers=config.num_hidden_layers,
        device=torch.device(device),
        window_size=1,
        stride=1,
        epochs=config.masked_epochs if masked else config.dense_epochs,
        lr=config.lr,
        data_name="masked" if masked else "init",
        causal_quantile=0.0,
    )


def _source_target_matrices(model, chunk):
    with torch.no_grad():
        coeffs = model._testing_step(chunk, add_u=False)[3]
        magnitude = torch.max(torch.median(torch.abs(coeffs), dim=0)[0], dim=0).values
        signed = coeffs.mean(dim=(0, 1))
    # Matrix-vector multiplication inside SENNGC indexes output before input.
    return magnitude.cpu().numpy().T.copy(), signed.cpu().numpy().T.copy()


def extract_aerca_matrices(data_chunks, variables, *, config=None):
    config = config or VerifierConfig()
    fit, _ = split_verifier_data(data_chunks, len(variables))
    model = _model(variables, config)
    model._training(fit, persist=False, calibrate=False)
    model.eval()
    magnitude, signed = _source_target_matrices(model, fit[0])
    np.fill_diagonal(magnitude, 0.0)
    return magnitude, signed, copy.deepcopy(model.state_dict())


def build_baseline_graph(est_matrix, signed_matrix, variables, q_threshold):
    """Threshold [source, target] strengths, excluding self/zero edges.

    Quantiles still include diagonal zeros for compatibility with the historical
    threshold setting. This is a directed temporal graph, not necessarily a DAG.
    """
    estimate, signed = np.asarray(est_matrix), np.asarray(signed_matrix)
    shape = (len(variables), len(variables))
    if len(variables) < 2 or len(set(variables)) != len(variables):
        raise ValueError("provide at least two unique variables")
    if estimate.shape != shape or signed.shape != shape:
        raise ValueError("matrix shapes must match the variables")
    if not np.isfinite(estimate).all() or not np.isfinite(signed).all():
        raise ValueError("matrices must be finite")
    if (estimate < 0).any() or not 0 <= q_threshold <= 1:
        raise ValueError("strengths must be nonnegative and quantile in [0, 1]")
    selected = (estimate >= np.quantile(estimate, q_threshold)) & (estimate > 0)
    np.fill_diagonal(selected, False)
    return [
        {"source": variables[i], "target": variables[j], "weight": float(signed[i, j])}
        for i, j in zip(*np.nonzero(selected))
    ]


def get_initial_graph_from_aerca(data_chunks, variables, default_threshold=0.75, *, config=None):
    magnitude, signed, weights = extract_aerca_matrices(data_chunks, variables, config=config)
    edges = build_baseline_graph(magnitude, signed, variables, default_threshold)
    return edges, weights, magnitude


def create_mask_matrix(edge_list, var2idx, n_vars):
    """Return a [source, target] mask. Autoregressive self effects stay enabled."""
    if len(var2idx) != n_vars or set(var2idx.values()) != set(range(n_vars)):
        raise ValueError("variable indices must be unique and contiguous")
    mask = torch.eye(n_vars)
    for edge in edge_list:
        if edge.get("source") not in var2idx or edge.get("target") not in var2idx:
            raise ValueError("edge contains an unknown variable")
        mask[var2idx[edge["source"]], var2idx[edge["target"]]] = 1.0
    return mask


def run_masked_aerca(data_chunks, edge_list, variables, init_weights=None, *, config=None):
    """Score a candidate on held-out graph-selection data, not a final test set.

    Warm-start weights must come from this protocol's fit partition. Externally
    supplied checkpoints cannot be automatically checked for data leakage.
    """
    config = config or VerifierConfig()
    fit, score = split_verifier_data(data_chunks, len(variables))
    var2idx = {name: i for i, name in enumerate(variables)}
    model = _model(variables, config, masked=True)
    if init_weights is not None:
        model.load_state_dict(init_weights)
    public_mask = create_mask_matrix(edge_list, var2idx, len(variables))
    model.causal_mask = public_mask.T.to(model.device)
    model._training(fit, persist=False, calibrate=False)
    model.eval()
    _, signed = _source_target_matrices(model, fit[0])
    signed *= public_mask.numpy()
    squared_error, count = 0.0, 0
    with torch.no_grad():
        for chunk in score:
            predictions, targets, *_ = model.forward(chunk, add_u=False)
            squared_error += torch.sum((predictions - targets) ** 2).item()
            count += targets.numel()
    mse = squared_error / count
    if not np.isfinite(mse):
        raise ValueError("nonfinite verifier score")
    updated = copy.deepcopy(edge_list)
    for edge in updated:
        edge["weight"] = float(signed[var2idx[edge["source"]], var2idx[edge["target"]]])
    return mse, len(updated), copy.deepcopy(model.state_dict()), updated
