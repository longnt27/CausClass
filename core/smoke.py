"""Small deterministic CPU integration run; NOT a reproduction of thesis results."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from core.aerca_verifier import VerifierConfig, get_initial_graph_from_aerca, run_masked_aerca
from utils.helpers import evaluate_dag
from utils.reproducibility import run_metadata


def run(output_dir: Path, seed: int = 42) -> dict:
    output_dir = Path(output_dir)
    # Refuse accidental replacement of an experiment record.
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = VerifierConfig(
        seed=seed,
        dense_epochs=2,
        masked_epochs=2,
        hidden_layer_size=8,
        num_hidden_layers=1,
        device="cpu",
    )
    rng = np.random.default_rng(seed)
    variables = ["x0", "x1", "x2", "x3"]
    transition = np.eye(4) * 0.4
    transition[0, 1], transition[1, 2], transition[2, 3] = 0.3, -0.25, 0.2
    values = np.zeros((264, 4), dtype=np.float32)
    for t in range(1, len(values)):
        values[t] = values[t - 1] @ transition + rng.normal(0, 0.2, 4)
    values = values[64:]  # Discard burn-in; diagonal eigenvalues are all 0.4.
    csv_path = output_dir / "time_series.csv"
    np.savetxt(csv_path, values, delimiter=",", header=",".join(variables), comments="")
    truth_path = output_dir / "ground_truth_graph.json"
    truth = {
        "edges_logic_from_llm": [
            {"source": "x0", "target": "x1"},
            {"source": "x1", "target": "x2"},
            {"source": "x2", "target": "x3"},
        ],
        "generator": "offline_linear_VAR_not_LLM",
    }
    truth_path.write_text(json.dumps(truth, indent=2) + "\n")
    xs = values[None, :, :]
    edges, weights, _ = get_initial_graph_from_aerca(xs, variables, config=config)
    mse, count, _, graph = run_masked_aerca(xs, edges, variables, weights, config=config)
    precision, recall, f1, *_ = evaluate_dag(graph, truth_path, variables)
    metrics = {
        "selection_mse": mse,
        "num_edges": count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "purpose": "software_smoke_test_only",
    }
    artifacts = {
        "metrics.json": metrics,
        "graph.json": graph,
        "config.json": asdict(config),
        "run_metadata.json": run_metadata(seed, [csv_path, truth_path]),
    }
    for name, contents in artifacts.items():
        (output_dir / name).write_text(json.dumps(contents, indent=2, allow_nan=False) + "\n")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        metrics = run(args.output_dir, args.seed)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
