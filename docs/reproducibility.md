# Reproduction protocol and evidence status

## Two different claims

A successful smoke run or CI job verifies software execution. It does not
reproduce a thesis table, validate a causal claim, establish generalization to
another classroom, or demonstrate parity with the upstream AERCA publication.
The report imported from `947714f3d1eb517c19e697f394763b1f7e8e9944` is historical.
The pre-upgrade code is retained at `c4c6e8622b4473a104a6b757a9ff2479708ec8c0`.

## Protocol v2: intentional result-changing corrections

1. **Direction:** public graph matrices use `[source, target]`. SENNGC predicts
   with `coefficient @ input`, so its coefficients use `[target, source]`.
   Extraction now transposes to the public convention; masks transpose back.
   Previously, labeled edges and applied masks could describe the opposite direction.
2. **Objective:** Gaussian KL against a standard normal now uses
   `0.5 * (trace(cov) + mean.T @ mean - dimension - logdet(cov))`.
   The historical helper used the opposite log-determinant sign and omitted 0.5.
   A dtype-sized diagonal jitter stabilizes the empirical covariance. Tests
   compare against PyTorch's Gaussian KL implementation, including gradients.
3. **Data isolation:** the last 20% of chronological chunks (or time points for
   a single chunk) is reserved for graph scoring BEFORE dense pretraining.
   Dense and masked training use only the earlier 80%. A separate internal
   early-stopping split is made within that fit partition. Previously, graph
   scoring data could influence dense model selection, and a single chunk could
   fall back to being scored on the training series.
4. **Boundary conditions:** zero strengths and diagonal entries never become
   reported baseline edges; autoregressive diagonal terms remain active inside
   the model. Unknown mask variables, nonfinite input and undersized data fail
   clearly. Empty graphs now export a header-only edge CSV without a KeyError.
5. **Ontology:** the standalone CSV discovery command now uses the current
   `Talk, Read, Phone, Write, Lean, Bow` ontology instead of the old Hand/Stand set.

Do not compare v2 output to report tables as though only file organization
changed. Regenerate results, review the conclusions and record the exact commit.
The upstream-style anomaly runner in `core/legacy/` is not a supported benchmark.

## What the score does and does not mean

The graph-scoring partition is reused to choose edits and hyperparameters. It is
**selection/validation data, not an untouched final test set**. Reserve independent
sessions or synthetic suites for final reporting, with all settings fixed before
accessing their results. Do not select settings using the final test F1.

The existing `calculate_bic` is a BIC-style MSE/edge penalty search proxy. Its
user-selected per-edge penalty is not a demonstrated parameter count of the
neural network and its sample-size convention is inherited from existing runners.
Treat it as a search objective, not a calibrated likelihood ratio or causal proof.
Temporal graphs may contain reciprocal lagged connections; the software does not
promise an instantaneous acyclic DAG despite historical function names.

The historical LLM prompt makes strong behavior-contagion assumptions. It is
retained for traceability, not endorsed as an empirical exclusion of confounding.
Observational video, compositional behavior shares, detection errors, omitted
variables, camera changes and temporal aggregation can all change conclusions.

## Executable reference run

```bash
uv sync --locked --extra dev
uv run --no-sync causclass-smoke --output-dir output/reference-001 --seed 42
uv run --no-sync pytest --cov
```

Smoke artifacts: `time_series.csv`, `ground_truth_graph.json`, `graph.json`,
`metrics.json`, `config.json`, and `run_metadata.json`. Metadata includes protocol,
seed, commit/dirty status, package versions and input SHA-256 digests, not secrets.
The ground-truth JSON keeps a legacy schema key for evaluator compatibility; its
`generator` field explicitly states that no LLM generated it.

The reusable `VerifierConfig` controls seed, CPU/CUDA, epochs, hidden layers and
learning rate. Defaults retain the prior sizes (150 dense / 15 masked epochs,
64 hidden units, two layers, lr 0.005); smoke uses two epochs and eight units.
Use the same config for dense and masked stages. Supply at least two variables,
20 finite time points per chunk, and chronologically ordered, non-overlapping
chunks. Existing video chunking drops an incomplete trailing chunk when full
chunks exist; record this preprocessing choice rather than silently changing it.

`seed_everything` seeds Python, NumPy and PyTorch and disables cuDNN benchmarking.
Bitwise equality across GPU/CPU, library versions, hardware or LLM providers is
not guaranteed. CI checks repeatability on one fixed CPU environment.

## Reproducing a report experiment

For each experiment, retain the original data/model checksum and access terms,
preprocessing and ontology, subject/session split, exact commit, environment lock
(or a GPU `pip freeze`), configuration and seed, provider/model ID, saved prompts
and responses, final metrics, and failed attempts. Never put credentials in an
experiment manifest. Use `utils.reproducibility.run_metadata` as a starting point;
legacy runners do not all produce that full manifest automatically.

The tracked tree does not contain the complete thesis training data, all original
synthetic suites, detector training procedure or original run manifests. LLM
regeneration is not a substitute for the exact original suite. Multi-seed runs,
confidence intervals, held-out session validation and regeneration of thesis
figures/tables remain research work, not claims made by the software checks.

For parallel GPU jobs, begin with one worker per GPU and monitor memory. This
revision removes transient verifier checkpoint writes and threshold calibration
that graph scoring never uses, but it does not implement distributed scheduling.

## Sources for the software contract

- Gaussian distributions and KL: https://docs.pytorch.org/docs/stable/distributions.html
- Reproducibility limitations: https://docs.pytorch.org/docs/stable/notes/randomness.html
- Original AERCA: https://openreview.net/forum?id=k38Th3x4d9
