# CausClass

CausClass turns classroom video into behavior time series, searches for plausible
temporal behavior graphs, and exports artifacts that can be reviewed by a human.
It uses YOLO/ByteTrack for perception, Gemini 3 Flash for synthetic data
generation, LLM-guided graph search for proposal generation, and AERCA as the
neural verifier for candidate causal graphs.

AERCA is used here only as the verification engine inside CausClass. Please cite
the original AERCA paper if you use this project:

```bibtex
@inproceedings{
   han2025root,
   title={Root Cause Analysis of Anomalies in Multivariate Time Series through Granger Causal Discovery},
   author={Xiao Han and Saima Absar and Lu Zhang and Shuhan Yuan},
   booktitle={The Thirteenth International Conference on Learning Representations},
   year={2025},
   url={https://openreview.net/forum?id=k38Th3x4d9}
}
```

## Installation

Use Python 3.11 if possible.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA training/inference, install the PyTorch build that matches your CUDA
driver from the official PyTorch install selector, then install this project’s
remaining dependencies.

Create a root-level `.env` file:

```bash
GEMINI_API_KEY=your_gemini_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
```

`GEMINI_API_KEY` is required for synthetic data generation. The generators use
Gemini 3 Flash with model id `gemini-3-flash-preview` by default. Override it with
`GEMINI_SYNTHETIC_MODEL=...` if Google changes the recommended model name.

`DEEPSEEK_API_KEY` is required for graph-edit proposals and the optional final
teacher-facing report.

Optional for local dry runs:

```bash
WANDB_MODE=disabled
```

## Common Commands

Run the end-to-end video pipeline:

```bash
python -m core.end_to_end_pipeline \
  --video /path/to/classroom.mp4 \
  --model data/best.pt \
  --output_dir output/runs/classroom_001
```

Extract only video dynamics:

```bash
python -m data.extract_video_dynamics /path/to/classroom.mp4 \
  --model data/best.pt \
  --output-dir output/phase1/classroom_001
```

Run causal discovery on an existing CSV:

```bash
python -m core.run_causal_discovery \
  --input output/phase1/classroom_001/classroom_multivariate_timeseries.csv \
  --output output/final_causal_graph.json
```

Generate aligned synthetic VAR data with Gemini 3 Flash:

```bash
python -m data.generate_synthetic_data -n 10 --output-dir data/synth_data
```

Run one synthetic-suite graph-search pass:

```bash
python -m core.run_synthetic_search --debug 1 --workers 1
```

Run BIC-lambda grid search:

```bash
python -m core.grid_search_bic_lambda --test_limit 10 --workers 1
```

Run verifier-threshold sweep:

```bash
python -m core.sweep_aerca_thresholds --workers 1
```

## Project Tree

Bulky data and output folders are shown by pattern instead of every generated
image, CSV, JSON, and frame file.

```text
.
├── LICENSE
├── README.md
├── requirements.txt
├── .gitignore
├── .env
├── core/
│   ├── __init__.py
│   ├── end_to_end_pipeline.py
│   ├── aerca_verifier.py
│   ├── run_synthetic_search.py
│   ├── run_synthetic_ablation.py
│   ├── grid_search_hyperparams.py
│   ├── grid_search_bic_lambda.py
│   ├── graph_edit_agent.py
│   ├── sweep_aerca_thresholds.py
│   ├── diagnose_synthetic_suite.py
│   ├── run_causal_discovery.py
│   ├── verify_gpu.py
│   ├── legacy/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── run_classroom_test.py
│   │   └── test_all_datasets.py
│   └── models/
│       ├── __init__.py
│       ├── aerca.py
│       └── senn.py
├── data/
│   ├── __init__.py
│   ├── best.pt
│   ├── prepare_cvat_dataset.py
│   ├── extract_video_dynamics.py
│   ├── generate_legacy_synthetic_data.py
│   ├── generate_synthetic_data.py
│   ├── cvat_dataset/
│   │   ├── classes.txt
│   │   ├── images/*.jpg
│   │   └── labels/*.txt
│   └── synth_data/
│       └── test_XX/
│           ├── ground_truth_graph.json
│           └── time_series_noisy.csv
├── output/
│   ├── ablation_phase2_test01/
│   ├── grid_search_results.csv
│   ├── lambda_grid_runs/
│   └── runs/
└── utils/
    ├── __init__.py
    ├── aerca_utils.py
    ├── helpers.py
    └── paths.py
```

## File Usage

Top-level files:

- `LICENSE`: Project license.
- `README.md`: This project guide.
- `requirements.txt`: Python dependencies for perception, Gemini generation, graph search, verification, plotting, and experiment logging.
- `.gitignore`: Ignores local secrets, bytecode caches, macOS metadata, and generated verifier checkpoints.
- `.env`: Local API keys and optional runtime settings.

Core runtime:

- `core/end_to_end_pipeline.py`: Full video-to-report workflow. Extracts behavior events, builds verifier input chunks, runs LLM-guided graph search, saves graph/time-series artifacts, and optionally asks for a teacher-facing report.
- `core/aerca_verifier.py`: Shared verifier execution layer. Trains/extracts the dense discovery matrix, builds thresholded baseline graphs, creates masks, and evaluates masked graph candidates.
- `core/run_synthetic_search.py`: Batch synthetic graph-search runner for `data/synth_data/test_*`. Writes per-suite logs to `output/synthetic_runs/<timestamp>/`.
- `core/run_synthetic_ablation.py`: Ablation runner for comparing verifier baseline, random edits, greedy LLM search, and beam LLM search. Writes summaries to `output/ablation_runs/` by default.
- `core/grid_search_hyperparams.py`: Older broad grid search over threshold, BIC lambda, and minimum delta.
- `core/grid_search_bic_lambda.py`: Cleaner lambda sweep that computes each suite baseline once and evaluates many BIC penalties.
- `core/graph_edit_agent.py`: DeepSeek chat API wrapper for graph-edit proposals.
- `core/sweep_aerca_thresholds.py`: Quantile-threshold sweep for the verifier baseline.
- `core/diagnose_synthetic_suite.py`: Single synthetic-suite diagnostic for baseline plus one LLM proposal pass.
- `core/run_causal_discovery.py`: Causal discovery on an existing behavior time-series CSV.
- `core/verify_gpu.py`: CUDA and verifier import sanity check.

Verifier model code:

- `core/models/aerca.py`: AERCA neural verifier adapted to save checkpoints and thresholds under `output/saved_models/`.
- `core/models/senn.py`: Self-explaining neural network component used by the verifier.
- `core/models/__init__.py`: Package marker.

Legacy reference files:

- `core/legacy/main.py`: Original upstream-style runner; retained for reference and exits clearly because the old `args/` and `datasets/` packages are not part of CausClass.
- `core/legacy/test_all_datasets.py`: Original multi-dataset smoke runner; retained only as historical reference.
- `core/legacy/run_classroom_test.py`: Older custom classroom verifier test. It expects `data/raw_exclusive_4s.csv` and `data/classroom_dag.json`.

Data utilities and assets:

- `data/best.pt`: YOLO weights used by the video perception commands.
- `data/prepare_cvat_dataset.py`: Extracts frames and pseudo-labels into CVAT YOLO format.
- `data/extract_video_dynamics.py`: Standalone Phase 1 video perception script. Writes micro-events, time series, and optional demo video under `output/phase1/<video_stem>/` by default.
- `data/generate_legacy_synthetic_data.py`: Older Gemini-backed synthetic generator using the `Talk/Read/Phone/Hand/Lean/Stand` ontology.
- `data/generate_synthetic_data.py`: Current Gemini-backed synthetic VAR generator using the `Talk/Read/Phone/Write/Lean/Bow` ontology.
- `data/cvat_dataset/`: CVAT-ready image/label dataset.
- `data/synth_data/test_XX/ground_truth_graph.json`: Ground-truth graph metadata for one synthetic suite.
- `data/synth_data/test_XX/time_series_noisy.csv`: Noisy multivariate behavior time series for one synthetic suite.

Shared utilities:

- `utils/aerca_utils.py`: AERCA math/evaluation helpers retained for verifier internals.
- `utils/helpers.py`: Project helpers for `.env` loading, BIC calculation, DAG evaluation, graph hashing, graph edits, and dual logging.
- `utils/paths.py`: Shared `PROJECT_ROOT`, `DATA_DIR`, `OUTPUT_DIR`, and `ENV_FILE` constants.
- `utils/__init__.py`: Package marker.

Generated outputs:

- `output/runs/`: End-to-end pipeline outputs from real-video runs.
- `output/lambda_grid_runs/`: Lambda grid-search summaries and logs.
- `output/ablation_phase2_test01/`: Existing ablation result artifacts moved out of the root.
- `output/grid_search_results.csv`: Existing grid-search summary moved out of the root.
- `output/saved_models/`: Created automatically by verifier training for checkpoints and threshold arrays.
