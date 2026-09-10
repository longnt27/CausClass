<div align="center">

# CausClass

### From classroom video to auditable temporal behavior graphs

A reproducible research framework that combines **video perception**, **LLM-guided graph search**, and **AERCA neural verification** to study temporal relationships between classroom behaviors.

[![Research CI](https://github.com/longnt27/CausClass/actions/workflows/ci.yml/badge.svg)](https://github.com/longnt27/CausClass/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11--3.13-3776AB?logo=python&logoColor=white)
![Protocol](https://img.shields.io/badge/research%20protocol-v2-7C3AED)
[![License](https://img.shields.io/badge/license-MIT-2EA44F)](LICENSE)

[Quick start](#quick-start) · [How it works](#how-it-works) · [Results & report](#results-and-thesis-report) · [Reproducibility](docs/reproducibility.md) · [Citation](#citation)

</div>

---

## What is CausClass?

CausClass is a research implementation for turning classroom video into behavior-share time series, searching over plausible temporal graph edits, and scoring candidate graphs with a neural verifier.

The repository contains both the research software and the complete thesis report. **The quantitative results in the thesis were produced from real experiment runs** and are part of the project's reported research evidence. The current repository also adds stronger packaging, testing, provenance, and CI around that work so future experiments are easier to reproduce and audit.

| Stage | Role | Main output |
| --- | --- | --- |
| **Perception** | YOLO + ByteTrack extract tracked classroom behavior signals | behavior-share time series |
| **Initial graph** | AERCA/SENNGC estimates a dense temporal dependency structure | weighted baseline graph |
| **Graph search** | An LLM proposes structured edge additions/deletions | candidate graphs |
| **Verification** | Masked AERCA scores candidates | ranked graph artifacts |
| **Interpretation** | Optional LLM pass summarizes the selected artifacts | observation report |
| **Provenance** | Research utilities record run metadata and input digests | reproducibility metadata |

## How it works

```mermaid
flowchart LR
    A[Classroom video] --> B[YOLO detections]
    B --> C[ByteTrack tracks]
    C --> D[Behavior-share time series]
    D --> E[AERCA dense baseline]
    E --> F[LLM graph-edit proposals]
    F --> G[Masked AERCA scoring]
    G --> H[Reviewed graph artifacts]
    H -. optional .-> I[LLM observation report]
```

The maintained public graph convention is **source row → target column**. Conversion to SENNGC's internal coefficient convention happens only at the verifier boundary and is regression-tested. See [architecture.md](docs/architecture.md) for the software contract.

> [!IMPORTANT]
> The thesis results are real experimental results from the implementation and protocol used for those runs. The current codebase includes protocol-v2 corrections and refactoring, so rerunning the same experiments on the newer implementation can produce different numbers. That distinction is about **versioned methodology**, not about whether the thesis results were actually run or verified.

## Quick start

The fastest path exercises the real verifier on CPU with **no video, GPU, API key, or paid service**.

### Requirements

- Python **3.11–3.13**
- [`uv`](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/longnt27/CausClass.git
cd CausClass

uv sync --locked --extra dev
uv run --no-sync causclass-smoke --output-dir output/smoke-001 --seed 42
uv run --no-sync pytest --cov
```

The smoke run generates a small deterministic VAR fixture and performs real dense + masked AERCA training on CPU. It writes:

```text
output/smoke-001/
├── time_series.csv
├── ground_truth_graph.json
├── graph.json
├── metrics.json
├── config.json
└── run_metadata.json
```

`run_metadata.json` records the protocol, seed, Git state, package versions, and SHA-256 digests of the input artifacts. The smoke F1 is a **software integration check**, separate from the research results reported in the thesis. Existing output directories are rejected so experiment records are not silently overwritten.

## Run the full research pipeline

Install the optional runtime groups you need:

```bash
uv sync --locked \
  --extra dev \
  --extra llm \
  --extra vision \
  --extra tracking \
  --extra dashboard

cp .env.example .env
```

Then run video → graph discovery:

```bash
uv run --no-sync python -m core.end_to_end_pipeline \
  --video /path/to/consented-classroom.mp4 \
  --model data/best.pt \
  --output_dir output/runs/classroom-001 \
  --skip_teacher_advice
```

Or run the stages independently:

```bash
# Video -> behavior time series
uv run --no-sync python -m data.extract_video_dynamics /path/to/consented-classroom.mp4 \
  --model data/best.pt \
  --output-dir output/phase1/classroom-001

# Existing CSV -> temporal graph
uv run --no-sync python -m core.run_causal_discovery \
  --input output/phase1/classroom-001/classroom_multivariate_timeseries.csv \
  --output output/final_causal_graph.json
```

The maintained behavior ontology for standalone CSV discovery is:

```text
Talk, Read, Phone, Write, Lean, Bow
```

optionally preceded by `time_bin_sec`.

### External services

- `DEEPSEEK_API_KEY` — graph-edit proposals and optional observation reports.
- `GEMINI_API_KEY` — synthetic-data generation workflows.
- `WANDB_MODE=disabled` — recommended unless experiment telemetry has been explicitly approved.

External-provider commands can incur charges and transmit data outside your machine. Environment variables take precedence over values in `.env`.

## Results and thesis report

The complete Vietnamese thesis source, figures, appendices, bibliography, and reported experimental results from the original `CausClass-report` repository are preserved under [`report/`](report/README.md), including source provenance.

Those results come from actual research runs. They should be read as results for the implementation, datasets, settings, and experimental protocol documented in the report. The newer protocol-v2 code should be treated as a revised implementation rather than as evidence that the original runs were invalid.

Build the thesis from the repository root with:

```bash
make report
```

The generated PDF is written to `report/build/main.pdf`. CI also builds the report and uploads the PDF and complete build log.

## Reproducible research workflow

CausClass now makes both the original reported work and future runs easier to inspect.

### What GitHub CI validates

The GitHub Actions research workflow checks:

- locked installs on Python 3.11, 3.12, and 3.13;
- linting and regression tests;
- a deterministic CPU verifier smoke run;
- source and wheel builds, including an installed-wheel smoke test outside the checkout;
- optional vision/runtime imports and CLI entry points;
- a clean thesis PDF build with unresolved citation/reference checks;
- a non-root Docker image running the smoke workflow with `--network none`.

CI validates the **current software and build pipeline**. It is not intended to replace or reinterpret the real experiment runs already reported in the thesis.

### Protocol v2 and future reruns

The refactored implementation intentionally fixes graph direction handling, Gaussian KL calculation, graph-selection leakage, baseline edge boundary cases, and ontology consistency. Because these changes affect methodology, fresh protocol-v2 reruns should be reported as a new result set rather than mixed with the original thesis numbers.

For future reportable runs, retain the exact commit, seed, split, preprocessing, configuration, provider/model IDs, input/model checksums, environment lock, prompts/responses when applicable, and final metrics. See [docs/reproducibility.md](docs/reproducibility.md).

## Container

Build the CPU reference image:

```bash
docker build -t causclass:research .
docker run --rm --network none causclass:research
```

The container runs as a non-root user and executes the API-free smoke workflow. It does not bundle private classroom videos or the detector checkpoint. Mount a writable directory at `/results` if you want to retain generated artifacts.

## Artifact viewer

```bash
uv run --no-sync streamlit run app.py
```

`app.py` is a local viewer for completed pipeline artifacts.

## Repository map

```text
CausClass/
├── core/                  # graph search, verifier, experiments, smoke CLI
│   ├── models/            # adapted AERCA / SENNGC model code
│   └── legacy/            # historical runners retained for traceability
├── data/                  # perception and synthetic-data utilities
├── utils/                 # graph, path and reproducibility helpers
├── tests/                 # API-free regression + CPU integration tests
├── docs/                  # architecture, reproducibility and ethics notes
├── report/                # thesis source, figures, results and provenance
├── refs/                  # retained third-party reference papers
├── app.py                 # local artifact viewer
├── pyproject.toml         # package and optional dependency groups
├── uv.lock                # resolved CPU reference environment
├── Dockerfile             # non-root CPU research image
└── CITATION.cff           # software citation metadata
```

## Research scope and responsible use

Classroom video can contain sensitive personal data. Use only data you are authorized to process, document consent/access conditions, and review [docs/data-and-ethics.md](docs/data-and-ethics.md) before redistributing images, detector weights, derived datasets, or third-party publications.

The project studies temporal behavior relationships from observational data. Detection quality, omitted variables, camera conditions, temporal aggregation, and model assumptions can affect interpretation. Use the reported results in the context of the study design documented in the thesis.

## Development

Before changing scientific behavior, read [CONTRIBUTING.md](CONTRIBUTING.md) and the [reproduction protocol](docs/reproducibility.md).

```bash
uv sync --locked --extra dev
make lint
uv run --no-sync pytest --cov
```

Existing `python -m core...` and `python -m data...` entry points are intentionally retained to avoid breaking experiment commands.

## Citation

If you use CausClass, cite the software using [`CITATION.cff`](CITATION.cff), record the exact commit, and cite the original AERCA work used by the verifier:

```bibtex
@inproceedings{han2025root,
  title={Root Cause Analysis of Anomalies in Multivariate Time Series through Granger Causal Discovery},
  author={Xiao Han and Saima Absar and Lu Zhang and Shuhan Yuan},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=k38Th3x4d9}
}
```

## License and third-party material

The repository retains its existing [MIT license](LICENSE). Adapted AERCA/SENNGC code, reference papers, datasets, images, and model weights may carry separate provenance or redistribution requirements. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before reuse or redistribution.
