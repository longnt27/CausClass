# CausClass

CausClass is a research implementation for extracting classroom behavior time
series from video, proposing temporal graph edits with an LLM, and scoring
candidate graphs with an AERCA neural verifier. The Vietnamese thesis source,
figures, appendices and bibliography now live in [`report/`](report/README.md).

**Research status:** this repository provides a tested software workflow, not
independent validation of the thesis findings. Protocol v2 corrects coefficient
orientation, Gaussian KL and graph-scoring data isolation. Historical results
must be rerun before being compared with this version. Temporal prediction and
LLM plausibility do **not** establish intervention-level causation, a student's
internal state, or a basis for high-stakes decisions.

## Quick start: no video, GPU, API key or paid service

Python 3.11-3.13 is supported. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/longnt27/CausClass.git
cd CausClass
uv sync --locked --extra dev
uv run --no-sync causclass-smoke --output-dir output/smoke-001 --seed 42
uv run --no-sync pytest --cov
```

The smoke command generates a stable toy VAR series, runs real dense and masked
AERCA training on CPU, and writes graph, metrics, configuration and provenance
JSON files. It uses two training epochs per stage; its F1 is **not** a benchmark.
An existing output directory is rejected to prevent accidental experiment loss.
Use a new run directory for each invocation.

`uv.lock` is the resolved environment. Linux/Windows uv installs use CPU PyTorch
by default. For CUDA, use a separate environment and the official PyTorch
installation selector; save its complete `pip freeze` with the results. Do not
silently mix GPU changes into the CPU reference lock. A compatibility
`pip install -r requirements.txt` path remains available but is not frozen.

## Online and perception workflows

```bash
uv sync --locked --extra dev --extra llm --extra vision --extra tracking --extra dashboard
cp .env.example .env
# Edit .env locally; existing environment variables take precedence.

uv run --no-sync python -m core.end_to_end_pipeline \
  --video /path/to/consented-classroom.mp4 --model data/best.pt \
  --output_dir output/runs/classroom-001 --skip_teacher_advice

uv run --no-sync python -m data.extract_video_dynamics /path/to/consented-classroom.mp4 \
  --model data/best.pt --output-dir output/phase1/classroom-001

uv run --no-sync python -m core.run_causal_discovery \
  --input output/phase1/classroom-001/classroom_multivariate_timeseries.csv \
  --output output/final_causal_graph.json

uv run --no-sync python -m data.generate_synthetic_data -n 10 --output-dir data/synth_data
uv run --no-sync python -m core.run_synthetic_search --debug 1 --workers 1
uv run --no-sync python -m core.grid_search_bic_lambda --test_limit 10 --workers 1
uv run --no-sync python -m core.sweep_aerca_thresholds --workers 1
```

Gemini generation requires `GEMINI_API_KEY`; graph proposals and optional
observation reports require `DEEPSEEK_API_KEY`. These commands may incur charges
and transmit data to external providers. `--skip_teacher_advice` skips only the
last report call, **not** the graph-search API calls. Set `WANDB_MODE=disabled`
unless experiment telemetry has been approved. Provider model IDs are recorded
historical defaults, not a guarantee that an endpoint is available today.

The CSV discovery command uses `Talk, Read, Phone, Write, Lean, Bow` percentage
columns, optionally preceded by `time_bin_sec`. See the
[reproduction protocol](docs/reproducibility.md) for splits, sample requirements,
seeds, provider variability, and the changes that invalidate historical scores.

## Inspect results, build the report, or use a container

```bash
uv run --no-sync streamlit run app.py    # local artifact viewer, not video inference
make report                            # report/build/main.pdf, requires TeX Live

docker build -t causclass:research .
docker run --rm --network none causclass:research
```

See [`report/README.md`](report/README.md) for TeX dependencies. CI builds and
uploads the PDF and build log; the bibliography/reference gate fails on missing
citations. The original thesis remains a historical document, with a separate
[import record](report/PROVENANCE.md), not silently rewritten results.

The CPU container runs the API-free smoke workflow as a non-root user. To retain
results, mount a writable host directory at `/results`; use an appropriate UID
for that directory. It does not bundle private videos or the detector checkpoint.

## Repository map

| Location | Purpose |
| --- | --- |
| `core/` | Discovery, graph search, ablations, verifier and offline smoke CLI |
| `core/models/` | Adapted AERCA/SENNGC models; original license retained |
| `core/legacy/` | Historical runners; not the maintained test suite |
| `data/` | Data generation/perception utilities and pre-existing detector weights |
| `utils/` | Graph, configuration and provenance helpers |
| `tests/` | API-free regression and CPU integration tests |
| `report/` | Imported thesis source, figures, appendices and bibliography |
| `docs/` | Architecture, reproducibility, data/ethics and model limitations |
| `output/` | Local generated runs; ignored by Git |
| `refs/` | Existing reference papers; third-party rights remain applicable |

## Contributing and attribution

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing scientific behavior.
Existing `python -m core...` and `python -m data...` entry points remain supported;
no broad file move was made merely to change the directory layout.

Use [CITATION.cff](CITATION.cff) for CausClass and cite the original AERCA work:

```bibtex
@inproceedings{han2025root,
  title={Root Cause Analysis of Anomalies in Multivariate Time Series through Granger Causal Discovery},
  author={Xiao Han and Saima Absar and Lu Zhang and Shuhan Yuan},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=k38Th3x4d9}
}
```

The existing [MIT license](LICENSE), including Xiao Han's copyright, is preserved.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[the data/model notes](docs/data-and-ethics.md) before redistributing weights,
classroom images or third-party publications.
