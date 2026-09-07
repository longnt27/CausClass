# Contributing

Use Python 3.11-3.13 and `uv sync --locked --extra dev --extra llm`. Run `make lint`,
`make test`, and a fresh `make smoke SMOKE_OUT=output/my-new-smoke`. Install TeX
Live and run `make report` for report changes. Optional perception tests need
`uv sync --locked --extra dev --extra llm --extra vision`.

Changes to objectives, graph orientation, data splits, thresholds, ontology,
normalization or prompts are scientific changes. Add a regression test, explain
result comparability in docs/reproducibility.md, and regenerate affected results.
Do not edit thesis metrics to match an unverified new run. Avoid layout-only
renames that break the documented module entry points.

Update dependencies in pyproject.toml, regenerate `uv.lock` with uv 0.8.22, and run
the CPU test matrix. Keep the lock in the same PR. CI runs without secrets and
must not download detector weights or call paid APIs. Use mocks for transports.

Only synthetic non-personal fixtures belong in tests. Put local data and outputs
under ignored paths. Do not force-add videos, checkpoints, API keys, cached LLM
responses or large generated binaries. For a defect, provide the commit, protocol,
Python/platform versions, redacted configuration, traceback and a minimal
synthetic reproduction. Report exposed keys privately and rotate them promptly.

The root license and upstream copyright notices must remain intact. New reports,
figures, models and datasets need explicit provenance and appropriate rights.
CI status checks exist in code; maintainers should separately make the test,
report and container jobs required in branch protection before merging releases.
