# Architecture

Video -> YOLO/ByteTrack detections -> behavior-share time series -> AERCA dense
proposal -> LLM add/delete proposals -> masked AERCA scoring -> reviewed graph
artifacts -> optional LLM observation report.

`core/end_to_end_pipeline.py` orchestrates these stages. `core/graph_edit_agent.py`
owns provider transport and proposal validation. `core/aerca_verifier.py` owns
fit/score isolation, coefficient orientation, masking and verifier configuration.
`core/models/` retains adapted upstream model code. `core/smoke.py` exercises the
real verifier without perception or network services.

Existing package/module paths are retained to avoid breaking published commands.
`pyproject.toml` explicitly discovers only core/data/utils packages and excludes
report sources, videos and checkpoints from wheels. Editable installs are the
primary research workflow. For a non-editable installation, set CAUSCLASS_HOME,
CAUSCLASS_DATA_DIR and CAUSCLASS_OUTPUT_DIR to writable locations as appropriate.

Public matrices use source rows / target columns; SENNGC coefficients use target
rows / source columns. Conversion happens only at the verifier boundary. Tests
assert this convention with known coefficients and a directional forward pass.
The graph is temporal and can contain cycles. Self effects are modeled, but are
not counted as reported cross-variable edges.

`app.py` is a local artifact viewer. It does not accept a video and pretend to
analyze it, call an LLM, train a model or expose a multi-user inference service.
Do not deploy it against an unrestricted filesystem as a public web application.
