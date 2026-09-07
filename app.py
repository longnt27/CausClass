"""Local viewer for existing pipeline artifacts; never performs video inference."""
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.paths import OUTPUT_DIR

st.set_page_config(page_title="CausClass Artifact Viewer", layout="wide")
st.title("CausClass Artifact Viewer")
st.info("This viewer reads completed local runs. It does not analyze uploaded video or call an LLM.")
st.caption("Temporal associations are exploratory evidence, not proof of causation or student intent.")
run_dir = Path(st.text_input("Completed run directory", str(OUTPUT_DIR / "runs"))).expanduser()
if not run_dir.is_dir():
    st.warning("Choose a local directory produced by core.end_to_end_pipeline.")
    st.stop()

edge_files = sorted(run_dir.glob("*_final_graph_edges.csv"))
if not edge_files:
    st.warning("No completed graph artifacts found in this directory.")
    st.stop()

selected = st.selectbox("Graph artifact", edge_files, format_func=lambda p: p.name)
stem = selected.name.removesuffix("_final_graph_edges.csv")
try:
    graph = pd.read_csv(selected)
    if not {"source", "target", "weight", "sign"}.issubset(graph.columns):
        raise ValueError("graph CSV requires source, target, weight and sign columns")
    st.subheader("Temporal graph edges")
    st.dataframe(graph, hide_index=True, use_container_width=True)
    if graph.empty:
        st.caption("No cross-variable edges survived this run's selection procedure.")
    candidates = sorted(run_dir.glob(f"{stem}*timeseries*.csv"))
    if candidates:
        series = pd.read_csv(candidates[0])
        st.subheader("Behavior time series")
        st.caption(candidates[0].name)
        if "time_bin_sec" in series:
            series = series.set_index("time_bin_sec")
        st.line_chart(series.select_dtypes(include="number"))
    report = run_dir / f"{stem}_observation_report.md"
    if report.is_file():
        st.subheader("Generated observation report - requires human review")
        st.markdown(report.read_text(encoding="utf-8"))
except (OSError, ValueError, pd.errors.ParserError) as exc:
    st.error(f"Cannot read this run: {exc}")
