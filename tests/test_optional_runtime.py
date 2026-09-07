import importlib.util

import pandas as pd
import pytest


@pytest.mark.vision
def test_empty_graph_exports_valid_csv(tmp_path):
    if importlib.util.find_spec("ultralytics") is None:
        pytest.skip("install the vision extra to test perception integration")
    from core.end_to_end_pipeline import save_graph_artifacts

    result = save_graph_artifacts([], ["Talk", "Read"], tmp_path, "empty")
    graph = pd.read_csv(result["edge_list_csv"])
    assert graph.empty and graph.columns.tolist() == ["source", "target", "weight", "sign"]
