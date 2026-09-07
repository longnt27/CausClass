import json

import numpy as np
import pytest

from utils import helpers, paths
from utils.reproducibility import run_metadata, sha256_file


def test_env_respects_existing_values_and_parses_quotes(monkeypatch, tmp_path):
    file = tmp_path / ".env"
    file.write_text('TEST_KEEP="file value"\nTEST_NEW="hello world" # comment\n')
    monkeypatch.setenv("TEST_KEEP", "existing")
    monkeypatch.delenv("TEST_NEW", raising=False)
    monkeypatch.setattr(helpers, "ENV_FILE", file)
    helpers.load_env_file()
    import os

    assert os.environ["TEST_KEEP"] == "existing"
    assert os.environ["TEST_NEW"] == "hello world"


def test_graph_edits_are_nonmutating_and_reject_self_loops():
    original = [{"source": "A", "target": "B", "weight": 0.2}]
    result = helpers.apply_edit(original, {"action": "delete", "source": "A", "target": "B"})
    assert result == [] and len(original) == 1
    assert helpers.apply_edit(original, {"action": "add", "source": "A", "target": "A"}) == original
    assert helpers.apply_edit(original, {"action": "add", "source": "A", "target": "B"}) == original
    assert helpers.hash_graph(original) == helpers.hash_graph(list(reversed(original)))


@pytest.mark.parametrize(
    "mse,edges,samples,penalty",
    [(np.nan, 1, 10, 2), (-1, 1, 10, 2), (0.1, -1, 10, 2), (0.1, 1, 1, 2), (0.1, 1, 10, -1)],
)
def test_invalid_bic_inputs(mse, edges, samples, penalty):
    with pytest.raises(ValueError):
        helpers.calculate_bic(mse, edges, samples, penalty)


def test_metrics_and_bic(tmp_path):
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({"edges_logic_from_llm": [{"source": "A", "target": "B"}]}))
    assert helpers.evaluate_dag([{"source": "A", "target": "B"}], truth, ["A", "B"])[:3] == (
        1.0,
        1.0,
        1.0,
    )
    assert helpers.calculate_bic(0.2, 2, 100, 3) == pytest.approx(
        100 * np.log(0.2 + 1e-8) + 6 * np.log(100)
    )


def test_output_cannot_escape_configured_root(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path)
    assert paths.output_path("nested", "metrics.json").parent.is_dir()
    with pytest.raises(ValueError):
        paths.output_path("..", "escape.json")


def test_manifest_has_hashes_but_no_secrets(monkeypatch, tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("fixture")
    monkeypatch.setenv("SECRET_SENTINEL", "must-not-appear")
    metadata = run_metadata(42, [source])
    assert metadata["inputs"]["input.txt"] == sha256_file(source)
    assert metadata["protocol_version"] == "2"
    assert "must-not-appear" not in json.dumps(metadata)
