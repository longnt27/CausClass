import json
import subprocess
import sys

import pytest

from core.smoke import run


@pytest.mark.integration
def test_real_cpu_smoke_is_repeatable_and_writes_provenance(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    a, b = run(first, 42), run(second, 42)
    assert a == b
    assert a["selection_mse"] >= 0
    assert (first / "time_series.csv").read_bytes() == (second / "time_series.csv").read_bytes()
    expected = {
        "time_series.csv",
        "ground_truth_graph.json",
        "graph.json",
        "metrics.json",
        "config.json",
        "run_metadata.json",
    }
    assert {p.name for p in first.iterdir()} == expected
    assert json.loads((first / "config.json").read_text())["device"] == "cpu"
    assert json.loads((first / "run_metadata.json").read_text())["protocol_version"] == "2"
    with pytest.raises(FileExistsError):
        run(first, 42)


def test_cli_existing_directory_exits_nonzero(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "core.smoke", "--output-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 2 and "error:" in result.stderr
