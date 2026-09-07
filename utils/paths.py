"""Configurable locations; existing source-checkout defaults are preserved."""
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("CAUSCLASS_HOME", Path(__file__).resolve().parents[1])).expanduser().resolve()
DATA_DIR = Path(os.environ.get("CAUSCLASS_DATA_DIR", PROJECT_ROOT / "data")).expanduser().resolve()
OUTPUT_DIR = Path(os.environ.get("CAUSCLASS_OUTPUT_DIR", PROJECT_ROOT / "output")).expanduser().resolve()
ENV_FILE = Path(os.environ.get("CAUSCLASS_ENV_FILE", PROJECT_ROOT / ".env")).expanduser().resolve()


def output_path(*parts: str) -> Path:
    """Return a contained output path and create its parent directory."""
    path = OUTPUT_DIR.joinpath(*parts).resolve()
    if not path.is_relative_to(OUTPUT_DIR):
        raise ValueError("output path must stay within CAUSCLASS_OUTPUT_DIR")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
