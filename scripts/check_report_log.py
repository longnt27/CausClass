"""Fail report CI on unresolved citations/references; retain all TeX diagnostics."""
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
patterns = [r"Citation .* undefined", r"Reference .* undefined", r"There were undefined references",
            r"Please \(re\)run Biber", r"LaTeX Error:"]
failures = [pattern for pattern in patterns if re.search(pattern, text)]
if failures:
    raise SystemExit("Unresolved report diagnostics: " + ", ".join(failures))
print("Report log: no unresolved references or citations.")
