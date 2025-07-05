#!/usr/bin/env python3
"""run_test_durations.py

Utility script to execute the unit-test suite and capture the runtime of every
individual test.  The results are written to a TAB-separated file named
`test_durations.txt` at the root of the *Environments* repository.

Execution (from the *Environments* directory):

    uv run python dev/run_test_durations.py

The script invokes *pytest* with `--durations=0` to obtain timing data for **all**
phases (`call`, `setup`, `teardown`).  It parses those lines and emits output in
```
<seconds>    <nodeid>
```
format, ordered as they appear in the pytest report.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Location of the tests directory – keep in sync with update_readme_metrics.py
TESTS_DIR = "tests/"

# Output file lives in the repository root so it can be inspected easily.
OUTPUT_FILENAME = "test_durations.txt"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_pytest_and_capture() -> str:
    """Run pytest and return its full captured (stdout + stderr) text."""
    cmd = [
        "pytest",
        TESTS_DIR,
        "--durations=0",  # report *all* test durations
        "--color=no",  # strip ANSI colour codes for easier parsing
        "-q",  # quiet ‑ only test results + durations
    ]
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,  # allow script to continue even if tests fail
    )
    return completed.stdout + completed.stderr


def extract_duration_lines(pytest_output: str) -> List[str]:
    """Return pytest duration lines (e.g. '0.04s call tests/test_foo.py::test_bar')."""
    pattern = re.compile(r"^\s*([0-9.]+)s\s+\w+\s+.+", re.MULTILINE)
    return [match.group(0) for match in pattern.finditer(pytest_output)]


def write_durations(lines: List[str], output_path: Path) -> None:
    """Write formatted duration lines to *output_path* (TAB-separated)."""
    with output_path.open("w", encoding="utf-8") as fh:
        for line in lines:
            # Example line: "0.04s call     tests/test_foo.py::test_bar"
            parts = line.strip().split(maxsplit=3)
            seconds = parts[0].rstrip("s")  # remove trailing 's'
            nodeid = parts[3] if len(parts) == 4 else " ".join(parts[2:])
            fh.write(f"{seconds}\t{nodeid}\n")


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    output_file = repo_root / OUTPUT_FILENAME

    print("🧪 Running pytest to collect duration data…", flush=True)
    raw_output = run_pytest_and_capture()

    duration_lines = extract_duration_lines(raw_output)
    write_durations(duration_lines, output_file)

    print(
        f"✅ Wrote {len(duration_lines)} duration entries to {output_file.relative_to(repo_root)}"
    )
