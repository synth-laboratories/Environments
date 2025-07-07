from __future__ import annotations

from pathlib import Path
from typing import Set

import pytest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SLOW_THRESHOLD_SECONDS = 2.0  # Any test longer than this is considered "slow"
DURATIONS_FILE = Path(__file__).parent / "test_durations.txt"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_slow_tests() -> Set[str]:
    """Return a set of test nodeids that exceeded SLOW_THRESHOLD_SECONDS."""
    slow: Set[str] = set()

    if not DURATIONS_FILE.exists():
        return slow  # No duration data available

    for line in DURATIONS_FILE.read_text().splitlines():
        if "\t" not in line:
            continue
        seconds_str, nodeid = line.split("\t", 1)
        try:
            if float(seconds_str) > SLOW_THRESHOLD_SECONDS:
                slow.add(nodeid.strip())
        except ValueError:
            continue  # Skip malformed line

    return slow

_SLOW_TESTS: Set[str] = _collect_slow_tests()

# ---------------------------------------------------------------------------
# Pytest hook implementations
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run tests marked as slow (runtime > 2s).",
    )
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Run only tests marked as fast (quick unit tests).",
    )

def pytest_configure(config: pytest.Config) -> None:
    # Register custom markers to avoid warnings
    config.addinivalue_line("markers", "slow: tests that are slow (>2 s)")
    config.addinivalue_line("markers", "fast: tests that are fast (quick unit tests)")

def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # Auto-mark previously identified slow tests
    for item in items:
        if item.nodeid in _SLOW_TESTS:
            item.add_marker("slow")

    # If --fast is passed, skip all non-fast tests
    if config.getoption("--fast"):
        skip_non_fast = pytest.mark.skip(reason="Skipped non-fast test (only running fast tests)")
        for item in items:
            if "fast" not in item.keywords:
                item.add_marker(skip_non_fast)
        return  # Nothing else to do

    # If --slow is not passed, skip all slow tests
    if not config.getoption("--slow"):
        skip_slow = pytest.mark.skip(reason="Skipped slow test (pass --slow to include)")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
