from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SLOW_THRESHOLD_SECONDS = 2.0  # anything longer is considered "slow"
DURATIONS_FILE = Path(__file__).parent / "test_durations.txt"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_slow_tests() -> set[str]:
    """Return a set of nodeids that exceeded *SLOW_THRESHOLD_SECONDS*."""
    slow: set[str] = set()

    if not DURATIONS_FILE.exists():
        # No duration data yet – treat everything as normal.
        return slow

    for line in DURATIONS_FILE.read_text().splitlines():
        if "\t" not in line:
            continue
        seconds_str, nodeid = line.split("\t", 1)
        try:
            if float(seconds_str) > SLOW_THRESHOLD_SECONDS:
                slow.add(nodeid.strip())
        except ValueError:
            # Malformed line; ignore and continue
            pass

    return slow


_SLOW_TESTS: set[str] = _collect_slow_tests()

# ---------------------------------------------------------------------------
# Pytest hook implementations
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run tests marked as slow (>2 s).",
    )
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Run only tests marked as fast (quick unit tests).",
    )


def pytest_configure(config: pytest.Config) -> None:
    # Register the marker so that pytest -m slow works and to silence warnings.
    config.addinivalue_line("markers", "slow: tests that are slow (>2 s)")
    config.addinivalue_line("markers", "fast: tests that are fast (quick unit tests)")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    # First, auto-mark the slow tests based on prior timing data.
    for item in items:
        if item.nodeid in _SLOW_TESTS:
            item.add_marker("slow")

    # Handle --fast flag: only run tests marked as fast
    if config.getoption("--fast"):
        skip_marker = pytest.mark.skip(
            reason="Skipped non-fast test (only running fast tests)"
        )
        for item in items:
            if "fast" not in item.keywords:
                item.add_marker(skip_marker)
        return

    # Handle --slow flag: skip slow tests unless explicitly requested
    if not config.getoption("--slow"):
        skip_marker = pytest.mark.skip(reason="Skipped slow test (pass --slow to run)")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_marker)
