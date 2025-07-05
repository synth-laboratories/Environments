#!/usr/bin/env python3
"""
Dev helper script to update README.md with current metrics.

This script:
1. Runs the test suite and calculates coverage (optional with --coverage)
2. Runs type checking on different parts of the codebase
3. Checks PyPI for latest package versions
4. Updates README.md with badges/widgets

Usage:
    python dev/update_readme_metrics.py
    python dev/update_readme_metrics.py --coverage  # Include coverage analysis

Requirements:
    - pytest, coverage, requests installed
    - uvx ty available
"""

import subprocess
import re
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import requests
import xml.etree.ElementTree as ET


def run_command(cmd: str, cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=cwd
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)


def get_test_coverage(include_coverage: bool = False) -> Tuple[float, int, int]:
    """Run tests and get coverage percentage, total tests, passed tests."""
    if include_coverage:
        print("🧪 Running test suite with coverage...")
        # Skip slow tests explicitly (those marked @pytest.mark.slow)
        cmd = 'uv run pytest tests/ -m "not slow" --cov=src --cov-report=xml --cov-report=term-missing -q'
    else:
        print("🧪 Running test suite (fast mode, no coverage)...")
        # Run only tests marked as fast
        cmd = "uv run pytest tests/ --fast -q --tb=no"

    exit_code, stdout, stderr = run_command(cmd)

    # Parse coverage from XML report (only if coverage was requested)
    coverage_pct = 0.0
    if include_coverage:
        try:
            if Path("coverage.xml").exists():
                tree = ET.parse("coverage.xml")
                root = tree.getroot()
                coverage_pct = float(root.attrib.get("line-rate", 0)) * 100
        except Exception as e:
            print(f"Warning: Could not parse coverage XML: {e}")
            # Fallback: parse from stdout
            coverage_match = re.search(r"TOTAL.*?(\d+)%", stdout)
            if coverage_match:
                coverage_pct = float(coverage_match.group(1))

    # Parse test results
    total_tests = 0
    passed_tests = 0

    # Look for pytest summary
    if "failed" in stdout.lower() or "error" in stdout.lower():
        test_status = "failing"
        # Try to extract numbers
        failed_match = re.search(r"(\d+) failed", stdout)
        passed_match = re.search(r"(\d+) passed", stdout)
        if failed_match and passed_match:
            failed = int(failed_match.group(1))
            passed = int(passed_match.group(1))
            total_tests = failed + passed
            passed_tests = passed
    else:
        test_status = "passing"
        # Look for passed count
        passed_match = re.search(r"(\d+) passed", stdout)
        if passed_match:
            passed_tests = int(passed_match.group(1))
            total_tests = passed_tests

    coverage_msg = (
        f" ({coverage_pct:.1f}% coverage)" if include_coverage else " (no coverage)"
    )
    print(f"✅ Tests: {passed_tests}/{total_tests} passed{coverage_msg}")
    return coverage_pct, total_tests, passed_tests


def get_type_check_results() -> Dict[str, Tuple[int, int]]:
    """Run type checking and return error counts for different areas."""
    print("🔍 Running type checking...")

    results = {}

    # Check examples directory
    print("  Checking src/synth_env/examples...")
    cmd = "uvx ty check src/synth_env/examples"
    exit_code, stdout, stderr = run_command(cmd)

    # Parse mypy output for error count
    examples_errors = 0
    examples_files = 0

    # Count errors and files
    error_lines = [
        line
        for line in stdout.split("\n")
        if ":" in line and ("error:" in line or "warning:" in line)
    ]
    examples_errors = len([line for line in error_lines if "error:" in line])

    # Count files checked
    file_lines = [
        line
        for line in stdout.split("\n")
        if line.strip().startswith("src/synth_env/examples") and ":" in line
    ]
    examples_files = len(set(line.split(":")[0] for line in file_lines if ":" in line))

    if examples_files == 0:
        # Fallback: count unique file mentions
        all_lines = stdout.split("\n") + stderr.split("\n")
        file_mentions = set()
        for line in all_lines:
            if "src/synth_env/examples" in line and ".py" in line:
                # Extract filename
                parts = line.split()
                for part in parts:
                    if "src/synth_env/examples" in part and ".py" in part:
                        file_mentions.add(part.split(":")[0])
        examples_files = len(file_mentions) or 1  # At least 1 to avoid division by zero

    results["examples"] = (examples_errors, examples_files)

    # Check core synth_env (excluding examples)
    print("  Checking src/synth_env (core)...")
    cmd = "uvx ty check src/synth_env --exclude src/synth_env/examples"
    exit_code, stdout, stderr = run_command(cmd)

    core_errors = 0
    core_files = 0

    error_lines = [
        line
        for line in stdout.split("\n")
        if ":" in line and ("error:" in line or "warning:" in line)
    ]
    core_errors = len([line for line in error_lines if "error:" in line])

    file_lines = [
        line
        for line in stdout.split("\n")
        if line.strip().startswith("src/synth_env")
        and "examples" not in line
        and ":" in line
    ]
    core_files = len(set(line.split(":")[0] for line in file_lines if ":" in line))

    if core_files == 0:
        # Fallback approach
        all_lines = stdout.split("\n") + stderr.split("\n")
        file_mentions = set()
        for line in all_lines:
            if "src/synth_env" in line and ".py" in line and "examples" not in line:
                parts = line.split()
                for part in parts:
                    if (
                        "src/synth_env" in part
                        and ".py" in part
                        and "examples" not in part
                    ):
                        file_mentions.add(part.split(":")[0])
        core_files = len(file_mentions) or 1

    results["core"] = (core_errors, core_files)

    print(
        f"✅ Type checking: Examples {examples_errors} errors in {examples_files} files, Core {core_errors} errors in {core_files} files"
    )
    return results


def get_pypi_version(
    package_name: str = "synth-env",
) -> Tuple[Optional[str], Optional[str]]:
    """Get latest stable and dev versions from PyPI."""
    print(f"📦 Checking PyPI for {package_name} versions...")

    try:
        response = requests.get(
            f"https://pypi.org/pypi/{package_name}/json", timeout=10
        )
        if response.status_code == 200:
            data = response.json()

            # Get all versions
            versions = list(data.get("releases", {}).keys())

            # Filter stable vs dev versions
            stable_versions = [
                v
                for v in versions
                if not any(
                    x in v.lower() for x in ["a", "b", "rc", "dev", "alpha", "beta"]
                )
            ]
            dev_versions = [
                v
                for v in versions
                if any(x in v.lower() for x in ["a", "b", "rc", "dev", "alpha", "beta"])
            ]

            # Get latest of each
            latest_stable = (
                max(
                    stable_versions,
                    key=lambda x: [int(n) for n in x.split(".") if n.isdigit()],
                )
                if stable_versions
                else None
            )
            latest_dev = max(dev_versions, key=lambda x: x) if dev_versions else None

            print(f"✅ PyPI versions: stable={latest_stable}, dev={latest_dev}")
            return latest_stable, latest_dev

    except Exception as e:
        print(f"Warning: Could not fetch PyPI versions: {e}")

    return None, None


def update_readme_badges(readme_path: Path, metrics: Dict) -> None:
    """Update README.md with new badge metrics."""
    print("📝 Updating README.md badges...")

    # Read current README
    content = readme_path.read_text()

    # Define badge patterns and replacements
    coverage_pct = metrics["coverage"]
    coverage_color = (
        "red" if coverage_pct < 50 else "yellow" if coverage_pct < 80 else "green"
    )

    test_pct = (
        (metrics["passed_tests"] / metrics["total_tests"] * 100)
        if metrics["total_tests"] > 0
        else 0
    )
    test_status = (
        "passing" if metrics["passed_tests"] == metrics["total_tests"] else "failing"
    )
    test_color = "brightgreen" if test_status == "passing" else "red"

    # Type checking badges
    examples_errors, examples_files = metrics["type_check"]["examples"]
    core_errors, core_files = metrics["type_check"]["core"]

    examples_pct = (
        max(0, (examples_files - examples_errors) / examples_files * 100)
        if examples_files > 0
        else 100
    )
    core_pct = (
        max(0, (core_files - core_errors) / core_files * 100) if core_files > 0 else 100
    )

    examples_color = (
        "red" if examples_pct < 70 else "yellow" if examples_pct < 90 else "green"
    )
    core_color = "red" if core_pct < 70 else "yellow" if core_pct < 90 else "green"

    # Badge replacements
    replacements = [
        # Coverage badge
        (
            r"!\[Coverage\]\(https://img\.shields\.io/badge/coverage-[^)]+\)",
            f"![Coverage](https://img.shields.io/badge/coverage-{coverage_pct:.1f}%25-{coverage_color})",
        ),
        # Tests badge
        (
            r"!\[Tests\]\(https://img\.shields\.io/badge/tests-[^)]+\)",
            f"![Tests](https://img.shields.io/badge/tests-{metrics['passed_tests']}/{metrics['total_tests']} {test_status}-{test_color})",
        ),
        # Type checking badges (replace existing or add new)
        (
            r"!\[Type Check Examples\]\([^)]+\)",
            f"![Type Check Examples](https://img.shields.io/badge/types (examples)-{examples_pct:.0f}%25 ({examples_errors} errors)-{examples_color})",
        ),
        (
            r"!\[Type Check Core\]\([^)]+\)",
            f"![Type Check Core](https://img.shields.io/badge/types (core)-{core_pct:.0f}%25 ({core_errors} errors)-{core_color})",
        ),
    ]

    # Add PyPI version badges if we have version info
    if metrics.get("stable_version"):
        replacements.append(
            (
                r"!\[PyPI Stable\]\([^)]+\)",
                f"![PyPI Stable](https://img.shields.io/badge/PyPI stable-{metrics['stable_version']}-blue)",
            )
        )

    if metrics.get("dev_version"):
        replacements.append(
            (
                r"!\[PyPI Dev\]\([^)]+\)",
                f"![PyPI Dev](https://img.shields.io/badge/PyPI dev-{metrics['dev_version']}-orange)",
            )
        )

    # Apply replacements
    new_content = content
    for pattern, replacement in replacements:
        new_content = re.sub(pattern, replacement, new_content)

    # Update coverage percentage in text if present
    new_content = re.sub(
        r"Current test coverage: \*\*[\d.]+%\*\*",
        f"Current test coverage: **{coverage_pct:.1f}%**",
        new_content,
    )

    # Write updated README
    readme_path.write_text(new_content)
    print("✅ README.md updated successfully!")


def main():
    """Main function to run all metric updates."""
    parser = argparse.ArgumentParser(description="Update README metrics")
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Include test coverage analysis (slower)",
    )
    args = parser.parse_args()

    print("🚀 Updating README metrics...\n")

    # Ensure we're in the right directory
    project_root = Path(__file__).parent.parent
    readme_path = project_root / "README.md"

    if not readme_path.exists():
        print(f"Error: README.md not found at {readme_path}")
        sys.exit(1)

    # Collect all metrics
    metrics = {}

    # Test coverage
    try:
        coverage, total_tests, passed_tests = get_test_coverage(args.coverage)
        metrics.update(
            {
                "coverage": coverage,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
            }
        )
    except Exception as e:
        print(f"Error getting test coverage: {e}")
        metrics.update({"coverage": 0.0, "total_tests": 0, "passed_tests": 0})

    # Type checking
    try:
        type_results = get_type_check_results()
        metrics["type_check"] = type_results
    except Exception as e:
        print(f"Error getting type check results: {e}")
        metrics["type_check"] = {"examples": (0, 1), "core": (0, 1)}

    # PyPI versions
    try:
        stable_version, dev_version = get_pypi_version()
        if stable_version:
            metrics["stable_version"] = stable_version
        if dev_version:
            metrics["dev_version"] = dev_version
    except Exception as e:
        print(f"Error getting PyPI versions: {e}")

    # Update README
    try:
        update_readme_badges(readme_path, metrics)
    except Exception as e:
        print(f"Error updating README: {e}")
        sys.exit(1)

    print(f"\n🎉 README metrics updated successfully!")
    print(f"📊 Summary:")
    if args.coverage:
        print(f"   Coverage: {metrics['coverage']:.1f}%")
    else:
        print(f"   Coverage: skipped (use --coverage to include)")
    print(f"   Tests: {metrics['passed_tests']}/{metrics['total_tests']}")

    if "type_check" in metrics:
        examples_errors, examples_files = metrics["type_check"]["examples"]
        core_errors, core_files = metrics["type_check"]["core"]
        print(
            f"   Type Check Examples: {examples_errors} errors in {examples_files} files"
        )
        print(f"   Type Check Core: {core_errors} errors in {core_files} files")

    if metrics.get("stable_version"):
        print(f"   PyPI Stable: {metrics['stable_version']}")
    if metrics.get("dev_version"):
        print(f"   PyPI Dev: {metrics['dev_version']}")


if __name__ == "__main__":
    main()
