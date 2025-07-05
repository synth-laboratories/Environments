#!/usr/bin/env python3
"""
Health check script for the codebase

Provides:
- Lines of code in Python
- uvx ty check violations
- ruff format issues
- ruff lint issues
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, capture_output: bool = True) -> tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, stderr."""
    result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
    return result.returncode, result.stdout, result.stderr


def count_python_lines() -> int:
    """Count lines of Python code."""
    print("📊 Counting Python lines of code...")

    # Use find to get all Python files and wc to count lines
    exit_code, stdout, stderr = run_command(
        "find . -name '*.py' -not -path './.*' -not -path './temp/*' -not -path './htmlcov/*' -not -path './build/*' -not -path './dist/*' | xargs wc -l | tail -1"
    )

    if exit_code != 0:
        print(f"❌ Failed to count lines: {stderr}")
        return 0

    # Extract total from wc output (format: "  12345 total")
    try:
        total_lines = int(stdout.strip().split()[0])
        return total_lines
    except (ValueError, IndexError):
        print(f"❌ Failed to parse line count: {stdout}")
        return 0


def check_ty_violations() -> tuple[int, dict[str, int]]:
    """Check for ty violations and return count per file."""
    print("🔍 Checking ty violations...")

    # Use concise format with --exit-zero to get all violations
    exit_code, stdout, stderr = run_command(
        "uvx ty check --output-format concise --exit-zero"
    )

    if not stdout.strip():
        print("✅ No ty violations found")
        return 0, {}

    # Count violations per file
    violations_per_file = {}
    total_violations = 0

    for line in stdout.strip().split("\n"):
        if line.strip() and ":" in line:
            file_path = line.split(":", 1)[0]
            violations_per_file[file_path] = violations_per_file.get(file_path, 0) + 1
            total_violations += 1

    # Show top 10 files with most violations
    if violations_per_file:
        print(f"📊 Top 10 files with ty violations:")
        sorted_files = sorted(
            violations_per_file.items(), key=lambda x: x[1], reverse=True
        )
        for i, (file_path, count) in enumerate(sorted_files[:10]):
            print(f"  {count:4d} {file_path}")

    return total_violations, violations_per_file


def check_ruff_format() -> tuple[int, list[str]]:
    """Check ruff format issues and return count and list of files."""
    print("🎨 Checking ruff format issues...")

    # Use --check to get list of files that need formatting
    exit_code, stdout, stderr = run_command("ruff format --check .")

    if exit_code == 0:
        print("✅ No format issues found")
        return 0, []

    # Get list of files that need formatting
    files_needing_format = [
        line.strip() for line in stdout.strip().split("\n") if line.strip()
    ]

    # Show top 10 files (or all if less than 10)
    if files_needing_format:
        print(f"📊 Files needing formatting (showing top 10):")
        for i, file_path in enumerate(files_needing_format[:10]):
            print(f"  {i + 1:2d}. {file_path}")

        if len(files_needing_format) > 10:
            print(f"     ... and {len(files_needing_format) - 10} more files")

    return len(files_needing_format), files_needing_format


def check_ruff_lint() -> int:
    """Check ruff lint issues."""
    print("🔧 Checking ruff lint issues...")

    exit_code, stdout, stderr = run_command("ruff check .")

    if exit_code == 0:
        print("✅ No lint issues found")
        return 0

    # Count lint violations (each line is typically one violation)
    lint_issues = len(
        [
            line
            for line in stdout.strip().split("\n")
            if line.strip() and not line.startswith("Found")
        ]
    )

    return lint_issues


def main():
    """Main health check function."""
    print("🏥 Codebase Health Check")
    print("=" * 40)

    # Count Python lines
    python_lines = count_python_lines()

    # Check ty violations
    ty_violations, ty_violations_per_file = check_ty_violations()

    # Check ruff format issues
    format_issues, format_files = check_ruff_format()

    # Check ruff lint issues
    lint_issues = check_ruff_lint()

    print("\n📋 Health Check Summary")
    print("=" * 40)
    print(f"📊 Python Lines of Code: {python_lines:,}")
    print(f"🔍 Ty Check Violations: {ty_violations}")
    if ty_violations_per_file:
        print(f"    └─ Across {len(ty_violations_per_file)} files")
    print(f"🎨 Ruff Format Issues: {format_issues}")
    if format_files:
        print(f"    └─ {len(format_files)} files need formatting")
    print(f"🔧 Ruff Lint Issues: {lint_issues}")

    # Calculate overall health score
    total_issues = ty_violations + format_issues + lint_issues

    if total_issues == 0:
        print("\n🎉 Perfect Health! No issues found.")
        health_status = "🟢 EXCELLENT"
    elif total_issues <= 5:
        print(f"\n✅ Good Health! Only {total_issues} minor issues.")
        health_status = "🟡 GOOD"
    elif total_issues <= 20:
        print(f"\n⚠️  Fair Health. {total_issues} issues to address.")
        health_status = "🟠 FAIR"
    else:
        print(f"\n❌ Poor Health. {total_issues} issues need attention.")
        health_status = "🔴 POOR"

    print(f"🏥 Overall Health: {health_status}")

    # Exit with non-zero code if there are issues
    if total_issues > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
