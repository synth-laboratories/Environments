#!/usr/bin/env python3
"""
Unified release script: increment version, build, and publish to PyPI

Usage:
    python scripts/release.py                    # Increment dev version and publish
    python scripts/release.py --minor            # Increment minor version and publish
    python scripts/release.py --patch            # Increment patch version and publish
    python scripts/release.py --dry-run          # Show what would happen without doing it
    python scripts/release.py --test-pypi        # Publish to TestPyPI instead
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, check: bool = True) -> tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, stderr."""
    print(f"🔧 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if check and result.returncode != 0:
        print(f"❌ Command failed: {cmd}")
        print(f"Exit code: {result.returncode}")
        print(f"Stderr: {result.stderr}")
        sys.exit(1)

    return result.returncode, result.stdout, result.stderr


def get_current_version() -> str:
    """Get current version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")

    if not pyproject_path.exists():
        print("❌ pyproject.toml not found")
        sys.exit(1)

    content = pyproject_path.read_text()
    match = re.search(r'^version = "([^"]*)"', content, re.MULTILINE)

    if not match:
        print("❌ Could not find version in pyproject.toml")
        sys.exit(1)

    return match.group(1)


def parse_version(version: str) -> tuple[int, int, int, str | None, int | None]:
    """Parse version string into components."""
    pattern = r"^(\d+)\.(\d+)\.(\d+)(?:\.(dev|alpha|beta|rc)(\d+))?$"
    match = re.match(pattern, version)

    if not match:
        print(f"❌ Invalid version format: {version}")
        sys.exit(1)

    major = int(match.group(1))
    minor = int(match.group(2))
    patch = int(match.group(3))
    pre_type = match.group(4)
    pre_num = int(match.group(5)) if match.group(5) else None

    return major, minor, patch, pre_type, pre_num


def increment_dev_version(version: str) -> str:
    """Increment dev version."""
    major, minor, patch, pre_type, pre_num = parse_version(version)

    if pre_type == "dev":
        new_pre_num = (pre_num or 0) + 1
        return f"{major}.{minor}.{patch}.dev{new_pre_num}"
    else:
        return f"{major}.{minor}.{patch}.dev1"


def increment_minor_version(version: str) -> str:
    """Increment minor version."""
    major, minor, patch, pre_type, pre_num = parse_version(version)
    return f"{major}.{minor + 1}.0"


def increment_patch_version(version: str) -> str:
    """Increment patch version."""
    major, minor, patch, pre_type, pre_num = parse_version(version)
    return f"{major}.{minor}.{patch + 1}"


def update_version_in_pyproject(new_version: str) -> None:
    """Update version in pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()

    new_content = re.sub(
        r'^version = "[^"]*"', f'version = "{new_version}"', content, flags=re.MULTILINE
    )

    pyproject_path.write_text(new_content)
    print(f"✅ Updated version to {new_version}")


def build_package() -> None:
    """Build the package."""
    print("🔨 Building package...")

    # Install build tools
    run_command("python -m pip install --upgrade build twine")

    # Clean old artifacts
    run_command("rm -rf dist/ build/ *.egg-info")

    # Build package
    run_command("python -m build")

    print("✅ Package built successfully")


def publish_package(test_pypi: bool = False) -> None:
    """Publish package to PyPI."""
    target = "TestPyPI" if test_pypi else "PyPI"
    print(f"🚀 Publishing to {target}...")

    if test_pypi:
        run_command("python -m twine upload --repository testpypi dist/*")
    else:
        run_command("python -m twine upload dist/*")

    print(f"✅ Published to {target} successfully!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Release script: increment version, build, and publish"
    )
    parser.add_argument("--minor", action="store_true", help="Increment minor version")
    parser.add_argument("--patch", action="store_true", help="Increment patch version")
    parser.add_argument(
        "--dev", action="store_true", help="Increment dev version (default)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would happen without doing it"
    )
    parser.add_argument(
        "--test-pypi", action="store_true", help="Publish to TestPyPI instead of PyPI"
    )

    args = parser.parse_args()

    # Default to dev if no specific increment type specified
    if not (args.minor or args.patch):
        args.dev = True

    print("🚀 Synth-Env Release Script")
    print("=" * 40)

    # Get current version and determine new version
    current_version = get_current_version()
    print(f"📦 Current version: {current_version}")

    if args.dev:
        new_version = increment_dev_version(current_version)
    elif args.minor:
        new_version = increment_minor_version(current_version)
    elif args.patch:
        new_version = increment_patch_version(current_version)

    print(f"🎯 New version: {new_version}")

    target = "TestPyPI" if args.test_pypi else "PyPI"

    if args.dry_run:
        print(f"\n🔍 DRY RUN - Would perform these actions:")
        print(f"  1. Update version: {current_version} → {new_version}")
        print(f"  2. Build package")
        print(f"  3. Publish to {target}")
        return

    # Confirm the release
    confirm = (
        input(f"\n❓ Release version {new_version} to {target}? (y/N): ")
        .strip()
        .lower()
    )
    if confirm != "y":
        print("❌ Release cancelled")
        return

    # Perform the release
    print(f"\n🎬 Starting release process...")

    # Step 1: Update version
    update_version_in_pyproject(new_version)

    # Step 2: Build package
    build_package()

    # Step 3: Publish to PyPI
    publish_package(args.test_pypi)

    print(f"\n🎉 Release {new_version} completed successfully!")
    print(f"📦 Package is now available on {target}")


if __name__ == "__main__":
    main()
