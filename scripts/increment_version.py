#!/usr/bin/env python3
"""
Simple version increment script for pyproject.toml

Usage:
    python scripts/increment_version.py          # Increment dev version (default)
    python scripts/increment_version.py --minor  # Increment minor version
    python scripts/increment_version.py --patch  # Increment patch version
"""

import argparse
import re
import sys
from pathlib import Path


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
    # Handle versions like "0.0.1.dev3" or "1.2.3"
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
        # Increment dev number
        new_pre_num = (pre_num or 0) + 1
        return f"{major}.{minor}.{patch}.dev{new_pre_num}"
    else:
        # Add dev suffix
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

    # Replace version line
    new_content = re.sub(
        r'^version = "[^"]*"', f'version = "{new_version}"', content, flags=re.MULTILINE
    )

    pyproject_path.write_text(new_content)
    print(f"✅ Updated version to {new_version}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Increment version in pyproject.toml")
    parser.add_argument("--minor", action="store_true", help="Increment minor version")
    parser.add_argument("--patch", action="store_true", help="Increment patch version")
    parser.add_argument(
        "--dev", action="store_true", help="Increment dev version (default)"
    )

    args = parser.parse_args()

    # Default to dev if no specific increment type specified
    if not (args.minor or args.patch):
        args.dev = True

    current_version = get_current_version()
    print(f"📦 Current version: {current_version}")

    if args.dev:
        new_version = increment_dev_version(current_version)
    elif args.minor:
        new_version = increment_minor_version(current_version)
    elif args.patch:
        new_version = increment_patch_version(current_version)

    print(f"🎯 New version: {new_version}")

    # Confirm the change
    confirm = (
        input(f"Update version from {current_version} to {new_version}? (y/N): ")
        .strip()
        .lower()
    )
    if confirm == "y":
        update_version_in_pyproject(new_version)
    else:
        print("❌ Cancelled")


if __name__ == "__main__":
    main()
