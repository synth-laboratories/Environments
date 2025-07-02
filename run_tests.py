#!/usr/bin/env python3
"""
Synth-Env Test Runner Script
Usage: python run_tests.py [options]
Options:
    -u, --unit          Run unit tests only
    -i, --integration   Run integration tests only
    -s, --skip-service  Skip service check
    -e, --env ENV_NAME  Run tests for specific environment only
    -v, --verbose       Verbose output
    -h, --help          Show help message
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import requests
from typing import List, Tuple, Optional

# ANSI color codes
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def print_header(msg: str):
    print(f"\n{Colors.BLUE}==== {msg} ===={Colors.NC}")

def print_success(msg: str):
    print(f"{Colors.GREEN}✓ {msg}{Colors.NC}")

def print_error(msg: str):
    print(f"{Colors.RED}✗ {msg}{Colors.NC}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.NC}")

def check_service(skip_check: bool = False) -> bool:
    """Check if the service is running."""
    if skip_check:
        return True
    
    print_header("Checking if service is running")
    
    try:
        response = requests.get("http://localhost:6532/health", timeout=2)
        if response.status_code == 200:
            print_success("Service is running on http://localhost:6532")
            return True
    except:
        pass
    
    print_warning("Service is not running on http://localhost:6532")
    print("To start the service, run: cd src && python -m synth_env.service.app")
    print("Continuing with tests that don't require the service...")
    return False

def run_pytest(test_path: str, verbose: bool = False, extra_args: List[str] = None) -> bool:
    """Run pytest on the given path."""
    cmd = [sys.executable, "-m", "pytest", test_path]
    
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    if extra_args:
        cmd.extend(extra_args)
    
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=not verbose)
        return result.returncode == 0
    except Exception as e:
        print_error(f"Failed to run pytest: {e}")
        return False

def run_type_check(env_path: str) -> bool:
    """Run type checking on the given environment."""
    try:
        cmd = ["uvx", "ty", "check", env_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if "error[" in result.stdout or "error[" in result.stderr:
            print_warning(f"Type errors found (non-blocking)")
            return False
        else:
            print_success("No type errors found")
            return True
    except FileNotFoundError:
        # uvx not installed
        return True
    except Exception as e:
        print_warning(f"Type check skipped: {e}")
        return True

def run_env_tests(env_name: str, run_unit: bool, run_integration: bool, verbose: bool) -> bool:
    """Run tests for a specific environment."""
    env_path = Path(f"src/synth_env/examples/{env_name}")
    
    if not env_path.exists():
        print_error(f"Environment '{env_name}' not found at {env_path}")
        return False
    
    print_header(f"Testing {env_name} environment")
    
    all_passed = True
    
    # Run unit tests
    if run_unit and (env_path / "units").exists():
        print(f"\n{Colors.YELLOW}Running unit tests for {env_name}...{Colors.NC}")
        if run_pytest(str(env_path / "units"), verbose):
            print_success(f"Unit tests passed for {env_name}")
        else:
            print_error(f"Unit tests failed for {env_name}")
            all_passed = False
    
    # Run integration tests (agent demos)
    if run_integration and (env_path / "agent_demos").exists():
        print(f"\n{Colors.YELLOW}Running integration tests for {env_name}...{Colors.NC}")
        
        test_files = list((env_path / "agent_demos").glob("test_*.py"))
        
        if not test_files:
            print_warning(f"No integration tests found for {env_name}")
        else:
            for test_file in test_files:
                print(f"Running {test_file.name}...")
                # Skip eval_ tests as they may take a long time
                if run_pytest(str(test_file), verbose, ["-k", "not eval_", "--tb=short"]):
                    print_success(f"{test_file.name} passed")
                else:
                    print_error(f"{test_file.name} failed")
                    all_passed = False
    
    # Run type checking
    print(f"\n{Colors.YELLOW}Running type check for {env_name}...{Colors.NC}")
    run_type_check(str(env_path))
    
    return all_passed

def main():
    parser = argparse.ArgumentParser(description="Synth-Env Test Runner")
    parser.add_argument("-u", "--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("-i", "--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("-s", "--skip-service", action="store_true", help="Skip service check")
    parser.add_argument("-e", "--env", type=str, help="Run tests for specific environment only")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set test types
    run_unit = True
    run_integration = True
    
    if args.unit and not args.integration:
        run_unit = True
        run_integration = False
    elif args.integration and not args.unit:
        run_unit = False
        run_integration = True
    
    print_header("Synth-Env Test Runner")
    
    # Check Python version
    print(f"Python version: {sys.version.split()[0]}")
    
    # Check if pytest is installed
    try:
        subprocess.run([sys.executable, "-m", "pytest", "--version"], 
                      capture_output=True, check=True)
    except:
        print_error("pytest is not installed. Please install it with: pip install pytest")
        sys.exit(1)
    
    # Check service status
    check_service(args.skip_service)
    
    # List of all environments
    environments = ["tictactoe", "sokoban", "verilog", "crafter_classic"]
    
    # If specific environment is requested
    if args.env:
        if args.env in environments:
            environments = [args.env]
        else:
            print_error(f"Unknown environment: {args.env}")
            print(f"Available environments: {', '.join(environments)}")
            sys.exit(1)
    
    # Track overall success
    all_passed = True
    
    # Run tests for each environment
    for env in environments:
        if not run_env_tests(env, run_unit, run_integration, args.verbose):
            all_passed = False
    
    # Run core tests if not testing specific environment
    if not args.env:
        print_header("Testing core modules")
        
        # Test core task module
        if run_unit and Path("src/synth_env/tasks/units").exists():
            print(f"\n{Colors.YELLOW}Running core task tests...{Colors.NC}")
            if run_pytest("src/synth_env/tasks/units", args.verbose):
                print_success("Core task tests passed")
            else:
                print_error("Core task tests failed")
                all_passed = False
    
    # Summary
    print_header("Test Summary")
    
    if all_passed:
        print_success("All tests passed!")
        sys.exit(0)
    else:
        print_error("Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()