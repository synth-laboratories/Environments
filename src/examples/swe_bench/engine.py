from __future__ import annotations
import os
import shutil
import subprocess
import tempfile
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from stateful.engine import StatefulEngine, StatefulEngineSnapshot


# Snapshot for SWEBenchEngine state
@dataclass
class SWEBenchEngineSnapshot(StatefulEngineSnapshot):
    repo: str
    base_commit: str
    current_commit: str
    pending_fail_tests: List[str]
    pass_tests: List[str]


# Assuming StatefulEngine is provided by the SWE-agent framework.
# Here we subclass it to create our SWEBenchEngine.
class SWEBenchEngine(StatefulEngine):
    def __init__(self, task_data: Dict[str, Any]):
        """
        Initialize the engine with task-specific data.
        task_data is expected to have keys:
          - 'repo': e.g. 'owner/name' of the GitHub repo (or full git URL)
          - 'base_commit': the commit hash to checkout
          - 'fail_tests': list of test identifiers expected to fail initially
          - 'pass_tests': list of test identifiers expected to pass initially
        """
        super().__init__()  # Initialize base StatefulEngine if needed
        self.repo = task_data["repo"]
        self.base_commit = task_data["base_commit"]
        self.fail_tests: List[str] = task_data.get("fail_tests", [])
        self.pass_tests: List[str] = task_data.get("pass_tests", [])
        self.repo_dir: Optional[str] = None  # Directory where repo is cloned
        self.docker_image: Optional[str] = (
            None  # Docker image to use for tests (set after cloning if needed)
        )

    async def _reset_engine(self) -> Dict[str, Any]:
        """
        Prepares the environment for a new task. Clones the repo and checks out the base commit.
        Returns an initial observation (e.g., confirming reset or initial test statuses).
        """
        # Clean up any old repo directory if exists
        if self.repo_dir and os.path.isdir(self.repo_dir):
            shutil.rmtree(self.repo_dir, ignore_errors=True)
        # Create a fresh temp directory for the repository
        self.repo_dir = tempfile.mkdtemp(prefix="swebench_repo_")
        # Construct clone URL (assume public GitHub; if auth needed, handle separately)
        repo_url = self.repo
        if not repo_url.endswith(".git"):
            # If given as "owner/name", convert to https URL
            repo_url = f"https://github.com/{repo_url}.git"
        # Clone the repository
        try:
            subprocess.run(
                ["git", "clone", "--quiet", repo_url, self.repo_dir],
                check=True,
                timeout=300,
            )
        except subprocess.CalledProcessError as e:
            return {"error": f"Git clone failed: {e}"}
        except subprocess.TimeoutExpired:
            return {"error": "Git clone timed out"}
        # Checkout the specific commit
        try:
            subprocess.run(
                ["git", "-C", self.repo_dir, "checkout", "--quiet", self.base_commit],
                check=True,
                timeout=60,
            )
        except subprocess.CalledProcessError as e:
            return {"error": f"Git checkout failed: {e}"}
        except subprocess.TimeoutExpired:
            return {"error": "Git checkout timed out"}

        # Determine a suitable Docker image based on repo content (simple heuristic)
        self.docker_image = self._detect_docker_image()
        # Return initial observation (could run initial tests to report failing tests, but we'll just acknowledge reset)
        return {"status": "reset", "repo": self.repo, "commit": self.base_commit}

    async def _step_engine(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single agent action on the environment.
        The action is a dict with a "type" key indicating the action type, plus any parameters.
        Returns a structured observation dict.
        """
        action_type = action.get("type")
        if action_type == "run_tests":
            # Run tests (if specific tests provided, use them; otherwise default to all relevant tests)
            tests = action.get("tests")  # Optional list of test identifiers
            return self._run_tests_action(tests)
        elif action_type == "apply_patch":
            patch = action.get("patch")  # Expected to be a unified diff string
            return self._apply_patch_action(patch)
        elif action_type == "read_file":
            filepath = action.get("path")
            return self._read_file_action(filepath)
        else:
            # Unknown action
            return {"error": f"Unknown action type: {action_type}"}

    async def _serialize_engine(self) -> SWEBenchEngineSnapshot:
        """
        Serialize the current engine state to a dict. This could include current commit hash,
        applied patches (if tracking), or other info needed to restore the state.
        """
        current_commit = None
        try:
            # Get current HEAD commit (after patches applied, etc.)
            result = subprocess.run(
                ["git", "-C", self.repo_dir, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            current_commit = result.stdout.strip()
        except Exception:
            current_commit = (
                self.base_commit
            )  # Fallback to base commit if we can't get HEAD
        return SWEBenchEngineSnapshot(
            repo=self.repo,
            base_commit=self.base_commit,
            current_commit=current_commit,
            pending_fail_tests=self.fail_tests,
            pass_tests=self.pass_tests,
        )

    def _detect_docker_image(self) -> str:
        """
        Simple heuristic to choose a Docker image based on repository content.
        This can be expanded to support multiple languages.
        """
        # Default to a Python image as a baseline
        # (In a full implementation, detect language from files or config)
        return "python:3.10-slim"

    def _run_tests_action(self, tests: Optional[List[str]]) -> Dict[str, Any]:
        """Run the specified tests inside a Docker container and return their results."""
        if tests is None:
            # If no specific tests provided, run all fail_tests and pass_tests
            tests = self.fail_tests + self.pass_tests
        if not tests:
            return {"tests": {}, "note": "No tests specified"}

        results: Dict[str, Dict[str, Any]] = {}
        # Ensure Docker is available
        if self.docker_image is None:
            self.docker_image = self._detect_docker_image()
        for test in tests:
            # Construct the test command. We assume using pytest for Python tests.
            # Quote the test name if needed (especially if it contains param brackets).
            test_identifier = test
            # Define the command to run inside Docker
            # We mount the repo into /app and run pytest on the test identifier.
            docker_cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{self.repo_dir}:/app",
                "-w",
                "/app",
                self.docker_image,
                "bash",
                "-c",
                # The command installs requirements (if any) then runs pytest for the test
                "pip install -q . && pytest -q -x -o cache_dir=/tmp/pytest_cache "
                + test_identifier,
            ]
            try:
                proc = subprocess.run(
                    docker_cmd, capture_output=True, text=True, timeout=600
                )
            except subprocess.TimeoutExpired:
                # Kill the container if still running (subprocess timeout will have stopped waiting, container might persist)
                # We attempt to force-remove any container (this is a simple approach; in practice Docker CLI might handle it).
                results[test] = {
                    "status": "TIMEOUT",
                    "output": "Test execution exceeded 10 minute limit.",
                }
                # Try to kill any running container (if we had container name we could do docker rm -f)
                continue
            # If we reach here, command completed (possibly with failure). Capture exit code and output.
            exit_code = proc.returncode
            out = proc.stdout.strip()
            err = proc.stderr.strip()
            # Determine test status
            if exit_code == 0:
                results[test] = {"status": "PASS", "output": out}
            else:
                # If non-zero, test failed or errored. Include stderr and stdout for debugging.
                # (Truncate output if very large to keep observation size reasonable)
                combined_output = (out + "\n" + err).strip()
                if len(combined_output) > 10000:  # limit output size
                    combined_output = combined_output[:10000] + "... <truncated>"
                results[test] = {"status": "FAIL", "output": combined_output}
        return {"tests": results}

    def _apply_patch_action(self, patch: str) -> Dict[str, Any]:
        """Apply a unified diff patch to the repository code."""
        if not patch:
            return {"error": "No patch provided"}
        # Write the patch to a temporary file
        patch_path = os.path.join(self.repo_dir, "temp_patch.diff")
        try:
            with open(patch_path, "w") as pf:
                pf.write(patch)
        except Exception as e:
            return {"error": f"Failed to write patch file: {e}"}
        # Apply the patch using git
        try:
            subprocess.run(
                [
                    "git",
                    "-C",
                    self.repo_dir,
                    "apply",
                    "--whitespace=nowarn",
                    patch_path,
                ],
                check=True,
                timeout=30,
            )
        except subprocess.CalledProcessError as e:
            # Capture error message from git apply
            err_msg = (
                e.stderr.decode("utf-8")
                if hasattr(e, "stderr") and e.stderr
                else str(e)
            )
            return {"patch_status": "failed", "error": err_msg.strip()}
        except subprocess.TimeoutExpired:
            return {"patch_status": "failed", "error": "Patch apply timed out"}
        finally:
            # Remove the temp patch file
            try:
                os.remove(patch_path)
            except OSError:
                pass
        # If apply succeeds, optionally we could return the diff of changes or just success
        return {"patch_status": "applied"}

    def _read_file_action(self, filepath: str) -> Dict[str, Any]:
        """Read a file from the repository and return its content (or error if not found)."""
        if not filepath:
            return {"error": "No file path provided"}
        full_path = os.path.join(self.repo_dir, filepath)
        try:
            # Limit file size to avoid huge outputs
            if os.path.getsize(full_path) > 1000000:  # 1 MB limit for reading
                return {"file": filepath, "error": "File too large to read"}
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except FileNotFoundError:
            return {"file": filepath, "error": "File not found"}
        except Exception as e:
            return {"file": filepath, "error": f"Could not read file: {e}"}
        # Truncate content if it's extremely long
        if len(content) > 10000:
            content = content[:10000] + "\n... [truncated]"
        return {"file": filepath, "content": content}

    @classmethod
    async def _deserialize_engine(
        cls, snapshot: SWEBenchEngineSnapshot
    ) -> "SWEBenchEngine":
        """
        Recreate a SWEBenchEngine from a snapshot.
        """
        task_data = {
            "repo": snapshot.repo,
            "base_commit": snapshot.base_commit,
            "fail_tests": snapshot.pending_fail_tests,
            "pass_tests": snapshot.pass_tests,
        }
        engine = cls(task_data)
        return engine

# Alias for backwards compatibility: allow importing SweBenchEngine from examples.swe_bench.engine
SweBenchEngine = SWEBenchEngine
