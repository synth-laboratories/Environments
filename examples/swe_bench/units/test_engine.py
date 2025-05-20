# examples/swe_bench/units/test_engine.py
"""
Smoke-tests for SweBenchEngine + SweBenchEnvironment.

Run with:
    pytest -q examples/swe_bench/units/test_engine.py
"""

from __future__ import annotations
import asyncio, textwrap, uuid, re
from pathlib import Path
from typing import Dict, Any

import pytest

from examples.swe_bench.taskset     import create_taskset
from examples.swe_bench.environment import (            # wrappers exposed by env
    SweBenchEnvironment,
    OpenFile,   OpenFileArgs,
    ApplyPatch, ApplyPatchArgs,
    RunTests,
    Submit,
)
from examples.swe_bench.engine      import SweBenchEngine


# ───────────────────────── helpers ──────────────────────────
def _first_test_file(row: Dict[str, Any]) -> str:
    """Return a file path that *surely* exists in the repo."""
    for lst_name in ("fail_tests", "pass_tests"):
        lst = row.get(lst_name, [])
        if lst:
            return lst[0].split("::", 1)[0]
    return "setup.py"            # fallback


# ───────────────────────── fixtures ─────────────────────────
@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def lite_instance():
    """Returns a single TaskInstance or *skips* if no data is available."""
    ts = await create_taskset(dataset="swe-bench", config="lite")
    if not ts.instances:                                        # <-- guard
        pytest.skip("SWE-Bench *lite* split unavailable (no internet?).")
    inst = min(ts.instances, key=lambda i: len(i.impetus.instructions))
    return inst


# ───────────────────────── tests ────────────────────────────
@pytest.mark.asyncio
async def test_engine_bootstrap(lite_instance):
    eng = SweBenchEngine(lite_instance)
    priv, pub = await eng._reset_engine()

    assert pub["last_action"] == "reset"
    assert isinstance(pub["problem"], str) and pub["problem"]
    assert priv["terminated"] is False


@pytest.mark.asyncio
async def test_env_command_roundtrip(lite_instance, tmp_path: Path):
    env = SweBenchEnvironment(lite_instance)
    obs = await env.initialize()

    row         = lite_instance.initial_engine_snapshot
    target_path = _first_test_file(row)

    # 1️⃣ open_file
    obs = await env.step(
        OpenFile(**OpenFileArgs(path=target_path, start=1, end=25).model_dump())
    )
    assert "file_snippet" in obs and obs["file_snippet"].strip()

    # 2️⃣ apply_patch  – prepend a trivial comment
    patch = (
        textwrap.dedent(f"""
        --- a/{target_path}
        +++ b/{target_path}
        @@
        +# added by swe-bench test ({uuid.uuid4().hex[:6]})
        """).strip()
        + "\n"
    )
    obs = await env.step(ApplyPatch(**ApplyPatchArgs(patch=patch).model_dump()))
    assert obs.get("edit_applied") is True

    # 3️⃣ run_tests
    obs = await env.step(RunTests())
    assert isinstance(obs.get("tests_passed"), bool)
    assert "test_output" in obs

    # 4️⃣ submit  – terminates
    obs = await env.step(Submit(reason="unit-test finish"))
    assert obs["terminated"] is True

    # 5️⃣ checkpoint round-trip
    snap = await env.checkpoint()
    eng2 = await SweBenchEngine._deserialize_engine(snap)
    assert isinstance(eng2, SweBenchEngine)


@pytest.mark.asyncio
async def test_apply_patch_validation(lite_instance):
    env = SweBenchEnvironment(lite_instance)
    _   = await env.initialize()

    obs = await env.step(ApplyPatch(patch="NOT A DIFF"))
    assert obs.get("error") and re.search("patch", obs["error"], re.I)
