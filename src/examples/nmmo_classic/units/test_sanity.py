# examples/nmmo3/units/test_engine_snapshot.py
"""
Quick sanity-check for the NMMO3Engine:

1.  Build an engine from a tiny TaskInstance and call _reset_engine.
2.  Execute 5 random actions.
3.  Serialize -> deserialize into a *second* engine.
4.  Step both engines through an identical 10-step rollout and
    assert their public states stay identical.

If any of the core snapshot plumbing (pickle blob, RNG state, etc.)
is broken this test will fail.
"""

from __future__ import annotations
import asyncio, random, uuid, pytest

from examples.nmmo_classic.engine import NMMO3Engine
from examples.nmmo_classic.state import NMMO3PublicState
from examples.nmmo_classic.taskset import (
    NMMO3TaskInstance,
    NMMO3TaskInstanceMetadata,
)
from tasks.core import Impetus, Intent

# we reuse the small ACTION_TEMPLATES Dict from test_synth_react
from examples.nmmo_classic.test_synth_react import ACTION_TEMPLATES


@pytest.mark.asyncio
async def test_snapshot_roundtrip():
    """Step → snapshot → restore → step; states must stay in lock-step."""

    # ------------------------------------------------------------------ setup
    seed = 314
    meta = NMMO3TaskInstanceMetadata(
        difficulty="easy",
        seed=seed,
        map_size=64,
        season="summer",
        resource_density=0.0,
        water_pct=0.0,
        hostiles_25=0,
        forage_tiles_25=0,
        spawn_biome="plains",
    )
    inst = NMMO3TaskInstance(
        id=uuid.uuid4(),
        impetus=Impetus(instructions="unit-test"),
        intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
        metadata=meta,
        is_reproducible=True,
        initial_engine_snapshot=None,
        config={"seed": seed, "map_size": 64, "tick_limit": 200},
    )

    eng1 = NMMO3Engine(inst)
    priv1, pub1 = await eng1._reset_engine()

    # ------------------------------------------------------------------ 1st rollout (5 random acts)
    act_names = list(ACTION_TEMPLATES.keys())
    rng = random.Random(123)
    for _ in range(5):
        name = rng.choice(act_names)
        action = ACTION_TEMPLATES[name]
        _, pub1 = await eng1._step_engine(action)

    # ------------------------------------------------------------------ snapshot / restore
    snap = await eng1._serialize_engine()
    eng2 = await NMMO3Engine._deserialize_engine(snap)

    # sanity: take a no-op step and compare public states
    _, pub1 = await eng1._step_engine({})
    _, pub2 = await eng2._step_engine({})
    assert pub1.diff(pub2) == {}, "States diverged immediately after restore"

    # ------------------------------------------------------------------ 2nd rollout (10 identical acts)
    for _ in range(10):
        name = rng.choice(act_names)
        action = ACTION_TEMPLATES[name]
        _, pub1 = await eng1._step_engine(action)
        _, pub2 = await eng2._step_engine(action)

        diff = pub1.diff(pub2)
        assert diff == {}, f"Snapshot mismatch after identical action – {diff}"

    # If we reach here the round-trip is consistent
