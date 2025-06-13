"""
Procedural NMMO 3 task-set generation.
Iterates over seeds, inspects the freshly-reset world, and buckets it into
easy / medium / hard depending on resource and hostile density.

Requires:

    pip install "pufferlib[nmmo]"  # installs both PufferLib and Neural MMO
"""

from __future__ import annotations
import asyncio
import random
from uuid import uuid4, UUID
from dataclasses import dataclass, asdict, fields
from typing import Dict, List, Any

import nmmo
import pufferlib.emulation as pe
import numpy as np

from synth_env.tasks.core import (
    Task,
    Impetus,
    Intent,
    SplitInfo,
    TaskInstance,
    TaskInstanceMetadata,
    TaskInstanceMetadataFilter,
    TaskInstanceSet,
)

# ──────────────────────────────────────────────────────────────
# Global task description (mirrors Crafter/Sokoban pattern)
# ──────────────────────────────────────────────────────────────
TASK = Task(
    global_premises="Procedurally-generated Neural MMO 3 worlds",
    global_constraints="",
    global_objectives="Survive and maximise personal + team score.",
    shared_env_params={},
)

# generation hyper-params
NUM_INSTANCES = 60
SEED_START = 0
MAP_SIZE_CHOICES = [128, 256]  # 256 ≈ 'harder' due to travel cost
TICK_LIMIT = 1_000

# difficulty buckets – tweak as you like
DIFF_BOUNDS = {
    "easy": dict(min_forage=30, max_hostiles=0, max_water=0.15),
    "medium": dict(min_forage=15, max_hostiles=8, max_water=0.25),
    "hard": dict(min_forage=0, max_hostiles=30, max_water=1.00),
}

# terrain type ids in NMMO default config (see nmmo.lib.terrain)
# TODO: Update terrain constants for nmmo 2.1.2 API
WATER_IDs = {1}  # placeholder - was nmmo.Terrain.WATER
FORAGEABLE_IDs = {
    2,
    3,
    4,
    5,
    6,
}  # placeholder - terrain constants changed in nmmo 2.1.2


# ──────────────────────────────────────────────────────────────
# Metadata & TaskInstance dataclasses
# ──────────────────────────────────────────────────────────────
@dataclass
class NeuralMMOTaskInstanceMetadata(TaskInstanceMetadata):
    difficulty: str
    seed: int
    map_size: int
    season: str
    resource_density: float
    water_pct: float
    hostiles_25: int
    forage_tiles_25: int
    spawn_biome: str


class DifficultyFilter(TaskInstanceMetadataFilter):
    """Filter TaskInstance by difficulty using DIFF_BOUNDS."""

    def __init__(self, tier: str):
        self.tier = tier

    def __call__(self, instance: NeuralMMOTaskInstance) -> bool:
        m = instance.metadata
        b = DIFF_BOUNDS[self.tier]
        return (
            m.forage_tiles_25 >= b["min_forage"]
            and m.hostiles_25 <= b["max_hostiles"]
            and m.water_pct <= b["max_water"]
        )


@dataclass
class NeuralMMOTaskInstance(TaskInstance):
    async def serialize(self) -> dict:
        d = asdict(self)
        if isinstance(d.get("id"), UUID):
            d["id"] = str(d["id"])
        return d

    @classmethod
    async def deserialize(cls, data: dict) -> "NeuralMMOTaskInstance":
        if "id" in data:
            try:
                data["id"] = UUID(str(data["id"]))
            except Exception:
                pass
        if "impetus" in data:
            data["impetus"] = Impetus(**data["impetus"])
        if "intent" in data:
            data["intent"] = Intent(**data["intent"])
        if "metadata" in data:
            data["metadata"] = NeuralMMOTaskInstanceMetadata(**data["metadata"])
        keep = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in keep})


# ──────────────────────────────────────────────────────────────
# World-analysis helpers
# ──────────────────────────────────────────────────────────────
def _analyse_world(cfg: nmmo.config.Config, obs: Dict[str, Any]) -> Dict[str, Any]:
    """Return static metrics computed from the very first observation."""
    terrain = obs["terrain"]  # (H×W) uint8
    h, w = terrain.shape
    passable_mask = terrain != nmmo.Terrain.LAVA  # ≈ impassable id

    # global metrics --------------------------------------------------
    water_pct = float(np.sum(np.isin(terrain, list(WATER_IDs))) / np.sum(passable_mask))
    resource_density = float(np.sum(np.isin(terrain, list(FORAGEABLE_IDs))) / (h * w))

    # local (radius 25) metrics --------------------------------------
    entities = obs["entities"]  # pandas-style table
    self_row = obs["self_row"]
    self_col = obs["self_col"]
    dR = np.abs(np.asarray(entities["row"]) - self_row)
    dC = np.abs(np.asarray(entities["col"]) - self_col)
    manhattan = dR + dC
    within25 = manhattan <= 25

    hostiles_25 = int(
        np.sum((entities["damage"] > 0) & within25)
    )  # NPCs with attack dmg
    forage25 = int(
        np.sum(
            np.isin(
                terrain[
                    max(0, self_row - 25) : self_row + 26,
                    max(0, self_col - 25) : self_col + 26,
                ],
                list(FORAGEABLE_IDs),
            )
        )
    )

    # spawn tile biome
    spawn_biome = int(terrain[self_row, self_col])

    return dict(
        map_size=cfg.MAP_SIZE,
        season=cfg.SEASON.name,  # e.g. "SPRING"
        resource_density=resource_density,
        water_pct=water_pct,
        hostiles_25=hostiles_25,
        forage_tiles_25=forage25,
        spawn_biome=str(spawn_biome),
    )


def _classify(metrics: Dict[str, Any]) -> str | None:
    """Return 'easy'/'medium'/'hard' if metrics fit, else None."""
    for tier, b in DIFF_BOUNDS.items():
        if (
            metrics["forage_tiles_25"] >= b["min_forage"]
            and metrics["hostiles_25"] <= b["max_hostiles"]
            and metrics["water_pct"] <= b["max_water"]
        ):
            return tier
    return None


# ──────────────────────────────────────────────────────────────
# Main generator
# ──────────────────────────────────────────────────────────────
async def create_neural_mmo_taskset(
    num_instances: int = NUM_INSTANCES,
) -> TaskInstanceSet:
    instances: List[NeuralMMOTaskInstance] = []
    seed = SEED_START

    while len(instances) < num_instances:
        # --- 1. build config ----------------------------------------
        cfg = nmmo.config.Alpha()
        cfg.MAP_SIZE = random.choice(MAP_SIZE_CHOICES)
        cfg.SEED = seed
        cfg.TICK_LIMIT = TICK_LIMIT  # so meta matches engine cfg

        # --- 2. reset single-env Puffer wrapper ---------------------
        env = pe.make("nmmo", config=cfg, num_envs=1)
        obs_vec = env.reset()  # list-of-dict for each env idx
        obs = obs_vec[0]  # we only use env 0

        # --- 3. analyse world & bucket ------------------------------
        metrics = _analyse_world(cfg, obs)
        # Determine difficulty via DifficultyFilter
        difficulty = None
        # Build a temporary instance for filtering
        temp_meta = NeuralMMOTaskInstanceMetadata(difficulty="", seed=seed, **metrics)
        temp_instance = NeuralMMOTaskInstance(
            id=uuid4(),
            impetus=Impetus(instructions=""),
            intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
            metadata=temp_meta,
            is_reproducible=True,
            initial_engine_snapshot=None,
            config={"seed": seed, "map_size": cfg.MAP_SIZE, "tick_limit": TICK_LIMIT},
        )
        for tier in DIFF_BOUNDS:
            if DifficultyFilter(tier)(temp_instance):
                difficulty = tier
                break
        if difficulty is None:
            seed += 1
            continue  # reject & try next seed

        # Reassign proper metadata and instructions, then append
        temp_instance.metadata.difficulty = difficulty
        temp_instance.impetus = Impetus(
            instructions=f"Survive and maximise score. Difficulty={difficulty}."
        )
        temp_instance.intent = Intent(rubric={"goal": "Score as high as possible"})
        instances.append(temp_instance)
        seed += 1

    # --- 5. train/val/test split (80/10/10) -------------------------
    random.shuffle(instances)
    n = len(instances)
    split = SplitInfo(
        val_instance_ids={i.id for i in instances[int(0.8 * n) : int(0.9 * n)]},
        test_instance_ids={i.id for i in instances[int(0.9 * n) :]},
        _is_split_defined=True,
    )

    return TaskInstanceSet(
        name="Neural MMO Classic Procedural TaskSet",
        description="Worlds bucketed by forage / hostile / water density.",
        instances=instances,
        split_info=split,
    )


# ──────────────────────────────────────────────────────────────
# CLI quick-check
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    async def _main():
        ts = await create_neural_mmo_taskset(12)
        print(
            f"Generated {len(ts.instances)} instances ➜ "
            f"{len(ts.split_info.val_instance_ids)} val / "
            f"{len(ts.split_info.test_instance_ids)} test"
        )

    asyncio.run(_main())
