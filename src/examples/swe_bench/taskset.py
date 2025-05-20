# Support for SWE-Bench Verifier/Lite OR SWE-Gym

# examples/swe_bench/taskset.py
"""
Unified Task-Instance loader for
  • SWE-Bench  (configs: "lite", "medium", "full", "verifier", ...)
  • SWE-Gym    (training-style split)

The module converts raw dataset rows into `SweBenchTaskInstance`s that plug
directly into the SweBenchEngine / SweBenchEnvironment you already have.

Usage
-----
```python
from examples.swe_bench.taskset import create_taskset

ts_lite   = await create_taskset(dataset="swe-bench", config="lite")
ts_verify = await create_taskset(dataset="swe-bench", config="verifier")
ts_gym    = await create_taskset(dataset="swe-gym")
```

Each call returns a TaskInstanceSet with train / val / test split info.
"""

from __future__ import annotations

import uuid, asyncio
from dataclasses import dataclass, asdict, fields
from typing import Dict, Any, List, Optional

from datasets import load_dataset                         # pip install datasets

from tasks.core import (                              # synth-ai task API
    Task,
    TaskInstance,
    TaskInstanceMetadata,
    TaskInstanceSet,
    Impetus,
    Intent,
    SplitInfo,
)

# ──────────────────────── benchmark-level Task object ─────────────────────────

swe_bench_task = Task(
    global_premises="GitHub issue → code-patch bug-fix tasks (SWE-Bench & SWE-Gym)",
    global_constraints="Use only the allowed code-navigation/edit/test tools",
    global_objectives="Make failing tests pass without regressions",
    shared_env_params={},
)

# ────────────────────────── dataset registry ───────────────────────────────────

_DATASETS: Dict[str, Dict[str, Any]] = {
    "swe-bench": {
        "hf_name": "princeton-nlp/swe-bench",        # HuggingFace hub
        "default_config": "lite",
        "splits_map": {"train":"train", "val":"validation", "test":"test"},
    },
    "swe-gym": {
        "hf_name": "princeton-nlp/swe-gym",          # identical schema
        "default_config": "main",
        "splits_map": {"train":"train", "val":"val", "test":"test"},
    },
}

# ───────────────────── metadata + instance classes ───────────────────────────

@dataclass
class SweBenchTaskInstanceMetadata(TaskInstanceMetadata):
    repo: str
    base_commit: str
    n_fail_tests: int
    n_pass_tests: int
    dataset_source: str  # "swe-bench" or "swe-gym"
    cfg_name: str        # e.g. "lite", "verifier", "main"
    split: str           # train / val / test

@dataclass
class SweBenchTaskInstance(TaskInstance):
    """
    initial_engine_snapshot stores the entire raw row from the dataset so
    the engine can access patch / tests / etc. with no extra I/O.
    """
    async def serialize(self) -> Dict[str, Any]:
        data = asdict(self)
        if isinstance(data.get("id"), uuid.UUID):
            data["id"] = str(data["id"])
        return data

    @classmethod
    async def deserialize(cls, data: Dict[str, Any]) -> "SweBenchTaskInstance":
        if "impetus" in data and isinstance(data["impetus"], dict):
            data["impetus"] = Impetus(**data["impetus"])
        if "intent" in data and isinstance(data["intent"], dict):
            data["intent"] = Intent(**data["intent"])
        if "metadata" in data and isinstance(data["metadata"], dict):
            data["metadata"] = SweBenchTaskInstanceMetadata(**data["metadata"])
        ctor_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in ctor_fields})

# ─────────────────────── row->instance helper ────────────────────────────────

def _row_to_instance(
    row: Dict[str, Any],
    split_name: str,
    dataset_nm: str,
    cfg_name: str,
) -> SweBenchTaskInstance:
    # normalize problem statement and truncate to 4k chars
    problem = row.get("problem_statement") or row.get("problem") or ""
    meta = SweBenchTaskInstanceMetadata(
        repo=row["repo"].replace("https://github.com/", "").rstrip(".git"),
        base_commit=row["base_commit"],
        n_fail_tests=len(row.get("fail_tests", [])),
        n_pass_tests=len(row.get("pass_tests", [])),
        dataset_source=dataset_nm,
        cfg_name=cfg_name,
        split=split_name,
    )
    impetus = Impetus(instructions=problem[:4096])
    intent = Intent(
        rubric={"goal": "Fix the bug so failing tests pass"},
        gold_trajectories=None,
        gold_state_diff={},  # evaluation is test-based
    )
    return SweBenchTaskInstance(
        id=uuid.uuid4(),
        impetus=impetus,
        intent=intent,
        metadata=meta,
        is_reproducible=True,
        initial_engine_snapshot=row,  # dump raw row for engine
    )

# ───────────────── public factory function ───────────────────────────────────

async def create_taskset(
    *,
    dataset: str = "swe-bench",      # "swe-bench" | "swe-gym"
    config: Optional[str] = None,      # "lite", "verifier", …
    cache_dir: Optional[str] = None,
    limit: Optional[int] = None,
    streaming: bool = False,
) -> TaskInstanceSet:
    """
    Build a TaskInstanceSet for the requested dataset / config.
    """
    if dataset not in _DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(_DATASETS)}")

    ds_info = _DATASETS[dataset]
    hf_name = ds_info["hf_name"]
    cfg_name = config or ds_info["default_config"]
    split_alias = ds_info["splits_map"]  # local->hf split mapping

    instances: List[SweBenchTaskInstance] = []
    for local_split, hf_split in split_alias.items():
        # HF 'validation' split may not exist for some configs → catch & skip
        try:
            ds = load_dataset(hf_name, cfg_name, split=hf_split, cache_dir=cache_dir, streaming=streaming)
        except (FileNotFoundError, ValueError):
            continue  # skip missing split

        for i, row in enumerate(ds):
            if limit is not None and i >= limit:
                break
            inst = _row_to_instance(
                row=row,
                split_name=local_split,
                dataset_nm=dataset,
                cfg_name=cfg_name,
            )
            instances.append(inst)

    val_ids = {i.id for i in instances if i.metadata.split == "val"}
    test_ids = {i.id for i in instances if i.metadata.split == "test"}
    split_info = SplitInfo(
        val_instance_ids=val_ids,
        test_instance_ids=test_ids,
        _is_split_defined=True,
    )

    return TaskInstanceSet(
        name=f"{dataset}-{cfg_name}",
        description=f"{dataset} ({cfg_name}) imported as SweBench tasks.",
        instances=instances,
        split_info=split_info,
    )

# ───────────────────── CLI sanity check ──────────────────────────────────────

if __name__ == "__main__":
    async def _demo():
        ts = await create_taskset(dataset="swe-bench", config="lite")
        print(f"{len(ts.instances):,} instances loaded ({len(ts.split_info.test_instance_ids)} test)")
        gym = await create_taskset(dataset="swe-gym")
        print(f"{len(gym.instances):,} SWE-Gym instances ready.")
    asyncio.run(_demo())