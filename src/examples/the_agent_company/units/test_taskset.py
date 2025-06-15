import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[4]
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "src"))

import asyncio
import pytest
from examples.the_agent_company.taskset import create_tac_taskset


def test_create_tac_taskset():
    taskset = asyncio.run(create_tac_taskset(max_tasks=3))
    assert len(taskset.instances) == 3
    for inst in taskset.instances:
        assert inst.metadata.image.startswith("ghcr.io/theagentcompany/")
