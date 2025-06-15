import csv
import uuid
from pathlib import Path
from typing import List

from src.tasks.core import Impetus, Intent, TaskInstanceSet, SplitInfo
from .schema import FinanceTaskInstance, FinanceTaskInstanceMetadata


def load_finance_tasks(csv_path: Path) -> List[dict]:
    tasks = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append(row)
    return tasks


async def create_finance_taskset() -> TaskInstanceSet:
    data_path = Path(__file__).parent / "data" / "public.csv"
    tasks = load_finance_tasks(data_path)
    instances: List[FinanceTaskInstance] = []
    for t in tasks:
        meta = FinanceTaskInstanceMetadata(
            question=t.get("Question", ""),
            answer=t.get("Answer", ""),
            question_type=t.get("Question Type"),
            expert_time_mins=int(t.get("Expert time (mins)", 0) or 0),
            rubric=t.get("Rubric"),
        )
        inst = FinanceTaskInstance(
            id=uuid.uuid4(),
            impetus=Impetus(instructions="Answer the financial question."),
            intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
            metadata=meta,
            is_reproducible=True,
            initial_engine_snapshot=None,
        )
        instances.append(inst)

    split_info = SplitInfo(set(), set(), _is_split_defined=False)
    return TaskInstanceSet(
        name="ValsAI Finance Agent Benchmark",
        description="Tasks derived from ValsAI finance agent benchmark dataset",
        instances=instances,
        split_info=split_info,
    )
