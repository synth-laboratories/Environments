from dataclasses import dataclass, asdict, fields
from typing import Optional, Dict, Any
from uuid import UUID

from src.tasks.core import TaskInstance, TaskInstanceMetadata, Impetus, Intent


@dataclass
class FinanceTaskInstanceMetadata(TaskInstanceMetadata):
    question: str
    answer: str
    question_type: Optional[str] = None
    expert_time_mins: Optional[int] = None
    rubric: Optional[str] = None


@dataclass
class FinanceTaskInstance(TaskInstance):
    async def serialize(self) -> Dict[str, Any]:
        data = asdict(self)
        if isinstance(data.get("id"), UUID):
            data["id"] = str(data["id"])
        return data

    @classmethod
    async def deserialize(cls, data: Dict[str, Any]) -> "FinanceTaskInstance":
        if "id" in data:
            try:
                data["id"] = UUID(str(data["id"]))
            except Exception:
                pass
        if "impetus" in data and isinstance(data["impetus"], dict):
            data["impetus"] = Impetus(**data["impetus"])
        if "intent" in data and isinstance(data["intent"], dict):
            data["intent"] = Intent(**data["intent"])
        if "metadata" in data and isinstance(data["metadata"], dict):
            data["metadata"] = FinanceTaskInstanceMetadata(**data["metadata"])
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})
