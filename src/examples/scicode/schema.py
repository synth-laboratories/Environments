from dataclasses import dataclass, asdict, fields
from uuid import UUID
from typing import Optional, Dict, Any

from src.tasks.core import TaskInstance, TaskInstanceMetadata, Impetus, Intent


@dataclass
class SciCodeTaskInstanceMetadata(TaskInstanceMetadata):
    category: Optional[str]
    difficulty: Optional[str]
    solution: str


@dataclass
class SciCodeTaskInstance(TaskInstance):
    async def serialize(self) -> Dict[str, Any]:
        data = asdict(self)
        if "id" in data and isinstance(data["id"], UUID):
            data["id"] = str(data["id"])
        return data

    @classmethod
    async def deserialize(cls, data: Dict[str, Any]) -> "SciCodeTaskInstance":
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
            data["metadata"] = SciCodeTaskInstanceMetadata(**data["metadata"])
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)
