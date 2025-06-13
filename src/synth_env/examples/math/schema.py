from dataclasses import dataclass, asdict, fields
from uuid import UUID
from typing import Optional, Dict, Any

from synth_env.tasks.core import TaskInstance, TaskInstanceMetadata, Impetus, Intent


@dataclass
class HendryksTaskInstanceMetadata(TaskInstanceMetadata):
    subject: Optional[str]
    category: Optional[str]
    level: Optional[str]
    solution: str


@dataclass
class HendryksTaskInstance(TaskInstance):
    async def serialize(self) -> Dict[str, Any]:
        data = asdict(self)
        if "id" in data and isinstance(data["id"], UUID):
            data["id"] = str(data["id"])
        return data

    @classmethod
    async def deserialize(cls, data: Dict[str, Any]) -> "HendryksTaskInstance":
        # Reconstruct UUID
        if "id" in data:
            try:
                data["id"] = UUID(str(data["id"]))
            except Exception:
                pass
        # Reconstruct impetus and intent
        if "impetus" in data and isinstance(data["impetus"], dict):
            data["impetus"] = Impetus(**data["impetus"])
        if "intent" in data and isinstance(data["intent"], dict):
            data["intent"] = Intent(**data["intent"])
        # Reconstruct metadata
        if "metadata" in data and isinstance(data["metadata"], dict):
            data["metadata"] = HendryksTaskInstanceMetadata(**data["metadata"])
        # Filter to constructor args
        field_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)
