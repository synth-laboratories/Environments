from dataclasses import dataclass, asdict, fields
from typing import Tuple, Optional, Dict, Any
from uuid import UUID

from src.tasks.core import TaskInstance, TaskInstanceMetadata, Impetus, Intent


@dataclass
class TreasureHunterTaskInstanceMetadata(TaskInstanceMetadata):
    grid_size: int = 5
    max_steps: int = 20
    treasure_pos: Tuple[int, int] = (4, 4)


@dataclass
class TreasureHunterTaskInstance(TaskInstance):
    async def serialize(self) -> Dict[str, Any]:
        data = asdict(self)
        if isinstance(data.get("id"), UUID):
            data["id"] = str(data["id"])
        return data

    @classmethod
    async def deserialize(cls, data: Dict[str, Any]) -> "TreasureHunterTaskInstance":
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
            data["metadata"] = TreasureHunterTaskInstanceMetadata(**data["metadata"])
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)
