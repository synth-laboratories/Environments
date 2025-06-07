from dataclasses import dataclass
from src.environment.tools import EnvToolCall


@dataclass
class RecallSequenceTool(EnvToolCall):
    answer: str

    def __str__(self) -> str:
        preview = self.answer[:50]
        if len(self.answer) > 50:
            preview += "..."
        return f"RecallSequenceTool(answer='{preview}')"
