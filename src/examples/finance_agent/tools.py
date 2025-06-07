from dataclasses import dataclass


@dataclass
class SubmitAnswerTool:
    answer: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"SubmitAnswerTool(answer='{self.answer[:50]}...')"
