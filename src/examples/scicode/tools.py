from src.environment.tools import EnvToolCall


class SubmitAnswerTool(EnvToolCall):
    """Simple wrapper holding the submitted answer."""

    def __init__(self, answer: str):
        # EnvToolCall inherits from Pydantic's BaseModel which forbids setting
        # undeclared attributes via normal assignment. Use object.__setattr__
        # to bypass validation as we just need a lightweight wrapper.
        object.__setattr__(self, "answer", answer)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"SubmitAnswerTool(answer='{self.answer[:50]}...')"
