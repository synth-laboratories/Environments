from dataclasses import dataclass
from synth_env.environment.tools import EnvToolCall


@dataclass
class SubmitAnswerTool(EnvToolCall):
    answer: str
    # Optional: reasoning: str

    def __str__(self):
        return f"SubmitAnswerTool(answer='{self.answer[:50]}...')"
