import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.stateful.engine import StatefulEngine, StatefulEngineSnapshot


@dataclass
class MemoryTextPublicState:
    prompt: str
    terminated: bool
    correct: Optional[bool]


@dataclass
class MemoryTextPrivateState:
    sequence: List[int]
    submitted_answer: Optional[str]
    correct: Optional[bool]
    terminated: bool


@dataclass
class MemoryTextEngineSnapshot(StatefulEngineSnapshot):
    sequence: List[int]
    submitted_answer: Optional[str]
    correct: Optional[bool]
    terminated: bool


class MemoryTextEngine(StatefulEngine):
    def __init__(self, sequence_length: int = 5):
        self.sequence_length = sequence_length
        self.sequence: List[int] = []
        self.submitted_answer: Optional[str] = None
        self.correct: Optional[bool] = None
        self.terminated: bool = False

    async def _reset_engine(
        self,
    ) -> Tuple[MemoryTextPrivateState, MemoryTextPublicState]:
        self.sequence = [random.randint(0, 9) for _ in range(self.sequence_length)]
        self.submitted_answer = None
        self.correct = None
        self.terminated = False
        priv = MemoryTextPrivateState(
            sequence=self.sequence,
            submitted_answer=None,
            correct=None,
            terminated=False,
        )
        pub = MemoryTextPublicState(
            prompt="Memorize: " + " ".join(map(str, self.sequence)),
            terminated=False,
            correct=None,
        )
        return priv, pub

    async def _step_engine(
        self, answer: str
    ) -> Tuple[MemoryTextPrivateState, MemoryTextPublicState]:
        self.submitted_answer = answer
        self.correct = answer.strip() == " ".join(map(str, self.sequence))
        self.terminated = True
        priv = MemoryTextPrivateState(
            sequence=self.sequence,
            submitted_answer=self.submitted_answer,
            correct=self.correct,
            terminated=True,
        )
        pub = MemoryTextPublicState(
            prompt="Result",
            terminated=True,
            correct=self.correct,
        )
        return priv, pub

    def get_current_states_for_observation(
        self,
    ) -> Tuple[MemoryTextPrivateState, MemoryTextPublicState]:
        priv = MemoryTextPrivateState(
            sequence=self.sequence,
            submitted_answer=self.submitted_answer,
            correct=self.correct,
            terminated=self.terminated,
        )
        pub = MemoryTextPublicState(
            prompt="Memorize: " + " ".join(map(str, self.sequence))
            if not self.terminated
            else "Result",
            terminated=self.terminated,
            correct=self.correct,
        )
        return priv, pub

    async def _serialize_engine(self) -> MemoryTextEngineSnapshot:
        return MemoryTextEngineSnapshot(
            sequence=self.sequence,
            submitted_answer=self.submitted_answer,
            correct=self.correct,
            terminated=self.terminated,
        )

    @classmethod
    async def _deserialize_engine(
        cls, snapshot: MemoryTextEngineSnapshot
    ) -> "MemoryTextEngine":
        engine = cls(len(snapshot.sequence))
        engine.sequence = list(snapshot.sequence)
        engine.submitted_answer = snapshot.submitted_answer
        engine.correct = snapshot.correct
        engine.terminated = snapshot.terminated
        return engine
