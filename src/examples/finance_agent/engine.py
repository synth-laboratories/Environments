import csv
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

from src.stateful.engine import StatefulEngine, StatefulEngineSnapshot
from .schema import FinanceTaskInstance


@dataclass
class FinancePublicState:
    question: str


@dataclass
class FinancePrivateState:
    gold_answer: str
    submitted_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    terminated: bool = False


@dataclass
class FinanceEngineSnapshot(StatefulEngineSnapshot):
    task_instance_dict: Dict[str, Any]


class FinanceEngine(StatefulEngine):
    def __init__(self, task_instance: FinanceTaskInstance):
        self.task_instance = task_instance
        self.submitted_answer: Optional[str] = None
        self.is_correct: Optional[bool] = None
        self.terminated: bool = False

    async def _reset_engine(
        self,
    ) -> Tuple[FinancePrivateState, FinancePublicState]:
        self.submitted_answer = None
        self.is_correct = None
        self.terminated = False
        priv = FinancePrivateState(gold_answer=self.task_instance.metadata.answer)
        pub = FinancePublicState(question=self.task_instance.metadata.question)
        return priv, pub

    async def _step_engine(
        self, submitted_answer: str
    ) -> Tuple[FinancePrivateState, FinancePublicState]:
        self.submitted_answer = submitted_answer
        self.is_correct = (
            submitted_answer.strip().lower()
            == self.task_instance.metadata.answer.strip().lower()
        )
        self.terminated = True
        priv = FinancePrivateState(
            gold_answer=self.task_instance.metadata.answer,
            submitted_answer=self.submitted_answer,
            is_correct=self.is_correct,
            terminated=True,
        )
        pub = FinancePublicState(question=self.task_instance.metadata.question)
        return priv, pub

    async def _serialize_engine(self) -> FinanceEngineSnapshot:
        data = await self.task_instance.serialize()  # type: ignore
        return FinanceEngineSnapshot(task_instance_dict=data)

    @classmethod
    async def _deserialize_engine(
        cls, snapshot: FinanceEngineSnapshot
    ) -> "FinanceEngine":
        task_instance = await FinanceTaskInstance.deserialize(
            snapshot.task_instance_dict
        )
        return cls(task_instance)
