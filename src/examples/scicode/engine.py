from __future__ import annotations
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from datasets import load_dataset

from src.stateful.engine import StatefulEngine, StatefulEngineSnapshot
from src.environment.shared_engine import GetObservationCallable, InternalObservation
from src.tasks.core import TaskInstance as BaseTaskInstance


# --- Task Loading and Scoring -------------------------------------------------
def load_tasks() -> List[Dict[str, Any]]:
    """Load SciCode tasks from cache or HuggingFace."""
    cache_path = Path(__file__).parent / "dataset" / "scicode.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    ds = load_dataset("allenai/scicode", split="train")
    tasks: List[Dict[str, Any]] = []
    for ex in ds:
        tasks.append(
            {
                "id": str(uuid.uuid4()),
                "prompt": ex.get("prompt") or ex.get("question"),
                "solution": ex.get("solution") or ex.get("answer"),
                "category": ex.get("category"),
                "difficulty": ex.get("difficulty"),
            }
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(tasks, f)

    return tasks


def score_response(response: str, gold: str) -> bool:
    return response.strip() == gold.strip()


# --- Engine States and Snapshot ----------------------------------------------
@dataclass
class SciCodePublicState:
    problem_id: str
    prompt: str
    category: Optional[str]
    difficulty: Optional[str]


@dataclass
class SciCodePrivateState:
    solution: str
    submitted_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    terminated: bool = False


@dataclass
class SciCodeEngineSnapshot(StatefulEngineSnapshot):
    task_instance_dict: Dict
    current_problem_id: str


# --- Observation Callables ----------------------------------------------------
class SynthSciCodeObservationCallable(GetObservationCallable):
    async def get_observation(
        self, pub: SciCodePublicState, priv: SciCodePrivateState
    ) -> InternalObservation:
        obs = {
            "problem_id": pub.problem_id,
            "prompt": pub.prompt,
            "category": pub.category,
            "difficulty": pub.difficulty,
            "terminated": priv.terminated,
        }
        if priv.submitted_answer is not None:
            obs["submitted_answer"] = priv.submitted_answer
            obs["is_correct"] = priv.is_correct
        return obs


# --- SciCode Engine -----------------------------------------------------------
class SciCodeEngine(StatefulEngine):
    def __init__(self, task_instance: BaseTaskInstance):
        super().__init__()
        self.task_instance = task_instance
        self._all_tasks: List[Dict[str, Any]] = load_tasks()
        self._task_lookup: Dict[str, Dict[str, Any]] = {
            t["id"]: t for t in self._all_tasks
        }

        self.current_problem_id: Optional[str] = None
        self.current_prompt: Optional[str] = None
        self.current_solution: Optional[str] = None
        self.current_category: Optional[str] = None
        self.current_difficulty: Optional[str] = None

        self.submitted_answer: Optional[str] = None
        self.is_correct: Optional[bool] = None
        self.terminated: bool = False

    def _load_problem(self, problem_id: str) -> bool:
        data = self._task_lookup.get(problem_id)
        if not data:
            return False
        self.current_problem_id = data["id"]
        self.current_prompt = data["prompt"]
        self.current_solution = data["solution"]
        self.current_category = data.get("category")
        self.current_difficulty = data.get("difficulty")
        self.submitted_answer = None
        self.is_correct = None
        self.terminated = False
        return True

    async def _reset_engine(
        self, problem_id: Optional[str] = None
    ) -> Tuple[SciCodePrivateState, SciCodePublicState]:
        pid = problem_id or next(iter(self._task_lookup))
        if not self._load_problem(pid):
            raise ValueError(f"Problem {pid} not found")

        pub = SciCodePublicState(
            problem_id=self.current_problem_id,
            prompt=self.current_prompt,
            category=self.current_category,
            difficulty=self.current_difficulty,
        )
        priv = SciCodePrivateState(solution=self.current_solution, terminated=False)
        return priv, pub

    async def _step_engine(
        self, submitted_answer: str
    ) -> Tuple[SciCodePrivateState, SciCodePublicState]:
        if self.current_solution is None or self.current_problem_id is None:
            raise RuntimeError("Engine not initialized")

        self.submitted_answer = submitted_answer
        self.is_correct = score_response(submitted_answer, self.current_solution)
        self.terminated = True

        pub = SciCodePublicState(
            problem_id=self.current_problem_id,
            prompt=self.current_prompt,
            category=self.current_category,
            difficulty=self.current_difficulty,
        )
        priv = SciCodePrivateState(
            solution=self.current_solution,
            submitted_answer=self.submitted_answer,
            is_correct=self.is_correct,
            terminated=True,
        )
        return priv, pub

    async def _serialize_engine(self) -> SciCodeEngineSnapshot:
        if not self.task_instance or not self.current_problem_id:
            raise RuntimeError("Engine not initialized")
        task_instance_dict = await self.task_instance.serialize()
        return SciCodeEngineSnapshot(
            task_instance_dict=task_instance_dict,
            current_problem_id=self.current_problem_id,
        )

    @classmethod
    async def _deserialize_engine(
        cls, snapshot: SciCodeEngineSnapshot
    ) -> "SciCodeEngine":
        from examples.scicode.schema import SciCodeTaskInstance

        task_instance = await SciCodeTaskInstance.deserialize(
            snapshot.task_instance_dict
        )
        engine = cls(task_instance)
        if not engine._load_problem(snapshot.current_problem_id):
            raise ValueError(f"Problem {snapshot.current_problem_id} not found")
        return engine
