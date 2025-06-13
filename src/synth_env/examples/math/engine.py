import json
import uuid
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from datasets import load_dataset

from synth_env.stateful.engine import StatefulEngine, StatefulEngineSnapshot
from synth_env.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from synth_env.tasks.core import TaskInstance as BaseTaskInstance


# --- Task Loading and Scoring (Moved from synth_env.environment.py) ---
def load_tasks() -> List[Dict[str, Any]]:
    """Load tasks from cache or HuggingFace dataset and cache them."""
    cache_path = Path(__file__).parent / "dataset" / "hendryks.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    configs = ["algebra", "number_theory", "counting_and_probability"]
    all_tasks: List[Dict[str, Any]] = []
    for cfg in configs:
        ds = load_dataset("EleutherAI/hendrycks_math", name=cfg, split="test")
        for ex in ds:
            all_tasks.append(
                {
                    "id": str(uuid.uuid4()),
                    "prompt": ex["problem"],
                    "solution": ex["solution"],
                    "tags": [cfg, ex.get("category")],
                    "difficulty": ex.get("level"),
                    "metadata": {
                        "subject": cfg,
                        "category": ex.get("category"),
                        "level": ex.get("level"),
                    },
                }
            )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(all_tasks, f)
    return all_tasks


def score_response(response: str, gold: str) -> bool:
    """Compare boxed answers or raw text for exact match."""
    response_match = re.search(r"\\boxed{((?:[^{}]|{[^{}]*})*)}", response)
    gold_match = re.search(r"\\boxed{((?:[^{}]|{[^{}]*})*)}", gold)
    if response_match and gold_match:
        return response_match.group(1).strip() == gold_match.group(1).strip()
    return response.strip() == gold.strip()


# --- Engine States and Snapshot ---
@dataclass
class HendryksPublicState:
    problem_id: str
    prompt: str
    tags: Optional[List[Optional[str]]]
    difficulty: Optional[str]
    # Potentially other metadata visible to agent


@dataclass
class HendryksPrivateState:
    solution: str  # For engine internal use
    submitted_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    terminated: bool = False  # True after one submission attempt
    # total_reward could be added if we quantify rewards


@dataclass
class HendryksEngineSnapshot(StatefulEngineSnapshot):
    task_instance_dict: Dict  # Serialized HendryksTaskInstance
    current_problem_id: str
    # Any other engine state to persist, e.g. attempts if tracked per problem


# --- Observation Callables ---
class SynthHendryksObservationCallable(GetObservationCallable):
    async def get_observation(
        self, pub: HendryksPublicState, priv: HendryksPrivateState
    ) -> InternalObservation:
        obs = {
            "problem_id": pub.problem_id,
            "prompt": pub.prompt,
            "tags": pub.tags,
            "difficulty": pub.difficulty,
            "terminated": priv.terminated,  # Key information for the agent
        }
        if priv.submitted_answer is not None:
            obs["submitted_answer"] = priv.submitted_answer
            obs["is_correct"] = priv.is_correct
        return obs


class SynthHendryksCheckpointObservationCallable(GetObservationCallable):
    async def get_observation(
        self, pub: HendryksPublicState, priv: HendryksPrivateState
    ) -> InternalObservation:
        # For a stateless env, checkpoint might be similar to regular observation after submission
        obs = {
            "problem_id": pub.problem_id,
            "prompt": pub.prompt,
            "is_correct": priv.is_correct,
            "submitted_answer": priv.submitted_answer,
        }
        return obs


# --- Hendryks Math Engine ---
class HendryksMathEngine(StatefulEngine):
    def __init__(self, task_instance: BaseTaskInstance):
        super().__init__()
        self.task_instance = (
            task_instance  # This specific instance the engine is tied to
        )
        self._all_tasks_data: List[Dict[str, Any]] = load_tasks()
        self._tasks_lookup: Dict[str, Dict[str, Any]] = {
            t["id"]: t for t in self._all_tasks_data
        }

        self.current_problem_id: Optional[str] = None
        self.current_prompt: Optional[str] = None
        self.current_solution: Optional[str] = None
        self.current_tags: Optional[List[Optional[str]]] = None
        self.current_difficulty: Optional[str] = None

        self.submitted_answer: Optional[str] = None
        self.is_correct: Optional[bool] = None
        self.terminated: bool = False

    def _load_problem(self, problem_id: str) -> bool:
        problem_data = self._tasks_lookup.get(problem_id)
        if not problem_data:
            return False
        self.current_problem_id = problem_data["id"]
        self.current_prompt = problem_data["prompt"]
        self.current_solution = problem_data["solution"]
        self.current_tags = problem_data["tags"]
        self.current_difficulty = problem_data["difficulty"]
        # Reset submission state for the new problem
        self.submitted_answer = None
        self.is_correct = None
        self.terminated = False
        return True

    async def _reset_engine(
        self, problem_id: Optional[str] = None
    ) -> Tuple[HendryksPrivateState, HendryksPublicState]:
        pid_to_load = problem_id
        if pid_to_load is None:
            # Default to task_instance's ID if it's a HendryksTaskInstance and has a valid one
            # Or, could default to the ID stored in initial_engine_snapshot if that pattern is adopted
            if hasattr(self.task_instance, "id") and self.task_instance.id:
                pid_to_load = str(self.task_instance.id)
            elif self._all_tasks_data:  # Fallback to the first loaded task
                pid_to_load = self._all_tasks_data[0]["id"]
            else:
                raise ValueError(
                    "No problem_id specified and no default task available."
                )

        if not self._load_problem(pid_to_load):
            raise ValueError(f"Failed to load problem with id: {pid_to_load}")

        pub_state = HendryksPublicState(
            problem_id=self.current_problem_id,
            prompt=self.current_prompt,
            tags=self.current_tags,
            difficulty=self.current_difficulty,
        )
        priv_state = HendryksPrivateState(
            solution=self.current_solution,  # Engine needs this for scoring
            terminated=self.terminated,
        )
        return priv_state, pub_state

    async def _step_engine(
        self, submitted_answer: str
    ) -> Tuple[HendryksPrivateState, HendryksPublicState]:
        if self.terminated:
            # Or raise an error, or allow re-submission with different handling
            # For now, assume one submission per reset/problem instance
            pass

        if self.current_solution is None or self.current_problem_id is None:
            raise RuntimeError(
                "Engine not initialized with a problem. Call _reset_engine first."
            )

        self.submitted_answer = submitted_answer
        self.is_correct = score_response(submitted_answer, self.current_solution)
        self.terminated = True  # Episode terminates after one submission

        pub_state = HendryksPublicState(
            problem_id=self.current_problem_id,
            prompt=self.current_prompt,
            tags=self.current_tags,
            difficulty=self.current_difficulty,
        )
        priv_state = HendryksPrivateState(
            solution=self.current_solution,
            submitted_answer=self.submitted_answer,
            is_correct=self.is_correct,
            terminated=self.terminated,
        )
        return priv_state, pub_state

    async def _serialize_engine(self) -> HendryksEngineSnapshot:
        if not self.task_instance or not self.current_problem_id:
            raise RuntimeError(
                "Cannot serialize engine without a task instance and current problem."
            )
        task_instance_dict = await self.task_instance.serialize()  # type: ignore
        return HendryksEngineSnapshot(
            task_instance_dict=task_instance_dict,
            current_problem_id=self.current_problem_id,
            # Add other relevant engine state if needed, e.g., submitted_answer, is_correct for full resumption
        )

    @classmethod
    async def _deserialize_engine(
        cls, snapshot: HendryksEngineSnapshot
    ) -> "HendryksMathEngine":
        # Lazy import to avoid circular dependency
        from examples.math.schema import HendryksTaskInstance

        task_instance = await HendryksTaskInstance.deserialize(
            snapshot.task_instance_dict
        )
        engine = cls(task_instance)  # Re-initializes _all_tasks_data and _tasks_lookup

        # Restore the specific problem state from the snapshot
        if not engine._load_problem(snapshot.current_problem_id):
            # This case should ideally not happen if snapshot was valid
            raise ValueError(
                f"Failed to load problem {snapshot.current_problem_id} from snapshot during deserialization"
            )

        # If snapshot also stored submission state, restore it here:
        # engine.submitted_answer = snapshot.get('submitted_answer')
        # engine.is_correct = snapshot.get('is_correct')
        # engine.terminated = snapshot.get('terminated', True) # Assuming if problem_id is set, it was at least attempted or reset to
        return engine
