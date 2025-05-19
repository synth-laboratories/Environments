# engine.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple, Optional, List
import os
from pydantic import BaseModel

from datasets import load_dataset
from examples.enron.art_helpers.types_enron import Email
from examples.enron.art_helpers.email_search_tools import search_emails, read_email, SearchResult
                                                # SQLite-backed helpers
from src.stateful.engine import StatefulEngine, StatefulEngineSnapshot
from examples.enron.taskset import EnronTaskInstance
from synth_ai.zyk import LM # Import LM class

# --------------------------------------------------------------------------- actions
ACTION_SEARCH, ACTION_READ, ACTION_ANSWER = range(3)


# --------------------------------------------------------------------------- snapshot
@dataclass
class EnronEngineSnapshot(StatefulEngineSnapshot):
    idx: int
    answered: bool
    total_reward: float
    partial_rewards: List[float]


# --------------------------------------------------------------------------- engine
class EnronEngine(StatefulEngine):
    """
    Minimal state-machine around the corbt/enron_emails_sample_questions dataset.
    Action is a tuple(kind, arg):

        (ACTION_SEARCH,  query: str)      → returns {"search_results": [message_ids]}
        (ACTION_READ,    message_id: str) → returns {"email_body": str}
        (ACTION_ANSWER,  answer: str)     → rewards +1 / -1 and terminates
    """

    # ----------------------------- init / helpers
    def __init__(self, task_instance: EnronTaskInstance):
        # Use the provided TaskInstance snapshot for this episode
        self.instance = task_instance
        self.answered = False
        self.total_reward = 0.0
        self.idx = 0
        # List to track each step's reward
        self.rewards_history: List[float] = []

    def _sample(self) -> Dict[str, Any]:
        # Return the snapshot dict from the TaskInstance
        return self.instance.initial_engine_snapshot

    # ----------------------------- step / reset
    async def _step_engine(
        self, action: Tuple[int, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        k, arg = action
        s = self._sample()
        # NOTE: anything we want the agent to see must be placed in `pub`
        obs: Dict[str, Any] = {}  # ← kept only for private logging, can delete
        r, term = 0.0, False

        # Initialize pub with details that are always present
        pub = {
            "question": s["question"],
            "tools": ["search_emails", "read_email", "answer_question"],
            "already_answered": self.answered,
            "query_date": s.get("query_date", "<unknown date>"),
            "inbox_address": s.get("inbox_address", "<unknown_inbox>"),
            "search_results": [], # Default to empty list
            "email": None # Default to None
        }

        if k == ACTION_SEARCH:
            # arg is a dict of kwargs for search_emails
            res: List[SearchResult] = search_emails(**arg)  # returns dataclass list
            pub["search_results"] = [asdict(item) for item in res] # surface the results to the agent

        elif k == ACTION_READ:
            email: Optional[Email] = read_email(arg)        # arg = message_id
            pub["email"] = email.dict() if email else None # surface the email to the agent

        elif k == ACTION_ANSWER:
            agent_ans = arg.strip()
            # Use LLM judge instead of simple substring matching
            correct = await determine_if_answer_is_correct(
                s["question"], s["answer"], agent_ans
            )
            r = 1.0 if correct else -1.0
            term, self.answered = True, True
        else:
            raise ValueError("unknown action")

        self.total_reward += r
        # Record this step's reward
        self.rewards_history.append(r)

        # Update pub with the potentially changed 'self.answered' status
        pub["already_answered"] = self.answered

        # keep the gold answer hidden from the agent → private state only
        priv = {
            "reward_last": r,
            "total_reward": self.total_reward,
            "terminated": term,
            "truncated": False,
            "gold_answer": s["answer"],
        }
        return priv, pub

    async def _reset_engine(
        self, *, seed: Optional[int] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Advance to the next Q-A pair and emit an initial observation **without**
        issuing an empty-keyword DB search (which would raise).
        """
        # Reset answered status and total reward for this instance
        self.answered = False
        self.total_reward = 0.0

        priv = {
            "reward_last": 0.0,
            "total_reward": 0.0,
            "terminated": False,
            "truncated": False,
            "gold_answer": self._sample()["answer"],
        }
        pub = {
            "question": self._sample()["question"],
            "tools": ["search_emails", "read_email", "answer_question"],
            "already_answered": False,
            "query_date": self._sample().get("query_date", "<unknown date>"),
            "inbox_address": self._sample().get("inbox_address", "<unknown_inbox>"),
        }
        # No index advancement needed when using a single TaskInstance
        return priv, pub

    # ----------------------------- serialization helpers
    async def _serialize_engine(self) -> EnronEngineSnapshot:
        # Include partial rewards history in the snapshot
        return EnronEngineSnapshot(
            self.idx,
            self.answered,
            self.total_reward,
            self.rewards_history,
        )

    @classmethod
    async def _deserialize_engine(cls, snap: EnronEngineSnapshot) -> "EnronEngine":
        eng = cls.__new__(cls)
        eng.data = load_dataset(
            "corbt/enron_emails_sample_questions",
            split="train",
            cache_dir=os.path.join(os.path.dirname(__file__), "dataset"),
        )
        eng.idx, eng.answered, eng.total_reward = snap.idx, snap.answered, snap.total_reward
        return eng

# ----------------------------- LLM Judge for answers
async def determine_if_answer_is_correct(question: str, gold_answer: str, agent_answer: str) -> bool:
    # Instantiate LM for the judge
    llm = LM(model_name="gpt-4.1-nano",formatting_model_name="gpt-4.1-nano", temperature=0.0)

    system_prompt = (
        "You will be given a question and two different answers to the question, "
        "the correct answer and the answer given by an AI. Your job is to determine "
        "if the answer given by the AI is correct."

    )
    user_message_content = f"Question: {question}\nCorrect answer: {gold_answer}\nAI answer: {agent_answer}"

    class CorrectnessResponse(BaseModel):
        correct: bool
    # Use LM.respond_async
    response = await llm.respond_async(
        system_message=system_prompt,
        user_message=user_message_content,
        response_model=CorrectnessResponse,
        # Caching is typically handled within the LM class or its underlying setup
    )
    return response.structured_output.correct

