from typing import Optional, List
from synth_env.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from synth_env.reproducibility.core import ReproducibleEnvironment
from synth_env.stateful.core import StatefulEnvironment
from synth_env.environment.tools import EnvToolCall
from synth_env.examples.math.tools import (
    SubmitAnswerTool,
)  # Specific tool for math submissions

from synth_env.examples.math.engine import (
    HendryksMathEngine,
    HendryksPublicState,
    HendryksPrivateState,
    HendryksEngineSnapshot,
    SynthHendryksObservationCallable,
    SynthHendryksCheckpointObservationCallable,
)
from synth_env.examples.math.schema import (
    HendryksTaskInstance,
)  # Import from schema to avoid cycle
# score_response and load_tasks are now in engine.py


class HendryksMathEnv(StatefulEnvironment, ReproducibleEnvironment[HendryksMathEngine]):
    def __init__(
        self,
        task_instance: HendryksTaskInstance,  # Explicitly use HendryksTaskInstance
        custom_step_obs: Optional[GetObservationCallable] = None,
        custom_ckpt_obs: Optional[GetObservationCallable] = None,
    ):
        self.name = "HendryksMath"
        self.task_instance = task_instance
        self.custom_step_observation_callable = custom_step_obs
        self.custom_checkpoint_observation_callable = custom_ckpt_obs
        # The actual engine is now HendryksMathEngine
        self.engine: HendryksMathEngine = HendryksMathEngine(task_instance)

    async def initialize(self) -> InternalObservation:
        """Initializes or resets the engine to the configured task_instance."""
        priv, pub = await self.engine._reset_engine()
        return await self._to_observation(
            priv,
            pub,
            self.custom_step_observation_callable or SynthHendryksObservationCallable(),
        )

    async def terminate(self) -> InternalObservation:
        """Terminates the environment (stub for stateless math)."""
        return {"terminated": True, "message": "Environment session terminated."}

    def validate_tool_calls(self, tool_calls: List[List[EnvToolCall]]) -> None:
        # Expect exactly the SubmitAnswerTool for math submissions
        if (
            not tool_calls
            or not tool_calls[0]
            or not isinstance(tool_calls[0][0], SubmitAnswerTool)
        ):
            raise ValueError(
                "Expected SubmitAnswerTool wrapped in a non-empty nested list for math environment step."
            )

    async def step(self, tool_calls: List[List[EnvToolCall]]) -> InternalObservation:
        self.validate_tool_calls(tool_calls)
        # Extract answer from SubmitAnswerTool
        submitted_answer: str = tool_calls[0][0].answer  # type: ignore[attr-defined]

        priv, pub = await self.engine._step_engine(submitted_answer)
        return await self._to_observation(
            priv,
            pub,
            self.custom_step_observation_callable or SynthHendryksObservationCallable(),
        )

    async def checkpoint(self) -> InternalObservation:
        """Returns the current state of the problem, including solution if submitted."""
        if not self.engine.current_problem_id or self.engine.is_correct is None:
            # Or handle this by returning a specific "not initialized" observation
            raise RuntimeError("Engine not initialized or no current problem loaded.")

        pub_state = HendryksPublicState(
            problem_id=self.engine.current_problem_id,
            prompt=self.engine.current_prompt or "N/A",
            tags=self.engine.current_tags,
            difficulty=self.engine.current_difficulty,
        )
        priv_state = HendryksPrivateState(
            solution="REDACTED",  # Do not expose gold solution in private state
            submitted_answer=self.engine.submitted_answer,
            is_correct=self.engine.is_correct,
            terminated=self.engine.terminated,
        )

        # Use SynthHendryksCheckpointObservationCallable by default for checkpoint
        obs_cb = (
            self.custom_checkpoint_observation_callable
            or SynthHendryksCheckpointObservationCallable()
        )
        return await obs_cb.get_observation(pub_state, priv_state)

    async def _to_observation(
        self,
        priv: HendryksPrivateState,
        pub: HendryksPublicState,
        obs_cb: GetObservationCallable,  # Made non-optional, pass explicitly
    ) -> InternalObservation:
        return await obs_cb.get_observation(pub, priv)

    async def _serialize_engine(self) -> HendryksEngineSnapshot:  # Return type updated
        return await self.engine._serialize_engine()

    @classmethod
    async def _deserialize_engine(
        cls, snapshot: HendryksEngineSnapshot
    ) -> "HendryksMathEnv":  # Param type updated
        # Deserialize the engine first
        engine = await HendryksMathEngine._deserialize_engine(snapshot)
        # The engine's task_instance is already deserialized, use it
        env = cls(task_instance=engine.task_instance)
        env.engine = engine  # Assign the deserialized engine to the new env instance
        return env


# Removed old HendryksMathEnv class structure, load_tasks, score_response, _task_lookup
# as they are now part of or managed by HendryksMathEngine
