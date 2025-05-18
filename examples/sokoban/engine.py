from typing import Optional, Dict, Any, Tuple

from src.environment.shared_engine import GetObservationCallable, InternalObservation
from src.stateful.engine import StatefulEngine, StatefulEngineSnapshot
from src.tasks.core import TaskInstance
import numpy as np
from dataclasses import dataclass
from examples.sokoban.taskset import SokobanTaskInstance # Assuming this is where SokobanTaskInstance is defined
from src.reproducibility.core import IReproducibleEngine # Added import


@dataclass
class SokobanEngineSnapshot(StatefulEngineSnapshot):
    task_instance_dict: Dict
    engine_snapshot: Dict

@dataclass
class SokobanPublicState:
    dim_room: Tuple[int, int]
    room_fixed: np.ndarray #numpy kinda sucks
    room_state: np.ndarray
    player_position: Tuple[int, int]
    boxes_on_target: int
    num_steps: int
    max_steps: int
    last_action_name: str

    def diff(self, prev_state: "SokobanPublicState") -> Dict:
        pass

    @property
    def room_text(self) -> str:
        """ASCII visualization of the room state"""
        return _grid_to_text(self.room_state)

@dataclass
class SokobanPrivateState:
    reward_last: float
    total_reward: float
    terminated: bool
    truncated: bool
    rng_state: dict | None = None

    def diff(self, prev_state: "SokobanPrivateState") -> Dict:
        pass

# Note - just how we roll! Show your agent whatever state you want
# Close to original
def _grid_to_text(grid: np.ndarray) -> str:
    """Pretty 3-char glyphs for each cell – same lookup the legacy renderer used."""
    return "\n".join(
        "".join(GRID_LOOKUP.get(int(cell), "?") for cell in row)  # type: ignore[arg-type]
        for row in grid
    )

class SynthSokobanObservationCallable(GetObservationCallable):
    
    def __init__(self):
        pass
    async def get_observation(self, pub: SokobanPublicState, priv: SokobanPrivateState) -> InternalObservation:  # type: ignore[override]
        board_txt = _grid_to_text(pub.room_state)
        return {
            "room_text": board_txt,
            "player_position": pub.player_position,
            "boxes_on_target": pub.boxes_on_target,
            "steps_taken": pub.num_steps,
            "max_steps": pub.max_steps,
            "last_action": pub.last_action_name,
            #
            "reward_last": priv.reward_last,
            "total_reward": priv.total_reward,
            "terminated": priv.terminated,
            "truncated": priv.truncated,
        }

# Close to original
class SynthSokobanCheckpointObservationCallable(GetObservationCallable):
    """
    Snapshot emitted once after the episode finishes.
    Mirrors the legacy 'final_observation' concept: full board + final tallies.
    """

    def __init__(self):
        pass

    async def get_observation(self, pub: SokobanPublicState, priv: SokobanPrivateState) -> InternalObservation:  # type: ignore[override]
        board_txt = _grid_to_text(pub.room_state)
        return {
            "room_text_final": board_txt,
            "boxes_on_target_final": pub.boxes_on_target,
            "steps_taken_final": pub.num_steps,
            #
            "total_reward": priv.total_reward,
            "terminated": priv.terminated,
            "truncated": priv.truncated,
        }

# Think of engine as the actual logic, then with hooks to update the public and private state
# Note - I don't really want to split up the transformation/engine logic from the instance information. There's already quite a bit of abstraction, so let's make the hard call here. I observe that this class does combine the responsibility of tracking engine state AND containing dynamics, but I think it's fine.


from gym_sokoban.envs.sokoban_env import ACTION_LOOKUP

GRID_LOOKUP = {0: " # ", 1: " _ ", 2: " O ", 3: " √ ", 4: " X ", 5: " P ", 6: " S "}

import numpy as np
from typing import Dict, Any
from gym_sokoban.envs.sokoban_env import (
    SokobanEnv as PackageSokobanEnv,
) # adjust import path as needed


def package_sokoban_env_from_engine_snapshot(
    engine_snapshot: Dict[str, Any],
) -> PackageSokobanEnv:
    """Instantiate SokobanEnv and load every field from a saved-state dict."""
    # 1. create empty env (skip reset)
    env = PackageSokobanEnv(
        dim_room=tuple(engine_snapshot["dim_room"]),
        max_steps=engine_snapshot.get("max_steps", 120),
        num_boxes=engine_snapshot.get("num_boxes", 1),
        reset=False,
    )

    # 2. restore core grids
    env.room_fixed = np.asarray(engine_snapshot["room_fixed"], dtype=int)
    env.room_state = np.asarray(engine_snapshot["room_state"], dtype=int)

    # 3. restore auxiliary data
    raw_map = engine_snapshot.get("box_mapping", {})
    if isinstance(raw_map, list):  # list-of-dict form
        env.box_mapping = {tuple(e["original"]): tuple(e["current"]) for e in raw_map}
    else:  # string-keyed dict form
        env.box_mapping = {
            tuple(map(int, k.strip("[]").split(","))): tuple(v)
            for k, v in raw_map.items()
        }

    env.player_position = np.argwhere(env.room_state == 5)[0]
    env.num_env_steps = engine_snapshot.get("num_env_steps", 0)
    env.boxes_on_target = engine_snapshot.get(
        "boxes_on_target", int(np.sum(env.room_state == 3))
    )
    env.reward_last = engine_snapshot.get("reward_last", 0)

    # 4. restore RNG (if provided)
    rng = engine_snapshot.get("np_random_state")
    if rng:
        env.seed()  # init env.np_random
        env.np_random.set_state(
            (
                rng["key"],
                np.array(rng["state"], dtype=np.uint32),
                rng["pos"],
                0,  # has_gauss
                0.0,  # cached_gaussian
            )
        )

    return env

class SokobanEngine(StatefulEngine, IReproducibleEngine):

    task_instance: TaskInstance
    package_sokoban_env: PackageSokobanEnv

    # sokoban stuff

    def __init__(self, task_instance: TaskInstance):
        self.task_instance = task_instance
        pass

    # gives the observation!
    # also final rewards when those are passed in
    async def _render(
        self,
        private_state: SokobanPrivateState,
        public_state: SokobanPublicState,
        get_observation: Optional[GetObservationCallable] = None,
    ) -> str:
        """
        1. choose the observation callable (default = SynthSokobanObservationCallable)
        2. fetch obs via callable(pub, priv)
        3. if callable returned a dict -> pretty-print board + footer
           if str -> forward unchanged
        """
        # 1 – pick callable
        obs_cb = get_observation or SynthSokobanObservationCallable()

        # 2 – pull observation
        obs = await obs_cb.get_observation(public_state, private_state)

        # 3 – stringify
        if isinstance(obs, str):
            return obs

        if isinstance(obs, dict):
            board_txt = (
                obs.get("room_text")
                or obs.get("room_text_final")
                or _grid_to_text(public_state.room_state)
            )
            footer = (
                f"steps: {public_state.num_steps}/{public_state.max_steps} | "
                f"boxes✓: {public_state.boxes_on_target} | "
                f"last_r: {private_state.reward_last:.2f} | "
                f"total_r: {private_state.total_reward:.2f}"
            )
            return f"{board_txt}\n{footer}"

        # unknown payload type -> fallback
        return str(obs)

    # yields private state, public state
    async def _step_engine(self, action: int) -> Tuple[SokobanPrivateState, SokobanPublicState]:
        if action not in self.package_sokoban_env.action_space:
            raise ValueError(f"Illegal action {action}")
        obs, r, terminated, info = self.package_sokoban_env.step(
            action
        )  # tiny_rgb_array default
        priv = SokobanPrivateState(
            reward_last=r,
            total_reward=self._total_reward,
            terminated=terminated,
            truncated=info.get("maxsteps_used", False),
            rng_state=self.package_sokoban_env.np_random.bit_generator.state,
        )
        pub = SokobanPublicState(
            dim_room=self.package_sokoban_env.dim_room,
            room_fixed=self.package_sokoban_env.room_fixed.copy(),
            room_state=self.package_sokoban_env.room_state.copy(),
            player_position=tuple(self.package_sokoban_env.player_position),
            boxes_on_target=self.package_sokoban_env.boxes_on_target,
            num_steps=self.package_sokoban_env.num_env_steps,
            max_steps=self.package_sokoban_env.max_steps,
            last_action_name=info["action.name"],
        )

        return priv, pub


    async def _reset_engine(
        self, *, seed: int | None = None
    ) -> Tuple[SokobanPrivateState, SokobanPublicState]:
        """
        (Re)build the wrapped PackageSokobanEnv in a fresh state.

        1.  Decide whether we have an initial snapshot in the TaskInstance.
        2.  If yes → hydrate env from it; otherwise call env.reset(seed).
        3.  Zero-out cumulative reward and emit fresh state objects.
        """
        self._total_reward = 0.0

        init_snap: dict | None = getattr(self.task_instance, "initial_engine_snapshot", None)

        if init_snap:
            # deterministic replay from provided snapshot
            self.package_sokoban_env = package_sokoban_env_from_engine_snapshot(
                init_snap
            )
        else:
            # brand-new level — pull dimensions / boxes from task config if present
            cfg = getattr(self.task_instance, "config", {})
            self.package_sokoban_env = PackageSokobanEnv(
                dim_room=tuple(cfg.get("dim_room", (5, 5))),
                max_steps=cfg.get("max_steps", 120),
                num_boxes=cfg.get("num_boxes", 1),
            )
            _ = self.package_sokoban_env.reset()

        # build first public/private views
        priv = SokobanPrivateState(
            reward_last=0.0,
            total_reward=0.0,
            terminated=False,
            truncated=False,
            rng_state=self.package_sokoban_env.np_random.bit_generator.state,
        )
        pub = SokobanPublicState(
            dim_room=self.package_sokoban_env.dim_room,
            room_fixed=self.package_sokoban_env.room_fixed.copy(),
            room_state=self.package_sokoban_env.room_state.copy(),
            player_position=tuple(self.package_sokoban_env.player_position),
            boxes_on_target=self.package_sokoban_env.boxes_on_target,
            num_steps=self.package_sokoban_env.num_env_steps,
            max_steps=self.package_sokoban_env.max_steps,
            last_action_name="reset",
        )
        return priv, pub
    
    async def _serialize_engine(self) -> SokobanEngineSnapshot:
        """Dump wrapped env + task_instance into a JSON-ready snapshot."""
        env = self.package_sokoban_env

        # helper – numpy RNG → dict
        def _rng_state(e):
            state = e.np_random.bit_generator.state
            state['state'] = state['state'].tolist()
            return state

        snap: Dict[str, Any] = {
            "dim_room": list(env.dim_room),
            "max_steps": env.max_steps,
            "num_boxes": env.num_boxes,
            "room_fixed": env.room_fixed.tolist(),
            "room_state": env.room_state.tolist(),
            "box_mapping": [
                {"original": list(k), "current": list(v)}
                for k, v in env.box_mapping.items()
            ],
            "player_position": env.player_position.tolist(),
            "num_env_steps": env.num_env_steps,
            "boxes_on_target": env.boxes_on_target,
            "reward_last": env.reward_last,
            "total_reward": getattr(self, "_total_reward", 0.0),
            # "np_random_state": _rng_state(env), # Assuming _rng_state is defined if needed
        }

        # Serialize the TaskInstance using its own serialize method
        task_instance_dict = await self.task_instance.serialize()

        return SokobanEngineSnapshot(
            task_instance_dict=task_instance_dict, # Store serialized TaskInstance
            engine_snapshot=snap,
        )

    @classmethod
    async def _deserialize_engine(
        cls, sokoban_engine_snapshot: "SokobanEngineSnapshot"
    ) -> "SokobanEngine":
        """
        Recreate a SokobanEngine (including wrapped env and TaskInstance) from a snapshot blob.
        """
        # --- 1. rebuild TaskInstance ----------------------------------- # 
        # Use the concrete SokobanTaskInstance.deserialize method
        instance = await SokobanTaskInstance.deserialize(sokoban_engine_snapshot.task_instance_dict)
        
        # --- 2. create engine shell ------------------------------------ #
        engine = cls.__new__(cls)  # bypass __init__
        StatefulEngine.__init__(engine)  # initialise mix-in parts
        engine.task_instance = instance  # assign restored TaskInstance

        # --- 3. hydrate env & counters --------------------------------- #
        engine.package_sokoban_env = package_sokoban_env_from_engine_snapshot(
            sokoban_engine_snapshot.engine_snapshot
        )
        engine._total_reward = sokoban_engine_snapshot.engine_snapshot.get(
            "total_reward", 0.0
        )
        return engine


if __name__ == "__main__":
    # // 0=wall, 1=floor, 2=target
    # // 4=box-not-on-target, 5=player
    # initial_room = {
    #     "dim_room": [5, 5],
    #     "max_steps": 120,
    #     "num_boxes": 1,
    #     "seed": 42,                         
    #     "room_fixed": [
    #         [0, 0, 0, 0, 0],
    #         [0, 1, 1, 2, 0],
    #         [0, 1, 0, 1, 0],
    #         [0, 1, 5, 1, 0],
    #         [0, 0, 0, 0, 0]
    #     ],                                   
    #     "room_state": [
    #         [0, 0, 0, 0, 0],
    #         [0, 1, 4, 1, 0],
    #         [0, 1, 0, 1, 0],
    #         [0, 1, 5, 1, 0],
    #         [0, 0, 0, 0, 0]
    #     ]                                    
    # }
    task_instance_dict = {
        "initial_engine_snapshot": {
            "dim_room": [5, 5],
            "max_steps": 120,
            "num_boxes": 1,
            "room_fixed": [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 2, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            "room_state": [
                [0, 0, 0, 0, 0],
                [0, 1, 4, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 5, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            "box_mapping": [{"original": [1, 2], "current": [3, 2]}],
            "boxes_on_target": 0,
            "np_random_state": {
                "key": "MT19937",
                "state": [1804289383, 846930886, 1681692777, 1714636915],
                "pos": 0,
            },
            "reward_last": 0,
            "num_env_steps": 0
        }
    }
    import random
    import asyncio
    async def sanity():
        task_instance = TaskInstance()
        engine = SokobanEngine(task_instance=task_instance)
        priv, pub = await engine._reset_engine()
        print(await engine._render(priv, pub))  # initial board

        for _ in range(10):  # play 10 random moves
            a = random.randint(0, 8)  # action range 0-8
            priv, pub = await engine._step_engine(a)
            print(f"\n### step {pub.num_steps} — {ACTION_LOOKUP[a]} ###")
            print("public:", pub)
            print("private:", priv)
            print(await engine._render(priv, pub))
            if priv.terminated or priv.truncated:
                break
    asyncio.run(sanity())
    # sokoban_engine = SokobanEngine.deserialize(
    #     engine_snapshot=SokobanEngineSnapshot(
    #         instance=instance_information,
    #         snapshot_dict=instance_information["initial_engine_snapshot"],
    #     )
    # )


# {
#   "dim_room": [5, 5],
#   "max_steps": 120,
#   "num_boxes": 1,

#   "room_fixed": [...],                // as above
#   "room_state": [...],                // current grid (3 = box-on-target)

#   "box_mapping": {
#     "[1,3]": [3,2]                    // origin-target → current-pos pairs
#   },
#   "player_position": [3, 2],          // row, col

#   "num_env_steps": 15,                // steps already taken
#   "boxes_on_target": 0,               // live counter

#   "np_random_state": {                // optional but makes replay bit-exact
#     "key": "MT19937",
#     "state": [1804289383, 846930886, ...],
#     "pos": 123
#   }
# }