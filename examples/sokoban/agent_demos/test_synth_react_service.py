import pytest
import asyncio
import json
import uuid

from httpx import AsyncClient

from examples.sokoban.agent_demos.test_synth_react import (
    ReActAgent,
    SIMPLE_SNAPSHOT,
)
from synth_sdk.tracing.abstractions import RewardSignal, Dataset, TrainingQuestion
from synth_sdk.tracing.utils import get_system_id
from synth_ai.zyk import LM

# Demo: drive Sokoban via FastAPI service endpoints

# HTTP-mode formatting for service-based observations
def format_obs_http(public: dict, private: dict, total_boxes: int) -> str:
    room_text = public.get("room_text") or public.get("room_text_final", "")
    return (
        f"{room_text}\n"
        f"Boxes on Target: {public.get('boxes_on_target', 0)} / {total_boxes}\n"
        f"Steps Taken: {public.get('steps_taken', 0)} / {public.get('max_steps', 0)}\n"
        f"Terminated: {private.get('terminated')}\n"
        f"Last Reward: {private.get('reward_last', 0)}"
    )

@pytest.mark.anyio
async def test_react_service_sokoban():
    # Launch the service with in-process AsyncClient
    async with AsyncClient(base_url="http://localhost:8000") as client:
        # 1) Health check
        health = await client.get("/env/health")
        assert health.status_code == 200
        supported = health.json()["supported_environments"]
        assert "Sokoban" in supported

        # 2) Create a Sokoban instance from a simple snapshot
        resp = await client.post(
            "/env/Sokoban/create",
            json={"initial_state": SIMPLE_SNAPSHOT},
        )
        assert resp.status_code == 200
        instance_id = resp.json()["instance_id"]

        # 3) Reset to get initial observation
        reset_resp = await client.post(f"/env/Sokoban/{instance_id}/reset")
        assert reset_resp.status_code == 200
        obs = reset_resp.json()
        private = obs["private"]
        public = obs["public"]

        # 4) Instantiate the LLM & ReAct agent
        llm = LM(model_name="gpt-4.1-nano", formatting_model_name="gpt-4.1-nano", temperature=0.0)
        agent = ReActAgent(llm)

        # Helper to track total boxes from the initial snapshot
        total_boxes = SIMPLE_SNAPSHOT.get("num_boxes", 0)

        # 5) Run episode loop via service step calls
        prompt = format_obs_http(public, private, total_boxes)
        for _ in range(agent.max_turns):
            action_idx = await agent.decide(prompt)
            # Agent signals termination
            if action_idx == -1:
                break

            # POST step with a single EnvToolCall JSON
            step_resp = await client.post(
                f"/env/Sokoban/{instance_id}/step",
                json=[
                    {"tool": "interact", "args": {"action": action_idx}}
                ],
            )
            assert step_resp.status_code == 200
            obs = step_resp.json()
            private = obs["private"]
            public = obs["public"]

            # Update prompt and check termination
            prompt = format_obs_http(public, private, total_boxes)
            if private.get("terminated"):
                break

        # 6) Final checkpoint (optional)
        ckpt = await client.get(f"/env/Sokoban/{instance_id}/checkpoint")
        assert ckpt.status_code == 200
        snapshot = ckpt.json().get("snapshot")

        # 7) Assertions: ensure solved state
        assert private.get("terminated") is True
        assert public.get("boxes_on_target") == total_boxes

        # 8) Optionally upload or record dataset
        dataset = Dataset(
            questions=[TrainingQuestion(id="sokoban_ep", intent="solve", criteria="solved")],
            reward_signals=[
                RewardSignal(
                    question_id="sokoban_ep",
                    run_id=agent.system_instance_id,
                    system_instance_id=agent.system_instance_id,
                    reward=1,
                    error_message="",
                    metadata={"agent_history": agent.history},
                )
            ],
        )
        # upload(dataset=dataset)  # Uncomment to send logs

# --- single-episode runner for service-based Sokoban ---
async def run_service_episode(client, agent, snapshot, total_boxes):
    # Create new instance
    resp = await client.post(
        "/env/Sokoban/create",
        json={"initial_state": snapshot},
    )
    instance_id = resp.json()["instance_id"]
    # Reset environment
    reset_resp = await client.post(f"/env/Sokoban/{instance_id}/reset")
    obs = reset_resp.json()
    private = obs["private"]
    public = obs["public"]
    # Initialize prompt
    prompt = format_obs_http(public, private, total_boxes)
    # Run one episode loop
    for _ in range(agent.max_turns):
        action_idx = await agent.decide(prompt)
        if action_idx == -1:
            break
        # Step via service
        step_resp = await client.post(
            f"/env/Sokoban/{instance_id}/step",
            json=[{"tool": "interact", "args": {"action": action_idx}}],
        )
        if step_resp.status_code != 200:
            print(f"ERROR in STEP: Status {step_resp.status_code}, Response: {step_resp.text}")
            # Decide how to handle error, e.g., raise or return False
            raise Exception(f"Step API call failed with status {step_resp.status_code}")
        obs = step_resp.json()
        private = obs["private"]
        public = obs["public"]
        prompt = format_obs_http(public, private, total_boxes)
        if private.get("terminated"):
            break
    # Optionally terminate (cleanup)
    await client.post(f"/env/Sokoban/{instance_id}/terminate")
    return bool(private.get("terminated"))

# --- batch evaluation helper for service-based Sokoban ---
async def eval_react_service_sokoban():
    from examples.sokoban.engine_helpers.room_utils import generate_room, get_shortest_action_path
    from tabulate import tabulate

    # Model & agent setup
    llm = LM(model_name="gpt-4.1-nano", formatting_model_name="gpt-4.1-nano", temperature=0.0)
    agent = ReActAgent(llm)
    total_boxes = SIMPLE_SNAPSHOT.get("num_boxes", 0)

    # Define difficulties and target solution lengths
    configs = [("ultra-easy", 1), ("easy", 3), ("medium", 5)]
    table_rows = []

    # Reuse a single AsyncClient for all episodes
    async with AsyncClient(base_url="http://localhost:8000") as client:
        for label, target_len in configs:
            # Generate snapshots matching target length
            snapshots = []
            seed = 0
            while len(snapshots) < 3:
                room_struct, room_state, _, _ = generate_room(
                    dim=(5, 5), initial_seed=seed, num_boxes=1, search_depth=max(10, target_len + 2)
                )
                path = get_shortest_action_path(room_struct, room_state, MAX_DEPTH=20)
                if len(path) == target_len:
                    # Convert numpy arrays to lists for JSON serialization
                    snapshots.append({
                        "dim_room": (5, 5),
                        "room_fixed": room_struct.tolist(),
                        "room_state": room_state.tolist(),
                        "boxes_on_target": 0,
                        "max_steps": 20,
                        "num_boxes": 1,
                    })
                seed += 1
            # Run episodes in parallel
            tasks = [run_service_episode(client, agent, snap, total_boxes) for snap in snapshots]
            solved = await asyncio.gather(*tasks)
            num_solved = sum(solved)
            rate = num_solved / len(solved)
            table_rows.append([label, f"{num_solved}/{len(solved)}", f"{rate:.0%}"])

    # Print aggregated results
    print(f"Model: {llm.model_name}, System: {agent.system_name}")
    print(tabulate(table_rows, headers=["Difficulty", "Solved", "Success Rate"], tablefmt="github"))

if __name__ == "__main__":
    import asyncio
    asyncio.run(eval_react_service_sokoban())
