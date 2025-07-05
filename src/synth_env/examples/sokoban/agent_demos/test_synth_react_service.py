import pytest
import asyncio
import json

from httpx import AsyncClient

from synth_env.examples.sokoban.agent_demos.test_synth_react_locally import (
    ReActAgent,
    SIMPLE_SNAPSHOT,
)
from synth_sdk.tracing.abstractions import RewardSignal, Dataset, TrainingQuestion
from synth_ai.zyk import LM

# Demo: drive Sokoban via FastAPI service endpoints


# === UNIT TESTS FOR SERVICE INTEGRATION ===

def create_valid_snapshot():
    """Create a valid Sokoban snapshot using proper room generation"""
    from synth_env.examples.sokoban.engine_helpers.room_utils import generate_room
    
    room_fixed, room_state, _, _ = generate_room(
        dim=(5, 5),
        initial_seed=42,
        num_boxes=1,
        search_depth=15
    )
    
    return {
        "dim_room": [5, 5],
        "room_fixed": room_fixed.tolist(),
        "room_state": room_state.tolist(),
        "boxes_on_target": 0,
        "max_steps": 50,
        "num_boxes": 1
    }


@pytest.mark.anyio
async def test_sokoban_service_new_api():
    """Test the new /env/Sokoban/initialize API - verifies the fix works"""
    async with AsyncClient(base_url="http://localhost:8000", timeout=30.0) as client:
        # 1) Health check
        health = await client.get("/health")
        assert health.status_code == 200
        supported = health.json()["supported_environments"]
        assert "Sokoban" in supported
        print("✅ Health check passed")

        # 2) Initialize a Sokoban instance using the new API with generated room
        snapshot = create_valid_snapshot()
        resp = await client.post(
            "/env/Sokoban/initialize",
            json={"initial_state": snapshot, "config": {}}
        )
        print(f"Initialize response status: {resp.status_code}")
        print(f"Initialize response: {resp.text}")
        assert resp.status_code == 200
        
        init_data = resp.json()
        instance_id = init_data["env_id"]
        initial_obs = init_data["observation"]
        print(f"✅ Created instance: {instance_id}")
        
        # 3) Critical test: Verify observation is NOT empty (this was the bug!)
        assert len(initial_obs) > 0, "Initial observation should not be empty!"
        assert "room_text" in initial_obs, "Should have room_text in observation"
        assert "player_position" in initial_obs, "Should have player_position in observation"
        print(f"✅ Observation has {len(initial_obs)} keys: {list(initial_obs.keys())}")
        
        # 4) Take a few steps to verify step observations are also not empty
        for step_num in range(3):
            # Try moving right (action 8)
            step_resp = await client.post(
                "/env/Sokoban/step",
                json={
                    "env_id": instance_id,
                    "request_id": f"test_step_{step_num}",
                    "action": {
                        "tool_calls": [{"tool": "interact", "args": {"action": 8}}]
                    }
                }
            )
            assert step_resp.status_code == 200
            
            step_data = step_resp.json()
            obs_data = step_data.get("observation", {})
            
            # Critical test: Step observations should NOT be empty
            assert len(obs_data) > 0, f"Step {step_num+1} observation should not be empty!"
            assert "room_text" in obs_data, f"Step {step_num+1} should have room_text"
            
            print(f"✅ Step {step_num+1}: Observation has {len(obs_data)} keys")
            print(f"   Boxes on target: {obs_data.get('boxes_on_target', 0)}")
            print(f"   Steps taken: {obs_data.get('steps_taken', 0)}")
            
            if obs_data.get("terminated"):
                print(f"Game terminated at step {step_num+1}")
                break

        # 5) Terminate the instance
        term_resp = await client.post(
            "/env/Sokoban/terminate",
            json={"env_id": instance_id}
        )
        assert term_resp.status_code == 200
        print("✅ Instance terminated successfully")


@pytest.mark.anyio
async def test_sokoban_service_legacy_api():
    """Test the legacy /Sokoban/create API - verifies both APIs work"""
    async with AsyncClient(base_url="http://localhost:8000", timeout=30.0) as client:
        # 1) Create using legacy API with generated room
        snapshot = create_valid_snapshot()
        resp = await client.post(
            "/Sokoban/create",
            json={"initial_state": snapshot}
        )
        print(f"Legacy create response status: {resp.status_code}")
        
        if resp.status_code == 200:
            create_data = resp.json()
            instance_id = create_data["instance_id"]
            print(f"✅ Created legacy instance: {instance_id}")
            
            # 2) Reset to get initial observation
            reset_resp = await client.post(f"/Sokoban/{instance_id}/reset")
            assert reset_resp.status_code == 200
            
            obs = reset_resp.json()
            private = obs["private"]
            public = obs["public"]
            
            # Verify observations are not empty
            assert len(private) > 0, "Private observation should not be empty!"
            assert len(public) > 0, "Public observation should not be empty!"
            assert "room_text" in public, "Should have room_text in public observation"
            
            print(f"✅ Legacy API: Public obs has {len(public)} keys, Private obs has {len(private)} keys")
            print(f"   Initial state: Boxes on target: {public.get('boxes_on_target', 0)}")
            
            # 3) Take a step using legacy API
            step_resp = await client.post(
                f"/Sokoban/{instance_id}/step",
                json=[{"tool": "interact", "args": {"action": 8}}]
            )
            assert step_resp.status_code == 200
            
            step_obs = step_resp.json()
            step_private = step_obs["private"]
            step_public = step_obs["public"]
            
            # Critical test: Step observations should NOT be empty
            assert len(step_private) > 0, "Step private observation should not be empty!"
            assert len(step_public) > 0, "Step public observation should not be empty!"
            print(f"✅ Legacy step: Public obs has {len(step_public)} keys, Private obs has {len(step_private)} keys")
            
            # 4) Terminate using legacy API
            term_resp = await client.post(f"/Sokoban/{instance_id}/terminate")
            assert term_resp.status_code == 200
            print("✅ Legacy instance terminated successfully")


@pytest.mark.anyio
async def test_generated_room():
    """Test with a generated room using proper room generation"""
    from synth_env.examples.sokoban.engine_helpers.room_utils import generate_room
    
    # Generate a simple room
    room_fixed, room_state, _, _ = generate_room(
        dim=(6, 6),
        initial_seed=123,
        num_boxes=2,
        search_depth=20
    )
    
    snapshot = {
        "dim_room": [6, 6],
        "room_fixed": room_fixed.tolist(),
        "room_state": room_state.tolist(),
        "boxes_on_target": 0,
        "max_steps": 100,
        "num_boxes": 2
    }
    
    async with AsyncClient(base_url="http://localhost:8000", timeout=30.0) as client:
        # Initialize with generated room
        resp = await client.post(
            "/env/Sokoban/initialize",
            json={"initial_state": snapshot, "config": {}}
        )
        print(f"Generated room initialize status: {resp.status_code}")
        
        if resp.status_code == 200:
            init_data = resp.json()
            instance_id = init_data["env_id"]
            obs = init_data["observation"]
            
            # Critical test: Verify observation is not empty
            assert len(obs) > 0, "Generated room observation should not be empty!"
            assert "room_text" in obs, "Should have room_text in observation"
            assert obs["num_boxes"] == 2, f"Should have 2 boxes, got {obs.get('num_boxes', 0)}"
            
            print(f"✅ Created instance with generated room: {instance_id}")
            print(f"✅ Observation has {len(obs)} keys including: {list(obs.keys())}")
            
            # Show the room
            if "room_text" in obs:
                print(f"🎮 Generated Room ({obs['num_boxes']} boxes):")
                print(obs["room_text"])
            
            # Terminate
            term_resp = await client.post(
                "/env/Sokoban/terminate",
                json={"env_id": instance_id}
            )
            assert term_resp.status_code == 200
            print("✅ Generated room test completed successfully")


@pytest.mark.anyio
async def test_observation_content_validation():
    """Test that observations contain expected content structure"""
    async with AsyncClient(base_url="http://localhost:8000", timeout=30.0) as client:
        snapshot = create_valid_snapshot()
        
        # Initialize
        resp = await client.post(
            "/env/Sokoban/initialize",
            json={"initial_state": snapshot, "config": {}}
        )
        assert resp.status_code == 200
        
        init_data = resp.json()
        instance_id = init_data["env_id"]
        obs = init_data["observation"]
        
        # Test specific observation structure
        required_keys = ["room_text", "player_position", "boxes_on_target", "steps_taken", "num_boxes"]
        for key in required_keys:
            assert key in obs, f"Missing required key: {key}"
        
        # Test data types
        assert isinstance(obs["player_position"], list), "player_position should be a list"
        assert isinstance(obs["boxes_on_target"], int), "boxes_on_target should be an int"
        assert isinstance(obs["steps_taken"], int), "steps_taken should be an int"
        assert isinstance(obs["room_text"], str), "room_text should be a string"
        assert len(obs["room_text"]) > 0, "room_text should not be empty"
        
        print(f"✅ Observation structure validation passed")
        print(f"   Player position: {obs['player_position']}")
        print(f"   Boxes on target: {obs['boxes_on_target']}/{obs['num_boxes']}")
        print(f"   Steps taken: {obs['steps_taken']}")
        print(f"   Room text length: {len(obs['room_text'])} characters")
        
        # Take a step and verify the observation still has proper structure
        step_resp = await client.post(
            "/env/Sokoban/step",
            json={
                "env_id": instance_id,
                "request_id": "validation_step",
                "action": {
                    "tool_calls": [{"tool": "interact", "args": {"action": 8}}]
                }
            }
        )
        assert step_resp.status_code == 200
        
        step_data = step_resp.json()
        step_obs = step_data.get("observation", {})
        
        # Verify step observation structure
        for key in required_keys:
            assert key in step_obs, f"Missing required key in step observation: {key}"
        
        # Verify steps_taken increased
        assert step_obs["steps_taken"] > obs["steps_taken"], "steps_taken should increase after step"
        
        print(f"✅ Step observation structure validation passed")
        print(f"   Steps taken after step: {step_obs['steps_taken']}")
        
        # Cleanup
        await client.post("/env/Sokoban/terminate", json={"env_id": instance_id})


# === END UNIT TESTS ===

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
        llm = LM(model_name="gpt-4.1", formatting_model_name="gpt-4.1", temperature=0.0)
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
                json=[{"tool": "interact", "args": {"action": action_idx}}],
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
            questions=[
                TrainingQuestion(id="sokoban_ep", intent="solve", criteria="solved")
            ],
            reward_signals=[
                RewardSignal(
                    question_id="sokoban_ep",
                    system_instance_id=agent.system_instance_id,
                    reward=1,
                    annotation=json.dumps({"agent_history": agent.history}),
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
        decision_record = await agent.decide(prompt)
        action_idx = decision_record.action_int
        if action_idx == -1:
            break
        # Step via service
        step_resp = await client.post(
            f"/env/Sokoban/{instance_id}/step",
            json=[{"tool": "interact", "args": {"action": action_idx}}],
        )
        if step_resp.status_code != 200:
            print(
                f"ERROR in STEP: Status {step_resp.status_code}, Response: {step_resp.text}"
            )
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
async def eval_react_service_sokoban(
    model_name: str = "gpt-4.1-nano",
    formatting_model_name: str = "gpt-4.1-nano",
    modes: list[str] = ["ultra-easy", "easy", "medium"],
):
    from examples.sokoban.engine_helpers.room_utils import (
        generate_room,
        get_shortest_action_path,
    )
    from tabulate import tabulate

    llm = LM(
        model_name=model_name,
        formatting_model_name=formatting_model_name,
        temperature=0.0,
    )
    agent = ReActAgent(llm)
    total_boxes = 1

    difficulty_to_length_map = {
        "ultra-easy": 1,
        "easy": 3,
        "medium": 5,
        "hard": 7,
        "ultra-hard": 10,
    }

    configs_for_modes = []
    for mode_label in modes:
        if mode_label in difficulty_to_length_map:
            configs_for_modes.append((mode_label, difficulty_to_length_map[mode_label]))
        else:
            print(
                f"Warning: Mode '{mode_label}' not found in difficulty_to_length_map. Skipping."
            )

    if not configs_for_modes:
        print("No valid modes selected for evaluation. Exiting.")
        return

    async def evaluate_single_mode(
        client,
        mode_label: str,
        target_len: int,
        agent_for_mode: ReActAgent,
        boxes_for_mode: int,
    ) -> dict:
        """Generates instances for a mode, runs episodes in parallel, and returns results for that mode."""
        print(
            f"  Starting evaluation for mode: {mode_label} (target_len: {target_len}) for model {model_name}..."
        )
        snapshots = []
        seed = 0
        # Generate 3 instances for this mode
        while len(snapshots) < 3:
            room_struct, room_state, _, _ = generate_room(
                dim=(5, 5),
                initial_seed=seed,
                num_boxes=1,
                search_depth=max(10, target_len + 2),
            )
            path = get_shortest_action_path(room_struct, room_state, MAX_DEPTH=20)
            if len(path) == target_len:
                snapshots.append(
                    {
                        "dim_room": (5, 5),
                        "room_fixed": room_struct.tolist(),
                        "room_state": room_state.tolist(),
                        "boxes_on_target": 0,
                        "max_steps": 20,
                        "num_boxes": 1,
                    }
                )
            seed += 1

        episode_tasks = [
            run_service_episode(client, agent_for_mode, snap, boxes_for_mode)
            for snap in snapshots
        ]
        solved_statuses = await asyncio.gather(*episode_tasks)
        num_solved = sum(solved_statuses)
        num_instances = len(snapshots)
        rate = num_solved / num_instances if num_instances > 0 else 0.0
        print(
            f"    Completed mode: {mode_label} for model {model_name} - Solved: {num_solved}/{num_instances} ({rate:.0%})"
        )
        return {
            "Difficulty": mode_label,
            "Solved": f"{num_solved}/{num_instances}",
            "Success Rate": f"{rate:.0%}",
        }

    all_mode_results_list = []
    async with AsyncClient(base_url="http://localhost:8000") as client:
        mode_evaluation_tasks = []
        for mode_label, target_len in configs_for_modes:
            # Create a new agent instance for each mode to ensure isolated history, if ReActAgent maintains state
            # If ReActAgent is stateless or history is reset per decide call, this might not be strictly necessary
            # but it is safer for parallel execution if there's any doubt.
            llm_for_mode = LM(
                model_name=model_name,
                formatting_model_name=formatting_model_name,
                temperature=0.0,
            )
            agent_for_mode = ReActAgent(llm_for_mode)
            mode_evaluation_tasks.append(
                evaluate_single_mode(
                    client, mode_label, target_len, agent_for_mode, total_boxes
                )
            )

        # Run evaluations for all modes in parallel
        all_mode_results_list = await asyncio.gather(*mode_evaluation_tasks)

    # Sort results by the original order in modes (optional, but good for consistent table output)
    # This requires knowing the original order. If gather changes it, we might need to re-sort.
    # For now, let's assume gather maintains order or sort based on a predefined difficulty order.
    # To simplify, we'll use the order from `configs_for_modes` if needed, though `all_mode_results_list` should be in order.

    # Build table_rows from the collected results
    table_rows = []
    for result_dict in all_mode_results_list:
        table_rows.append(
            [
                result_dict["Difficulty"],
                result_dict["Solved"],
                result_dict["Success Rate"],
            ]
        )

    print(
        f"\nModel: {llm.model_name}, System: {agent.system_name}"
    )  # agent here is the one from the outer scope
    print(
        tabulate(
            table_rows,
            headers=["Difficulty", "Solved", "Success Rate"],
            tablefmt="github",
        )
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(
        eval_react_service_sokoban(
            model_name="gpt-4.1-mini",
            formatting_model_name="gpt-4.1-mini",
            modes=["ultra-easy", "easy"],
        )
    )
