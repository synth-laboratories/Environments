import json
from fastapi.testclient import TestClient

from horizons.environments.service.app import app


def init_env(client: TestClient, env_type: str, config=None):
    r = client.post(f"/env/{env_type}/initialize", json={"config": config})
    r.raise_for_status()
    data = r.json()
    return data["env_id"], data["observation"]


def step_env(client: TestClient, env_type: str, env_id: str, tool: str, args: dict):
    payload = {"env_id": env_id, "request_id": "parity", "action": {"tool_calls": [{"tool": tool, "args": args}]}}
    r = client.post(f"/env/{env_type}/step", json=payload)
    r.raise_for_status()
    return r.json()


def extract_public(obs):
    if isinstance(obs, dict) and "public_observation" in obs:
        return obs["public_observation"]
    return obs


def main():
    client = TestClient(app)

    # Initialize both envs
    sok_id, sok_obs = init_env(client, "Sokoban")
    rust_id, rust_obs = init_env(client, "Sokoban_PyO3")

    pub_sok = extract_public(sok_obs)
    pub_rust = extract_public(rust_obs)

    # Basic key parity
    required = {
        "num_env_steps",
        "max_steps",
        "boxes_on_target",
        "num_boxes",
    }
    missing_sok = [k for k in required if k not in pub_sok]
    missing_rust = [k for k in required if k not in pub_rust]
    assert not missing_sok, f"Pure missing keys: {missing_sok}"
    assert not missing_rust, f"PyO3 missing keys: {missing_rust}"

    # Step once with action=0 (push up) and compare flags
    sok_step = step_env(client, "Sokoban", sok_id, "interact", {"action": 0})
    rust_step = step_env(client, "Sokoban_PyO3", rust_id, "interact", {"action": 0})

    sok_pub = extract_public(sok_step.get("observation", sok_step))
    rust_pub = extract_public(rust_step.get("observation", rust_step))

    # Ensure both carry termination flags; content may differ by initial layout
    assert "terminated" in sok_pub and "truncated" in sok_pub
    assert "terminated" in rust_pub and "truncated" in rust_pub

    print("Parity smoke test passed: key presence and flags available")


if __name__ == "__main__":
    main()
