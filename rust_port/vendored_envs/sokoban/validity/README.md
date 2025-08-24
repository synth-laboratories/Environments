Sokoban Validity Cases

Purpose
- Provide strict, deterministic checks to validate the behavior of the vendored Sokoban engine.
- Designed to flag any semantic drift when the engine is refactored/replaced.

Format (cases.jsonl)
- Each line is a JSON object with:
  - name: Unique case identifier
  - config: { room_fixed, room_state, max_steps, num_boxes }
  - scripts: [
      { name, steps: [
          { call: { action:int } | { direction:str, mode:str }, expect: { boxes_on_target:int, terminated:bool, truncated:bool, room_text_contains?:str, reward_last?:number } }
        ] }
    ]

Notes
- room_fixed/state use the same codes as the Python engine:
  - 0=Wall, 1=Empty, 2=Target, 3=BoxOnTarget, 4=Box, 5=Player
- Scripts focus on observable public state via the service: boxes_on_target, flags, and room_text.
- Tests run through the PyO3-backed service (Sokoban_PyO3) for exact parity.

