from src.tasks.core import (
    Task,
    TaskInstance,
    TaskInstanceMetadata,
    TaskInstanceSet,
    SplitInfo,
    Impetus,
    Intent,
)
from uuid import uuid4, UUID
from dataclasses import dataclass, asdict, fields
from typing import Tuple, Optional
from pathlib import Path
import tempfile
import os

verilog_task = Task(
    global_premises="Implement and verify Verilog hardware designs",
    global_constraints="Must pass testbench verification",
    global_objectives="Write correct Verilog code that passes all tests",
    shared_env_params={},
)


@dataclass
class VerilogTaskInstanceMetadata(TaskInstanceMetadata):
    problem_name: str
    difficulty: str
    description: str
    files_provided: list[str]


@dataclass
class VerilogTaskInstance(TaskInstance):
    pristine_dir: Optional[str] = None
    snapshot_dir: Optional[str] = None

    async def serialize(self) -> dict:
        data = asdict(self)
        if "id" in data and isinstance(data["id"], UUID):
            data["id"] = str(data["id"])
        if "intent" in data and data["intent"] is not None:
            if "deterministic_eval_functions" in data["intent"]:
                data["intent"]["deterministic_eval_functions"] = []
        return data

    @classmethod
    async def deserialize(cls, data: dict) -> "VerilogTaskInstance":
        """Gracefully accept non-UUID ids and rebuild required objects."""
        if "id" in data:
            try:
                data["id"] = UUID(str(data["id"]))
            except (ValueError, TypeError, AttributeError):
                pass  # keep original string

        if "impetus" in data and isinstance(data["impetus"], dict):
            data["impetus"] = Impetus(**data["impetus"])

        if "intent" in data and isinstance(data["intent"], dict):
            intent_data = data["intent"]
            intent_data["deterministic_eval_functions"] = []
            data["intent"] = Intent(**intent_data)

        if "metadata" in data and isinstance(data["metadata"], dict):
            data["metadata"] = VerilogTaskInstanceMetadata(**data["metadata"])

        constructor_field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in constructor_field_names}

        return cls(**filtered_data)


async def create_verilog_taskset() -> TaskInstanceSet:
    """Create a simple Verilog task set with basic examples."""
    instances = []

    # Example 1: Simple adder
    adder_instance = _create_adder_task()
    instances.append(adder_instance)

    # Example 2: Simple AND gate
    and_gate_instance = _create_and_gate_task()
    instances.append(and_gate_instance)

    # Create split info - for demo purposes
    split_info = SplitInfo(
        val_instance_ids={instances[0].id},
        test_instance_ids={instances[1].id},
        _is_split_defined=True,
    )

    return TaskInstanceSet(
        name="Verilog Basic TaskSet",
        description="Basic Verilog design tasks for testing the environment",
        instances=instances,
        split_info=split_info,
    )


def _create_adder_task() -> VerilogTaskInstance:
    """Create a simple 4-bit adder task."""
    instance_id = uuid4()
    
    # Create temporary directory for this task
    temp_dir = tempfile.mkdtemp(prefix=f"verilog_adder_{instance_id}_")
    
    # Write adder testbench
    adder_tb_content = '''`timescale 1ns/1ps
module adder4_tb;
    reg [3:0] a, b;
    wire [4:0] sum;
    
    adder4 dut(.a(a), .b(b), .sum(sum));
    
    initial begin
        a = 4'b0000; b = 4'b0000; #10;
        if (sum != 5'b00000) $fatal("Test failed: 0 + 0 != 0");
        
        a = 4'b0001; b = 4'b0001; #10;
        if (sum != 5'b00010) $fatal("Test failed: 1 + 1 != 2");
        
        a = 4'b1111; b = 4'b0001; #10;
        if (sum != 5'b10000) $fatal("Test failed: 15 + 1 != 16");
        
        $display("ALL_TESTS_PASSED");
        $finish;
    end
endmodule'''
    
    # Write incomplete adder module (for student to complete)
    adder_content = '''module adder4(
    input [3:0] a,
    input [3:0] b,
    output [4:0] sum
);
    // TODO: Implement 4-bit adder
    // assign sum = ?;
endmodule'''
    
    pristine_dir = Path(temp_dir)
    pristine_dir.mkdir(exist_ok=True)
    
    (pristine_dir / "adder4_tb.v").write_text(adder_tb_content)
    (pristine_dir / "adder4.v").write_text(adder_content)
    
    impetus = Impetus(
        instructions="Implement a 4-bit adder module that takes two 4-bit inputs 'a' and 'b' and produces a 5-bit output 'sum'."
    )
    
    intent = Intent(
        rubric={"goal": "Implement correct 4-bit adder that passes testbench"},
        gold_trajectories=None,
        gold_state_diff={},
    )
    
    metadata = VerilogTaskInstanceMetadata(
        problem_name="adder4",
        difficulty="easy",
        description="4-bit adder implementation",
        files_provided=["adder4.v", "adder4_tb.v"]
    )
    
    return VerilogTaskInstance(
        id=instance_id,
        impetus=impetus,
        intent=intent,
        metadata=metadata,
        is_reproducible=True,
        initial_engine_snapshot=None,
        pristine_dir=str(pristine_dir),
        snapshot_dir=tempfile.mkdtemp(prefix=f"verilog_snapshot_{instance_id}_")
    )


def _create_and_gate_task() -> VerilogTaskInstance:
    """Create a simple AND gate task."""
    instance_id = uuid4()
    
    # Create temporary directory for this task
    temp_dir = tempfile.mkdtemp(prefix=f"verilog_and_{instance_id}_")
    
    # Write AND gate testbench
    and_tb_content = '''`timescale 1ns/1ps
module and_gate_tb;
    reg a, b;
    wire y;
    
    and_gate dut(.a(a), .b(b), .y(y));
    
    initial begin
        a = 0; b = 0; #10;
        if (y != 0) $fatal("Test failed: 0 AND 0 != 0");
        
        a = 0; b = 1; #10;
        if (y != 0) $fatal("Test failed: 0 AND 1 != 0");
        
        a = 1; b = 0; #10;
        if (y != 0) $fatal("Test failed: 1 AND 0 != 0");
        
        a = 1; b = 1; #10;
        if (y != 1) $fatal("Test failed: 1 AND 1 != 1");
        
        $display("ALL_TESTS_PASSED");
        $finish;
    end
endmodule'''
    
    # Write incomplete AND gate module
    and_content = '''module and_gate(
    input a,
    input b,
    output y
);
    // TODO: Implement AND gate
    // assign y = ?;
endmodule'''
    
    pristine_dir = Path(temp_dir)
    pristine_dir.mkdir(exist_ok=True)
    
    (pristine_dir / "and_gate_tb.v").write_text(and_tb_content)
    (pristine_dir / "and_gate.v").write_text(and_content)
    
    impetus = Impetus(
        instructions="Implement an AND gate module that takes two inputs 'a' and 'b' and produces output 'y'."
    )
    
    intent = Intent(
        rubric={"goal": "Implement correct AND gate that passes testbench"},
        gold_trajectories=None,
        gold_state_diff={},
    )
    
    metadata = VerilogTaskInstanceMetadata(
        problem_name="and_gate",
        difficulty="easy",
        description="Basic AND gate implementation",
        files_provided=["and_gate.v", "and_gate_tb.v"]
    )
    
    return VerilogTaskInstance(
        id=instance_id,
        impetus=impetus,
        intent=intent,
        metadata=metadata,
        is_reproducible=True,
        initial_engine_snapshot=None,
        pristine_dir=str(pristine_dir),
        snapshot_dir=tempfile.mkdtemp(prefix=f"verilog_snapshot_{instance_id}_")
    )


# Example usage
if __name__ == "__main__":
    import asyncio
    import json

    async def main():
        taskset = await create_verilog_taskset()
        
        serialized = await asyncio.gather(
            *(inst.serialize() for inst in taskset.instances)
        )
        
        print(f"Created {len(serialized)} Verilog task instances")
        
        # Print summary
        for i, inst in enumerate(taskset.instances):
            print(f"Task {i+1}: {inst.metadata.problem_name} ({inst.metadata.difficulty})")
            print(f"  Description: {inst.metadata.description}")
            print(f"  Files: {inst.metadata.files_provided}")
            print()

    asyncio.run(main())