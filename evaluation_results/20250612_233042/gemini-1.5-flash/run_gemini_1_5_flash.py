
import asyncio
import sys
import os

# Add the Environments src directory to Python path
sys.path.insert(0, "/Users/joshuapurtell/Documents/GitHub/Environments/src")

# Import using the installed synth_env package
from synth_env.examples.verilog.agent_demos.test_synth_react import eval_verilog_react

async def main():
    try:
        result = await eval_verilog_react(
            model_name="gemini-1.5-flash",
            formatting_model_name="gemini-1.5-flash",
            n_instances=2,  # Reduced from 5 to make it run faster
            debug_mode=False
        )
        print("\n=== EVALUATION RESULTS ===")
        print(f"Model: {result['model']}")
        print(f"Overall Success Rate: {result['overall_success_rate']:.1%}")
        print(f"Total Instances: {result['total_instances']}")
        print(f"Successful Instances: {result['successful_instances']}")
        return result
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    asyncio.run(main())
