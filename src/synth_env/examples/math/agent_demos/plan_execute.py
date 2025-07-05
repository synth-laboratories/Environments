import asyncio
import logging

# Configure logging immediately
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from synth_ai.zyk import LM
from synth_env.examples.math.environment import HendryksMathEnv
from synth_env.examples.math.taskset import (
    create_hendryks_taskset,
    HendryksSubjectFilter,
)
from synth_env.examples.math.schema import (
    HendryksTaskInstance,
    HendryksTaskInstanceMetadata,
)
from synth_env.examples.math.tools import SubmitAnswerTool
from tabulate import tabulate

logger = logging.getLogger(__name__)


async def main():
    logger.info("Starting main function...")

    logger.info("Creating Hendryks taskset...")
    task_set = await create_hendryks_taskset()
    total_instances = len(task_set.instances)
    logger.info(f"Total task instances loaded in taskset: {total_instances}")

    if total_instances == 0:
        logger.warning("No task instances found in the taskset. Exiting.")
        return

    # Log count per available subject from the taskset instances
    logger.info("Task counts per subject in taskset:")
    subject_counts = {}
    for instance in task_set.instances:
        if isinstance(instance.metadata, HendryksTaskInstanceMetadata):
            subj = (
                instance.metadata.subject
                if instance.metadata.subject is not None
                else "Unknown"
            )
            subject_counts[subj] = subject_counts.get(subj, 0) + 1
    for subj, count in sorted(subject_counts.items()):
        logger.info(f"  - {subj}: {count}")

    # Instantiate the LLM model
    llm = LM(
        model_name="gpt-4.1-nano", formatting_model_name="gpt-4.1-nano", temperature=0.0
    )

    topics = ["algebra", "geometry"]  # topics to evaluate
    samples_per_topic = 5
    table_rows = []

    logger.info(f"Evaluating topics: {topics}")
    for topic in topics:
        logger.debug(f"Processing topic: {topic}")

        subject_filter = HendryksSubjectFilter(subjects=[topic])
        topic_instances = [inst for inst in task_set.instances if subject_filter(inst)]

        count = len(topic_instances)
        logger.debug(f"Task instances matching topic '{topic}': {count}")

        selected_instances = topic_instances[:samples_per_topic]

        logger.debug(f"Selected {len(selected_instances)} instances for '{topic}'")
        results_correctness = []  # Store boolean correctness

        if not selected_instances:
            logger.warning(
                f"No instances selected for topic '{topic}'. Skipping LLM calls."
            )

        for instance_data in selected_instances:
            if not isinstance(instance_data, HendryksTaskInstance):
                logger.warning(
                    f"Skipping non-HendryksTaskInstance: {type(instance_data)}"
                )
                continue

            # Create a new environment instance for each task instance from the taskset
            env = HendryksMathEnv(task_instance=instance_data)
            problem_id = str(instance_data.id)

            logger.debug(f"Initializing environment for task ID: {problem_id}")
            # Initialize with the specific problem ID from the instance
            # obs = await env.initialize(problem_id=problem_id)
            # HendryksMathEngine._reset_engine uses task_instance.id by default if problem_id is None
            # and task_instance is passed to HendryksMathEnv constructor.
            # So, initializing with problem_id from task_instance.id is implicitly handled.
            init_obs = await env.initialize()
            prompt = f"Solve the following {topic} problem (id={init_obs['problem_id']}): {init_obs['prompt']}"
            logger.debug(f"Prompt: {prompt}")

            llm_response = await llm.respond_async(
                system_message="You are a math problem solver.", user_message=prompt
            )
            answer_text = llm_response.raw_response.strip()
            logger.debug(f"LLM answer: {answer_text}")

            tool_call = SubmitAnswerTool(answer=answer_text)
            step_obs = await env.step(
                [[tool_call]]
            )  # Pass as list of lists of tool calls

            logger.debug(f"Step observation for {init_obs['problem_id']}: {step_obs}")
            results_correctness.append(step_obs.get("is_correct", False))

        solved_count = sum(results_correctness)
        num_attempted = len(selected_instances)
        rate = solved_count / num_attempted if num_attempted else 0
        logger.info(
            f"Results for topic '{topic}': Solved {solved_count}/{num_attempted} ({rate:.0%})"
        )
        table_rows.append([topic, f"{solved_count}/{num_attempted}", f"{rate:.0%}"])

    print("Model: gpt-4.1-nano")
    print(
        tabulate(
            table_rows, headers=["Topic", "Solved", "Success Rate"], tablefmt="github"
        )
    )
    logger.info("Main function finished.")


if __name__ == "__main__":
    logger.info("Script execution started.")
    asyncio.run(main())
    logger.info("Script execution finished.")
