# The Agent Company Benchmark

This example integrates the [TheAgentCompany](https://github.com/TheAgentCompany/TheAgentCompany) benchmark.
Only the task list is retrieved to demonstrate loading of the benchmark metadata.

`create_tac_taskset()` will read a cached list of Docker images from `data/tasks.md`.
If the file is missing it is downloaded once from the official repository and cached locally.
