hello-py
===

A small eval pipeline: an LLM agent is given a coding task, writes files into a workspace, runs tests via a tool, and is judged on whether tests passed (no access to test code).

## What it does (current use case)

1. **Prompt:** e.g. "Create add.py in the workspace with add(a, b) that returns a + b. Use write_file, then run_tests, then submit_answer."
2. **Agent** uses tools: `write_file` (creates `workspace/add.py`), `run_tests` (runs `eval/test.py` with workspace on `PYTHONPATH`), `submit_answer` (signals done).
3. **Judge:** Success/failure is determined only by the **last `run_tests` result** from the agent loop (`all_passed`). The agent never sees the test file; tests live in `eval/test.py` and are run by the tool.
4. The pipeline runs **N times** (e.g. 10). Between runs, the workspace is cleaned so each run starts from an empty workspace.

## Repo layout

- **`workspace/`** – Where the agent may write files (e.g. `add.py`). Cleared at the start and end of each run.
- **`eval/test.py`** – Test script for the current task. It imports from `workspace` (e.g. `import add`) and asserts. Change this file when you switch tasks; the agent cannot read it.
- **`main.py`** – Agent loop, tool definitions, and the N-run harness.


Setup instructions:

1. Clone the repository:
   ```
   git clone https://github.com/preferencemodel/hello-py.git
   ```

2. Navigate to the project directory:
   ```
   cd hello-py
   ```

3. Set up `ANTHROPIC_API_KEY` environment variable:
   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

4. Run the agent:
   ```
   uv run main.py
   ```

## Execution Modes

The test suite supports both concurrent and sequential execution. 

To change modes, edit the `concurrent` parameter at the bottom of `main.py`:

```python
asyncio.run(main(concurrent=True))
asyncio.run(main(concurrent=False))
```

When running concurrently, results print as they complete (not in run order) for faster overall execution.
