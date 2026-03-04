import asyncio
import json
from collections.abc import Callable
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, TypedDict, Optional
import pprint
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam
from pathlib import Path
import sys

# Eval layout: one task at a time; tests live in eval/test.py
REPO_ROOT = Path(__file__).resolve().parent
WORKSPACE_DIR = REPO_ROOT / "workspace"
EVAL_TEST_FILE = REPO_ROOT / "eval" / "test.py"

ADD_PY_PATH = WORKSPACE_DIR / "quantize.py"

MAX_TOKENS = 2000


class PythonExpressionToolResult(TypedDict):
    result: Any
    error: str | None


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool

class WriteFileToolResult(TypedDict, total=False):
    status: str
    path: str
    message: str

class RunTestsToolResult(TypedDict):
    passed: int
    total: int
    all_passed: bool
    output: str

def write_file_tool(path: str, contents: str) -> WriteFileToolResult:
    """
    Writes content to a file at the given path. Path must be under workspace.
    """
    WORKSPACE_DIR.mkdir(exist_ok=True)
    full = (WORKSPACE_DIR / path).resolve()
    if not str(full).startswith(str(WORKSPACE_DIR.resolve())):
        return {"status": "error", "message": "Path must be inside workspace"}
    try:
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(contents, encoding="utf-8")
        return {"status": "success", "path": str(full)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
def run_tests_tool() -> RunTestsToolResult:
    """
    Runs the test file at eval/test.py. That file defines the current task's
    tests and can import code from workspace. Change eval/test.py when you
    switch tasks. Returns all_passed and output.
    """
    import subprocess
    import os

    if not EVAL_TEST_FILE.exists():
        return {
            "passed": 0,
            "total": 0,
            "all_passed": False,
            "output": f"Test file not found: {EVAL_TEST_FILE}",
            "metrics": {},
        }

    sep = ";" if os.name == "nt" else ":"
    env = {**os.environ, "PYTHONPATH": str(WORKSPACE_DIR) + sep + os.environ.get("PYTHONPATH", "")}

    try:
        r = subprocess.run(
            [os.sys.executable, str(EVAL_TEST_FILE)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=env,
            timeout=30,
        )
        stdout = (r.stdout or "").strip()
        stderr = (r.stderr or "").strip()
        output = stdout + ("\n" + stderr if stderr else "")

        metrics = {}
        if stdout:
            *_, last = stdout.splitlines()
            import json
            try:
                metrics = json.loads(last)
            except Exception:
                metrics = {}
                # Debug: print test run output so tracebacks are visible
        print("[run_tests] --- begin ---")
        print(output)
        print("[run_tests] --- end ---")
        return {
            "passed": 0,
            "total": 0,
            "all_passed": r.returncode == 0,
            "output": output,
            "metrics": metrics,
        }
    except subprocess.TimeoutExpired:
        return {"passed": 0, "total": 0, "all_passed": False, "output": "Tests timed out (30s).", "metrics": {}}
    except Exception as e:
        return {"passed": 0, "total": 0, "all_passed": False, "output": str(e), "metrics": {}}


def python_expression_tool(expression: str) -> PythonExpressionToolResult:
    """
    Tool that evaluates Python expressions using exec.
    Use print(...) to emit output; stdout will be captured and returned.
    """
    try:
        namespace = {}
        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(expression, namespace, namespace)
        return {"result": stdout.getvalue(), "error": None}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"result": None, "error": str(e)}


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """
    Tool for submitting the final answer.
    """
    return {"answer": answer, "submitted": True}


def _print_add_py_for_run(run_id: int) -> None:
    """Print contents of workspace/quantize.py for this run if it exists."""
    if not ADD_PY_PATH.exists():
        print(f"[Run {run_id}] quantize.py not found in workspace.")
        return
    text = ADD_PY_PATH.read_text(encoding="utf-8")
    print(f"[Run {run_id}] add.py content:\n---\n{text}\n---")


def _clean_workspace() -> None:
    """Remove all contents of workspace so the next run starts clean."""
    import shutil
    if WORKSPACE_DIR.exists():
        shutil.rmtree(WORKSPACE_DIR)
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

def judge_run(submitted_answer: Any, tests_result: RunTestsToolResult | None) -> dict:
    """
    Judge a single run based on metrics from eval/test.py.

    tests_result["metrics"] is expected to contain:
      - base_model_size_bytes
      - quantized_model_size_bytes
      - base_ppl
      - quantized_ppl
      - has_fp16_params
    """
    if tests_result is None:
        return {
            "final_score": 0.0,
            "compression": 0.0,
            "ppl_increase": float("inf"),
            "passed": False,
        }

    m = tests_result.get("metrics", {}) or {}

    base_size = float(m.get("base_model_size_bytes", 0.0))
    quant_size = float(m.get("quantized_model_size_bytes", 0.0))
    base_ppl = float(m.get("base_ppl", 0.0))
    quant_ppl = float(m.get("quantized_ppl", 0.0))
    has_fp16 = bool(m.get("has_fp16_params", True))  # default to True (fail-safe)

    # 1) compression ratio
    if quant_size <= 0 or base_size <= 0:
        compression = 0.0
    else:
        compression = base_size / quant_size

    # 2) perplexity degradation
    if base_ppl <= 0:
        ppl_increase = float("inf")
    else:
        ppl_increase = (quant_ppl - base_ppl) / base_ppl

    # 3) compression_score (expect ~2x for fp16 -> int8)
    compression_score = min(compression / 2.0, 1.0)

    # 4) accuracy_score (tolerate up to 20% increase)
    accuracy_score = max(0.0, 1.0 - (ppl_increase / 0.20))

    # 5) combined score
    final_score = 0.6 * accuracy_score + 0.4 * compression_score

    # 6) hard fail conditions
    hard_fail = (
        compression < 1.8
        or ppl_increase > 0.25
        or has_fp16
    )

    passed = (not hard_fail)

    return {
        "final_score": float(final_score),
        "compression": float(compression),
        "ppl_increase": float(ppl_increase),
        "passed": bool(passed),
    }


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 20,
    model: str = "claude-haiku-4-5",
    verbose: bool = True,
) -> Any | None:
    """
    Runs an agent loop with the given prompt and tools.

    Args:
        prompt: The initial prompt for the agent
        tools: List of tool definitions for Anthropic API
        tool_handlers: Dictionary mapping tool names to their handler functions
        max_steps: Maximum number of steps before stopping (default 5)
        model: The Anthropic model to use
        verbose: Whether to print detailed output (default True)

    Returns:
        A dict with submitted_answer and last_run_tests
    """
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]
    last_run_tests: RunTestsToolResult | None = None

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")

        response = await client.messages.create(
            model=model, max_tokens=MAX_TOKENS, tools=tools, messages=messages
        )
        pprint.pprint(response)
        assert response.stop_reason in ["max_tokens", "tool_use", "end_turn"], (
            f"unsupported stop_reason {response.stop_reason}"
        )
        if response.stop_reason == "max_tokens":
            print(
                f"Model reached max_tokens limit {MAX_TOKENS}. Increase "
                "MAX_TOKENS, simplify your task, or update the code to provide "
                "a message back to the model when it exceeds MAX_TOKENS."
            )

        # Track if we need to continue
        has_tool_use = False
        tool_results = []
        submitted_answer = None

        # Process the response
        for content in response.content:
            if content.type == "text":
                if verbose:
                    print(f"Assistant: {content.text}")
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name

                if tool_name in tool_handlers:
                    if verbose:
                        print(f"Using tool: {tool_name}")

                    # Extract arguments based on tool
                    handler = tool_handlers[tool_name]
                    tool_input = content.input

                    # Call the appropriate tool handler
                    if tool_name == "python_expression":
                        assert (
                            isinstance(tool_input, dict) and "expression" in tool_input
                        )
                        if verbose:
                            print("\nInput:")
                            print("```")
                            for line in tool_input["expression"].split("\n"):
                                print(f"{line}")
                            print("```")
                        result = handler(tool_input["expression"])
                        if verbose:
                            print("\nOutput:")
                            print("```")
                            print(result)
                            print("```")
                    elif tool_name == "submit_answer":
                        assert isinstance(tool_input, dict) and "answer" in tool_input
                        result = handler(tool_input["answer"])
                        submitted_answer = result["answer"]
                    else:
                        # Generic handler call
                        result = (
                            handler(**tool_input)
                            if isinstance(tool_input, dict)
                            else handler(tool_input)
                        )

                    if tool_name == "run_tests":
                        last_run_tests = result 

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": json.dumps(result),
                        }
                    )

        # If we have tool uses, add them to the conversation
        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})

            messages.append({"role": "user", "content": tool_results})

            # If an answer was submitted, return it
            if submitted_answer is not None:
                if verbose:
                    print(f"\nAgent submitted answer: {submitted_answer}")
                break
        else:
            # No tool use, conversation might be complete
            if verbose:
                print("\nNo tool use in response, ending loop.")
            break

    if verbose and submitted_answer is None:
        print(f"\nReached maximum steps ({max_steps}) without submitting answer.")
    return {
        "submitted_answer": submitted_answer,
        "last_run_tests": last_run_tests,
    }


async def run_single_test(
    run_id: int,
    num_runs: int,
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    verbose: bool = True,
) -> tuple[int, bool, Any]:
    if verbose:
        print(f"\n\n{'=' * 20} RUN {run_id}/{num_runs} {'=' * 20}")
    
    _clean_workspace()

    result = await run_agent_loop(
        prompt=prompt,
        tools=tools,
        tool_handlers=tool_handlers,
        max_steps=10,
        verbose=verbose,
    )

    tests = result.get("last_run_tests") if isinstance(result, dict) else None
    judge = judge_run(
        submitted_answer=result.get("submitted_answer") if isinstance(result, dict) else None,
        tests_result=tests,
    )

    success = judge["passed"]

    if success:
        print(f"✓ Run {run_id}: SUCCESS - final_score={judge['final_score']:.3f}")
    else:
        print(f"✗ Run {run_id}: FAILURE - final_score={judge['final_score']:.3f}")
        print(f"  compression={judge['compression']:.3f}, ppl_increase={judge['ppl_increase']:.3f}")
        if tests and tests.get("output"):
            print(f"  --- run_tests output ---")
            print(tests["output"])
            print(f"  --- end run_tests output ---")
    _print_add_py_for_run(run_id)
    _clean_workspace()

    return run_id, success, judge


async def main(concurrent: bool = False):
    tools: list[ToolUnionParam] = [
        {
            "name": "write_file",
            "description": "Write text content to a file. Path is relative to workspace (e.g. add.py or src/add.py).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to workspace, e.g. add.py",
                    },
                    "contents": {
                        "type": "string",
                        "description": "Full file contents to write.",
                    },
                },
                "required": ["path", "contents"],
            },
        },
        {
            "name": "run_tests",
            "description": "Run the predefined test cases for the code in the workspace (e.g. add.py). Returns passed/total and output.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit the final answer",
            "input_schema": {
                "type": "object",
                "properties": {"answer": {"description": "The final answer to submit"}},
                "required": ["answer"],
            },
        },
    ]

    tool_handlers = {
        "submit_answer": submit_answer_tool,
        "write_file": write_file_tool,
        "run_tests": run_tests_tool,
    }

    # Run the test 10 times and track success rate
    num_runs = 1
    prompt = (
    "Create a Python file quantize.py in the workspace with a function "
    "quantize_model(model) that takes a fp16 HuggingFace causal language model "
    "and returns an 8-bit/int8 version of the same model.\n\n"
    "Requirements:\n"
    "1) Do not change the model architecture.\n"
    "2) The returned model must actually use 8-bit/int8 weights (e.g. via bitsandbytes "
    "or another standard quantization approach).\n"
    "3) Keep the function in a single file quantize.py so it can be imported as "
    "`import quantize; quantize.quantize_model(model)`.\n\n"
    "Use the write_file tool to create or update workspace/quantize.py with your code. "
    "Then use the run_tests tool to run eval/test.py, which will check size reduction "
    "and perplexity on a sample prompt."
)
    execution_mode = "concurrently" if concurrent else "sequentially"
    print(f"Running {num_runs} test iterations {execution_mode}...")
    print("=" * 60)

    # Create all test coroutines
    tasks = [
        run_single_test(
            run_id=i + 1,
            num_runs=num_runs,
            prompt=prompt,
            tools=tools,
            tool_handlers=tool_handlers,
            verbose=True,
        )
        for i in range(num_runs)
    ]

    # Run concurrently or sequentially based on the flag
    if concurrent:
        # Process results as they complete
        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
    else:
        # Run sequentially by awaiting each task in order
        results = []
        for task in tasks:
            result = await task
            results.append(result)

    # Count successes
    successes = sum(success for _, success, _ in results)

    # Calculate and display pass rate
    pass_rate = (successes / num_runs) * 100
    print(f"\n{'=' * 60}")
    print("Test Results:")
    print(f"  Passed: {successes}/{num_runs}")
    print(f"  Failed: {num_runs - successes}/{num_runs}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Set to True for concurrent execution, False for sequential execution
    asyncio.run(main(concurrent=False))
