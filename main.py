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
import math

# Eval layout: one task at a time; tests live in eval/test.py
REPO_ROOT = Path(__file__).resolve().parent
WORKSPACE_DIR = REPO_ROOT / "workspace"
EVAL_TEST_FILE = REPO_ROOT / "eval" / "test.py"

ADD_PY_PATH = WORKSPACE_DIR / "quantize.py"

MAX_TOKENS = 100000


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

class RunTestsToolResult(TypedDict, total=False):
    passed: int
    total: int
    all_passed: bool
    output: str
    metrics: dict[str, Any]
    judge: dict[str, Any]

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
    Runs eval/test.py, parses JSON metrics from stdout, then applies judge_run().
    Returns judge-aligned all_passed so the model sees real status.
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
            "judge": {
                "final_score": 0.0,
                "compression": 0.0,
                "ppl_increase": float("inf"),
                "passed": False,
            },
        }

    sep = ";" if os.name == "nt" else ":"
    env = {
        **os.environ,
        "PYTHONPATH": str(WORKSPACE_DIR) + sep + os.environ.get("PYTHONPATH", ""),
    }

    try:
        r = subprocess.run(
            [sys.executable, str(EVAL_TEST_FILE)],   # <- use sys.executable
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=env,
            timeout=3000,
        )

        stdout = (r.stdout or "").strip()
        stderr = (r.stderr or "").strip()
        output = stdout + ("\n" + stderr if stderr else "")

        # If test script crashed, fail immediately with traceback
        if r.returncode != 0:
            judge = {
                "final_score": 0.0,
                "compression": 0.0,
                "ppl_increase": float("inf"),
                "passed": False,
            }
            return {
                "passed": 0,
                "total": 0,
                "all_passed": False,
                "output": output or f"eval/test.py failed with exit code {r.returncode}",
                "metrics": {},
                "judge": judge,
            }

        # Parse JSON metrics from last stdout line
        metrics: dict[str, Any] = {}
        parse_error = None
        if stdout:
            *_, last = stdout.splitlines()
            try:
                metrics = json.loads(last)
            except Exception as e:
                parse_error = str(e)

        if not metrics:
            judge = {
                "final_score": 0.0,
                "compression": 0.0,
                "ppl_increase": float("inf"),
                "passed": False,
            }
            msg = "No valid JSON metrics produced by eval/test.py."
            if parse_error:
                msg += f" JSON parse error: {parse_error}"
            if output:
                msg += f"\nRaw output:\n{output}"
            return {
                "passed": 0,
                "total": 0,
                "all_passed": False,
                "output": msg,
                "metrics": {},
                "judge": judge,
            }

        # Judge based on metrics (single source of truth)
        judge = judge_run(submitted_answer=None, tests_result={"metrics": metrics})

        # Give model a clear picture
        judge_summary = (
            f"judge_passed={judge['passed']}, final_score={judge['final_score']:.3f}, "
            f"compression={judge['compression']:.3f}, ppl_increase={judge['ppl_increase']:.3f}"
        )
        full_output = (output + "\n" if output else "") + judge_summary

        return {
            "passed": 0,
            "total": 0,
            "all_passed": bool(judge["passed"]),  # <- judge, not subprocess code
            "output": full_output,
            "metrics": metrics,
            "judge": judge,
        }

    except subprocess.TimeoutExpired:
        judge = {
            "final_score": 0.0,
            "compression": 0.0,
            "ppl_increase": float("inf"),
            "passed": False,
        }
        return {
            "passed": 0,
            "total": 0,
            "all_passed": False,
            "output": "Tests timed out (3000s).",
            "metrics": {},
            "judge": judge,
        }
    except Exception as e:
        judge = {
            "final_score": 0.0,
            "compression": 0.0,
            "ppl_increase": float("inf"),
            "passed": False,
        }
        return {
            "passed": 0,
            "total": 0,
            "all_passed": False,
            "output": str(e),
            "metrics": {},
            "judge": judge,
        }

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


def _print_quantize_py_for_run(run_id: int) -> None:
    """Print contents of workspace/quantize.py for this run if it exists."""
    if not ADD_PY_PATH.exists():
        print(f"[Run {run_id}] quantize.py not found in workspace.")
        return
    text = ADD_PY_PATH.read_text(encoding="utf-8")
    print(f"[Run {run_id}] quantize.py content:\n---\n{text}\n---")


def _clean_workspace() -> None:
    """Remove all contents of workspace so the next run starts clean."""
    import shutil
    if WORKSPACE_DIR.exists():
        shutil.rmtree(WORKSPACE_DIR)
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


def ppl_accuracy_score(ppl_increase: float) -> float:
    p = float(ppl_increase)
    # Better than base model
    if p <= 0.0:
        return 1.0
    # "0.00x" region: still high, around 0.8
    if p <= 0.005:
        # 0.0 -> 1.0, 0.005 -> 0.8
        return 1.0 - (p / 0.005) * 0.2
    # 0.005 -> 0.8, 0.013 -> 0.5
    if p <= 0.013:
        return 0.8 - ((p - 0.005) / 0.008) * 0.3
    # 0.013 -> 0.5, 0.016 -> 0.4
    if p <= 0.016:
        return 0.5 - ((p - 0.013) / 0.003) * 0.1
    # 0.016 -> 0.4, 0.019 -> 0.3
    if p <= 0.019:
        return 0.4 - ((p - 0.016) / 0.003) * 0.1
    # After 0.019, decay quickly toward zero
    return max(0.0, 0.3 * math.exp(-(p - 0.019) / 0.004))

def judge_run(submitted_answer: Any, tests_result: RunTestsToolResult | None) -> dict:
    """
    Judge a single run based on metrics from eval/test.py.

    tests_result["metrics"] is expected to contain:
      - base_model_size_bytes
      - quantized_model_size_bytes
      - base_ppl
      - quantized_ppl
      - has_non_int8_params
      - has_sub_int8_params
      - has_quant_scales
      - same_layer_count
      - per_channel_scales
      - activation_quantization_modules_exists
      - activation_quantization_during_forward
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
    has_non_int8 = bool(m.get("has_non_int8_params", True))  # default to True (fail-safe)
    has_sub_int8 = bool(m.get("has_sub_int8_params", True))  # default to True (fail-safe)
    has_quant_scales = bool(m.get("has_quant_scales", False))  # default to False (fail-safe)
    same_layer_count = bool(m.get("same_layer_count", False))  # default to False (fail-safe)
    per_channel_scales = bool(m.get("per_channel_scales", False))  # default to False (fail-safe)
    detected_int8_tensor_during_forward = bool(m.get("detected_int8_tensor_during_forward", False))  # default to False (fail-safe)
    activation_quantization_during_forward = bool(m.get("activation_quantization_during_forward", False))  # default to False (fail-safe)
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

    # 6) hard fail conditions
    hard_fail = (
        compression < 1.6
        or ppl_increase > 0.25
        or has_sub_int8
        or not same_layer_count
        or not per_channel_scales
        or not detected_int8_tensor_during_forward
        or not activation_quantization_during_forward
    )

    passed = (not hard_fail)
    # 5) final score
    if passed:
        final_score = ppl_accuracy_score(ppl_increase)
    else:
        final_score = 0.0
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
    max_steps: int = 50,
    model: str = "claude-opus-4-6",
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

        response = None
        async with client.messages.stream(
            model=model,
            max_tokens=MAX_TOKENS,
            tools=tools,
            messages=messages,
            extra_headers={"anthropic-beta": "context-1m-2025-08-07"},
        ) as stream:
            response = await stream.get_final_message()

        assert response is not None
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
        max_steps=50,
        verbose=verbose,
    )

    tests = result.get("last_run_tests") if isinstance(result, dict) else None
    judge = (tests or {}).get("judge")
    if judge is None:
        judge = judge_run(submitted_answer=None, tests_result=tests)

    success = judge["passed"]

    if success:
        print(f"✓ Run {run_id}: SUCCESS - final_score={judge['final_score']:.3f}")
        print(f"  compression={judge['compression']:.3f}, ppl_increase={judge['ppl_increase']:.3f}, passed={judge['passed']}")
    else:
        print(f"✗ Run {run_id}: FAILURE - final_score={judge['final_score']:.3f}")
        print(f"  compression={judge['compression']:.3f}, ppl_increase={judge['ppl_increase']:.3f}, passed={judge['passed']}")
    if tests and tests.get("output"):
        print(f"  --- run_tests output ---")
        print(tests["output"])
        print(f"  --- end run_tests output ---")
    _print_quantize_py_for_run(run_id)
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
    ]

    tool_handlers = {
        "write_file": write_file_tool,
        "run_tests": run_tests_tool,
    }

    # Run the test 10 times and track success rate
    num_runs = 10
    prompt = (
    "Create a Python file quantize.py in the workspace with a function "
    "quantize_model(model) that takes a fp16 language model "
    "and returns an 8-bit/int8 version of the same model.\n\n"
    "Requirements:\n"
    "1) Do not change the model architecture.\n"
    "2) Keep the function in a single file quantize.py so it can be imported as "
    "`import quantize; quantize.quantize_model(model)`.\n\n"
    "3) Use per-channel INT8 quantization for linear layer weights.\n"
    "4) Implement activation quantization (W8A8). Activations should be "
    "quantized to int8 using dynamic scaling during the forward pass. "
    "The implementation should introduce activation quantization "
    "modules or parameters (e.g., activation scales) so that activations "
    "are quantized before linear operations.\n"
    "Use the write_file tool to create or update workspace/quantize.py with your code. "
    "Then use the run_tests tool to run eval/test.py, which will check size reduction "
    "and perplexity and other metrics on a sample prompt."
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
