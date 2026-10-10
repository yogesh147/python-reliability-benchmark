import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import requests

TASKS_FILE = Path("tasks.json")
RESULTS_DIR = Path("results")
TIMEOUT_SECONDS = 3
MAX_RESPONSE_CHARS = 20000

SYSTEM_PROMPT = """
You are participating in a Python reliability benchmark.
Return the complete Python function requested by the task.
Use the exact requested function name.
Do not include markdown fences or explanatory text.
Follow all constraints in the prompt.
"""


def load_models():
    """
    Configure models using the MODELS environment variable.

    Example:
    MODELS='[
      {
        "name": "Model A",
        "model": "provider-model-id",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY"
      }
    ]'
    """

    OPENAI_API_KEY = "your-api-key"
    MODELS = [{
        "name": "Model A",
        "model": "your-model-id-a",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY"
    }, {
        "name": "Model B",
        "model": "your-model-id-b",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY"
    }]

    raw = os.getenv("MODELS", "[]")
    models = json.loads(raw)

    if not models:
        raise ValueError(
            "Configure at least one model in the MODELS "
            "environment variable."
        )

    for model in models:
        for key in ("name", "model", "base_url", "api_key_env"):
            if not model.get(key):
                raise ValueError(f"Model configuration missing {key}")

    return models


def query_model(model, prompt):
    """Call an OpenAI-compatible chat-completions endpoint."""
    api_key = os.getenv(model["api_key_env"])

    if not api_key:
        raise ValueError(
            f"Missing API key environment variable: "
            f"{model['api_key_env']}"
        )

    url = model["base_url"].rstrip("/") + "/chat/completions"

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model["model"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        },
        timeout=60,
    )
    response.raise_for_status()

    result = response.json()
    content = result["choices"][0]["message"]["content"]

    if not isinstance(content, str):
        raise ValueError("Model returned no text response.")

    return content[:MAX_RESPONSE_CHARS]


def extract_code(response):
    """Remove common markdown fences from a model response."""
    match = re.search(
        r"```(?:python)?\s*(.*?)```",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return match.group(1).strip() if match else response.strip()


def check_syntax(code):
    """Check whether the response is valid Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def check_forbidden_imports(code):
    """Detect imports for tasks that explicitly forbid them."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True

    return any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        for node in ast.walk(tree)
    )


def execute_tests(code, task):
    """
    Execute a candidate function in a child process.

    WARNING: This is not a secure sandbox for untrusted code.
    Use an isolated container with restricted permissions and
    no network access when evaluating public model outputs.
    """
    if not check_syntax(code):
        return {
            "passed": 0,
            "total": len(task["tests"]),
            "error": "SyntaxError",
        }

    function_name = task["function_name"]

    harness = r'''
import contextlib
import io
import json
import sys

namespace = {}
source = sys.stdin.read()
'''

    # Pass code and test data as JSON to avoid quoting issues.
    payload = json.dumps({
        "code": code,
        "tests": task["tests"],
        "function_name": function_name,
        "check_immutability": task["id"] == "INS001",
    })

    runner = r'''
import copy
import contextlib
import io
import json
import sys
import traceback

payload = json.loads(sys.stdin.read())
namespace = {}

try:
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(payload["code"], "<candidate>", "exec"),
             namespace)

    fn = namespace.get(payload["function_name"])
    if not callable(fn):
        raise ValueError("Requested function not found")

    results = []
    for case in payload["tests"]:
        args = copy.deepcopy(case["input"])
        original = copy.deepcopy(args)

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                actual = fn(*args)

            passed = actual == case["expected"]

            if payload["check_immutability"] and args != original:
                passed = False

            results.append({
                "passed": bool(passed),
                "error": None
            })
        except Exception as exc:
            results.append({
                "passed": False,
                "error": type(exc).__name__
            })

    print(json.dumps({"results": results}))

except Exception as exc:
    print(json.dumps({
        "error": type(exc).__name__,
        "results": []
    }))
'''

    with tempfile.TemporaryDirectory() as directory:
        script_path = Path(directory) / "runner.py"
        script_path.write_text(runner, encoding="utf-8")

        try:
            completed = subprocess.run(
                [sys.executable, str(script_path)],
                input=payload,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                cwd=directory,
                check=False,
            )

            if completed.returncode != 0:
                return {
                    "passed": 0,
                    "total": len(task["tests"]),
                    "error": "ProcessError",
                }

            output = json.loads(completed.stdout.strip())
            results = output.get("results", [])

            return {
                "passed": sum(r["passed"] for r in results),
                "total": len(task["tests"]),
                "error": output.get("error"),
            }

        except subprocess.TimeoutExpired:
            return {
                "passed": 0,
                "total": len(task["tests"]),
                "error": "Timeout",
            }
        except (json.JSONDecodeError, OSError):
            return {
                "passed": 0,
                "total": len(task["tests"]),
                "error": "ExecutionError",
            }


def evaluate_one(model, task):
    """Evaluate one model on one benchmark task."""
    start = time.perf_counter()

    record = {
        "model": model["name"],
        "task_id": task["id"],
        "category": task["category"],
        "passed_tests": 0,
        "total_tests": len(task["tests"]),
        "pass_rate": 0.0,
        "syntax_valid": False,
        "constraint_compliant": None,
        "error": None,
        "latency_seconds": None,
    }

    try:
        response = query_model(model, task["prompt"])
        code = extract_code(response)

        record["syntax_valid"] = check_syntax(code)

        if task["id"] == "INS001":
            record["constraint_compliant"] = (
                    record["syntax_valid"]
                    and not check_forbidden_imports(code)
            )

        # Invalid syntax or a violated hard constraint fails the task.
        if not record["syntax_valid"]:
            record["error"] = "SyntaxError"
        elif record["constraint_compliant"] is False:
            record["error"] = "ForbiddenImport"
        else:
            result = execute_tests(code, task)
            record["passed_tests"] = result["passed"]
            record["error"] = result["error"]

        record["pass_rate"] = (
            record["passed_tests"] / record["total_tests"]
            if record["total_tests"] else 0.0
        )

    except Exception as exc:
        record["error"] = type(exc).__name__

    record["latency_seconds"] = round(
        time.perf_counter() - start, 3
    )

    return record


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with TASKS_FILE.open(encoding="utf-8") as file:
        tasks = json.load(file)

    models = load_models()
    records = []

    for model in models:
        print(f"\nEvaluating {model['name']}")

        for task in tasks:
            result = evaluate_one(model, task)
            records.append(result)

            print(
                f"  {task['id']}: "
                f"{result['passed_tests']}/"
                f"{result['total_tests']} tests passed"
                f" | error={result['error']}"
            )

            # Avoid excessive API request rates.
            time.sleep(0.2)

    raw = pd.DataFrame(records)
    raw.to_csv(RESULTS_DIR / "raw_results.csv", index=False)

    summary = (
        raw.groupby(["model", "category"], as_index=False)
        .agg(
            tasks=("task_id", "count"),
            mean_pass_rate=("pass_rate", "mean"),
            syntax_valid_rate=("syntax_valid", "mean"),
            mean_latency_seconds=("latency_seconds", "mean"),
        )
    )

    summary.to_csv(
        RESULTS_DIR / "model_summary.csv", index=False
    )

    leaderboard = (
        raw.groupby("model", as_index=False)
        .agg(
            tasks_evaluated=("task_id", "count"),
            overall_pass_rate=("pass_rate", "mean"),
            syntax_valid_rate=("syntax_valid", "mean"),
            mean_latency_seconds=("latency_seconds", "mean"),
        )
        .sort_values(
            "overall_pass_rate", ascending=False
        )
    )

    leaderboard.to_csv(
        RESULTS_DIR / "leaderboard.csv", index=False
    )

    print("\n=== Benchmark leaderboard ===")
    print(leaderboard.to_string(index=False))
    print(f"\nResults saved to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()