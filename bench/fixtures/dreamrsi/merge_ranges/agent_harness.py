"""Public, fixture-specific tool runner. It contains no model or server code."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
STARTER = HERE / "starter" / "merge_ranges.py"
CHECKER = HERE / "checker.py"
SYSTEM_PROMPT = HERE / "system_prompt.txt"
TASK_PROMPT = HERE / "task.md"
SEEDS = (101, 211, 307, 401, 503)
MAX_COMPLETION_TOKENS = 512
MAX_TOOL_CALLS = 4
MAX_FILE_BYTES = 64 * 1024
TOOL_TIMEOUT_SECONDS = 5


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the current merge_ranges.py file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Replace merge_ranges.py with the supplied source text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the fixture's independent exhaustive checker.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]


def fresh_workspace(work_root: str | Path, seed: int) -> Path:
    """Create one private, never-reused workspace for a preregistered seed."""
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    root = Path(work_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("work root must already exist as a non-symlink directory")
    root_mode = root.stat().st_mode & 0o777
    if root_mode & 0o077:
        raise PermissionError("work root must be owner-only (mode 0700)")
    workspace = root / f"seed-{seed}"
    workspace.mkdir(mode=0o700)
    workspace.chmod(0o700)
    shutil.copyfile(STARTER, workspace / "merge_ranges.py")
    (workspace / "merge_ranges.py").chmod(0o600)
    return workspace


class FixtureTools:
    """Allowlisted read/write/test operations rooted in one seed workspace."""

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError("workspace must be a directory")
        self.test_passed = False
        self.invocations = 0

    @staticmethod
    def _object_args(raw_args: str) -> dict[str, Any]:
        if not isinstance(raw_args, str):
            raise ValueError("arguments must be a JSON string")
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSON arguments: {exc.msg}") from exc
        if type(args) is not dict:
            raise ValueError("arguments must decode to a JSON object")
        return args

    @staticmethod
    def _exact_keys(args: dict[str, Any], expected: set[str]):
        if set(args) != expected:
            raise ValueError(f"expected argument keys {sorted(expected)}")

    def invoke(self, name: str, raw_args: str) -> dict[str, Any]:
        self.invocations += 1
        try:
            args = self._object_args(raw_args)
            if name == "read_file":
                self._exact_keys(args, {"path"})
                if args["path"] != "merge_ranges.py":
                    raise ValueError("path is not allowlisted")
                data = (self.workspace / "merge_ranges.py").read_bytes()
                if len(data) > MAX_FILE_BYTES:
                    raise ValueError("file exceeds the 64 KiB fixture limit")
                return {"ok": True, "content": data.decode("utf-8")}

            if name == "write_file":
                self._exact_keys(args, {"path", "content"})
                if args["path"] != "merge_ranges.py":
                    raise ValueError("path is not allowlisted")
                if not isinstance(args["content"], str):
                    raise ValueError("content must be a string")
                data = args["content"].encode("utf-8")
                if len(data) > MAX_FILE_BYTES:
                    raise ValueError("file exceeds the 64 KiB fixture limit")
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=self.workspace, prefix=".write-", delete=False
                ) as temporary:
                    temporary.write(data)
                    temporary_path = Path(temporary.name)
                temporary_path.chmod(0o600)
                os.replace(temporary_path, self.workspace / "merge_ranges.py")
                self.test_passed = False
                return {"ok": True, "bytes_written": len(data)}

            if name == "run_tests":
                self._exact_keys(args, set())
                try:
                    completed = subprocess.run(
                        [sys.executable, str(CHECKER), str(self.workspace / "merge_ranges.py")],
                        cwd=self.workspace,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=TOOL_TIMEOUT_SECONDS,
                        check=False,
                        shell=False,
                    )
                except subprocess.TimeoutExpired:
                    self.test_passed = False
                    return {"ok": False, "timed_out": True, "timeout_seconds": TOOL_TIMEOUT_SECONDS}
                try:
                    result = json.loads(completed.stdout)
                except json.JSONDecodeError:
                    self.test_passed = False
                    return {
                        "ok": False,
                        "exit_code": completed.returncode,
                        "error": "checker returned non-JSON output",
                        "stderr": completed.stderr[-2000:],
                    }
                self.test_passed = completed.returncode == 0 and result.get("ok") is True
                return {"ok": self.test_passed, "checker": result}

            raise ValueError("tool name is not allowlisted")
        except (UnicodeDecodeError, OSError, ValueError, TypeError) as exc:
            return {"ok": False, "error": str(exc)}


class ChatCompletionsClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: int = 120):
        self.endpoint = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        payload = dict(request)
        payload["model"] = self.model
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        body = json.dumps(payload).encode("utf-8")
        http_request = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Chat Completions request failed with HTTP {exc.code}") from exc


def _tool_call_id(call: dict[str, Any], index: int) -> str:
    value = call.get("id")
    return value if isinstance(value, str) and value else f"invalid-call-{index}"


def _completion_usage(response: dict[str, Any]) -> int:
    usage = response.get("usage")
    if type(usage) is not dict or type(usage.get("completion_tokens")) is not int:
        raise ValueError("response is missing integer usage.completion_tokens")
    if usage["completion_tokens"] < 0:
        raise ValueError("usage.completion_tokens must be nonnegative")
    return usage["completion_tokens"]


def run_agent(client, workspace: str | Path, seed: int) -> dict[str, Any]:
    """Run one seeded, bounded agent session using an injected completions client."""
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    tools = FixtureTools(workspace)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.read_text(encoding="utf-8")},
        {"role": "user", "content": TASK_PROMPT.read_text(encoding="utf-8")},
    ]
    generated_tokens = 0
    tool_calls = 0
    unexecuted_tool_calls = 0
    final_text = ""
    reason = "incomplete"
    forced_final = False
    while True:
        remaining = MAX_COMPLETION_TOKENS - generated_tokens
        if remaining <= 0:
            reason = "token_budget_exhausted"
            break
        request = {
            "messages": messages,
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "seed": seed,
            "max_completion_tokens": remaining,
            "enable_thinking": True,
            "preserve_thinking": False,
        }
        if forced_final or tool_calls == MAX_TOOL_CALLS:
            request["tool_choice"] = "none"
        else:
            request["tools"] = TOOL_DEFINITIONS
            request["tool_choice"] = "auto"

        try:
            response = client.complete(request)
            used = _completion_usage(response)
        except Exception as exc:
            reason = f"completion_error:{type(exc).__name__}"
            break
        if used > remaining:
            generated_tokens += used
            reason = "server_exceeded_token_budget"
            break
        generated_tokens += used

        try:
            choice = response["choices"][0]
            assistant_message = choice["message"]
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError):
            reason = "malformed_completion_response"
            break
        calls = assistant_message.get("tool_calls") or []
        if not isinstance(calls, list):
            reason = "malformed_tool_calls"
            break

        if forced_final:
            if calls:
                reason = "unexpected_tool_calls_in_forced_final"
                break
            final_text = assistant_message.get("content") or ""
            if finish_reason != "stop":
                reason = "final_did_not_stop"
            elif tools.test_passed:
                reason = "success"
            else:
                reason = "tests_not_passed"
            break

        if calls:
            remaining_calls = MAX_TOOL_CALLS - tool_calls
            overflowed = len(calls) > remaining_calls
            allowed_calls = calls[:remaining_calls]
            if overflowed:
                unexecuted_tool_calls += len(calls) - remaining_calls
            # Preserve the model-authored call list before returning tool results.
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.get("content"),
                    "tool_calls": allowed_calls,
                }
            )
            for index, call in enumerate(allowed_calls):
                tool_calls += 1
                if not isinstance(call, dict):
                    result = {"ok": False, "error": "tool call must be an object"}
                    call_id = f"invalid-call-{index}"
                else:
                    call_id = _tool_call_id(call, index)
                    function = call.get("function")
                    if not isinstance(function, dict):
                        result = {"ok": False, "error": "function must be an object"}
                    else:
                        name = function.get("name")
                        arguments = function.get("arguments")
                        if not isinstance(name, str):
                            result = {"ok": False, "error": "function name must be a string"}
                        elif not isinstance(arguments, str):
                            result = {"ok": False, "error": "function arguments must be a JSON string"}
                        else:
                            result = tools.invoke(name, arguments)
                messages.append(
                    {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result, sort_keys=True)}
                )
            if overflowed:
                reason = "tool_budget_exceeded_in_response"
                break
            if tool_calls == MAX_TOOL_CALLS:
                forced_final = True
            if generated_tokens >= MAX_COMPLETION_TOKENS:
                reason = "token_budget_exhausted"
                break
            continue

        final_text = assistant_message.get("content") or ""
        if finish_reason == "stop":
            reason = "success" if tools.test_passed else "tests_not_passed"
        elif generated_tokens >= MAX_COMPLETION_TOKENS:
            reason = "token_budget_exhausted"
        else:
            reason = "final_did_not_stop"
        break

    return {
        "seed": seed,
        "reason": reason,
        "success": reason == "success",
        "completion_tokens": generated_tokens,
        "tool_calls": tool_calls,
        "unexecuted_tool_calls": unexecuted_tool_calls,
        "tests_passed": tools.test_passed,
        "final_text": final_text,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    args = parser.parse_args()
    workspace = fresh_workspace(args.work_root, args.seed)
    client = ChatCompletionsClient(
        args.base_url,
        args.model,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    result = run_agent(client, workspace, args.seed)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
