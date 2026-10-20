import json
import tempfile
import unittest
from pathlib import Path

import agent_harness as harness


CORRECT = '''\
def merge_ranges(ranges):
    ordered = sorted((start, end) for start, end in ranges)
    merged = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
'''


def completion(message, tokens=12, finish_reason="stop"):
    return {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {"completion_tokens": tokens},
    }


def call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected completion request")
        return self.responses.pop(0)


class HarnessSelfTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "private-root"
        self.root.mkdir(mode=0o700)
        self.root.chmod(0o700)

    def tearDown(self):
        self.temp.cleanup()

    def workspace(self, seed=101):
        return harness.fresh_workspace(self.root, seed)

    def test_fresh_seed_workspaces_are_private_and_never_reused(self):
        first = self.workspace(101)
        second = self.workspace(211)
        self.assertNotEqual(first, second)
        self.assertEqual(first.stat().st_mode & 0o777, 0o700)
        self.assertEqual((first / "merge_ranges.py").stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            self.workspace(101)

    def test_sequential_tools_run_checker_then_final_summary(self):
        workspace = self.workspace()
        write_call = call("w1", "write_file", json.dumps({"path": "merge_ranges.py", "content": CORRECT}))
        test_call = call("t1", "run_tests", "{}")
        client = ScriptedClient(
            [
                completion({"role": "assistant", "content": None, "tool_calls": [write_call, test_call]}, tokens=40, finish_reason="tool_calls"),
                completion({"role": "assistant", "content": "Implemented and all tests pass.", "tool_calls": []}, tokens=8),
            ]
        )
        result = harness.run_agent(client, workspace, 101)
        self.assertTrue(result["success"], result)
        self.assertTrue(result["tests_passed"])
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual(client.requests[0]["tool_choice"], "auto")
        self.assertEqual(client.requests[0]["seed"], 101)
        self.assertEqual(client.requests[0]["max_completion_tokens"], 512)
        self.assertNotIn("parallel_tool_calls", client.requests[0])
        tool_messages = [m for m in client.requests[1]["messages"] if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_messages], ["w1", "t1"])

    def test_invalid_calls_count_and_recover_with_allowlisted_test(self):
        workspace = self.workspace()
        (workspace / "merge_ranges.py").write_text(CORRECT, encoding="utf-8")
        invalid = [
            call("bad-name", "shell", "{}"),
            call("bad-json", "read_file", "{"),
            call("bad-path", "read_file", '{"path":"../merge_ranges.py"}'),
        ]
        test = call("test", "run_tests", "{}")
        client = ScriptedClient(
            [
                completion({"role": "assistant", "content": None, "tool_calls": invalid}, tokens=30, finish_reason="tool_calls"),
                completion({"role": "assistant", "content": None, "tool_calls": [test]}, tokens=30, finish_reason="tool_calls"),
                completion({"role": "assistant", "content": "Done.", "tool_calls": []}, tokens=5),
            ]
        )
        result = harness.run_agent(client, workspace, 211)
        self.assertTrue(result["success"], result)
        self.assertEqual(result["tool_calls"], 4)
        self.assertEqual(client.requests[2]["tool_choice"], "none")

    def test_four_call_budget_forces_final_without_tools(self):
        workspace = self.workspace()
        bad_calls = [call(f"bad-{i}", "unknown", "{}") for i in range(4)]
        client = ScriptedClient(
            [
                completion({"role": "assistant", "content": None, "tool_calls": bad_calls}, tokens=20, finish_reason="tool_calls"),
                completion({"role": "assistant", "content": "Could not complete.", "tool_calls": []}, tokens=3),
            ]
        )
        result = harness.run_agent(client, workspace, 307)
        self.assertEqual(result["reason"], "tests_not_passed")
        self.assertEqual(result["tool_calls"], 4)
        self.assertEqual(client.requests[1]["tool_choice"], "none")
        self.assertNotIn("tools", client.requests[1])

    def test_tool_calls_in_forced_final_are_rejected_without_execution(self):
        workspace = self.workspace()
        bad_calls = [call(f"bad-{i}", "unknown", "{}") for i in range(4)]
        client = ScriptedClient(
            [
                completion({"role": "assistant", "content": None, "tool_calls": bad_calls}, tokens=20, finish_reason="tool_calls"),
                completion(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [call("extra", "read_file", '{"path":"merge_ranges.py"}')],
                    },
                    tokens=3,
                    finish_reason="tool_calls",
                ),
            ]
        )
        result = harness.run_agent(client, workspace, 307)
        self.assertEqual(result["reason"], "unexpected_tool_calls_in_forced_final")
        self.assertEqual(result["tool_calls"], 4)
        self.assertEqual(result["unexecuted_tool_calls"], 0)

    def test_excess_tool_calls_in_one_response_are_not_executed(self):
        workspace = self.workspace()
        five = [call(f"r{i}", "read_file", '{"path":"merge_ranges.py"}') for i in range(5)]
        client = ScriptedClient([completion({"role": "assistant", "content": None, "tool_calls": five}, tokens=20, finish_reason="tool_calls")])
        result = harness.run_agent(client, workspace, 401)
        self.assertEqual(result["reason"], "tool_budget_exceeded_in_response")
        self.assertEqual(result["tool_calls"], 4)
        self.assertEqual(result["unexecuted_tool_calls"], 1)

    def test_exact_token_cap_stops_without_another_completion(self):
        workspace = self.workspace()
        read = call("r1", "read_file", '{"path":"merge_ranges.py"}')
        client = ScriptedClient([completion({"role": "assistant", "content": None, "tool_calls": [read]}, tokens=512, finish_reason="tool_calls")])
        result = harness.run_agent(client, workspace, 503)
        self.assertEqual(result["reason"], "token_budget_exhausted")
        self.assertEqual(result["completion_tokens"], 512)
        self.assertEqual(len(client.requests), 1)

    def test_rejects_server_usage_over_remaining_token_budget(self):
        workspace = self.workspace()
        client = ScriptedClient([completion({"role": "assistant", "content": "too much", "tool_calls": []}, tokens=513)])
        result = harness.run_agent(client, workspace, 503)
        self.assertEqual(result["reason"], "server_exceeded_token_budget")
        self.assertEqual(len(client.requests), 1)


if __name__ == "__main__":
    unittest.main()
