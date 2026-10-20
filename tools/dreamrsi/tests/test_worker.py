from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tarfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lifecycle import ManagedProcess  # noqa: E402
from evaluator import (  # noqa: E402
    EvaluationError,
    RequestRejectedError,
    _classify_failure,
    _expected_settings,
    _first_token_seconds,
    _match_request,
    _parse_capture_window,
    _raise_for_rejected_replay,
    _run_plan,
    _sanitized_environment,
    _server_command,
    _verify_artifact,
    _validate_derived_capsule_descriptors,
)
from compat_replay import (  # noqa: E402
    _prepare_flat_replay_import_package,
    _load_provenance,
    inspect_capsule,
)
from submit import (  # noqa: E402
    SubmissionError,
    enqueue_submission,
    freeze_submission,
    parse_override,
)


def init_worktree(path: Path) -> None:
    subprocess.run(["git", "init", "--initial-branch=main", str(path)],
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Worker Test"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "worker@example.test"], check=True)
    (path / "src").mkdir()
    (path / "src" / "main.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
    (path / "src" / "models").mkdir()
    (path / "src" / "models" / "model.cpp").write_text("// model implementation\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "src/main.cpp"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "baseline"],
                   check=True, stdout=subprocess.DEVNULL)


class SubmissionTests(unittest.TestCase):
    def test_freeze_includes_worktree_patch_and_untracked_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worktree = root / "worktree"
            worktree.mkdir()
            init_worktree(worktree)
            (worktree / "src" / "main.cpp").write_text("int main() { return 1; }\n", encoding="utf-8")
            (worktree / "candidate.py").write_text("candidate = True\n", encoding="utf-8")
            (worktree / "models").mkdir()
            (worktree / "models" / "large.ninfer").write_bytes(b"not packaged")
            (worktree / "build").mkdir()
            (worktree / "build" / "cache").write_bytes(b"not packaged")

            package = freeze_submission(
                source=worktree,
                package_dir=root / "packages",
                role="candidate",
                baseline_id="baseline-a",
                artifact_path="/models/fixture.ninfer",
                artifact_sha256="a" * 64,
                runtime_id="test-runtime",
                overrides={"max_concurrency": 1},
            )
            manifest = package["manifest"]
            self.assertEqual(manifest["source"]["git_commit"],
                             subprocess.check_output(["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True).strip())
            self.assertNotEqual(manifest["source"]["worktree_patch_sha256"], "0" * 64)
            self.assertGreaterEqual(manifest["source"]["file_count"], 2)
            with tarfile.open(package["archive_path"], "r") as archive:
                names = archive.getnames()
            self.assertIn("candidate.py", names)
            self.assertIn("src/models/model.cpp", names)
            self.assertNotIn("models/large.ninfer", names)
            self.assertNotIn("build/cache", names)

    def test_fifo_accepts_only_matching_immutable_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worktree = root / "worktree"
            worktree.mkdir()
            init_worktree(worktree)
            package_a = freeze_submission(
                source=worktree, package_dir=root / "packages", role="baseline",
                baseline_id="baseline-a", artifact_path="/model.ninfer",
                artifact_sha256="b" * 64, runtime_id="fnn-test",
            )
            package_b = freeze_submission(
                source=worktree, package_dir=root / "packages", role="control",
                baseline_id="baseline-a", artifact_path="/model.ninfer",
                artifact_sha256="b" * 64, runtime_id="fnn-test",
            )
            queue = root / "queue"
            first = enqueue_submission(
                queue_root=queue,
                manifest_path=Path(package_a["manifest_path"]),
                archive_path=Path(package_a["archive_path"]),
            )
            second = enqueue_submission(
                queue_root=queue,
                manifest_path=Path(package_b["manifest_path"]),
                archive_path=Path(package_b["archive_path"]),
            )
            self.assertEqual((first["sequence"], second["sequence"]), (1, 2))
            pending = sorted((queue / "pending").glob("*.json"))
            self.assertEqual([json.loads(path.read_text())["role"] for path in pending],
                             ["baseline", "control"])
            self.assertEqual(Path(first["snapshot_path"]), Path(second["snapshot_path"]))
            self.assertEqual(os.stat(queue).st_mode & 0o777, 0o700)
            self.assertEqual(os.stat(first["snapshot_path"]).st_mode & 0o777, 0o400)

            broken = root / "broken.tar"
            broken.write_bytes(b"different")
            with self.assertRaises(SubmissionError):
                enqueue_submission(
                    queue_root=queue,
                    manifest_path=Path(package_a["manifest_path"]),
                    archive_path=broken,
                )

    def test_overrides_are_closed_and_typed(self) -> None:
        self.assertEqual(parse_override("max_concurrency=1"), ("max_concurrency", 1))
        self.assertEqual(parse_override("kv_capacity=auto"), ("kv_capacity", "auto"))
        self.assertEqual(parse_override("no_cuda_graph=true"), ("no_cuda_graph", True))
        with self.assertRaises(Exception):
            parse_override("host=0.0.0.0")
        with self.assertRaises(Exception):
            parse_override("max_concurrency=9")
        with self.assertRaises(Exception):
            parse_override("kv_dtype=unknown")


class ProcessGroupTests(unittest.TestCase):
    def test_terminate_process_group_reaps_child(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "child.log"
            managed = ManagedProcess.start(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                log_path=log,
            )
            self.assertTrue(managed.is_running())
            time.sleep(0.05)
            stopped = managed.terminate_group(timeout_seconds=2)
            self.assertIsNotNone(stopped["returncode"])
            self.assertFalse(stopped["process_group_alive"])


class EvaluatorContractTests(unittest.TestCase):
    def test_reuse_derived_capsule_requires_exact_transform_and_parent_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture_dir = Path(temporary) / "captures"
            capture_dir.mkdir(mode=0o700)
            capsule_dir = capture_dir / "derived-test.capsule"
            capsule_dir.mkdir(mode=0o700)
            body = json.dumps({
                "model": "claude-sonnet-5",
                "max_tokens": 96,
                "thinking": {"type": "adaptive", "display": "summarized"},
                "output_config": {"effort": "xhigh"},
                "messages": [{"role": "user", "content": "private prompt"}],
            }, separators=(",", ":")).encode()
            request_sha256 = hashlib.sha256(body).hexdigest()
            parent_sha256 = "a" * 64
            (capsule_dir / "request.body").write_bytes(body)
            (capsule_dir / "request.body").chmod(0o400)
            capsule_manifest = {
                "schema_version": 1,
                "capsule_id": "derived-test",
                "request": {
                    "method": "POST",
                    "target": "/v1/messages?beta=true",
                    "body_file": "request.body",
                    "body_sha256": request_sha256,
                },
                "derivation": {
                    "semantic_transform": True,
                    "production_compatible": False,
                    "parent_capsule_id": "parent-test",
                    "parent_raw_request_sha256": parent_sha256,
                    "transformed_body_sha256": request_sha256,
                    "diff": [
                        {"path": "$.thinking.display", "from": "omitted", "to": "summarized"},
                        {"path": "$.output_config.effort", "from": "high", "to": "xhigh"},
                    ],
                },
            }
            (capsule_dir / "manifest.json").write_text(json.dumps(capsule_manifest), encoding="utf-8")
            (capsule_dir / "manifest.json").chmod(0o400)
            descriptors = capture_dir / "derived-descriptors.jsonl"
            descriptors.write_text(json.dumps({
                "schema_version": 1,
                "capsule_id": "derived-test",
                "capsule_path": str(capsule_dir),
                "request_sha256": request_sha256,
                "output_budget": 96,
                "metadata": {"semantic_transform": True, "production_compatible": False},
            }) + "\n", encoding="utf-8")
            descriptors.chmod(0o600)

            summary = _validate_derived_capsule_descriptors(capture_dir, descriptors, parent_sha256)
            self.assertEqual(summary["request_sha256"], request_sha256)
            self.assertTrue(summary["semantic_transform"])
            self.assertFalse(summary["production_compatible"])
            self.assertEqual(summary["diff"], [
                {"path": "$.thinking.display", "from": "omitted", "to": "summarized"},
                {"path": "$.output_config.effort", "from": "high", "to": "xhigh"},
            ])
            self.assertNotIn("private prompt", json.dumps(summary))
            with self.assertRaisesRegex(EvaluationError, "approved summarized-thinking"):
                _validate_derived_capsule_descriptors(capture_dir, descriptors, "b" * 64)

    def test_flat_deployment_imports_replay_capture_module(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            research_root = Path(temporary) / "research"
            tools_dir = research_root / "tools"
            tools_dir.mkdir(parents=True)
            (tools_dir / "capture.py").write_text("SCHEMA_VERSION = 1\n", encoding="utf-8")
            replay_script = tools_dir / "replay.py"
            replay_script.write_text(
                "from tools.dreamrsi.capture import SCHEMA_VERSION\n"
                "print(SCHEMA_VERSION)\n",
                encoding="utf-8",
            )

            package = _prepare_flat_replay_import_package(research_root, tools_dir)
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(research_root)
            replay = subprocess.run(
                [sys.executable, str(replay_script)],
                cwd=research_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(replay.returncode, 0, replay.stderr)
            self.assertEqual(replay.stdout.strip(), "1")
            self.assertEqual(Path(package["module_path"]).read_text(), "SCHEMA_VERSION = 1\n")

    def test_replay_tool_failure_is_reported_before_telemetry_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tools_dir = root / "tools"
            tools_dir.mkdir()
            (tools_dir / "replay.py").write_text("", encoding="utf-8")
            capsule_path = root / "capsule-a"
            capsule = {
                "capsule_id": "capsule-a",
                "capsule_path": str(capsule_path),
                "output_budget": 96,
                "request_sha256": "a" * 64,
                "metadata": {"cache_state": "unknown"},
            }
            descriptors = root / "descriptors.jsonl"
            descriptors.write_text(json.dumps(capsule) + "\n", encoding="utf-8")
            request_log = root / "requests.jsonl"
            request_log.write_text("", encoding="utf-8")
            run_dir = root / "run"
            run_dir.mkdir()
            plan = {
                "plan_id": "plan-a",
                "cache_recipe_id": "cache-a",
                "selected": [{
                    "capsule_id": "capsule-a",
                    "capsule_path": str(capsule_path),
                    "request_sha256": "a" * 64,
                    "capture_output_budget": 96,
                }],
            }

            with (
                mock.patch("evaluator._run", return_value={
                    "returncode": 1,
                    "output_tail": "ModuleNotFoundError: no module named tools",
                }),
                mock.patch("evaluator._request_id_set", return_value=set()),
                mock.patch("evaluator._match_request") as match_request,
            ):
                with self.assertRaisesRegex(EvaluationError, "replay tool exited with status 1"):
                    _run_plan(
                        root / "quick_eval.py", tools_dir, plan, descriptors,
                        request_log, run_dir, mock.Mock(samples=[]), {},
                    )
            match_request.assert_not_called()
            self.assertFalse((run_dir / "response-capsule-a.tmp").exists())

    def test_terminal_replay_4xx_is_classified_before_telemetry_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tools_dir = root / "tools"
            tools_dir.mkdir()
            (tools_dir / "replay.py").write_text("", encoding="utf-8")
            descriptors = root / "descriptors.jsonl"
            capsule = {
                "capsule_id": "capsule-a",
                "capsule_path": str(root / "capsule-a"),
                "output_budget": 96,
                "request_sha256": "a" * 64,
                "metadata": {"cache_state": "unknown"},
            }
            descriptors.write_text(json.dumps(capsule) + "\n", encoding="utf-8")
            request_log = root / "requests.jsonl"
            request_log.write_text("", encoding="utf-8")
            run_dir = root / "run"
            run_dir.mkdir()
            error_body = {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": "unsupported_parameter",
                    "param": "messages[0].content[0].type",
                    "message": "Unsupported field; Bearer sk-ant-test-secret",
                },
            }
            response = run_dir / "response-capsule-a.tmp"

            def fake_replay(command: list[str], **_kwargs: object) -> dict[str, object]:
                response_path = Path(command[command.index("--response-output") + 1])
                response_path.write_text(json.dumps(error_body), encoding="utf-8")
                return {
                    "returncode": 0,
                    "output_tail": json.dumps({"status": 400}),
                }

            plan = {
                "plan_id": "plan-a",
                "cache_recipe_id": "cache-a",
                "selected": [{
                    "capsule_id": "capsule-a",
                    "capsule_path": capsule["capsule_path"],
                    "request_sha256": capsule["request_sha256"],
                    "capture_output_budget": 96,
                }],
            }
            with (
                mock.patch("evaluator._run", side_effect=fake_replay),
                mock.patch("evaluator._request_id_set", return_value=set()),
                mock.patch("evaluator._match_request") as match_request,
            ):
                with self.assertRaises(RequestRejectedError) as caught:
                    _run_plan(
                        root / "quick_eval.py", tools_dir, plan, descriptors,
                        request_log, run_dir, mock.Mock(samples=[]), {},
                    )
                match_request.assert_not_called()

            self.assertFalse(response.exists())
            failure = _classify_failure(caught.exception)
            self.assertEqual(failure["classification"], "request_rejected")
            self.assertEqual(failure["http_status"], 400)
            self.assertEqual(failure["response_error"], {
                "type": "invalid_request_error",
                "code": "unsupported_parameter",
                "param": "messages[0].content[0].type",
                "message": "Unsupported field; [redacted]",
            })
            self.assertNotIn("sk-ant-test-secret", json.dumps(failure))

    def test_compatibility_capsule_inspection_checks_hash_and_hides_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capsule = Path(temporary) / "capsule"
            capsule.mkdir()
            body = json.dumps({
                "model": "claude-sonnet-5",
                "max_tokens": 96,
                "thinking": {"type": "adaptive", "display": "omitted"},
                "messages": [{"role": "user", "content": "private prompt"}],
            }, separators=(",", ":")).encode()
            digest = hashlib.sha256(body).hexdigest()
            (capsule / "request.body").write_bytes(body)
            (capsule / "manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "capsule_id": "compat-a",
                "cache_state": "unknown",
                "request": {
                    "method": "POST",
                    "target": "/v1/messages?beta=true",
                    "body_file": "request.body",
                    "body_sha256": digest,
                },
            }), encoding="utf-8")

            summary = inspect_capsule(capsule, digest)
            self.assertEqual(summary["request_sha256"], digest)
            self.assertEqual(summary["max_tokens"], 96)
            self.assertEqual(summary["thinking"], {"type": "adaptive", "display": "omitted"})
            self.assertNotIn("private prompt", json.dumps(summary))
            with self.assertRaisesRegex(EvaluationError, "expected identity"):
                inspect_capsule(capsule, "0" * 64)

    def test_two_field_compatibility_transform_is_verified_against_parent_body(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture_root = Path(temporary)
            parent_id = "parent-capsule"
            parent_value = {
                "model": "claude-sonnet-5",
                "max_tokens": 96,
                "thinking": {"type": "adaptive", "display": "omitted"},
                "output_config": {"effort": "high"},
                "messages": [{"role": "user", "content": "private prompt"}],
            }
            derived_value = json.loads(json.dumps(parent_value))
            derived_value["thinking"]["display"] = "summarized"
            derived_value["output_config"]["effort"] = "xhigh"

            def write_capsule(name: str, value: dict[str, object], *, derivation: dict[str, object] | None = None) -> tuple[Path, str]:
                capsule = capture_root / f"{name}.capsule"
                capsule.mkdir(mode=0o700)
                body = json.dumps(value, separators=(",", ":")).encode()
                digest = hashlib.sha256(body).hexdigest()
                (capsule / "request.body").write_bytes(body)
                request: dict[str, object] = {
                    "method": "POST",
                    "target": "/v1/messages?beta=true",
                    "body_file": "request.body",
                    "body_sha256": digest,
                }
                manifest: dict[str, object] = {
                    "schema_version": 1,
                    "capsule_id": name,
                    "cache_state": "unknown",
                    "request": request,
                }
                if derivation is not None:
                    manifest["derivation"] = derivation
                (capsule / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                return capsule, digest

            parent_capsule, parent_sha256 = write_capsule(parent_id, parent_value)
            diff = [
                {"path": "$.thinking.display", "from": "omitted", "to": "summarized"},
                {"path": "$.output_config.effort", "from": "high", "to": "xhigh"},
            ]
            derived_value_sha256 = hashlib.sha256(
                json.dumps(derived_value, separators=(",", ":")).encode()
            ).hexdigest()
            derived_id = "derived-capsule"
            derived_capsule, derived_sha256 = write_capsule(
                derived_id,
                derived_value,
                derivation={
                    "semantic_transform": True,
                    "production_compatible": False,
                    "parent_capsule_id": parent_id,
                    "parent_raw_request_sha256": parent_sha256,
                    "transformed_body_sha256": derived_value_sha256,
                    "diff": diff,
                },
            )
            summary = inspect_capsule(derived_capsule, derived_sha256)
            provenance = _load_provenance(
                None, summary, parent_sha256, "thinking-summarized-effort-xhigh"
            )
            self.assertTrue(provenance["semantic_transform"])
            self.assertFalse(provenance["production_compatible"])
            self.assertEqual(provenance["diff"], diff)
            self.assertTrue(parent_capsule.is_dir())

            tampered_value = json.loads(json.dumps(derived_value))
            tampered_value["max_tokens"] = 95
            tampered_capsule, tampered_sha256 = write_capsule(
                "tampered-capsule",
                tampered_value,
                derivation={
                    "semantic_transform": True,
                    "production_compatible": False,
                    "parent_capsule_id": parent_id,
                    "parent_raw_request_sha256": parent_sha256,
                    "transformed_body_sha256": hashlib.sha256(
                        json.dumps(tampered_value, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "diff": diff,
                },
            )
            tampered_summary = inspect_capsule(tampered_capsule, tampered_sha256)
            with self.assertRaisesRegex(EvaluationError, "beyond its declared transform"):
                _load_provenance(
                    None, tampered_summary, parent_sha256, "thinking-summarized-effort-xhigh"
                )

    def test_capture_window_is_bounded_to_fifteen_minutes(self) -> None:
        self.assertEqual(_parse_capture_window("900"), 900)
        self.assertEqual(_parse_capture_window("60"), 60)
        with self.assertRaises(Exception):
            _parse_capture_window("59")
        with self.assertRaises(Exception):
            _parse_capture_window("901")

    def test_production_profile_is_explicit_and_deterministically_seeded(self) -> None:
        settings = _expected_settings({"overrides": {}})
        argv = _server_command(Path("/research/ninfer-serve"), "/models/explicit.ninfer",
                               Path("/research/requests.jsonl"), settings, 18081)
        self.assertIn("188416", argv)
        self.assertIn("k8v4", argv)
        self.assertIn("--spec", argv)
        self.assertIn("mtp", argv)
        self.assertIn("7", argv)
        self.assertIn("7008", argv)
        self.assertIn("--lm-head-draft", argv)
        self.assertEqual(settings["max_concurrency"], 1)

    def test_child_environment_drops_credentials_but_keeps_cuda_runtime(self) -> None:
        with mock.patch.dict(os.environ, {
            "PATH": "/usr/bin",
            "CUDA_VISIBLE_DEVICES": "0",
            "ANTHROPIC_API_KEY": "secret-a",
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "96",
            "AWS_SECRET_ACCESS_KEY": "secret-b",
            "NINFER_API_KEY": "secret-c",
        }, clear=True):
            environment = _sanitized_environment()
        self.assertEqual(environment["PATH"], "/usr/bin")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0")
        self.assertNotIn("ANTHROPIC_API_KEY", environment)
        self.assertNotIn("CLAUDE_CODE_MAX_OUTPUT_TOKENS", environment)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", environment)
        self.assertNotIn("NINFER_API_KEY", environment)

    def test_model_artifact_hash_is_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.ninfer"
            path.write_bytes(b"fixture model")
            import hashlib

            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            verified = _verify_artifact({"path": str(path), "sha256": expected})
            self.assertEqual(verified["sha256"], expected)
            self.assertEqual(verified["size_bytes"], len(b"fixture model"))
            with self.assertRaisesRegex(Exception, "differs from the queued manifest"):
                _verify_artifact({"path": str(path), "sha256": "0" * 64})

    def test_server_ttft_is_taken_from_correlated_terminal_event(self) -> None:
        terminal = {"timings_seconds": {"ttft": 0.25}}
        self.assertEqual(_first_token_seconds(terminal), 0.25)
        self.assertIsNone(_first_token_seconds(None))

    def test_serve_telemetry_maps_by_request_id_without_prompt_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "requests.jsonl"
            request = {"request_id": 7, "protocol": "anthropic_messages", "model": "qwen/qwen3.8-27b",
                       "sampling": {"seed": 7008}, "requested_output_tokens": 96}
            events = [
                {"schema_version": 21, "event": "request_start", "server_instance_id": "s1",
                 "request": request},
                {"schema_version": 21, "event": "request_done", "server_instance_id": "s1",
                 "request": request, "result": {"completion_tokens": 11}},
            ]
            log.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            start, terminal = _match_request(log, set())
            self.assertEqual(start["request"]["request_id"], 7)
            self.assertEqual(terminal["event"], "request_done")
            self.assertNotIn("content", json.dumps(start))


if __name__ == "__main__":
    unittest.main()
