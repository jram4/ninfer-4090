from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import types
from typing import Any

import pytest

from tools.dreamrsi import evaluator, lifecycle


def _install_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        import tools.dreamrsi.contracts as module
    except ImportError:
        try:
            import contracts as module  # type: ignore[no-redef]
        except ImportError:
            module = types.ModuleType("tools.dreamrsi.contracts")

            def canonical_hash(value: Any) -> str:
                encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

            def sha256_file(path: Path) -> str:
                digest = hashlib.sha256()
                with Path(path).open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                return digest.hexdigest()

            def read_json(path: Path) -> Any:
                return json.loads(Path(path).read_text(encoding="utf-8"))

            def write_json(path: Path, value: Any) -> None:
                destination = Path(path)
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                temporary.chmod(0o600)
                os.replace(temporary, destination)

            def submission_identity(value: dict[str, Any]) -> dict[str, Any]:
                identity = dict(value)
                identity.pop("submission_id", None)
                identity.pop("created_at", None)
                identity["source"] = dict(identity["source"])
                identity["source"].pop("archive_path", None)
                return identity

            def validate_submission(value: dict[str, Any]) -> dict[str, Any]:
                if (value.get("schema_version") != 1
                        or canonical_hash(submission_identity(value)) != value.get("submission_id")
                        or canonical_hash(value["source"]["manifest"]) != value["source"]["snapshot_sha256"]):
                    raise ValueError("invalid submission identity")
                return json.loads(json.dumps(value))

            def validate_policy(value: dict[str, Any]) -> dict[str, Any]:
                required = {"schema_version", "policy_id", "contract_id", "contract", "baseline_id",
                            "corpus_id", "toolchain_id", "stages", "qualification_enabled"}
                if (value.keys() != required
                        or canonical_hash(value["contract"]) != value["contract_id"]
                        or canonical_hash({k: v for k, v in value.items() if k != "policy_id"}) != value["policy_id"]):
                    raise ValueError("invalid policy")
                return json.loads(json.dumps(value))

            def validate_receipt(value: dict[str, Any]) -> dict[str, Any]:
                required = {"schema_version", "submission_id", "policy_id", "contract_id",
                            "source_snapshot_sha256", "verdict", "stage", "failure_class",
                            "evidence", "created_at"}
                optional = {"candidate_commit", "base_commit", "qualification_enabled", "promotion_authorized"}
                if not required <= value.keys() or value.keys() - (required | optional):
                    raise ValueError("invalid receipt fields")
                for item in value["evidence"]:
                    if set(item) - {"path", "sha256", "kind", "stage", "media_type"}:
                        raise ValueError("invalid evidence reference")
                    if not {"path", "sha256"} <= item.keys():
                        raise ValueError("incomplete evidence reference")
                return dict(value)

            def compare_contract(expected: dict[str, Any], observed: dict[str, Any]) -> list[str]:
                if expected == observed:
                    return []
                return ["contract differs"]

            module.canonical_hash = canonical_hash
            module.sha256_file = sha256_file
            module.read_json = read_json
            module.write_json = write_json
            module.submission_identity = submission_identity
            module.validate_submission = validate_submission
            module.validate_policy = validate_policy
            module.validate_receipt = validate_receipt
            module.compare_contract = compare_contract
    monkeypatch.setitem(sys.modules, "tools.dreamrsi.contracts", module)


def _archive(path: Path, *, unsafe: bool = False) -> str:
    source = path.parent / "src-tree"
    source.mkdir()
    (source / "CMakeLists.txt").write_text("project(frozen_source)\n", encoding="utf-8")
    archive_path = path
    with tarfile.open(archive_path, "w") as archive:
        member = tarfile.TarInfo("../escape.txt" if unsafe else "src-tree/CMakeLists.txt")
        content = b"project(frozen_source)\n"
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    return hashlib.sha256(archive_path.read_bytes()).hexdigest()


def _contract() -> dict[str, Any]:
    return {
        "model_artifact": {"sha256": "a" * 64, "size_bytes": 100},
        "runtime": {
            "binary": {"sha256": "d" * 64, "size_bytes": 101},
            "release_source_commit": "7" * 40,
            "kv_dtype": "k8v4",
            "max_context": 188416,
            "kv_capacity": 188416,
            "sampling_capabilities": {"mtp": True},
            "sampling_defaults": {
                "greedy": False,
                "omitted_seed": "random",
                "server_overrides": {"frequency_penalty": None, "min_p": None,
                                     "presence_penalty": None, "seed": None,
                                     "temperature": None, "top_k": None, "top_p": None},
                "thinking": {"frequency_penalty": 0.0, "min_p": 0.0, "presence_penalty": 0.0,
                             "temperature": 1.0, "top_k": 20, "top_p": 0.949999988079071},
                "non_thinking": {"frequency_penalty": 0.0, "min_p": 0.0, "presence_penalty": 1.5,
                                 "temperature": 0.699999988079071, "top_k": 20,
                                 "top_p": 0.800000011920929},
            },
        },
        "template": {"sha256": "b" * 64},
        "tokenizer": {"sha256": "c" * 64},
    }


def _outcome(receipt: dict[str, Any]) -> dict[str, Any]:
    item = next(item for item in receipt["evidence"] if item.get("kind") == "evaluator_diagnostics")
    return json.loads(Path(item["path"]).read_text(encoding="utf-8"))


def _submission(archive_path: Path, archive_sha256: str, policy: dict[str, Any]) -> dict[str, Any]:
    source_tree = archive_path.parent / "src-tree"
    file_path = source_tree / "CMakeLists.txt"
    manifest = [{
        "path": "src-tree/CMakeLists.txt",
        "type": "file",
        "mode": 0o100644,
        "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
    }]
    contracts = sys.modules["tools.dreamrsi.contracts"]
    submission = {
        "schema_version": 1,
        "submission_id": "0" * 64,
        "epoch_id": "epoch-test",
        "worker_id": "worker-test",
        "hypothesis_id": "hypothesis-test",
        "base_commit": "2" * 40,
        "candidate_commit": "3" * 40,
        "source": {
            "archive_path": str(archive_path.resolve()),
            "archive_sha256": archive_sha256,
            "snapshot_sha256": contracts.canonical_hash(manifest),
            "git_commit": "3" * 40,
            "dirty": False,
            "manifest": manifest,
        },
        "contract_id": policy["contract_id"],
        "policy_id": policy["policy_id"],
        "created_at": "2026-10-20T00:00:00+00:00",
        "idempotency_key": "idempotency-test",
    }
    submission["submission_id"] = contracts.canonical_hash(contracts.submission_identity(submission))
    return submission


def _stage_command(stage: str, *, omit_report: bool = False,
                   status: str = "pass", wrong_contract: bool = False,
                   wrong_toolchain: bool = False,
                   wrong_raw_hash: bool = False,
                   observed_contract: dict[str, Any] | None = None) -> list[str]:
    if omit_report:
        code = "raise SystemExit(0)"
    else:
        candidate_binary = (
            "b=os.path.join(e['NINFER_BUILD_DIR'],'ninfer-serve');"
            "open(b,'wb').write(b'candidate-binary');os.chmod(b,0o755);"
            "binary={'sha256':hashlib.sha256(b'candidate-binary').hexdigest(),"
            "'size_bytes':len(b'candidate-binary')};"
        )
        observed = (
            "observed=json.loads(" + repr(json.dumps(observed_contract or _contract())) + ");"
            "observed['runtime']=dict(observed['runtime']);"
            "observed['runtime']['binary']=binary;"
            "observed['runtime']['release_source_commit']='3'*40;"
            "observed['runtime']['max_context']=131072 if " + repr(bool(wrong_contract))
            + " else observed['runtime']['max_context'];"
        )
        code = (
            "import hashlib,json,os; e=os.environ; "
            + candidate_binary
            + "binding={k:e[v] for k,v in {"
            "'schema_version':'NINFER_REPORT_SCHEMA','submission_id':'NINFER_SUBMISSION_ID',"
            "'policy_id':'NINFER_POLICY_ID','contract_id':'NINFER_CONTRACT_ID',"
            "'source_snapshot_sha256':'NINFER_SOURCE_SNAPSHOT_SHA256',"
            "'attempt_id':'NINFER_ATTEMPT_ID','stage':'NINFER_STAGE'}.items()}; "
            "binding['schema_version']=int(e['NINFER_REPORT_SCHEMA']); "
            "binding['status']=" + repr(status) + "; "
            "binding['toolchain_id']='toolchain-test' if e['NINFER_STAGE']=='admission' else None; "
            "binding['toolchain']="
            + repr({"cmake": "4.3" if wrong_toolchain else "4.4", "cuda": "13"})
            + " if e['NINFER_STAGE']=='admission' else None; "
            "binding['build_key']=e['NINFER_BUILD_KEY'] if e['NINFER_STAGE']=='build' else None; "
            + observed
            + "binding['observed_contract']=observed if e['NINFER_STAGE']=='smoke' else None; "
            "raw_path=os.path.join(e['NINFER_STAGE_RAW_DIR'],'probe.txt'); "
            "raw_data=e['NINFER_STAGE'].encode(); open(raw_path,'wb').write(raw_data); "
            "required=json.loads(e['NINFER_STAGE_REQUIRED_EVIDENCE']); "
            "binding['evidence']={name:({'path':os.path.relpath(b,e['NINFER_ATTEMPT_DIR']),"
            "'sha256':binary['sha256']} if name=='candidate_binary' else "
            "{'path':os.path.relpath(raw_path,e['NINFER_ATTEMPT_DIR']),"
            "'sha256':(('0'*64) if " + repr(wrong_raw_hash)
            + " else hashlib.sha256(raw_data).hexdigest())}) for name in required}; "
            "binding['semantic_result']='pass' if e['NINFER_STAGE']=='correctness' else None; "
            "binding['task_result']='pass' if e['NINFER_STAGE']=='smoke' else None; "
            "binding['holdout_result']='pass' if e['NINFER_STAGE']=='full' else None; "
            "binding['independent_sessions']=3 if e['NINFER_STAGE']=='screen' else None; "
            "binding['holdout_groups']=2 if e['NINFER_STAGE']=='full' else None; "
            "open(e['NINFER_STAGE_EVIDENCE_PATH'],'w',encoding='utf-8').write(json.dumps(binding))"
        )
    return [sys.executable, "-c", code]


def _policy(*, qualification_enabled: bool = True, omit_report_stage: str | None = None,
            wrong_contract: bool = False, noisy_screen: bool = False,
            wrong_toolchain: bool = False, wrong_raw_hash_stage: str | None = None,
            gpu_stage: str | None = "full",
            contract_override: dict[str, Any] | None = None) -> dict[str, Any]:
    gates = {"correctness": ["semantic"], "smoke": ["task"], "full": ["holdout"]}
    expected_contract = contract_override or _contract()
    criteria_by_stage: dict[str, dict[str, Any]] = {
        "admission": {
            "equals": {"status": "pass"},
            "toolchain": {"cmake": "4.4", "cuda": "13"},
        },
        "build": {
            "equals": {"status": "pass"},
            "build_identity": {"flags": {"build_type": "Release"}, "dependencies": []},
        },
        "correctness": {
            "equals": {"status": "pass", "semantic_result": "pass"},
            "gates": ["semantic"],
        },
        "smoke": {
            "equals": {"status": "pass", "task_result": "pass"},
            "gates": ["task"],
            "observes_contract": True,
        },
        "screen": {
            "equals": {"status": "pass"},
            "minimum": {"independent_sessions": 3},
        },
        "full": {
            "equals": {"status": "pass", "holdout_result": "pass"},
            "minimum": {"holdout_groups": 2},
            "gates": ["holdout"],
        },
    }
    if gpu_stage is not None:
        criteria_by_stage[gpu_stage]["requires_gpu"] = True
        criteria_by_stage[gpu_stage]["gpu_admission"] = {
            "lock_path": "/tmp/opencode/dreamrsi-gpu.lock",
            "production_units": ["fnn4090service.service"],
            "target_gpu_uuid": "GPU-test",
        }
    if noisy_screen:
        criteria_by_stage["screen"]["minimum"] = {"independent_sessions": 3}
    stages = []
    for name in evaluator.STAGE_ORDER:
        required = [f"{name}_evidence"]
        if name == "build":
            required.append("candidate_binary")
        if criteria_by_stage[name].get("observes_contract"):
            required.append("contract_observation")
        criteria = json.loads(json.dumps(criteria_by_stage[name]))
        if name == "admission":
            criteria["toolchain"] = {"cmake": "4.4", "cuda": "13"}
        stage: dict[str, Any] = {
            "name": name,
            "argv": _stage_command(
                name,
                omit_report=omit_report_stage == name,
                status="inconclusive" if noisy_screen and name == "screen" else "pass",
                wrong_contract=wrong_contract,
                wrong_toolchain=wrong_toolchain,
                wrong_raw_hash=wrong_raw_hash_stage == name,
                observed_contract=expected_contract,
            ),
            "timeout_seconds": 10,
            "required_evidence": required,
            "criteria": criteria,
        }
        stages.append(stage)
    policy = {
        "schema_version": 1,
        "policy_id": "0" * 64,
        "contract_id": hashlib.sha256(json.dumps(expected_contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "contract": expected_contract,
        "baseline_id": "baseline-test",
        "corpus_id": "corpus-test",
        "toolchain_id": "toolchain-test",
        "qualification_enabled": qualification_enabled,
        "stages": stages,
    }
    contracts = sys.modules["tools.dreamrsi.contracts"]
    policy["contract_id"] = contracts.canonical_hash(expected_contract)
    policy["policy_id"] = contracts.canonical_hash({key: value for key, value in policy.items()
                                                      if key != "policy_id"})
    return policy


def _rehash_policy(policy: dict[str, Any]) -> None:
    contracts = sys.modules["tools.dreamrsi.contracts"]
    policy["policy_id"] = "0" * 64
    policy["policy_id"] = contracts.canonical_hash({key: value for key, value in policy.items()
                                                      if key != "policy_id"})


@contextmanager
def _admitted_gpu(*args: Any, **kwargs: Any):
    yield {"production_units_before": [], "gpu_inventory": {"available": True}}


def test_success_shadow_records_all_gates_without_promotion(tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    _install_contracts(monkeypatch)
    monkeypatch.setattr(lifecycle, "gpu_admission", _admitted_gpu)
    archive = tmp_path / "candidate.tar"
    archive_sha256 = _archive(archive)
    policy = _policy()
    receipt = evaluator.evaluate_submission(
        _submission(archive, archive_sha256, policy), policy, tmp_path / "runs"
    )

    assert receipt["verdict"] == "qualified"
    assert receipt["stage"] == "full"
    assert receipt["failure_class"] is None
    assert _outcome(receipt)["candidate_performance_eligible"] is True
    assert "not a promotion" in _outcome(receipt)["detail"]
    assert {item.get("stage") for item in _outcome(receipt)["stages"]} == set(evaluator.STAGE_ORDER)
    archive_evidence = next(item for item in receipt["evidence"]
                            if item.get("kind") == "source_archive")
    attempt_dir = Path(archive_evidence["path"]).parent
    assert (attempt_dir / "receipt.json").is_file()
    assert (attempt_dir / "source" / "src-tree" / "CMakeLists.txt").read_text() == (
        "project(frozen_source)\n"
    )


def test_qualification_disabled_never_turns_exit_zero_into_qualified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_contracts(monkeypatch)
    archive = tmp_path / "candidate.tar"
    archive_sha256 = _archive(archive)
    policy = _policy(qualification_enabled=False, gpu_stage=None)
    receipt = evaluator.evaluate_submission(
        _submission(archive, archive_sha256, policy),
        policy,
        tmp_path / "runs",
    )
    assert receipt["verdict"] == "promising"
    assert receipt["failure_class"] is None


def test_missing_host_preflight_evidence_is_infrastructure_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_contracts(monkeypatch)
    archive = tmp_path / "candidate.tar"
    archive_sha256 = _archive(archive)
    policy = _policy(omit_report_stage="admission")
    receipt = evaluator.evaluate_submission(
        _submission(archive, archive_sha256, policy),
        policy,
        tmp_path / "runs",
    )
    assert receipt["verdict"] == "blocked"
    assert receipt["stage"] == "admission"
    assert receipt["failure_class"] == "infrastructure"


def test_missing_candidate_stage_evidence_blocks_zero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_contracts(monkeypatch)
    archive = tmp_path / "candidate.tar"
    archive_sha256 = _archive(archive)
    policy = _policy(omit_report_stage="correctness")
    receipt = evaluator.evaluate_submission(
        _submission(archive, archive_sha256, policy),
        policy,
        tmp_path / "runs",
    )
    assert receipt["verdict"] == "inconclusive"
    assert receipt["stage"] == "correctness"
    assert receipt["failure_class"] is None


def test_unverified_raw_artifact_digest_cannot_qualify_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_contracts(monkeypatch)
    archive = tmp_path / "candidate.tar"
    archive_sha256 = _archive(archive)
    policy = _policy(wrong_raw_hash_stage="correctness")
    receipt = evaluator.evaluate_submission(
        _submission(archive, archive_sha256, policy),
        policy,
        tmp_path / "runs",
    )
    assert receipt["verdict"] == "inconclusive"
    assert receipt["stage"] == "correctness"


def test_wrong_actual_runtime_contract_is_a_candidate_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_contracts(monkeypatch)
    archive = tmp_path / "candidate.tar"
    archive_sha256 = _archive(archive)
    policy = _policy(wrong_contract=True)
    receipt = evaluator.evaluate_submission(
        _submission(archive, archive_sha256, policy),
        policy,
        tmp_path / "runs",
    )
    assert receipt["verdict"] == "regression"
    assert receipt["stage"] == "smoke"
    assert receipt["failure_class"] == "candidate"
    assert any("contract_verification" in item for item in _outcome(receipt)["stages"])


def test_wrong_host_toolchain_is_blocked_as_infrastructure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_contracts(monkeypatch)
    archive = tmp_path / "candidate.tar"
    archive_sha256 = _archive(archive)
    policy = _policy(wrong_toolchain=True)
    receipt = evaluator.evaluate_submission(
        _submission(archive, archive_sha256, policy),
        policy,
        tmp_path / "runs",
    )
    assert receipt["verdict"] == "blocked"
    assert receipt["stage"] == "admission"
    assert receipt["failure_class"] == "infrastructure"


def test_public_model_name_alone_cannot_define_the_expected_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_contracts(monkeypatch)
    archive = tmp_path / "candidate.tar"
    archive_sha256 = _archive(archive)
    policy = _policy(contract_override={"public_model_name": "qwen/qwen3.8-27b"})
    receipt = evaluator.evaluate_submission(
        _submission(archive, archive_sha256, policy), policy, tmp_path / "runs"
    )
    assert receipt["verdict"] == "blocked"
    assert receipt["stage"] == "admission"
    assert receipt["failure_class"] == "infrastructure"
    assert "artifact" in _outcome(receipt)["detail"]


def test_nested_model_and_engine_contract_locks_accept_mixed_weight_formats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_contracts(monkeypatch)
    monkeypatch.setattr(lifecycle, "gpu_admission", _admitted_gpu)
    contract = _contract()
    contract["model_artifact"]["formats"] = [
        "q4_g64_fp16", "q5_g64_fp16", "q8_g32_fp16", "bf16"
    ]
    archive = tmp_path / "candidate.tar"
    archive_sha256 = _archive(archive)
    policy = _policy(contract_override=contract)
    receipt = evaluator.evaluate_submission(
        _submission(archive, archive_sha256, policy),
        policy,
        tmp_path / "runs",
    )
    assert receipt["verdict"] == "qualified"


def test_build_identity_changes_with_trusted_flags_and_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_contracts(monkeypatch)
    archive = tmp_path / "candidate.tar"
    archive_sha256 = _archive(archive)
    dependency = tmp_path / "compiler-wrapper.txt"
    dependency.write_text("wrapper-v1", encoding="utf-8")
    policy_v1 = _policy(qualification_enabled=False, gpu_stage=None)
    build_criteria_v1 = next(stage for stage in policy_v1["stages"] if stage["name"] == "build")["criteria"]
    build_criteria_v1["build_identity"]["dependencies"] = [{
        "path": str(dependency),
        "sha256": hashlib.sha256(dependency.read_bytes()).hexdigest(),
    }]
    policy_v2 = json.loads(json.dumps(policy_v1))
    build_criteria_v2 = next(stage for stage in policy_v2["stages"] if stage["name"] == "build")["criteria"]
    build_criteria_v2["build_identity"]["flags"]["build_type"] = "RelWithDebInfo"
    _rehash_policy(policy_v1)
    _rehash_policy(policy_v2)
    first = evaluator.evaluate_submission(
        _submission(archive, archive_sha256, policy_v1), policy_v1, tmp_path / "runs-v1"
    )
    second = evaluator.evaluate_submission(
        _submission(archive, archive_sha256, policy_v2), policy_v2, tmp_path / "runs-v2"
    )
    assert first["verdict"] == second["verdict"] == "promising"
    first_key = _outcome(first)["build"]["build_key"]
    second_key = _outcome(second)["build"]["build_key"]
    assert first_key != second_key


def test_invalid_archive_paths_are_rejected_without_writing_outside_run_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_contracts(monkeypatch)
    archive = tmp_path / "unsafe.tar"
    archive_sha256 = _archive(archive, unsafe=True)
    outside = tmp_path / "escape.txt"
    policy = _policy()
    receipt = evaluator.evaluate_submission(
        _submission(archive, archive_sha256, policy), policy, tmp_path / "runs"
    )
    assert receipt["verdict"] == "rejected"
    assert receipt["failure_class"] == "candidate"
    assert not outside.exists()


def test_active_production_service_blocks_gpu_stage(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    _install_contracts(monkeypatch)
    archive = tmp_path / "candidate.tar"
    archive_sha256 = _archive(archive)
    policy = _policy()

    def active_service(_units: Any) -> list[dict[str, str]]:
        raise lifecycle.ProductionServiceActive("production is active")

    monkeypatch.setattr(lifecycle, "assert_service_units_inactive", active_service)
    receipt = evaluator.evaluate_submission(
        _submission(archive, archive_sha256, policy), policy, tmp_path / "runs"
    )
    assert receipt["verdict"] == "blocked"
    assert receipt["stage"] == "full"
    assert receipt["failure_class"] == "infrastructure"
    assert "production is active" in _outcome(receipt)["detail"]


def test_noisy_screen_result_is_inconclusive_not_a_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_contracts(monkeypatch)
    archive = tmp_path / "candidate.tar"
    archive_sha256 = _archive(archive)
    policy = _policy(noisy_screen=True, gpu_stage=None)
    receipt = evaluator.evaluate_submission(
        _submission(archive, archive_sha256, policy),
        policy,
        tmp_path / "runs",
    )
    assert receipt["verdict"] == "inconclusive"
    assert receipt["stage"] == "screen"
    assert receipt["failure_class"] is None


def test_missing_holdout_gate_blocks_qualification(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    _install_contracts(monkeypatch)
    monkeypatch.setattr(lifecycle, "gpu_admission", _admitted_gpu)
    archive = tmp_path / "candidate.tar"
    archive_sha256 = _archive(archive)
    policy = _policy()
    policy["stages"][-1]["criteria"]["gates"] = []
    _rehash_policy(policy)
    receipt = evaluator.evaluate_submission(
        _submission(archive, archive_sha256, policy), policy, tmp_path / "runs"
    )
    assert receipt["verdict"] == "inconclusive"
    assert receipt["stage"] == "full"
    assert "holdout" in _outcome(receipt)["detail"]


@pytest.mark.parametrize(
    ("failed_stage", "expected_verdict"),
    [("build", "rejected"), ("correctness", "regression")],
)
def test_candidate_build_and_test_failures_are_not_host_preflight_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_stage: str,
    expected_verdict: str,
) -> None:
    _install_contracts(monkeypatch)
    archive = tmp_path / "candidate.tar"
    archive_sha256 = _archive(archive)
    policy = _policy(qualification_enabled=False, gpu_stage=None)
    target = next(stage for stage in policy["stages"] if stage["name"] == failed_stage)
    target["argv"] = [sys.executable, "-c", "raise SystemExit(7)"]
    _rehash_policy(policy)
    receipt = evaluator.evaluate_submission(
        _submission(archive, archive_sha256, policy), policy, tmp_path / "runs"
    )
    assert receipt["verdict"] == expected_verdict
    assert receipt["stage"] == failed_stage
    assert receipt["failure_class"] == "candidate"


def test_submission_cannot_supply_the_trusted_stage_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_contracts(monkeypatch)
    archive = tmp_path / "candidate.tar"
    archive_sha256 = _archive(archive)
    policy = _policy(qualification_enabled=False, gpu_stage=None)
    submission = _submission(archive, archive_sha256, policy)
    marker = tmp_path / "candidate-command-ran"
    submission["command"] = [sys.executable, "-c", f"open({str(marker)!r},'w').write('ran')"]
    receipt = evaluator.evaluate_submission(
        submission, policy, tmp_path / "runs"
    )
    assert receipt["verdict"] == "rejected"
    assert not marker.exists()


def test_live_endpoint_stage_is_labelled_and_cannot_qualify_candidate_performance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_contracts(monkeypatch)
    archive = tmp_path / "candidate.tar"
    archive_sha256 = _archive(archive)
    policy = _policy(qualification_enabled=False, gpu_stage=None)
    policy["stages"][2]["criteria"]["execution_mode"] = "live_endpoint"
    policy["stages"][2]["criteria"]["label"] = "existing-service-plumbing-only"
    _rehash_policy(policy)
    receipt = evaluator.evaluate_submission(
        _submission(archive, archive_sha256, policy), policy, tmp_path / "runs"
    )
    assert receipt["verdict"] == "promising"
    assert _outcome(receipt)["candidate_performance_eligible"] is False
