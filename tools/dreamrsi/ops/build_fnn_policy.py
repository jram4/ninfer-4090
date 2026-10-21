#!/usr/bin/env python3
"""Build the trusted FNN evaluation policy for a corpus plan.

The policy is what a candidate must pass. It is generated here so the operator
never hand-writes a policy, and so the stage commands, the corpus plan, and the
baseline rows always refer to the same plan identity.

The baseline rows are produced by replaying the same plan against the running
qualified release, so the two arms differ only in the server that answered.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.dreamrsi import contracts, policy as policy_builder  # noqa: E402


DEFAULT_REMOTE_ROOT = "/home/aramirezfamily/.local/share/ninfer-bench"
REMOTE_PYTHON = ("/home/aramirezfamily/.local/share/uv/python/"
                 "cpython-3.11-linux-x86_64-gnu/bin/python3.11")


def build_specification(args: argparse.Namespace) -> dict:
    plan = args.plan
    # The stage runs on the evaluation host, so the plan is referenced by the
    # path it has there.
    remote_plan = args.remote_plan or str(Path(DEFAULT_REMOTE_ROOT) / "corpus-plan-screen.json")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    stages = [
        {
            "name": "admission",
            "argv": [REMOTE_PYTHON, "{trusted_harness}/stage.py", "toolchain"],
            "timeout_seconds": 300,
            "required_evidence": ["toolchain_evidence"],
            "criteria": {"equals": {"status": "pass"},
                         "toolchain": {"cmake": "4.4", "cuda": "13"}},
        },
        {
            "name": "build",
            "argv": [REMOTE_PYTHON, "{trusted_harness}/project_stages.py", "build"],
            "timeout_seconds": 10800,
            "required_evidence": ["build_evidence", "candidate_binary"],
            "criteria": {
                "equals": {"status": "pass"},
                "build_identity": {
                    "flags": {"build_type": "Release", "cuda_architectures": "89",
                              "apps": "ON", "benchmarks": "ON", "testing": "ON"},
                    "dependencies": [],
                },
            },
        },
        {
            "name": "correctness",
            "argv": [REMOTE_PYTHON, "{trusted_harness}/project_stages.py", "correctness"],
            "timeout_seconds": 7200,
            "required_evidence": ["correctness_evidence"],
            "gates": ["semantic"],
            "criteria": {"equals": {"status": "pass", "semantic_result": "pass"}},
        },
        {
            "name": "smoke",
            "argv": [REMOTE_PYTHON, "{trusted_harness}/project_stages.py", "smoke"],
            "timeout_seconds": 1800,
            "required_evidence": ["smoke_evidence", "contract_observation"],
            "gates": ["task"],
            # This is the one stage whose observed contract is compared with
            # the trusted contract.
            "criteria": {"equals": {"status": "pass", "task_result": "pass"},
                         "observes_contract": True},
        },
    ]
    for name, split, evidence, gate, extra in (
        ("screen", "screen", "screen_evidence", None,
         {"minimum": {"independent_sessions": 8},
          "thresholds": {"NINFER_MIN_PAIRS": "12", "NINFER_MIN_SESSIONS": "8"}}),
        ("full", "holdout", "full_evidence", "holdout",
         {"minimum": {"holdout_groups": 2},
          "thresholds": {"NINFER_MIN_PAIRS": "8", "NINFER_MIN_SESSIONS": "4"}}),
    ):
        stage = {
            "name": name,
            "argv": [
                REMOTE_PYTHON, "{trusted_harness}/project_stages.py", name,
                "--plan", remote_plan,
                "--capsule-root", "/",
                "--binary", "{build_dir}/apps/ninfer-serve",
                "--baseline-binary", args.baseline_binary,
                "--rows", "{run_dir}/rows.jsonl",
            ],
            "timeout_seconds": 14400,
            "required_evidence": [evidence],
            # A fixed generation seed, so the two arms perform the same work.
            "criteria": {"equals": {"status": "pass"},
                         "env": {"NINFER_EVAL_SEED": args.seed, **extra.pop("thresholds", {})},
                         **extra},
        }
        if gate:
            stage["gates"] = [gate]
            stage["criteria"]["equals"]["holdout_result"] = "pass"
        if name == "screen":
            stage["criteria"]["minimum"] = {"independent_sessions": 8}
        stages.append(stage)
    return {
        "contract": contract,
        "baseline_id": args.baseline_id,
        "corpus_id": args.corpus_id,
        "toolchain_id": "fnn-cuda13-cmake44",
        "qualification_enabled": args.qualification_enabled,
        "stages": stages,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--baseline-binary", required=True)
    parser.add_argument("--remote-plan", default="",
                        help="path of the corpus plan on the evaluation host")
    parser.add_argument("--harness", default="",
                        help="ignored; the transport resolves the trusted harness at run time")
    parser.add_argument("--baseline-id", default="baseline-sampled-k7-70ebb12")
    parser.add_argument("--corpus-id", default="")
    parser.add_argument("--seed", default="7008",
                        help="fixed generation seed used by both arms of every comparison")
    parser.add_argument("--qualification-enabled", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        # Check the policy against the evaluator's own rules before it is
        # used, so a wiring mistake does not cost a full evaluation run.
        from tools.dreamrsi import evaluator as evaluator_module

        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        args.corpus_id = args.corpus_id or plan["corpus_id"]
        specification = build_specification(args)
        built = policy_builder.build_policy(specification)
        evaluator_module._policy_stages(built)
        contracts.write_json(args.output, built)
        print(json.dumps({
            "policy_id": built["policy_id"],
            "contract_id": built["contract_id"],
            "corpus_id": built["corpus_id"],
            "qualification_enabled": built["qualification_enabled"],
            "stages": [stage["name"] for stage in built["stages"]],
            "output": str(args.output),
        }, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"build_fnn_policy.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
