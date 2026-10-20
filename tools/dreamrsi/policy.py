#!/usr/bin/env python3
"""Build a trusted evaluation policy from a compact operator specification.

The policy is the only place that decides what a candidate must pass. It lives
outside candidate source, so this builder is part of the trusted control plane:
it computes the content identities an operator would otherwise have to hash by
hand, and it refuses specifications that cannot be evaluated safely.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from . import contracts


STAGE_ORDER = ("admission", "build", "correctness", "smoke", "screen", "full")
QUALIFICATION_GATES = ("semantic", "task", "holdout")


class PolicyError(ValueError):
    """The operator specification cannot produce an evaluable policy."""


def _stage(spec: Any, name: str) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise PolicyError(f"{name} stage specification must be an object")
    argv = spec.get("argv")
    if not isinstance(argv, list) or not argv or any(
        not isinstance(item, str) or not item or "\0" in item for item in argv
    ):
        raise PolicyError(f"{name} stage argv must be a nonempty string array")
    timeout = spec.get("timeout_seconds", 3600)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise PolicyError(f"{name} stage timeout_seconds must be positive")
    criteria = spec.get("criteria")
    if not isinstance(criteria, dict) or not criteria:
        raise PolicyError(f"{name} stage must define criteria")
    required = spec.get("required_evidence")
    if not isinstance(required, list) or not required:
        raise PolicyError(f"{name} stage must list required evidence names")
    if any(not isinstance(item, str) or not item.strip() for item in required):
        raise PolicyError(f"{name} stage evidence names must be nonempty strings")
    stage = {
        "name": name,
        "argv": list(argv),
        "timeout_seconds": float(timeout),
        "required_evidence": list(required),
        "criteria": json.loads(json.dumps(criteria)),
    }
    if "gates" in spec:
        gates = spec["gates"]
        if not isinstance(gates, list) or any(gate not in QUALIFICATION_GATES for gate in gates):
            raise PolicyError(f"{name} stage gates must be drawn from {QUALIFICATION_GATES}")
        stage["criteria"]["gates"] = list(gates)
    return stage


def build_policy(specification: dict[str, Any]) -> dict[str, Any]:
    """Return a validated policy document from a compact operator specification."""
    if not isinstance(specification, dict):
        raise PolicyError("policy specification must be an object")
    contract = specification.get("contract")
    if not isinstance(contract, dict):
        raise PolicyError("policy specification must carry a contract object")
    stages = specification.get("stages")
    if not isinstance(stages, list):
        raise PolicyError("policy specification must carry a stages list")
    by_name: dict[str, Any] = {}
    for entry in stages:
        if not isinstance(entry, dict) or "name" not in entry:
            raise PolicyError("each stage specification needs a name")
        name = entry["name"]
        if name not in STAGE_ORDER:
            raise PolicyError(f"unsupported stage name: {name!r}")
        if name in by_name:
            raise PolicyError(f"duplicate stage name: {name!r}")
        by_name[name] = entry
    if tuple(by_name) != STAGE_ORDER:
        raise PolicyError(
            "stages must be exactly " + ", ".join(STAGE_ORDER) + " in that order"
        )
    qualification = specification.get("qualification_enabled", False)
    if type(qualification) is not bool:
        raise PolicyError("qualification_enabled must be boolean")
    policy = {
        "schema_version": contracts.SCHEMA_VERSION,
        "policy_id": "0" * 64,
        "contract_id": contracts.canonical_hash(contract),
        "contract": contract,
        "baseline_id": specification.get("baseline_id", "baseline-unnamed"),
        "corpus_id": specification.get("corpus_id", "corpus-unnamed"),
        "toolchain_id": specification.get("toolchain_id", "toolchain-unnamed"),
        "stages": [_stage(by_name[name], name) for name in STAGE_ORDER],
        "qualification_enabled": qualification,
    }
    policy["policy_id"] = contracts.canonical_hash(
        {key: value for key, value in policy.items() if key != "policy_id"}
    )
    return contracts.validate_policy(policy)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build and validate a trusted policy")
    build.add_argument("--spec", required=True, type=Path,
                       help="operator specification JSON with contract and stages")
    build.add_argument("--output", required=True, type=Path)
    show = commands.add_parser("show", help="print a policy identity summary")
    show.add_argument("--policy", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            specification = contracts.read_json(args.spec)
            policy = build_policy(specification)
            contracts.write_json(args.output, policy)
            print(json.dumps({
                "policy_id": policy["policy_id"],
                "contract_id": policy["contract_id"],
                "qualification_enabled": policy["qualification_enabled"],
                "output": str(args.output),
            }, sort_keys=True))
            return 0
        policy = contracts.validate_policy(contracts.read_json(args.policy))
        print(json.dumps({
            "policy_id": policy["policy_id"],
            "contract_id": policy["contract_id"],
            "baseline_id": policy["baseline_id"],
            "corpus_id": policy["corpus_id"],
            "toolchain_id": policy["toolchain_id"],
            "qualification_enabled": policy["qualification_enabled"],
            "stages": [stage["name"] for stage in policy["stages"]],
        }, sort_keys=True))
        return 0
    except (PolicyError, OSError, ValueError) as exc:
        print(f"policy.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
