#!/usr/bin/env python3
"""Relabel the stratum of captured descriptors.

The stratum is an operator-assigned label for a traffic class. A window's
session label is not the same thing: several sessions can belong to one
stratum, and a stratum needs more than one session before a holdout can be
drawn from it.

This rewrites only the ``stratum`` field of a sanitized descriptor manifest.
Request bodies, hashes, sequence numbers, session labels, and every other
field are left untouched, and the before/after summary is printed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from collections import Counter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--map", required=True, type=Path,
                        help="JSON object mapping session_id to stratum")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        mapping = json.loads(args.map.read_text(encoding="utf-8"))
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("the map must be a nonempty JSON object")
        rows = []
        for line in args.manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("every descriptor line must be an object")
            rows.append(row)
        before = Counter(row.get("stratum") for row in rows)
        changed = 0
        for row in rows:
            session = row.get("session_id")
            if session in mapping:
                if row.get("stratum") != mapping[session]:
                    changed += 1
                row["stratum"] = mapping[session]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        after = Counter(row.get("stratum") for row in rows)
        print(json.dumps({
            "descriptors": len(rows),
            "relabelled": changed,
            "before": dict(before),
            "after": dict(after),
            "sessions_per_stratum": {
                stratum: len({row.get("session_id") for row in rows if row.get("stratum") == stratum})
                for stratum in sorted(after)
            },
        }, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"relabel_manifest.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
