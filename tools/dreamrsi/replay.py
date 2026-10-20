#!/usr/bin/env python3
"""Directly replay a Dream-RSI capsule to an HTTP model-service endpoint."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

try:
    from tools.dreamrsi.capture import (
        SCHEMA_VERSION,
        Upstream,
        _REQUEST_HOP_HEADERS,
        _RESPONSE_ID_HEADERS,
        _has_secret_query,
        _safe_headers,
    )
except ModuleNotFoundError:  # Support direct `python tools/dreamrsi/replay.py ...` execution.
    script_dir = Path(__file__).resolve().parent
    if (script_dir / "capture.py").is_file():
        # FNN research deployments may place capture.py and replay.py together
        # in one owner-only tools directory instead of preserving the repo tree.
        sys.path.insert(0, str(script_dir))
        from capture import (  # type: ignore[no-redef]
            SCHEMA_VERSION,
            Upstream,
            _REQUEST_HOP_HEADERS,
            _RESPONSE_ID_HEADERS,
            _has_secret_query,
            _safe_headers,
        )
    else:
        sys.path.insert(0, str(script_dir.parents[1]))
        from tools.dreamrsi.capture import (  # type: ignore[no-redef]
            SCHEMA_VERSION,
            Upstream,
            _REQUEST_HOP_HEADERS,
            _RESPONSE_ID_HEADERS,
            _has_secret_query,
            _safe_headers,
        )


READ_CHUNK_BYTES = 64 * 1024
_METHOD_RE = re.compile(r"^[A-Z]{1,16}$")


class ReplayError(RuntimeError):
    """Invalid capsule or replay transport failure."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_capsule(capsule_dir: str | Path) -> tuple[Path, dict[str, Any], bytes]:
    capsule = Path(capsule_dir).expanduser().resolve(strict=True)
    manifest_path = capsule / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayError("capsule manifest could not be read") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ReplayError("unsupported capsule schema")
    request = manifest.get("request")
    if not isinstance(request, dict):
        raise ReplayError("capsule request is missing")
    method = request.get("method")
    target = request.get("target")
    body_name = request.get("body_file")
    if not isinstance(method, str) or not _METHOD_RE.fullmatch(method):
        raise ReplayError("capsule request method is invalid")
    if not isinstance(target, str) or not target.startswith("/") or target.startswith("//"):
        raise ReplayError("capsule request target is invalid")
    if "\r" in target or "\n" in target:
        raise ReplayError("capsule request target is invalid")
    if not isinstance(body_name, str) or Path(body_name).name != body_name:
        raise ReplayError("capsule request body path is invalid")
    try:
        body = (capsule / body_name).read_bytes()
    except OSError as exc:
        raise ReplayError("capsule request body could not be read") from exc
    body_hash = request.get("body_sha256")
    if not isinstance(body_hash, str) or _sha256(body) != body_hash:
        raise ReplayError("capsule request body hash does not match")
    if request.get("body_bytes") != len(body):
        raise ReplayError("capsule request body length does not match")
    return capsule, manifest, body


def replay_capsule(
    capsule_dir: str | Path,
    *,
    upstream_url: str,
    response_output: str | Path | None = None,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Send one captured request directly to a model-service endpoint.

    The replay omits credential and hop-by-hop headers even if a capsule was
    edited by hand. Response bytes are hashed in memory; an optional output file
    is written atomically with owner-only permissions.
    """
    if timeout <= 0:
        raise ReplayError("timeout must be positive")
    capsule, manifest, body = _load_capsule(capsule_dir)
    request = manifest["request"]
    upstream = Upstream.parse(upstream_url)
    if _has_secret_query(request["target"]):
        raise ReplayError("credential-like query parameters are not replayed")
    target = upstream.target(request["target"])
    raw_headers = request.get("headers", [])
    if not isinstance(raw_headers, list):
        raise ReplayError("capsule request headers are invalid")
    pairs: list[tuple[str, str]] = []
    for item in raw_headers:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not isinstance(item.get("value"), str):
            raise ReplayError("capsule request headers are invalid")
        pairs.append((item["name"], item["value"]))
    safe_headers = _safe_headers(pairs, excluded=_REQUEST_HOP_HEADERS)

    output_path: Path | None = Path(response_output).expanduser().resolve() if response_output else None
    output_handle = None
    temporary_output: Path | None = None
    if output_path is not None:
        output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".partial", dir=output_path.parent)
        os.fchmod(fd, 0o600)
        output_handle = os.fdopen(fd, "wb")
        temporary_output = Path(tmp_name)

    digest = hashlib.sha256()
    response_bytes = 0
    conn = upstream.connect(timeout)
    try:
        conn.putrequest(request["method"], target, skip_host=True, skip_accept_encoding=True)
        for item in safe_headers:
            conn.putheader(item["name"], item["value"])
        host = f"[{upstream.host}]" if ":" in upstream.host else upstream.host
        default_port = 443 if upstream.scheme == "https" else 80
        if upstream.port != default_port:
            host = f"{host}:{upstream.port}"
        conn.putheader("Host", host)
        conn.putheader("Content-Length", str(len(body)))
        conn.putheader("Connection", "close")
        conn.endheaders(body)
        response = conn.getresponse()
        response_headers = response.getheaders()
        response_map = {name.lower(): value for name, value in response_headers}
        response_identity: dict[str, str] = {}
        for name in _RESPONSE_ID_HEADERS:
            value = response_map.get(name)
            if value:
                response_identity = {"header_name": name, "header": value}
                break
        while True:
            try:
                chunk = response.read1(READ_CHUNK_BYTES)
            except http.client.IncompleteRead as exc:
                chunk = exc.partial
                if chunk:
                    digest.update(chunk)
                    response_bytes += len(chunk)
                    if output_handle is not None:
                        output_handle.write(chunk)
                complete = False
                break
            if not chunk:
                complete = True
                break
            digest.update(chunk)
            response_bytes += len(chunk)
            if output_handle is not None:
                output_handle.write(chunk)
        if output_handle is not None:
            output_handle.flush()
            os.fsync(output_handle.fileno())
            output_handle.close()
            output_handle = None
            assert temporary_output is not None and output_path is not None
            temporary_output.replace(output_path)
            os.chmod(output_path, 0o400)
        result = {
            "schema_version": SCHEMA_VERSION,
            "capsule_id": manifest.get("capsule_id"),
            "status": response.status,
            "response_identity": response_identity,
            "response_content_type": response_map.get("content-type"),
            "response_bytes": response_bytes,
            "response_sha256": digest.hexdigest(),
            "complete": complete,
        }
        if output_path is not None:
            result["response_output"] = str(output_path)
        return result
    except (OSError, http.client.HTTPException, TimeoutError) as exc:
        raise ReplayError(f"direct replay failed: {type(exc).__name__}") from exc
    finally:
        conn.close()
        if output_handle is not None:
            output_handle.close()
        if temporary_output is not None and temporary_output.exists():
            temporary_output.unlink()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule", type=Path, help="Restricted capsule directory")
    parser.add_argument("--upstream", required=True, help="Model-service origin, e.g. http://127.0.0.1:18081")
    parser.add_argument("--response-output", type=Path, help="Optional restricted path for replay response bytes")
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        result = replay_capsule(
            args.capsule,
            upstream_url=args.upstream,
            response_output=args.response_output,
            timeout=args.timeout,
        )
    except (OSError, ValueError, ReplayError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
