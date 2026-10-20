#!/usr/bin/env python3
"""Directly replay a private OpenAI-chat or Anthropic-Messages capsule.

Replay sends the captured JSON request bytes without semantic rewriting and
strips credentials even if a capsule was manually edited. Redacted capsules do
not contain a replayable request body. Use an explicit local/service endpoint;
this tool does not change client configuration or production services.
"""

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
        DEFAULT_MAX_REQUEST_BYTES,
        DEFAULT_MAX_RESPONSE_BYTES,
        SCHEMA_VERSION,
        Upstream,
        _REQUEST_HOP_HEADERS,
        _RESPONSE_ID_HEADERS,
        _SECRET_HEADER,
        _capsule_hash,
        _has_secret_query,
        _valid_capture_route,
        _safe_headers,
    )
except ModuleNotFoundError:  # Support direct `python tools/dreamrsi/replay.py ...` execution.
    script_dir = Path(__file__).resolve().parent
    if (script_dir / "capture.py").is_file():
        # FNN research deployments may place capture.py and replay.py together
        # in one owner-only tools directory instead of preserving the repo tree.
        sys.path.insert(0, str(script_dir))
        from capture import (  # type: ignore[no-redef]
            DEFAULT_MAX_REQUEST_BYTES,
            DEFAULT_MAX_RESPONSE_BYTES,
            SCHEMA_VERSION,
            Upstream,
            _REQUEST_HOP_HEADERS,
            _RESPONSE_ID_HEADERS,
            _SECRET_HEADER,
            _capsule_hash,
            _has_secret_query,
            _valid_capture_route,
            _safe_headers,
        )
    else:
        sys.path.insert(0, str(script_dir.parents[1]))
        from tools.dreamrsi.capture import (  # type: ignore[no-redef]
            DEFAULT_MAX_REQUEST_BYTES,
            DEFAULT_MAX_RESPONSE_BYTES,
            SCHEMA_VERSION,
            Upstream,
            _REQUEST_HOP_HEADERS,
            _RESPONSE_ID_HEADERS,
            _SECRET_HEADER,
            _capsule_hash,
            _has_secret_query,
            _valid_capture_route,
            _safe_headers,
        )


READ_CHUNK_BYTES = 64 * 1024
_METHOD_RE = re.compile(r"^[A-Z]{1,16}$")
_CAPSULE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,200}$")
_SAFE_HEADER_VALUE = re.compile(r"^[A-Za-z0-9!#$&^_.+:/;= -]{1,160}$")


class ReplayError(RuntimeError):
    """Invalid capsule or replay transport failure."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_capsule(capsule_dir: str | Path) -> tuple[Path, dict[str, Any], bytes]:
    capsule = Path(capsule_dir).expanduser().resolve(strict=True)
    if not capsule.is_dir():
        raise ReplayError("capsule path is not a directory")
    manifest_path = capsule / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ReplayError("capsule manifest is not a regular file")
    if manifest_path.stat().st_size > 2 * 1024 * 1024:
        raise ReplayError("capsule manifest exceeds the validation size limit")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayError("capsule manifest could not be read") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ReplayError("unsupported capsule schema")
    capsule_id = manifest.get("capsule_id")
    if not isinstance(capsule_id, str) or not _CAPSULE_ID_RE.fullmatch(capsule_id):
        raise ReplayError("capsule ID is invalid")
    origin = manifest.get("upstream_origin")
    if not isinstance(origin, str):
        raise ReplayError("capsule upstream origin is invalid")
    try:
        Upstream.parse(origin)
    except ValueError as exc:
        raise ReplayError("capsule upstream origin is invalid") from exc
    preliminary_request = manifest.get("request")
    if isinstance(preliminary_request, dict):
        preliminary_target = preliminary_request.get("target")
        if isinstance(preliminary_target, str) and _has_secret_query(preliminary_target):
            raise ReplayError("capsule contains a credential-like query parameter")
    for section_name in ("request", "response"):
        section = manifest.get(section_name)
        headers = section.get("headers") if isinstance(section, dict) else None
        if headers is None:
            continue
        if not isinstance(headers, list) or len(headers) > 256:
            raise ReplayError(f"capsule {section_name} headers are invalid")
        for item in headers:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise ReplayError(f"capsule {section_name} headers are invalid")
            lowered = item["name"].lower()
            if lowered in {"cookie2", "set-cookie2"} or _SECRET_HEADER.search(lowered):
                raise ReplayError("capsule contains a credential header")
    recorded_capsule_hash = manifest.get("capsule_sha256")
    if not isinstance(recorded_capsule_hash, str) or _capsule_hash(manifest) != recorded_capsule_hash:
        raise ReplayError("capsule manifest hash does not match")
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
    if not _valid_capture_route(method, target):
        raise ReplayError("capsule request route is not supported")
    if request.get("identity_scope") != "exact_body_no_credentials":
        raise ReplayError("redacted or unknown request identity cannot be replayed")
    if not isinstance(body_name, str) or Path(body_name).name != body_name:
        raise ReplayError("capsule request body path is invalid")
    body_path = capsule / body_name
    if body_path.is_symlink() or not body_path.is_file():
        raise ReplayError("capsule request body is not a regular file")
    if body_path.stat().st_size > DEFAULT_MAX_REQUEST_BYTES:
        raise ReplayError("capsule request body exceeds the replay size limit")
    try:
        body = body_path.read_bytes()
    except OSError as exc:
        raise ReplayError("capsule request body could not be read") from exc
    body_hash = request.get("body_sha256")
    if not isinstance(body_hash, str) or _sha256(body) != body_hash:
        raise ReplayError("capsule request body hash does not match")
    if request.get("body_bytes") != len(body):
        raise ReplayError("capsule request body length does not match")
    raw_headers = request.get("headers")
    if not isinstance(raw_headers, list) or len(raw_headers) > 256:
        raise ReplayError("capsule request headers are invalid")
    pairs: list[tuple[str, str]] = []
    for item in raw_headers:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not isinstance(item.get("value"), str)
            or len(item["name"]) > 256
            or len(item["value"]) > 8192
        ):
            raise ReplayError("capsule request headers are invalid")
        pairs.append((item["name"], item["value"]))
    safe_headers = _safe_headers(pairs, excluded=_REQUEST_HOP_HEADERS)
    content_types = [item["value"] for item in safe_headers if item["name"].lower() == "content-type"]
    if len(content_types) != 1 or "json" not in content_types[0].lower():
        raise ReplayError("capsule request is not marked as JSON")
    try:
        request_json = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayError("capsule request body is not valid JSON") from exc
    if not isinstance(request_json, dict):
        raise ReplayError("capsule request body must be a JSON object")

    response = manifest.get("response")
    if not isinstance(response, dict):
        raise ReplayError("capsule response is missing")
    response_body_name = response.get("body_file")
    if response_body_name is not None:
        if not isinstance(response_body_name, str) or Path(response_body_name).name != response_body_name:
            raise ReplayError("capsule response body path is invalid")
        response_body_path = capsule / response_body_name
        if response_body_path.is_symlink() or not response_body_path.is_file():
            raise ReplayError("capsule response body is not a regular file")
        if response_body_path.stat().st_size > DEFAULT_MAX_RESPONSE_BYTES:
            raise ReplayError("capsule response body exceeds the validation size limit")
        response_body = response_body_path.read_bytes()
        if len(response_body) != response.get("body_bytes") or _sha256(response_body) != response.get("body_sha256"):
            raise ReplayError("capsule response body hash or length does not match")
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
    safe_headers = _safe_headers(
        [(item["name"], item["value"]) for item in request["headers"]],
        excluded=_REQUEST_HOP_HEADERS,
    )
    upstream = Upstream.parse(upstream_url)
    if _has_secret_query(request["target"]):
        raise ReplayError("credential-like query parameters are not replayed")
    target = upstream.target(request["target"])
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
            if value and _CAPSULE_ID_RE.fullmatch(value):
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
            "capsule_sha256": manifest.get("capsule_sha256"),
            "status": response.status,
            "response_identity": response_identity,
            "response_content_type": (
                response_map.get("content-type")
                if isinstance(response_map.get("content-type"), str)
                and _SAFE_HEADER_VALUE.fullmatch(response_map["content-type"])
                else None
            ),
            "response_bytes": response_bytes,
            "response_sha256": digest.hexdigest(),
            "complete": complete,
        }
        if output_path is not None:
            result["response_output"] = str(output_path)
        return result
    except (OSError, http.client.HTTPException, TimeoutError, ValueError) as exc:
        raise ReplayError(f"direct replay failed: {type(exc).__name__}") from exc
    finally:
        conn.close()
        if output_handle is not None:
            output_handle.close()
        if temporary_output is not None and temporary_output.exists():
            temporary_output.unlink()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Example: python -m tools.dreamrsi.replay /private/id.capsule "
            "--upstream http://127.0.0.1:18081"
        ),
    )
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
