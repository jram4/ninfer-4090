from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.convert.qwen3_6.common.official_resources import (
    OFFICIAL_RESOURCE_SHA256,
    validate_official_resource_hashes,
)
from tools.convert.qwen3_6_27b import convert as convert_27b


MODEL_27B = Path(os.environ.get("NINFER_QWEN3_6_27B_SOURCE", ""))
UNSLOTH_TOKENIZER_SHA256 = (
    "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4"
)


def test_official_27b_source_passes_the_shared_preflight():
    if not os.environ.get("NINFER_QWEN3_6_27B_SOURCE") or not MODEL_27B.is_dir():
        pytest.skip("set NINFER_QWEN3_6_27B_SOURCE to test the official checkpoint")
    resources = convert_27b.load_resources(MODEL_27B)

    assert tuple(resource.name for resource in resources) == tuple(
        OFFICIAL_RESOURCE_SHA256
    )


def test_unsloth_tokenizer_hash_is_rejected():
    hashes = dict(OFFICIAL_RESOURCE_SHA256)
    hashes["frontend/tokenizer.json"] = UNSLOTH_TOKENIZER_SHA256

    with pytest.raises(
        ValueError,
        match=(
            "tokenizer.json.*expected "
            + OFFICIAL_RESOURCE_SHA256["frontend/tokenizer.json"]
            + ".*got "
            + UNSLOTH_TOKENIZER_SHA256
        ),
    ):
        validate_official_resource_hashes(hashes)
