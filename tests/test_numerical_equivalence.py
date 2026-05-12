"""Numerical equivalence tests for the SDPA-based ``FlashAttention``.

These tests load reference predictions that were produced with the previous
``flash-attn``-based implementation (see ``tests/generate_fixtures.py``) and
assert that the new ``torch.nn.functional.scaled_dot_product_attention``-based
implementation produces matching outputs when loaded with the same weights.

The tolerances are intentionally fp16-loose: different attention kernels (the
``flash-attn`` CUDA kernel vs. PyTorch's SDPA dispatch) compute the same
math but with slightly different reduction orders, so bit-identity is not
expected.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from yorzoi.config import BorzoiConfig
from yorzoi.model.attn_modules import FlashAttention
from yorzoi.model.borzoi import Borzoi

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Yorzoi inference paths assume CUDA (autocast / fp16).",
)


def _load_fixture(name: str) -> dict:
    path = FIXTURE_DIR / name
    if not path.exists():
        pytest.skip(
            f"Reference fixture {path.name} missing. "
            "Run `python tests/generate_fixtures.py` to regenerate."
        )
    return torch.load(path, weights_only=False, map_location="cpu")


def test_attention_block_matches_flash_attn_reference():
    """Single FlashAttention block: SDPA output matches the flash-attn output."""
    fixture = _load_fixture("attention_block.pt")
    cfg = fixture["config"]

    block = FlashAttention(
        dim=cfg["dim"], heads=cfg["heads"], dropout=0.0, pos_dropout=0.0
    )
    missing, unexpected = block.load_state_dict(fixture["state_dict"], strict=False)
    assert missing == [], f"Unexpected missing keys: {missing}"
    assert unexpected == [], f"Unexpected extra keys: {unexpected}"

    block = block.to("cuda").half().eval()
    x = fixture["input"].to("cuda")
    with torch.no_grad():
        actual = block(x).cpu().float()
    expected = fixture["output"].float()

    diff = (actual - expected).abs()
    assert diff.max().item() < 1e-2, (
        f"Attention block output drifted: max abs diff {diff.max().item():.6f} "
        f"(mean {diff.mean().item():.6f}, reference max {expected.abs().max().item():.6f})"
    )


def test_borzoi_forward_matches_flash_attn_reference():
    """Full Borzoi forward through one transformer block matches reference."""
    fixture = _load_fixture("borzoi_forward.pt")
    cfg = BorzoiConfig(**fixture["config_dict"])

    model = Borzoi(cfg)
    missing, unexpected = model.load_state_dict(fixture["state_dict"], strict=False)
    assert missing == [], f"Unexpected missing keys: {missing}"
    assert unexpected == [], f"Unexpected extra keys: {unexpected}"

    model = model.to("cuda").eval()
    x = fixture["input"].to("cuda")
    with torch.no_grad(), torch.autocast(device_type="cuda"):
        actual = model(x).cpu().float()
    expected = fixture["output"].float()

    diff = (actual - expected).abs()
    # Borzoi output passes through Softplus -> values are smooth and small;
    # 1e-3 absolute is generous given fp16 autocast through the transformer.
    assert diff.max().item() < 1e-3, (
        f"Borzoi forward drifted: max abs diff {diff.max().item():.6f} "
        f"(mean {diff.mean().item():.6f}, reference max {expected.abs().max().item():.6f})"
    )


def test_attention_state_dict_keys_unchanged():
    """The new module's state_dict keys must exactly match the old layout.

    This guards against accidental renames that would break HF checkpoint
    loading (``tom-ellis-lab/yorzoi`` on the Hub).
    """
    fixture = _load_fixture("attention_block.pt")
    cfg = fixture["config"]
    block = FlashAttention(
        dim=cfg["dim"], heads=cfg["heads"], dropout=0.0, pos_dropout=0.0
    )
    assert set(block.state_dict().keys()) == set(fixture["state_dict"].keys())
