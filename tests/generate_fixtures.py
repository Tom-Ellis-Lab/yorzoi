"""Generate reference predictions from the flash-attn-based Yorzoi.

Run this script ONCE while ``flash-attn`` is still installed, before swapping the
attention implementation. The committed fixtures are then used by
``tests/test_numerical_equivalence.py`` to verify the new SDPA-based
implementation produces matching outputs.

The fixtures are intentionally generated from a *small* Borzoi config (depth=1,
short input) so they stay under ~5 MB and can be committed.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from yorzoi.config import BorzoiConfig
from yorzoi.model.attn_modules import FlashAttention
from yorzoi.model.borzoi import Borzoi

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = "cuda"


def small_borzoi_config() -> BorzoiConfig:
    """Production-shape attention with the rest of the model shrunk for size."""
    return BorzoiConfig(
        dim=512,
        resolution=10,
        depth=1,
        heads=4,
        return_center_bins_only=True,
        attn_dim_key=64,
        attn_dim_value=192,
        dropout_rate=0.0,
        attn_dropout=0.0,
        pos_dropout=0.0,
        enable_mouse_head=False,
        enable_human_head=True,
        flashed=True,
        horizontal_conv0={"in_channels": 448, "out_channels": 324},
        upsampling_unet0={"in_channels": 324, "out_channels": 324},
        separable1={
            "conv1d": {"out_channels": 324},
            "separ_conv": {"in_channels": 512, "out_channels": 512},
        },
        separable0={"in_channels": 324, "out_channels": 324},
        final_joined_convs={"in_channels": 324, "out_channels": 16},
        head={"in_channels": 16, "out_channels": 16},
    )


def generate_attention_fixture() -> None:
    """FlashAttention block in isolation: same dim/heads/rotary as production."""
    torch.manual_seed(SEED)
    dim, heads, seq_len, batch = 512, 4, 32, 2

    block = FlashAttention(dim=dim, heads=heads, dropout=0.0, pos_dropout=0.0)
    block = block.to(DEVICE).half().eval()

    torch.manual_seed(SEED + 1)
    x = torch.randn(batch, seq_len, dim, device=DEVICE, dtype=torch.float16)

    with torch.no_grad():
        y = block(x)

    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in block.state_dict().items()},
            "input": x.detach().cpu(),
            "output": y.detach().cpu(),
            "config": {"dim": dim, "heads": heads, "seq_len": seq_len, "batch": batch},
        },
        FIXTURE_DIR / "attention_block.pt",
    )
    print(f"  attention_block.pt: output shape {tuple(y.shape)}, dtype {y.dtype}")


def generate_borzoi_fixture() -> None:
    """Full Borzoi forward, small config, autocast fp16 to match real inference."""
    torch.manual_seed(SEED)
    cfg = small_borzoi_config()
    model = Borzoi(cfg).to(DEVICE).eval()

    seq_len = 4992
    torch.manual_seed(SEED + 2)
    x = torch.zeros(1, seq_len, 4, device=DEVICE)
    idx = torch.randint(0, 4, (1, seq_len), device=DEVICE)
    x.scatter_(2, idx.unsqueeze(-1), 1.0)

    with torch.no_grad(), torch.autocast(device_type="cuda"):
        y = model(x)

    torch.save(
        {
            "config_dict": cfg.to_dict(),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "input": x.detach().cpu(),
            "output": y.detach().cpu().float(),
        },
        FIXTURE_DIR / "borzoi_forward.pt",
    )
    print(f"  borzoi_forward.pt: output shape {tuple(y.shape)}, dtype {y.dtype}")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (flash-attn only runs on CUDA).")
    print(f"Generating fixtures into {FIXTURE_DIR} ...")
    generate_attention_fixture()
    generate_borzoi_fixture()
    print("Done.")


if __name__ == "__main__":
    main()
