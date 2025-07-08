import torch
import torch.nn as nn
from yorzoi.config import TrainConfig, BorzoiConfig
from yorzoi.model.baseline import DNAConvNet


def get_model(cfg: TrainConfig) -> nn.Module:
    # Initialize the model
    model = None
    if cfg.model_name == "baseline":
        model = DNAConvNet()
    if cfg.model_name == "yorzoi":
        from yorzoi.model.borzoi import Borzoi

        config = BorzoiConfig(
            dim=cfg.borzoi_cfg["dim"],
            depth=cfg.borzoi_cfg["depth"],
            heads=cfg.borzoi_cfg["heads"],
            resolution=cfg.borzoi_cfg["resolution"],
            return_center_bins_only=cfg.borzoi_cfg["return_center_bins_only"],
            attn_dim_key=cfg.borzoi_cfg["attn_dim_key"],
            attn_dim_value=cfg.borzoi_cfg["attn_dim_value"],
            dropout_rate=cfg.borzoi_cfg["dropout_rate"],
            attn_dropout=cfg.borzoi_cfg["attn_dropout"],
            pos_dropout=cfg.borzoi_cfg["pos_dropout"],
            enable_mouse_head=cfg.borzoi_cfg["enable_mouse_head"],
            flashed=cfg.borzoi_cfg["flashed"],
            separable0=cfg.borzoi_cfg["separable0"],
            separable1=cfg.borzoi_cfg["separable1"],
            head=cfg.borzoi_cfg["head"],
            final_joined_convs=cfg.borzoi_cfg["final_joined_convs"],
            upsampling_unet0=cfg.borzoi_cfg["upsampling_unet0"],
            horizontal_conv0=cfg.borzoi_cfg["horizontal_conv0"],
        )

        model = Borzoi(config)

        if cfg.checkpoint_path:
            model.load_state_dict(torch.load(cfg.checkpoint_path, map_location="cpu"))

        model.to(cfg.device)
    else:
        raise ValueError("Unknown model! Valid options are 'DNAConvNet' and 'clex'")

    return model
