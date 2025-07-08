import torch
import torch.nn as nn
from yorzoi.config import TrainConfig
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ConstantLR,
)


def pick_optimizer(cfg: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    if cfg.optimizer["method"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer["lr"])
        print("Using Adam.")
    elif cfg.optimizer["method"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer["lr"],
            weight_decay=cfg.optimizer["weight_decay"],
        )
        print("Using AdamW.")
    else:
        raise ValueError("Unknown optimizer! Available options are 'adam' and 'adamw'.")

    return optimizer


def pick_scheduler(
    cfg: TrainConfig, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    if cfg.scheduler == "steplr":
        scheduler = StepLR(optimizer, step_size=18, gamma=0.1)
    elif cfg.scheduler == "cosineannealinglr":
        scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    elif cfg.scheduler == "cosineannealingwr":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    elif cfg.scheduler == "constant":
        scheduler = ConstantLR(optimizer, factor=1, total_iters=1)
    else:
        raise ValueError(
            "Unknown scheduler value. Available options are 'steplr', 'cosineannealinglr', 'cosineannealingwr', and 'constant'."
        )

    return scheduler
