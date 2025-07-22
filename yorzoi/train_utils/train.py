import torch
import torch.nn as nn
from yorzoi.config import TrainConfig


def get_optimizer(cfg: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
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


def get_scheduler(
    cfg: TrainConfig, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    from torch.optim.lr_scheduler import (
        StepLR,
        CosineAnnealingLR,
        CosineAnnealingWarmRestarts,
        ConstantLR,
    )

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


def get_criterion(cfg: TrainConfig) -> nn.Module:
    from yorzoi.loss import poisson_multinomial

    def criterion(output, targets):
        return poisson_multinomial(
            output,
            targets,
            poisson_weight=cfg.loss["poisson_weight"],
            epsilon=cfg.loss["epsilon"],
            rescale=False,
            reduction=cfg.loss["reduction"],
        )

    return criterion
