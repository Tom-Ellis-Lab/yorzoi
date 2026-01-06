import torch
import torch.nn as nn


def _randomise_tracks(outputs: torch.Tensor, targets: torch.Tensor):
    """Apply the **same** random permutation to outputs & targets.

    The permutation keeps the upper and lower halves separate to avoid mixing
    5'->3' with 3'->5' tracks.

    Args:
        outputs: (B, C, L) tensor predicted by the model.
        targets: (B, C, L) tensor of ground-truth tracks.

    Returns:
        Tuple[Tensor, Tensor] with the permuted outputs & targets.
    """

    # We only permute if there is a genuine channel dimension to shuffle.
    if outputs.dim() != 3:
        return outputs, targets

    n_tracks = outputs.size(1)
    if n_tracks < 2:
        return outputs, targets

    assert n_tracks % 2 == 0, f"Expected an even number of tracks, got {n_tracks}."

    half = n_tracks // 2
    device = outputs.device
    perm_upper = torch.randperm(half, device=device)
    perm_lower = torch.randperm(half, device=device) + half
    perm = torch.cat((perm_upper, perm_lower))

    outputs = outputs[:, perm, :]
    targets = targets[:, perm, :]
    return outputs, targets


def _freeze_backbone(model: nn.Module):
    """Freeze all layers except the prediction head(s).

    Works for the Borzoi model wrapped as ``nn.Sequential(TargetLengthCrop, Borzoi)``
    and for the simple ``DNAConvNet`` baseline.
    """
    from yorzoi.model.baseline import DNAConvNet

    # blanket-freeze
    for p in model.parameters():
        p.requires_grad = False

    heads = []

    # Wrapped Borzoi
    if isinstance(model, nn.Sequential) and len(model) == 2:
        borzoi = model[1]
        if hasattr(borzoi, "human_head"):
            heads.append(borzoi.human_head)
        if getattr(borzoi, "enable_mouse_head", False):
            heads.append(borzoi.mouse_head)

    # Baseline network
    if isinstance(model, DNAConvNet):
        heads.append(model.dense)

    # Re-enable gradients for the heads
    for head in heads:
        for p in head.parameters():
            p.requires_grad = True
