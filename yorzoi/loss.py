import torch
from torch.nn import functional as F


def poisson_multinomial(
    y_pred: torch.Tensor,  # Shape: bs x tracks x seq
    y_true: torch.Tensor,  # Shape: bs x tracks x seq
    poisson_weight: float = 1,
    epsilon: float = 1e-7,
    rescale: bool = False,
    reduction: str = "sum",
    resolution: int = 10,
):
    """Poisson decomposition with multinomial specificity term.

    Args:
        total_weight (float): Weight of the Poisson total term.
        epsilon (float): Added small value to avoid log(0).
        rescale (bool): Rescale loss after re-weighting.
    """
    assert (
        y_pred.shape == y_true.shape
    ), f"y_pred.shape: {y_pred.shape} but y_true.shape: {y_true.shape}"
    if torch.isnan(y_pred).any():
        print("NaN detected in model output")
        raise ValueError("NaN in model output")

    # Create mask for valid tracks (no negative values)
    valid_track_mask = ~torch.all(y_true == -resolution, dim=-1)  # Shape: bs x tracks

    # Sum across lengths (only for valid values)
    s_true = y_true.sum(dim=-1, keepdim=True) + epsilon  # Shape: (bs, tracks, 1)
    s_pred = y_pred.sum(dim=-1, keepdim=True) + epsilon  # Shape: (bs, tracks, 1)

    # Poisson term (on all data, we'll mask later)
    poisson_term = F.poisson_nll_loss(
        s_pred.squeeze(-1), s_true.squeeze(-1), reduction="none", log_input=False
    )  # Shape: (bs, tracks)

    # Multinomial term (on all data, we'll mask later)
    p_pred = y_pred / s_pred  # Shape: (bs, track, seq)
    multinomial_term = -torch.sum(
        y_true * torch.log(p_pred + epsilon), dim=-1
    )  # Shape: (bs, tracks)

    # Apply mask to both terms - zeroing out invalid tracks
    multinomial_term = multinomial_term * valid_track_mask
    poisson_term = poisson_term * valid_track_mask

    # Combine terms
    loss_raw = multinomial_term + poisson_weight * poisson_term

    # Optional rescaling
    if rescale:
        loss_rescale = loss_raw * 2 / (1 + poisson_weight)
    else:
        loss_rescale = loss_raw

    # Count number of valid tracks for proper normalization
    valid_track_count = valid_track_mask.sum()

    # Loss reduction
    if reduction == "sum":
        return loss_rescale.sum(), loss_rescale, multinomial_term, poisson_term
    elif reduction == "mean":
        # Return mean only over valid tracks
        if valid_track_count > 0:
            return (
                loss_rescale.sum() / valid_track_count,
                loss_rescale,
                multinomial_term.sum() / valid_track_count,
                poisson_term.sum() / valid_track_count,
            )
        else:
            raise ValueError("No valid tracks!")
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    y_pred = torch.randint(0, 100, (1, 4, 20)).float()
    y_true = torch.randint(0, 100, (1, 4, 20)).float()
    total_weight = 1
    epsilon = 1e-7
    rescale = False
    print("Loss:", poisson_multinomial(y_pred, y_true, total_weight, epsilon, rescale))
