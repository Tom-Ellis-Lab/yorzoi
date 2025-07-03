import torch
from typing import Literal
from yorzoi.constants import onehot2nucleotide


def _borzoi_transform(x: torch.Tensor):
    expd = torch.pow(input=x, exponent=0.75)

    return torch.where(
        expd <= 384, expd, torch.minimum(expd, 384 + torch.sqrt(expd - 384))
    )


def _borzoi_transform_inv(y: torch.Tensor):
    # Step 1: Undo the min operation
    expd = torch.where(y <= 384, y, 384 + torch.pow(y - 384, 2))

    # Step 2: Undo the power operation
    x = torch.pow(expd, 1 / 0.75)

    return x


# Bin y_true into 4bp bins
def _bin_tensor(y_true):
    BIN_SIZE = 4

    bs, tracks, seq_len = y_true.shape

    # Ensure the sequence length is divisible by bin_size
    if seq_len % BIN_SIZE != 0:
        raise ValueError(
            f"Sequence length ({seq_len}) must be divisible by bin_size ({BIN_SIZE})"
        )

    # Reshape to group every 4 consecutive values
    y_reshaped = y_true.reshape(bs, tracks, seq_len // BIN_SIZE, BIN_SIZE)

    # Sum over the bin_size dimension
    y_binned = y_reshaped.sum(dim=-1)

    return y_binned


def _unbin_tensor(y_binned, resolution):
    bs, tracks, binned_len = y_binned.shape

    # Expand the binned tensor to the original sequence length
    y_unbinned = y_binned.repeat_interleave(resolution, dim=-1)

    # Divide by four to get the average value
    y_unbinned /= resolution

    return y_unbinned


# Convert true to pred (no cropping)
def bin_then_transform(y_true):
    return _borzoi_transform(_bin_tensor(y_true))


def untransform_then_unbin(y_pred, resolution):
    return _unbin_tensor(_borzoi_transform_inv(y_pred), resolution=resolution)


def build_hom_graph_seq_id(
    chrom: str, strand: Literal["+", "-"], start_sample: int, end_sample: int
):
    """In the homology graph, nodes are named like this:
    'III_+_100299_105298.csv' which this function constructs
    from the arguments.

    Args:
        chrom (str): name of the chromosome, roman numerals
        strand (Literal[): +/-
        start_sample (_type_): _description_
        end_sample (_type_

    Raises:
        NotImplementedError: _description_
    """
    return f"{chrom}_{strand}_{start_sample}_{end_sample}.csv"


def _one_hot_decode(sequence: torch.tensor):
    assert sequence.shape[0] == 4, "Input shape must be 4xsequence_length"
    return "".join(
        onehot2nucleotide[tuple(sequence[:, i].tolist())]
        for i in range(sequence.shape[1])
    )


def batch_one_hot_decode(sequences: torch.tensor):
    """
    Decode a batch of one-hot encoded sequences.

    Args:
        sequences (torch.tensor): Batch of one-hot encoded sequences.
                                 Expected shape: [batch_size, sequence_length, 4]
                                 or [batch_size, 4, sequence_length]

    Returns:
        list: List of decoded nucleotide sequences as strings
    """
    batch_size = sequences.shape[0]
    decoded_sequences = []

    # Determine the format of the input tensor
    if sequences.shape[1] == 4 and sequences.shape[2] > 4:
        # Format is [batch_size, 4, sequence_length]
        for i in range(batch_size):
            seq = sequences[i]  # shape: [4, sequence_length]
            decoded_sequences.append(_one_hot_decode(seq))

    elif sequences.shape[2] == 4:
        # Format is [batch_size, sequence_length, 4]
        for i in range(batch_size):
            seq = sequences[i].permute(1, 0)  # Transpose to [4, sequence_length]
            decoded_sequences.append(_one_hot_decode(seq))

    else:
        raise ValueError(
            f"Unexpected shape for one-hot encoded sequences: {sequences.shape}. "
            "Expected [batch_size, sequence_length, 4] or [batch_size, 4, sequence_length]"
        )

    return decoded_sequences
