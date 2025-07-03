import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from yorzoi.constants import nucleotide2onehot
from yorzoi.utils import _borzoi_transform
import matplotlib.pyplot as plt
from itertools import islice
import math


class GenomicDataset(Dataset):
    def __init__(
        self,
        samples: pd.DataFrame,
        rc_aug: bool = False,
        noise_tracks: bool = False,
        split_name=None,
        resolution=10,
        sample_length=5_000,
        context_length=1_000,
    ):
        print(f"\tSetting up {split_name} GenomicDataset")
        self.split_name = split_name
        self.samples = samples

        self.mean_track_values, self.std_track_values = 0.0, 0.0
        n_seen = 0
        for p in self.samples["track_values"].sample(100):
            arr = np.load(p)["a"]
            subset = arr.flat[:: max(1, arr.size // 10_000)]
            self.mean_track_values += subset.mean()
            self.std_track_values += subset.std()
            n_seen += 1
        self.mean_track_values /= n_seen
        self.std_track_values /= n_seen

        # TODO: Remove hard-coded values as needed
        self.resolution = resolution
        self.sample_length = sample_length
        self.context_length = context_length
        self.reverse_complement_aug = rc_aug
        self.noise_tracks = noise_tracks
        print("\t\tDone.")

    def plot_track_values(self):
        plt.hist(
            np.concatenate(self.samples["track_values"].values), bins=100, log=True
        )
        plt.yscale("log")
        plt.savefig(f"track_values_histogram_{self.split_name}.png")
        plt.close()

    def reverse_complement_sequence(self, sequence):
        """Returns the reverse complement of a DNA sequence."""
        complement = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
        return "".join(complement.get(base, "N") for base in reversed(sequence))

    def reverse_complement_tracks(self, track_values):
        """
        Rearranges tracks and reverses values as specified.
        For tracks of shape (n_tracks, pred_length), where n_tracks is even:
        - Swaps first half with second half (0<->n/2, 1<->n/2+1, etc.)
        - Reverses the order of values in each track
        """
        n_tracks = track_values.shape[0]

        # Assert that n_tracks is even
        assert n_tracks % 2 == 0, f"Number of tracks must be even, got {n_tracks}"

        # Create a copy to avoid modifying the original
        modified_tracks = track_values.clone()

        # Swap first half with second half
        half_idx = n_tracks // 2
        for i in range(half_idx):
            modified_tracks[i], modified_tracks[i + half_idx] = (
                modified_tracks[i + half_idx].clone(),
                modified_tracks[i].clone(),
            )

        # Reverse each track's values
        for i in range(n_tracks):
            modified_tracks[i] = modified_tracks[i].flip(0)

        return modified_tracks

    def apply_noise(self, track_values):
        """
        Applies Gaussian noise to each track value that is not -1 * resolution.
        Noise is sampled from N(0, 0.1 * value).
        """
        # Get mask for valid values (not equal to -1 * resolution)
        valid_mask = track_values != -1 * self.resolution

        # Create noise tensor of the same shape
        noise = torch.zeros_like(track_values)

        # Calculate standard deviation as 10% of each value
        std_devs = 0.1 * torch.abs(track_values)

        # Generate noise only for valid values
        noise[valid_mask] = torch.normal(mean=0.0, std=std_devs[valid_mask])

        # Apply noise to track values
        noisy_tracks = track_values.clone() + noise

        return noisy_tracks

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples.iloc[idx]
        sample_sequence = sample["sample_sequence"]

        # Pad sample sequence if necessary
        if len(sample_sequence) < self.sample_length:
            start_sample = sample["start_sample"]
            start_loss = sample["start_loss"]
            left_context_len = start_loss - start_sample
            right_context_len = sample["end_sample"] - sample["end_loss"]

            if sample["strand_loss"] == "+":
                if left_context_len < self.context_length:  # pad from left up to 1000
                    to_add = self.context_length - left_context_len
                    sample_sequence = "N" * to_add + sample_sequence
                if right_context_len < self.context_length:
                    to_add = self.context_length - right_context_len
                    sample_sequence = sample_sequence + "N" * to_add
            elif sample["strand_loss"] == "-":
                # Negative strand (reverse complemented sequence)
                if left_context_len < self.context_length:  # pad from right up to 1000
                    to_add = self.context_length - left_context_len
                    sample_sequence = sample_sequence + "N" * to_add
                if right_context_len < self.context_length:
                    to_add = self.context_length - right_context_len
                    sample_sequence = "N" * to_add + sample_sequence
            else:
                raise ValueError("Strand not recognized")

        # Process track values
        tv_path = sample["track_values"]
        track_values = np.load(tv_path)["a"]

        # Determine prediction length given context
        pred_length = int(
            (self.sample_length - 2 * self.context_length) / self.resolution
        )

        # Reshape and sum adjacent windows (e.g. for 10 bp resolution)
        num_tracks = track_values.shape[0]
        track_values = (
            np.array(track_values)
            .reshape((num_tracks, pred_length, self.resolution))
            .sum(axis=2)
        )

        track_values = torch.tensor(track_values, dtype=torch.float32)

        # Apply reverse complement augmentation if enabled
        if (
            self.reverse_complement_aug
            and np.random.random() < 0.5
            and sample["chr_loss"]
            not in {
                "NC_001147.6",
                "NC_001138.5",
                "NC_001137.3",
                "E533_NC_000007.14",
                "E1068_NC_000007.14",
            }
        ):
            sample_sequence = self.reverse_complement_sequence(sample_sequence)
            track_values = self.reverse_complement_tracks(track_values)

        # Apply noise augmentation if enabled
        if self.noise_tracks:
            track_values = self.apply_noise(track_values)

        # Gather additional features
        chrom = sample["chr_loss"]
        strand = sample["strand_loss"]
        start_sample = sample["start_sample"]
        end_sample = sample["end_sample"]
        start_loss = sample["start_loss"]
        end_loss = sample["end_loss"]

        return (
            torch.tensor(
                GenomicDataset.one_hot_encode(sample_sequence), dtype=torch.float32
            ),
            track_values,
            idx,
            chrom,
            strand,
            start_sample,
            end_sample,
            start_loss,
            end_loss,
        )

    @staticmethod
    def one_hot_encode(seq: str) -> np.ndarray:
        return np.array([nucleotide2onehot.get(base, [0, 0, 0, 0]) for base in seq])


def custom_collate_factory(resolution=10):
    def custom_collate_fn(batch):
        """
        Collate function to batch samples and apply the borzoi_transform
        on the track_values only once per batch.
        """
        # Unpack batch items
        (
            sequences,
            track_values_list,
            indices,
            chroms,
            strands,
            start_samples,
            end_samples,
            start_losses,
            end_losses,
        ) = zip(*batch)

        # Stack sequences and track_values to form a batch

        sequences = torch.stack(sequences)
        track_values = torch.stack(
            track_values_list
        )  # shape: (batch_size, tracks, pred_length)

        # Apply the borzoi_transform for each sample in the batch
        for i in range(track_values.size(0)):
            tv = track_values[i]
            # Identify rows that are all -10 (i.e., -1 * resolution)
            all_neg_ten_rows = torch.all(tv == -1 * resolution, dim=1)
            # Temporarily set these rows to 0 to avoid transforming them
            tv[all_neg_ten_rows] = 0
            # Apply the transformation
            tv = _borzoi_transform(tv)
            # Set the previously identified rows back to -10
            tv[all_neg_ten_rows] = -1 * resolution
            track_values[i] = tv

        return (
            sequences,
            track_values,
            indices,
            chroms,
            strands,
            start_samples,
            end_samples,
            start_losses,
            end_losses,
        )

    return custom_collate_fn
