import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


from yorzoi.dataset import GenomicDataset


class Shalem2015Dataset(Dataset):

    def __init__(self, data_path: str):
        self.data = (
            pd.read_csv(data_path, sep="\t")
            .dropna()
            .reset_index(drop=True)
            .reset_index()
        )

    def construct_sequence(self, insert: str):

        # 665 bp
        GAL1_10_prom = "TTTCAAAAATTCTTACTTTTTTTTTGGATGGACGCAAAGAAGTTTAATAATCATATTACATGGCATTACCACCATATACATATCCATATACATATCCATATCTAATCTTACTTATATGTTGTGGAAATGTAAAGAGCCCCATTATCTTAGCCTAAAAAAACCTTCTCTTTGGAACTTTCAGTAATACGCTTAACTGCTCATTGCTATATTGAAGTACGGATTAGAAGCCGCCGAGCGGGTGACAGCCCTCCGAAGGAAGACTCTCCTCCGTGCGTCCTCGTCTTCACCGGTCGCGTTCCTGAAACGCAGATGTGCCTCGCGCCGCACTGCTCCGAACAATAAAGATTCTACAATACTAGCTTTTATGGTTATGAAGAGGAAAAATTGGCAGTAACCTGGCCCCACAAACCTTCAAATGAACGAATCAAATTAACAACCATAGGATGATAATGCGATTAGTTTTTTAGCCTTATTTCTGGGGTAATTAATCAGCGAAGCGATGATTTTTGATCTATTAACAGATATATAAATGCAAAAACTGCATAACCACTTTAACTAATACTTTCAACATTTTCGGTTTGTATTACTTCTTATTCAAATGTAATAAAAGTATCAACAAAAAATTGTTAATATACCTCTATACTTTAACGTCAAGGAGAAAAAAC"
        # 717 bp
        YFP = "ATGTCTAAAGGTGAAGAATTATTCACTGGTGTTGTCCCAATTTTGGTTGAATTAGATGGTGATGTTAATGGTCACAAATTTTCTGTCTCCGGTGAAGGTGAAGGTGATGCTACTTACGGTAAATTGACCTTAAAATTGATTTGTACTACTGGTAAATTGCCAGTTCCATGGCCAACCTTAGTCACTACTTTAGGTTATGGTTTGCAATGTTTTGCTAGATACCCAGATCATATGAAACAACATGACTTTTTCAAGTCTGCCATGCCAGAAGGTTATGTTCAAGAAAGAACTATTTTTTTCAAAGATGACGGTAACTACAAGACCAGAGCTGAAGTCAAGTTTGAAGGTGATACCTTAGTTAATAGAATCGAATTAAAAGGTATTGATTTTAAAGAAGATGGTAACATTTTAGGTCACAAATTGGAATACAACTATAACTCTCACAATGTTTACATCACTGCTGACAAACAAAAGAATGGTATCAAAGCTAACTTCAAAATTAGACACAACATTGAAGATGGTGGTGTTCAATTAGCTGACCATTATCAACAAAATACTCCAATTGGTGATGGTCCAGTCTTGTTACCAGACAACCATTACTTATCCTATCAATCTGCCTTATCCAAAGATCCAAACGAAAAGAGAGACCACATGGTCTTGTTAGAATTTGTTACTGCTGCTGGTATTACCCATGGTATGGATGAATTGTACAAATAA"
        # 50 bp
        GC_tail = "GC" * 50

        core_seq = f"{GAL1_10_prom}{YFP}{insert}{GC_tail}"

        padding = 4992 - len(core_seq)
        left_padding = 1000
        right_padding = padding - left_padding

        full_seq = f"{'N' * left_padding}{core_seq}{'N' * right_padding}"

        assert len(full_seq) == 4992, f"Sequence {len(full_seq)} long not 4992"

        return full_seq

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        construct_sequence = self.construct_sequence(
            self.data.iloc[idx]["Oligo Sequence"]
        ).upper()

        return (
            torch.tensor(
                GenomicDataset.one_hot_encode(construct_sequence), dtype=torch.float32
            ),
            torch.tensor([self.data.iloc[idx]["Expression"]]),
        )


def plot_tensor_tracks(
    tensor,
    save_path=None,
    figsize=(15, 10),
    plot_titles=None,
    track_names=None,
    vertical_lines=None,
):
    """
    Creates batch plots from a PyTorch tensor, where each plot contains multiple subplots for different tracks.

    Args:
        tensor (torch.Tensor): A tensor of shape (batch_size, tracks, seq_len) on GPU.
        save_path (str, optional): Path to save the plots. If None, plots will be displayed.
        figsize (tuple, optional): Figure size for each plot (width, height).
        plot_titles (list, optional): List of titles for each batch plot. If None, defaults to "Batch {i}".
        track_names (list, optional): List of names for each track. If None, defaults to "Track {j}".
        vertical_lines (dict, optional): Dictionary where keys are names and values are positions
                                        at which to draw vertical lines. Example: {"name": 120}
    """
    # Ensure tensor is the correct shape
    if len(tensor.shape) != 3:
        raise ValueError(
            f"Expected tensor of shape (batch_size, tracks, seq_len), got {tensor.shape}"
        )

    # Move tensor to CPU and convert to numpy for plotting
    tensor_np = tensor.detach().cpu().numpy()

    batch_size, num_tracks, seq_len = tensor.shape

    # Create x-axis values (0 to seq_len-1)
    x_values = np.arange(seq_len)

    # Set default titles if not provided
    if plot_titles is None:
        plot_titles = [f"Batch {i}" for i in range(batch_size)]

    # Set default track names if not provided
    if track_names is None:
        track_names = [f"Track {j}" for j in range(num_tracks)]

    # Iterate through each batch
    for batch_idx in range(batch_size):
        # Create a figure with subplots for each track
        fig, axes = plt.subplots(num_tracks, 1, figsize=figsize, sharex=True)

        # Handle the case when there's only one track
        if num_tracks == 1:
            axes = [axes]

        # Set the main title for the plot
        fig.suptitle(plot_titles[batch_idx], fontsize=16)

        # Plot each track in its own subplot
        for track_idx in range(num_tracks):
            track_data = tensor_np[batch_idx, track_idx]
            axes[track_idx].plot(x_values, track_data, linewidth=2)
            axes[track_idx].set_title(track_names[track_idx])
            axes[track_idx].grid(True)

            # Add vertical lines if specified
            if vertical_lines is not None:
                for name, position in vertical_lines.items():
                    # Check if position is within the sequence length
                    if 0 <= position < seq_len:
                        axes[track_idx].axvline(
                            x=position, linestyle="--", alpha=0.7, label=name
                        )

                # Add legend only if vertical lines were added
                if (
                    vertical_lines and track_idx == 0
                ):  # Add legend to first subplot only
                    axes[track_idx].legend(loc="upper right")

            # Add x-axis label to the bottom subplot only
            if track_idx == num_tracks - 1:
                axes[track_idx].set_xlabel("Sequence Index")

            # Add y-axis label to all subplots
            axes[track_idx].set_ylabel("Value")

        # Adjust layout for better spacing
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the main title

        # Save or display the plot
        if save_path:
            # Save each batch plot with a unique filename
            plt.savefig(f"{save_path}_batch_{batch_idx}.png")
            plt.close(fig)
        else:
            plt.show()


def evaluate(
    model, output_dir_base, device: str = "cuda:0", model_resolution: int = 10
):
    import os
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from scipy.stats import pearsonr

    dataset = Shalem2015Dataset(
        data_path="/home/tds122/clex/clex/eval/shalem/segal_2015.tsv"
    )
    test_loader = DataLoader(
        dataset,
        batch_size=100,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        shuffle=False,
    )

    yfp_start, yfp_end = 665 // model_resolution, (665 + 717) // model_resolution

    os.makedirs(output_dir_base, exist_ok=True)

    # ---------------------------------------------------------------------------
    # inference loop – keep only tiny per-track region sums
    # ---------------------------------------------------------------------------
    region_sums_accum = []  # list of (batch, tracks) CPU tensors
    expr_accum = []  # list of (batch, 1)   CPU tensors
    sample_ids = []  # list of str

    with torch.no_grad(), torch.cuda.amp.autocast():
        for b_idx, (seqs, exp_val) in enumerate(tqdm(test_loader, desc="Inference")):
            preds = model(seqs.to(device, non_blocking=True))

            # Reduce *inside* the loop to avoid holding the big tensor
            region_sum = preds[:, :, yfp_start:yfp_end].sum(dim=2).cpu()  # (B, tracks)

            region_sums_accum.append(region_sum)
            expr_accum.append(exp_val.cpu())

            offset = b_idx * test_loader.batch_size
            sample_ids.extend([f"sample_{offset + j}" for j in range(seqs.size(0))])

    # ---------------------------------------------------------------------------
    # build the final DataFrame
    # ---------------------------------------------------------------------------
    region_sums_full = torch.cat(region_sums_accum, dim=0).numpy()  # (N, tracks)
    expr_full = torch.cat(expr_accum, dim=0).squeeze(1).numpy()

    num_tracks = region_sums_full.shape[1]

    data = {
        "sample_id": sample_ids,
        "expression_value": expr_full,
    }
    for t in range(num_tracks):
        data[f"track_{t}"] = region_sums_full[:, t]

    final_df = pd.DataFrame(data)

    csv_path = os.path.join(output_dir_base, "all_samples_results.csv")
    final_df.to_csv(csv_path, index=False)
    print(f"Saved complete results to {csv_path}")

    track_cols = [c for c in final_df.columns if c.startswith("track_")]
    stats_df = final_df[track_cols + ["expression_value"]].describe()
    stats_df.to_csv(os.path.join(output_dir_base, "track_statistics.csv"))
    print("Saved track statistics.")

    corr_records = [
        {
            "track": col,
            "correlation_with_expression": pearsonr(
                final_df[col], final_df["expression_value"]
            )[0],
            "p_value": pearsonr(final_df[col], final_df["expression_value"])[1],
        }
        for col in track_cols
    ]
    pd.DataFrame(corr_records).to_csv(
        os.path.join(output_dir_base, "track_expression_correlations.csv"), index=False
    )
    print("Saved track-expression correlations.")
