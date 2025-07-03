import math
from pathlib import Path
import torch
import numpy as np
from typing import List
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
import os
import json


def plot_seq_identity_against_pred_true_corr(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    homology_graph_path: Path,
    seq_ids: List[str],
    out: Path,
) -> None:
    """
    We want to know, which percentage of predictive performance
    is due to sequences in the training set similar to sequences
    in the test set that have similar track values.

    E.g.: if we have a real duplication between training and test set,
    then, the performance on that sample most likely doesn't come
    from good generalization but rather from fitting a single training
    example very well.

    Here, we produce a plot that takes the most similar sequence
    from the training set for each sequence in the test set
    and plots their homology, computed using blat, on the x-axis and
    the pearson correlation between the predicted and true track values
    on the y-axis.

    # TODO: the similarity of track values also plays a role here!

    Arguments:
        y_pred: the predictions of the model on the test set
        y_true: the real track values of the test set
        homology_graph: path to a graph file storing homolgies between all samples
        seq_ids: list of sequence ids that identify the DNA sequence to use with the homology graph
        out: path were the plot is saved
    """

    def _compute_sample_corr(y_pred: torch.tensor, y_true: torch.tensor):
        output = y_pred.detach().cpu().numpy()
        target = y_true.detach().cpu().numpy()

        # Compute pearson corr using numpy
        correlation = np.corrcoef(output, target)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        return correlation

    def _get_closest_sample_homology(homology_graph: nx.Graph, node_id: str) -> float:
        """
        Returns the *minimum* edge weight (highest similarity) from `node_id` to any neighbor.
        We assume that each edge has a float attribute "weight", representing distance or
        dissimilarity (small = high sequence identity).
        """
        if node_id not in homology_graph:
            # Node might be missing if there's no homology entry for it.
            print("Warning! Could not find node in homology graph!")
            return float("nan")

        neighbors_data = homology_graph[node_id]
        if not neighbors_data:  # no neighbors => no edges
            return float("nan")

        # Each item is (neighbor_id, edge_attrs_dict)
        # We choose the largest "weight" among them.
        max_weight = max(
            edge_attrs.get("weight", float("inf"))
            for _, edge_attrs in neighbors_data.items()
        )
        return max_weight

    # Load homology graph
    homology_graph = nx.read_gml(path=homology_graph_path)

    # Compute all pearson correlations
    correlations = []
    homologies = []
    num_samples = y_true.shape[0]
    for i in range(num_samples):
        correlations.append(_compute_sample_corr(y_pred=y_pred[i], y_true=y_true[i]))
        homologies.append(
            _get_closest_sample_homology(
                homology_graph=homology_graph, node_id=seq_ids[i]
            )
        )

    # Convert to numpy arrays for plotting
    correlations = np.array(correlations)
    homologies = np.array(homologies)

    # Create a scatter plot of homologies vs. correlations
    plt.figure(figsize=(6, 6))
    plt.scatter(homologies, correlations, alpha=0.6)
    plt.xlabel("Sequence Distance / Dissimilarity (weight)")
    plt.ylabel("Predicted vs. True Pearson Correlation")
    plt.title("Sequence Homology vs. Model Performance")

    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def _evaluate_mrae(
    true_values: torch.Tensor,
    pred_values: torch.Tensor,
    track_name_order: List[str],
    model_resolution: int = 1,
):
    """Compute mean relative absolute error for each track and each sample and return DataFrames."""
    import numpy as np
    import pandas as pd

    num_samples, num_tracks = true_values.shape[:2]

    track_mrae = []
    sample_mrae = []

    for track_idx in range(num_tracks):
        track_results_mrae = []

        for sample_idx in range(num_samples):
            true_track = true_values[sample_idx, track_idx].numpy()
            pred_track = pred_values[sample_idx, track_idx].numpy()

            # Skip tracks consisting solely of missing values
            if np.all(true_track == -1 * model_resolution):
                continue

            valid_mask = true_track != -1 * model_resolution
            if np.sum(valid_mask) <= 1:
                continue

            true_valid = true_track[valid_mask]
            pred_valid = pred_track[valid_mask]

            with np.errstate(divide="ignore", invalid="ignore"):
                denom = np.where(true_valid == 0, np.nan, true_valid)
                rel_err = np.abs((pred_valid - true_valid) / denom)

            mrae = np.nanmean(rel_err)
            track_results_mrae.append(mrae)

            sample_mrae.append(
                {
                    "sample_idx": sample_idx,
                    "track_idx": track_idx,
                    "track_name": track_name_order[track_idx],
                    "mrae": mrae,
                }
            )

        track_mrae.append(
            {
                "track_idx": track_idx,
                "track_name": track_name_order[track_idx],
                "mean_mrae": np.nanmean(track_results_mrae)
                if len(track_results_mrae) > 0
                else float("nan"),
                "median_mrae": np.nanmedian(track_results_mrae)
                if len(track_results_mrae) > 0
                else float("nan"),
                "std_mrae": np.std(track_results_mrae)
                if len(track_results_mrae) > 0
                else float("nan"),
                "num_valid_samples": len(track_results_mrae),
            }
        )

    track_mrae_df = pd.DataFrame(track_mrae)
    sample_mrae_df = pd.DataFrame(sample_mrae)
    return track_mrae_df, sample_mrae_df

def _evaluate_correlation(
    true_values: torch.Tensor,
    pred_values: torch.Tensor,
    track_name_order: List[str],
    test_loader: DataLoader,
    model_resolution: int = 1,
):
    """Compute Pearson correlation for each track and sample and return DataFrames plus count array."""
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr

    num_samples, num_tracks = true_values.shape[:2]

    # Handle case where DataLoader.batch_size is None
    batch_size = test_loader.batch_size if test_loader.batch_size is not None else 1

    track_correlations = []
    sample_correlations = []
    valid_correlations_count = np.zeros(num_tracks)

    for track_idx in range(num_tracks):
        track_results_corr = []

        for sample_idx in range(num_samples):
            true_track = true_values[sample_idx, track_idx].numpy()
            pred_track = pred_values[sample_idx, track_idx].numpy()

            if np.all(true_track == -1 * model_resolution):
                continue

            valid_mask = true_track != -1 * model_resolution
            if np.sum(valid_mask) <= 1:
                continue

            true_valid = true_track[valid_mask]
            pred_valid = pred_track[valid_mask]

            if np.std(true_valid) == 0 or np.std(pred_valid) == 0:
                continue

            corr, _ = pearsonr(true_valid, pred_valid)
            track_results_corr.append(corr)

            sample_correlations.append(
                {
                    "sample_idx": sample_idx,
                    "track_idx": track_idx,
                    "track_name": track_name_order[track_idx],
                    "correlation": corr,
                    "batch_sample_idx": sample_idx % batch_size,
                    "batch_idx": sample_idx // batch_size,
                }
            )

            valid_correlations_count[track_idx] += 1

        if track_results_corr:
            track_correlations.append(
                {
                    "track_idx": track_idx,
                    "track_name": track_name_order[track_idx],
                    "mean_correlation": np.nanmean(track_results_corr),
                    "median_correlation": np.nanmedian(track_results_corr),
                    "std_correlation": np.nanstd(track_results_corr),
                    "num_valid_samples": len(track_results_corr),
                }
            )
        else:
            track_correlations.append(
                {
                    "track_idx": track_idx,
                    "track_name": track_name_order[track_idx],
                    "mean_correlation": float("nan"),
                    "median_correlation": float("nan"),
                    "std_correlation": float("nan"),
                    "num_valid_samples": 0,
                }
            )

    track_corr_df = pd.DataFrame(track_correlations)
    sample_corr_df = pd.DataFrame(sample_correlations)

    return track_corr_df, sample_corr_df, valid_correlations_count

def _evaluate_re_fc(
    true_values: torch.Tensor,
    pred_values: torch.Tensor,
    track_name_order: List[str],
    model_resolution: int = 1,
): 
    """
    Compute the relative error (as fold change) between true and predicted values for each track and each sample and return DataFrames. 
    I.e. let k be the sequence position for sample i and track m, then the relative error as fold changeis:
    re_fc = log2(pred_value / true_value)
    """

    # Local imports to avoid global dependency bloat and to be consistent with other helpers
    import numpy as np
    import pandas as pd

    num_samples, num_tracks = true_values.shape[:2]

    track_re_fc = []  # aggregated per-track statistics
    sample_re_fc = []  # per-sample, per-track metrics

    for track_idx in range(num_tracks):
        track_results_re_fc = []

        for sample_idx in range(num_samples):
            true_track = true_values[sample_idx, track_idx].numpy()
            pred_track = pred_values[sample_idx, track_idx].numpy()

            # Skip tracks consisting solely of missing values (encoded as -1 * model_resolution)
            if np.all(true_track == -1 * model_resolution):
                continue

            valid_mask = true_track != -1 * model_resolution
            if np.sum(valid_mask) <= 1:
                # Not enough valid positions to compute a meaningful metric
                continue

            true_valid = true_track[valid_mask]
            pred_valid = pred_track[valid_mask]

            # Guard against division by zero or negative / invalid values by masking those entries
            # We require both true and predicted values to be strictly positive for log2 ratio.
            positive_mask = (true_valid > 0) & (pred_valid > 0)

            if np.sum(positive_mask) == 0:
                # No positions with positive values in both true & pred – skip
                continue

            true_positive = true_valid[positive_mask]
            pred_positive = pred_valid[positive_mask]

            with np.errstate(divide="ignore", invalid="ignore"):
                # Compute log2 fold-change
                re_fc_values = np.log2(pred_positive / true_positive)

            re_fc = np.nanmean(re_fc_values)

            # Record per-track aggregation as well as per-sample value
            track_results_re_fc.append(re_fc)

            sample_re_fc.append(
                {
                    "sample_idx": sample_idx,
                    "track_idx": track_idx,
                    "track_name": track_name_order[track_idx],
                    "re_fc": re_fc,
                }
            )

        # Per-track statistics collection
        track_re_fc.append(
            {
                "track_idx": track_idx,
                "track_name": track_name_order[track_idx],
                "mean_re_fc": np.nanmean(track_results_re_fc)
                if len(track_results_re_fc) > 0
                else float("nan"),
                "median_re_fc": np.nanmedian(track_results_re_fc)
                if len(track_results_re_fc) > 0
                else float("nan"),
                "std_re_fc": np.nanstd(track_results_re_fc)
                if len(track_results_re_fc) > 0
                else float("nan"),
                "num_valid_samples": len(track_results_re_fc),
            }
        )

    track_re_fc_df = pd.DataFrame(track_re_fc)
    sample_re_fc_df = pd.DataFrame(sample_re_fc)

    return track_re_fc_df, sample_re_fc_df

def _make_predictions(model, test_loader, device, model_resolution):
    from clex.utils import untransform_then_unbin

    all_true_values = []
    all_pred_values = []
    all_sample_info = []
    sampled_rows = []

    # Run inference
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Extract data from batch
            sequences, targets = batch[0], batch[1]

            # Extract sample metadata
            sample_info = {
                "indices": batch[2],
                "chroms": batch[3],
                "strands": batch[4],
                "start_samples": batch[5],
                "end_samples": batch[6],
                "start_losses": batch[7],
                "end_losses": batch[8],
            }

            sequences = sequences.to(device)
            targets = targets.to(device)

            # Keep CPU copy of sequences for potential saving
            sequences_cpu = sequences.clone()

            # Forward pass - using autocast for mixed precision
            with torch.cuda.amp.autocast():
                # Convert to half precision for FlashAttention compatibility
                outputs = model(sequences)

                # Handle squeeze if necessary (from the model architecture)
                if len(outputs.shape) == 4:  # If shape is [batch, 1, tracks, bins]
                    outputs = outputs.squeeze(1)

                # Unbin and untransform model outputs
                outputs = untransform_then_unbin(outputs, resolution=model_resolution)

            outputs_cpu = outputs.float().cpu()

            # Save tensors for later aggregation
            all_true_values.append(targets.float().cpu())
            all_pred_values.append(outputs_cpu)

            all_sample_info.append(sample_info)

            # -----------------------------------------------------------
            # Randomly (5% probability) store detailed per-sample info
            # -----------------------------------------------------------
            batch_size_current = sequences_cpu.shape[0]
            for i in range(batch_size_current):
                if np.random.rand() < 0.05:
                    sampled_rows.append(
                        {
                            "true_values": targets[i].cpu().numpy(),
                            "pred_values": outputs_cpu[i].numpy(),
                            "chrom": sample_info["chroms"][i],
                            "start": sample_info["start_samples"][i],
                            "end": sample_info["end_samples"][i],
                        }
                    )

    # Concatenate batches
    true_values = torch.cat(all_true_values, dim=0)
    pred_values = torch.cat(all_pred_values, dim=0)

    return true_values, pred_values, sampled_rows


# Define evaluation function
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    track_name_order: List[str],
    device: str = "cuda:0",
    output_dir: str = "evaluation_results",
    create_visualizations: bool = True,
    correlation_threshold: float = 0.0,
    model_resolution=1,
):
    """
    Evaluate the model by computing Pearson correlation between true and predicted values.

    Args:
        model: The trained PyTorch model
        test_loader: DataLoader containing test data
        device: Device to run evaluation on
        output_dir: Directory to save evaluation results
        create_visualizations: Whether to create visualizations
        correlation_threshold: Minimum correlation to visualize (if create_visualizations is True)

    Returns:
        dict: Dictionary containing evaluation metrics
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    model.to(device)

    # Initialize lists to store results
    
    # Storage for a random subset (≈5%) of samples with detailed info

    true_values, pred_values, sampled_rows = _make_predictions(model, test_loader, device, model_resolution)

    # --------------------------------------------------------------
    # Compute evaluation metrics using dedicated helper functions
    # --------------------------------------------------------------
    num_samples = true_values.shape[0]
    num_tracks = true_values.shape[1]

    track_corr_df, sample_corr_df, valid_correlations_count = _evaluate_correlation(
        true_values=true_values,
        pred_values=pred_values,
        track_name_order=track_name_order,
        test_loader=test_loader,
        model_resolution=model_resolution,
    )

    track_mrae_df, sample_mrae_df = _evaluate_mrae(
        true_values=true_values,
        pred_values=pred_values,
        track_name_order=track_name_order,
        model_resolution=model_resolution,
    )
    
    track_re_fc_df, sample_re_fc_df = _evaluate_re_fc(
        true_values=true_values,
        pred_values=pred_values,
        track_name_order=track_name_order,
        model_resolution=model_resolution,
    )

    # Convert NumPy types to native Python types for JSON serialization
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    # Output metrics
    metrics = {
        "track_correlations": track_corr_df.to_dict("records"),
        "overall_mean_correlation": float(track_corr_df["mean_correlation"].mean()),
        "overall_mean_re_fc": float(track_re_fc_df["mean_re_fc"].mean()),
        "valid_tracks_count": int((valid_correlations_count > 0).sum()),
    }

    # Save metrics to JSON
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)

    # Save track correlations to CSV
    track_corr_df.to_csv(
        os.path.join(output_dir, "track_correlations.csv"), index=False
    )

    # Save track mrae to CSV
    track_mrae_df.to_csv(os.path.join(output_dir, "track_mrae.csv"), index=False)

    # Save relative error (fold-change) metrics to CSV
    track_re_fc_df.to_csv(os.path.join(output_dir, "track_re_fc.csv"), index=False)
    sample_re_fc_df.to_csv(os.path.join(output_dir, "sample_re_fc.csv"), index=False)

    # Display results in notebook
    display_results(track_corr_df, metrics, track_order=track_name_order)

    # sample_corr_df is already a DataFrame
    sample_corr_df.to_csv(
        os.path.join(output_dir, "sample_correlations.csv"), index=False
    )
    make_per_sample_corr_hist(sample_corr_df, output_dir)

    # sample_mrae_df is already a DataFrame
    sample_mrae_df.to_csv(os.path.join(output_dir, "sample_mrae.csv"), index=False)

    # ------------------------------------------------------------------
    # Save sampled subset with raw sequences, predictions and metadata
    # ------------------------------------------------------------------
    if len(sampled_rows) > 0:
        subset_df = pd.DataFrame(sampled_rows)
        subset_df.to_pickle(os.path.join(output_dir, "sample_subset.pkl"))
        print(
            f"Saved {len(sampled_rows)} randomly selected samples (≈5%) to 'sample_subset.pkl'"
        )

    # Create and save visualizations
    if create_visualizations:
        # Plot track correlations
        plt.figure(figsize=(12, 6))
        plt.bar(range(num_tracks), track_corr_df["mean_correlation"])
        plt.xticks(range(num_tracks), [f"Track {i}" for i in range(num_tracks)])
        plt.xlabel("Track")
        plt.ylabel("Mean Pearson Correlation")
        plt.title("Mean Pearson Correlation by Track")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "track_correlations.png"))
        plt.show()

        # Create visualization directory for individual samples
        viz_dir = os.path.join(output_dir, "sample_visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        # Find high-correlation samples for visualization
        if not sample_corr_df.empty:
            high_corr_samples = sample_corr_df[
                sample_corr_df["correlation"] > correlation_threshold
            ]

            # Group by sample and get mean correlation
            sample_mean_corr = (
                high_corr_samples.groupby("sample_idx")["correlation"]
                .mean()
                .reset_index()
            )

            # Take top 5 samples with highest mean correlation for notebook display
            top_samples = sample_mean_corr.sort_values(
                "correlation", ascending=False
            ).head(5)

            # Visualize top samples
            for _, row in top_samples.iterrows():
                sample_idx = int(row["sample_idx"])

                # Get data for this sample
                true = true_values[sample_idx].numpy()
                pred = pred_values[sample_idx].numpy()

                # Plot each track
                plt.figure(figsize=(15, 10))

                for track_idx in range(num_tracks):
                    true_track = true[track_idx]
                    pred_track = pred[track_idx]

                    # Skip tracks with no valid data
                    if np.all(true_track == -model_resolution):
                        continue

                    # Mask for valid values
                    valid_mask = true_track != -model_resolution

                    plt.subplot(math.ceil(num_tracks / 3), 3, track_idx + 1)
                    plt.plot(
                        np.arange(len(true_track))[valid_mask],
                        true_track[valid_mask],
                        "b-",
                        label="True",
                    )
                    plt.plot(
                        np.arange(len(pred_track))[valid_mask],
                        pred_track[valid_mask],
                        "r-",
                        label="Predicted",
                    )

                    # Compute correlation for this track in this sample
                    if (
                        np.sum(valid_mask) > 5
                        and np.std(true_track[valid_mask]) > 0
                        and np.std(pred_track[valid_mask]) > 0
                    ):
                        corr, _ = pearsonr(
                            true_track[valid_mask], pred_track[valid_mask]
                        )
                        plt.title(f"Track {track_idx} (Corr: {corr:.3f})")
                    else:
                        plt.title(f"Track {track_idx}")

                    if track_idx == 0:
                        plt.legend()

                plt.suptitle(
                    f"Sample {sample_idx} (Mean Corr: {row['correlation']:.3f})"
                )
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f"sample_{sample_idx}.png"))
                plt.show()

            # Create predictions vs. true value scatter plots for the best sample
            if len(top_samples) > 0:
                best_sample_idx = int(top_samples.iloc[0]["sample_idx"])
                true = true_values[best_sample_idx].numpy()
                pred = pred_values[best_sample_idx].numpy()

                plt.figure(figsize=(15, 10))
                for track_idx in range(num_tracks):
                    true_track = true[track_idx]
                    pred_track = pred[track_idx]

                    # Skip tracks with no valid data
                    if np.all(true_track == -model_resolution):
                        continue

                    # Mask for valid values
                    valid_mask = true_track != -model_resolution

                    if np.sum(valid_mask) > 5:
                        plt.subplot(math.ceil(num_tracks / 3), 3, track_idx + 1)
                        plt.scatter(
                            true_track[valid_mask], pred_track[valid_mask], alpha=0.5
                        )

                        # Add diagonal line
                        min_val = min(
                            true_track[valid_mask].min(), pred_track[valid_mask].min()
                        )
                        max_val = max(
                            true_track[valid_mask].max(), pred_track[valid_mask].max()
                        )
                        plt.plot([min_val, max_val], [min_val, max_val], "k--")

                        # Compute correlation
                        if (
                            np.std(true_track[valid_mask]) > 0
                            and np.std(pred_track[valid_mask]) > 0
                        ):
                            corr, _ = pearsonr(
                                true_track[valid_mask], pred_track[valid_mask]
                            )
                            plt.title(f"Track {track_idx} (Corr: {corr:.3f})")
                        else:
                            plt.title(f"Track {track_idx}")

                        plt.xlabel("True Values")
                        plt.ylabel("Predicted Values")

                plt.suptitle("True vs Predicted Values (Best Sample)")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "true_vs_pred_best_sample.png"))
                plt.show()

    return metrics, track_corr_df, true_values, pred_values


def make_per_sample_corr_hist(sample_corr_df: pd.DataFrame, output_dir: str):
    from pypalettes import load_cmap
    import matplotlib.pyplot as plt
    from pyfonts import load_google_font

    per_sequence_mean = sample_corr_df.groupby("sample_idx").agg(
        {"correlation": ["mean"]}
    )

    per_sequence_mean.reset_index(inplace=True)

    per_sequence_mean.columns = ["sample_idx", "mean_correlation"]

    # choose weight / style if you need: weight="bold", italic=True ...
    inter = load_google_font("Inter")  # FontProperties object

    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = inter.get_name()

    cmap = load_cmap("Acanthurus_leucosternon")  # note underscore, exact name
    main_colour = cmap(0.6)  # any float 0 – 1 gives a colour
    plt.figure(figsize=(10, 10), dpi=300)  # Increased DPI to 300
    plt.hist(
        per_sequence_mean["mean_correlation"],
        bins=50,
        color=main_colour,  # <- a single RGBA colour
        edgecolor="black",
        histtype="stepfilled",
    )

    # Calculate mean and median
    mean_val = per_sequence_mean["mean_correlation"].mean()
    median_val = per_sequence_mean["mean_correlation"].median()

    # Add vertical lines for mean and median
    plt.axvline(
        mean_val, color=cmap(0.2), linestyle="--", label=f"Mean: {mean_val:.3f}"
    )
    plt.axvline(
        median_val, color=cmap(0.8), linestyle="--", label=f"Median: {median_val:.3f}"
    )

    plt.title("Distribution of Sample Correlations", font=inter)
    plt.xlabel("Pearson R", font=inter, fontsize=12)
    plt.ylabel("Count", font=inter, fontsize=12)
    plt.grid(False, alpha=0.3)
    plt.legend(prop=inter)
    plt.savefig(os.path.join(output_dir, "correlation_distribution.png"))
    plt.close()


def display_results(track_corr_df, metrics, track_order: List[str]):
    """
    Display evaluation results in a notebook-friendly format
    """
    # Display overall metrics
    print(f"Overall mean correlation: {metrics['overall_mean_correlation']:.4f}")
    print(f"Valid tracks: {metrics['valid_tracks_count']} out of {len(track_corr_df)}")
    print("\n")

    # Create a styled dataframe for display
    display_df = track_corr_df[
        ["track_idx", "mean_correlation", "median_correlation", "num_valid_samples"]
    ].copy()
    display_df.columns = [
        "Track",
        "Mean Correlation",
        "Median Correlation",
        "Valid Samples",
    ]
    display_df["Track"] = display_df["Track"].apply(lambda x: f"Track {track_order[x]}")

    # Display with styling
    try:
        from IPython.display import display

        display(
            display_df.style.background_gradient(
                subset=["Mean Correlation"], cmap="viridis"
            )
        )
    except:
        display_df.to_string(index=False)


def load_model_for_testing(base_path: str, skip_data=False, device="cuda:0"):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import pandas as pd

    from clex.train import Config
    from clex.model.borzoi import Borzoi
    from clex.model.utils import TargetLengthCrop
    from clex.config import BorzoiConfig


    # Import your model classes and dataset
    from clex.dataset import GenomicDataset

    model_path = f"{base_path}/model_best.pth"
    cfg_path = f"{base_path}/train_config.json"

    cfg = Config.read_from_json(cfg_path)

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

    model = nn.Sequential(TargetLengthCrop(4992), Borzoi(config))

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    model.eval()
    model.to(device)

    if not skip_data:
        # Load dataset
        samples = pd.read_pickle(cfg.path_to_samples)

        # Filter for test set
        test_samples = samples[samples["fold"] == "test"]
        test_dataset = GenomicDataset(test_samples, resolution=1, split_name="test")

        # Create dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=10,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        with open(f"{base_path}/track_annotation.json", "r") as f:
            row_annotation = json.load(f)
            track_order = row_annotation["+"] + row_annotation["-"]

        return model, test_loader, track_order, cfg.borzoi_cfg["resolution"]
    return model, None, None, cfg.borzoi_cfg["resolution"]
