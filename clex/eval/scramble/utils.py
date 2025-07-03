import torch
import pandas as pd
import pathlib
from typing import Union, Optional, Any, Dict
import numpy as np
import pyBigWig
from torch.utils.data import DataLoader
import os, json
from clex.utils import untransform_then_unbin
from clex.eval.brooks.dataset import BrooksDataset

def _make_predictions(
    brooks_eval_df_path: str,
    model,
    dataloader: DataLoader,
    device: "torch.device",
    model_resolution: int = 10,
    track_annotation_path: Union[str, pathlib.Path] = "data/track_annotation.json",
):
    """Run *model* on *dataloader* and attach per-sample predictions.

    Only predictions corresponding to the track of each sample's +/- bigWig
    are kept.  The dataloader is expected to yield a tuple of *(x, meta)* where
    *meta* is a dict that at least contains: ``row_idx``, ``+_file``,
    ``-_file``.  This matches the index of the original DataFrame built by
    :pyfunc:`create_eval_df`.

    Returns the augmented DataFrame loaded from *brooks_eval_df.pkl* with two
    new columns ``+_pred`` and ``-_pred`` (numpy arrays).
    """

    # ------------------------------------------------------------------
    # 1.  Load eval dataframe (so we can append predictions)
    # ------------------------------------------------------------------
    if not pathlib.Path(brooks_eval_df_path).exists():
        raise FileNotFoundError(f"{brooks_eval_df_path} not found. Run create_eval_df() first.")

    df = pd.read_pickle(brooks_eval_df_path).copy()
    df["+_pred"] = None
    df["-_pred"] = None

    # ------------------------------------------------------------------
    # 2.  Track annotation ➜ mapping from bigWig basename ➜ track index
    # ------------------------------------------------------------------
    with open(track_annotation_path, "r") as fh:
        annot = json.load(fh)

    plus_tracks = annot.get("+", [])
    minus_tracks = annot.get("-", [])
    n_plus = len(plus_tracks)

    plus_map = {name: idx for idx, name in enumerate(plus_tracks)}
    minus_map = {name: idx for idx, name in enumerate(minus_tracks)}

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, meta = batch
            else:
                raise ValueError("Expected dataloader to yield (inputs, meta) tuples.")

            x = x.to(device)
            # Forward pass with mixed precision for efficiency
            with torch.cuda.amp.autocast():
                preds_t = model(x)  # (B, tracks, bins) or (B,1,tracks,bins)

                # Squeeze any singleton dimension
                if preds_t.dim() == 4 and preds_t.shape[1] == 1:
                    preds_t = preds_t.squeeze(1)

                # Unbin & untransform so that predictions are in raw coverage space
                preds_t = untransform_then_unbin(preds_t, resolution=model_resolution)

            preds = preds_t.cpu().numpy()

            for i in range(len(meta)):
                row_idx = int(meta[i]["row_idx"])  # original dataframe index
                plus_file = os.path.basename(meta[i]["+_file"])
                minus_file = os.path.basename(meta[i]["-_file"])

                plus_pred, minus_pred = None, None

                if plus_file in plus_map:
                    track_idx = plus_map[plus_file]
                    plus_pred = preds[i, track_idx]
                else:
                    print(f"[warning] + track '{plus_file}' not found in annotation.")

                if minus_file in minus_map:
                    track_idx = minus_map[minus_file] + n_plus  # offset in model output
                    minus_pred = preds[i, track_idx]
                else:
                    print(f"[warning] - track '{minus_file}' not found in annotation.")

                df.at[row_idx, "+_pred"] = plus_pred
                df.at[row_idx, "-_pred"] = minus_pred

    return df

def evaluate(model, device, track_annotation_path, model_resolution=10, brooks_eval_df_path="brooks_eval_df.pkl", output_dir: str = ""):
    print("Running Brooks eval...")
    print(f"\tBrooks eval df path: {brooks_eval_df_path}")

    def _collate(batch):
        seqs = [b[0] for b in batch]
        metas = [b[1] for b in batch]
        return torch.stack(seqs), metas

    dataset = BrooksDataset(brooks_eval_df_path)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, collate_fn=_collate)

    df = _make_predictions(
        brooks_eval_df_path=brooks_eval_df_path,
        model=model,
        dataloader=dataloader,
        device=device,
        model_resolution=model_resolution,
        track_annotation_path=track_annotation_path,
    )

    df.to_pickle(f"{output_dir}/brooks_eval_df_with_preds.pkl")

    print("\tDone.")


def get_bigwig_dict(
    df_intervals: "pd.DataFrame",
    bigwig_dir: Union[str, pathlib.Path] = "/home/tds122/clex/clex/eval/brooks/bigwigs",
) -> dict:
    """Return mapping from strain name to its '+' and '-' bigWig files.

    Parameters
    ----------
    df_intervals : pandas.DataFrame
        Must contain a column named ``strain`` with the strain IDs.
    bigwig_dir : str | pathlib.Path, optional
        Directory that holds the bigWig files.  Defaults to the canonical
        Brooks path inside the repo.

    Returns
    -------
    dict[str, dict[str, str]]
        ``{"STRAIN": {"+": "/path/to/plus.bw", "-": "/path/to/minus.bw"}, ...}``

    Raises
    ------
    FileNotFoundError
        If a plus or minus file for a strain cannot be located.
    """

    from pathlib import Path

    bigwig_dir = Path(bigwig_dir)
    if not bigwig_dir.is_dir():
        raise FileNotFoundError(f"BigWig directory does not exist: {bigwig_dir}")

    strain_to_paths: dict[str, dict[str, str]] = {}

    for strain in sorted(df_intervals["strain"].unique()):
        try:
           # Disambiguation 
            if strain == "JS710":
                strain = "JS710_20181122"
            if strain == "JS731":
                strain = "JS731_20181122"

            plus_files = list(bigwig_dir.glob(f"{strain}_*.plus.bw"))
            minus_files = list(bigwig_dir.glob(f"{strain}_*.minus.bw"))

            if len(plus_files) == 0 or len(minus_files) == 0:
                raise FileNotFoundError(
                    f"Missing BigWig file(s) for strain '{strain}'.\n"
                    f"  plus candidates: {plus_files}\n  minus candidates: {minus_files}"
                )
            if len(plus_files) > 1 or len(minus_files) > 1:
                raise FileExistsError(
                    f"Multiple BigWig files matched for strain '{strain}'. Please disambiguate."
                )

            strain_to_paths[strain] = {
                "+": str(plus_files[0]),
                "-": str(minus_files[0]),
            }
        except FileNotFoundError:
            print(f"Skipping {strain}")

    return strain_to_paths


import pandas as pd
from pathlib import Path
import pandas as pd
from pyfaidx import Fasta 
from intervaltree import Interval, IntervalTree

def build_trees_by_strain(df):
    """
    Parameters
    ----------
    df : pandas.DataFrame
         Must contain columns 'strain', 'clusterStart', and 'clusterEnd '
         (note the trailing space in the original header).

    Returns
    -------
    dict[str, IntervalTree]
         One IntervalTree per strain with all raw (unmerged) intervals added.
    """
    trees = {}

    for _, row in df.iterrows():
        strain = row["strain"]
        chrom = row["chrom"]
        # cast to int just in case—they often come through as NumPy scalars
        start = int(row["clusterStart"])
        end = int(row["clusterEnd "])  # keep the exact header!

        # put the interval into the appropriate tree
        trees.setdefault((strain, chrom), IntervalTree()).add(Interval(start, end))

    return trees


def merge_overlaps(
    trees_by_strain,
    max_len: int = 3000,
):
    """Merge overlapping intervals but keep each merged block shorter than *max_len*.

    If extending a block would exceed the limit, the current block is closed and a
    new one started.  Single raw intervals longer than *max_len* are **kept as
    is** but a warning is printed.

    Parameters
    ----------
    trees_by_strain : dict[(strain, chrom) -> IntervalTree]
    max_len         : int, optional
        Maximum allowed length of a merged interval.  Default = 3000.

    Returns
    -------
    dict[(strain, chrom) -> list[tuple[int, int]]]
        Each entry is a list of (start, end) tuples, sorted by start.
    """

    merged: dict[tuple[str, str], list[tuple[int, int]]] = {}

    for key, tree in trees_by_strain.items():
        # Sort raw intervals by start coordinate
        intervals = sorted(tree, key=lambda iv: iv.begin)

        out: list[tuple[int, int]] = []

        current_start = -1  # inactive state
        current_end = -1

        for iv in intervals:
            start, end = iv.begin, iv.end

            # Flag very long raw interval
            if end - start > max_len:
                print(
                    f"[warning] Raw interval >{max_len} bp ({end-start}) in {key}: {start}-{end}. Keeping as-is."
                )
                # flush current active block if any
                if current_start >= 0:
                    out.append((current_start, current_end))
                    current_start, current_end = -1, -1
                out.append((start, end))
                continue

            if current_start < 0:  # start new block
                current_start, current_end = start, end
                continue

            # Check overlap and resulting length
            if start <= current_end and (max(current_end, end) - current_start) <= max_len:
                current_end = max(current_end, end)
            else:
                out.append((current_start, current_end))
                current_start, current_end = start, end

        # append last active block
        if current_start >= 0:
            out.append((current_start, current_end))

        merged[key] = out

    return merged

def _load_genomes(genome_dir, fasta_suffix):
    """
    Returns
    -------
    dict[strain -> pyfaidx.Fasta]
    """
    genomes = {}
    for fp in Path(genome_dir).glob(f"*{fasta_suffix}"):
        # strip the common suffix to get the 'clean' strain name
        strain = fp.name.removesuffix(fasta_suffix)
        genomes[strain] = Fasta(fp, as_raw=True, sequence_always_upper=True)
    if not genomes:
        raise FileNotFoundError(
            f"No genomes ending with {fasta_suffix} found in {genome_dir}"
        )
    return genomes

def df_from_interval_trees(
    merged_trees,
    genome_dir="/home/tds122/clex/clex/eval/brooks/genomes",
    fasta_suffix="_ERCC92.fasta",
    coord_mode="0-based-half-open",  # just metadata for your own tracking
):
    """
    Parameters
    ----------
    merged_trees : dict
        Two common layouts are supported out-of-the-box.

        **Layout A**
        { strain → { chrom → IntervalTree } }

        **Layout B**
        { (strain, chrom) → IntervalTree }

        If you used some other layout, adapt the two inner `for`-loops below.

    genome_dir    : str | Path
        Folder with the FASTA (+ .fai) files.
    fasta_suffix  : str
        Common suffix shared by all FASTA filenames.
    coord_mode    : str
        Descriptive label stored in the output DataFrame.

    Returns
    -------
    pandas.DataFrame
        Columns: strain, chrom, start, end, sequence, sequence_rc
    """
    genomes = _load_genomes(genome_dir, fasta_suffix)

    def _centered_sequence(
        fa: "Fasta",
        chrom: str,
        iv_start: int,
        iv_end: int,
        window: int = 5000,
    ) -> tuple[str, str, int, int]:
        """Return padded sequence, its reverse-complement, seq_start, seq_end.

        The returned sequence is exactly *window* bp long (unless the raw
        interval itself already exceeds that length) and is centred on the
        provided interval.
        """

        chrom_len = len(fa[chrom])
        chunk_len = iv_end - iv_start

        if chunk_len >= window:
            # interval itself longer than window – return raw sequence
            seq_start = iv_start
            seq_end = iv_end
            seq_str = str(fa[chrom][seq_start:seq_end])
            print(
                f"[warning] Chunk length ({chunk_len}) >= window ({window}); returning raw interval for {chrom}:{iv_start}-{iv_end}."
            )
            return seq_str, _reverse_complement(seq_str), seq_start, seq_end

        flank_total = window - chunk_len
        left_pad = flank_total // 2
        right_pad = flank_total - left_pad

        seq_start = max(0, iv_start - left_pad)
        seq_end = min(chrom_len, iv_end + right_pad)

        # Adjust pads if we hit chromosome boundaries
        actual_left_pad = iv_start - seq_start
        actual_right_pad = seq_end - iv_end
        missing_left = left_pad - actual_left_pad
        missing_right = right_pad - actual_right_pad

        seq_parts = []
        if missing_left > 0:
            seq_parts.append("N" * missing_left)
        seq_parts.append(str(fa[chrom][seq_start:seq_end]))
        if missing_right > 0:
            seq_parts.append("N" * missing_right)

        seq_str = "".join(seq_parts)
        return seq_str, _reverse_complement(seq_str), seq_start, seq_end


    def _reverse_complement(seq: str) -> str:
        comp = str.maketrans("ATCGNatcgn", "TAGCNtagcn")
        return seq.translate(comp)[::-1]

    rows = []
    for (strain, chrom), itree in merged_trees.items():
        if strain not in genomes:
            print(f"Skipping {strain}")
            continue
        fa = genomes[strain]
        for iv in sorted(itree):
            seq_str, seq_rc, seq_start, seq_end = _centered_sequence(
                fa, chrom, iv[0], iv[1]
            )
            rows.append(
                {
                    "strain": strain,
                    "chrom": chrom,
                    "start": iv[0],
                    "end": iv[1],
                    "seq_start": seq_start,
                    "seq_end": seq_end,
                    "sequence": seq_str,
                    "sequence_rc": seq_rc,
                }
            )

    df = pd.DataFrame(rows)
    df.attrs["coordinates"] = coord_mode
    return df


def add_bigwig_coverage(
    df_intervals: "pd.DataFrame",
    bigwig_dict: Optional[dict] = None,
    bigwig_dir: Union[str, pathlib.Path] = "/home/tds122/clex/clex/eval/brooks/bigwigs",
) -> "pd.DataFrame":
    """Augment *df_intervals* with per-base coverage arrays (+ / – strands).

    New columns added:
        + "_cov"   : numpy.ndarray of coverage values (dtype float, length = end-start)
        + "-_cov"  : idem for minus strand
        + "+_file" : absolute path of the bigWig used
        + "-_file" : idem for minus strand

    Any failure while reading coverage prints a warning and leaves the *_cov
    entry as ``None``.

    Parameters
    ----------
    df_intervals : pandas.DataFrame
        Must contain columns  *strain*, *chrom*, *start*, *end*.
    bigwig_dict : dict, optional
        Mapping as returned by :pyfunc:`get_bigwig_dict`.  If *None*, the
        mapping will be rebuilt on-the-fly using *bigwig_dir*.
    bigwig_dir : str | pathlib.Path, optional
        Directory holding the ``*.plus.bw`` / ``*.minus.bw`` files.
    """

    if bigwig_dict is None:
        bigwig_dict = get_bigwig_dict(df_intervals, bigwig_dir=bigwig_dir)  # type: ignore[arg-type]

    # Re-use open pyBigWig handles to avoid filesystem overhead
    bw_cache: dict[str, pyBigWig.pyBigWig] = {}

    def _get_bw_handle(path: str) -> pyBigWig.pyBigWig:
        if path not in bw_cache:
            try:
                bw_cache[path] = pyBigWig.open(path)
            except Exception as err:  # pragma: no cover
                raise FileNotFoundError(f"Could not open bigWig: {path}\n{err}")
        return bw_cache[path]

    # Prepare containers for new columns
    plus_cov, minus_cov, plus_file, minus_file = [], [], [], []

    for _, row in df_intervals.iterrows():
        strain_raw: str = row["strain"]

        # Direct lookup; if missing, try special aliases handled in get_bigwig_dict
        strain_key = strain_raw if strain_raw in bigwig_dict else None
        if strain_key is None and f"{strain_raw}_20181122" in bigwig_dict:
            strain_key = f"{strain_raw}_20181122"

        if strain_key is None:
            print(f"[warning] No bigWig entry for strain '{strain_raw}'. Skipping coverage.")
            plus_cov.append(None)
            minus_cov.append(None)
            plus_file.append(None)
            minus_file.append(None)
            continue

        paths = bigwig_dict[strain_key]

        chrom = row["chrom"]
        start = int(row["start"])
        end = int(row["end"])

        # ---------------- plus strand ----------------
        try:
            bw_plus = _get_bw_handle(paths["+"])
            vals_plus = np.array(bw_plus.values(chrom, start, end), dtype=float)
            # Replace NaN values with 0
            vals_plus = np.nan_to_num(vals_plus, nan=0.0)
        except Exception as e:  # pragma: no cover
            print(f"[warning] Failed to extract + coverage for {strain_raw}:{chrom}:{start}-{end}: {e}")
            vals_plus = None

        # ---------------- minus strand ----------------
        try:
            bw_minus = _get_bw_handle(paths["-"])
            vals_minus = np.array(bw_minus.values(chrom, start, end), dtype=float)
            # Replace NaN values with 0
            vals_minus = np.nan_to_num(vals_minus, nan=0.0)
        except Exception as e:  # pragma: no cover
            print(f"[warning] Failed to extract - coverage for {strain_raw}:{chrom}:{start}-{end}: {e}")
            vals_minus = None

        plus_cov.append(vals_plus)
        minus_cov.append(vals_minus)
        plus_file.append(paths["+"])
        minus_file.append(paths["-"])

    df_aug = df_intervals.copy()
    df_aug["+_cov"] = plus_cov
    df_aug["-_cov"] = minus_cov
    df_aug["+_file"] = plus_file
    df_aug["-_file"] = minus_file

    # Close all opened bigWig handles
    for bw in bw_cache.values():
        try:
            bw.close()
        except Exception:
            pass

    return df_aug

def create_eval_df(brooks_sup_table: str = "science.abg0162_table_s3.txt"):
    import pandas as pd
    from clex.eval.brooks.utils import build_trees_by_strain, merge_overlaps, df_from_interval_trees, get_bigwig_dict, add_bigwig_coverage

    df = pd.read_csv(
        brooks_sup_table,
        sep="\t",
        header=1,
    )

    trees = build_trees_by_strain(df[df["noveljcn"] & (df["clusterEnd "] - df["clusterStart"] < 5000)])
    merged = merge_overlaps(trees)

    df_intervals = df_from_interval_trees(merged)

    bw_paths = get_bigwig_dict(df_intervals)
    df_cov = add_bigwig_coverage(df_intervals, bw_paths)

    df_cov.dropna(subset=['+_file', '-_file'], inplace=True)
    df_cov.to_pickle("brooks_eval_df.pkl")

    return df_cov

# ----------------------------------------------------------------------
# Visualisation helpers
# ----------------------------------------------------------------------


def _plot_all_predictions(df: "pd.DataFrame", strand: str = "+", ncols: int = 5, figsize_per_plot=(4, 2)):
    """Create one big matplotlib figure comparing predicted vs. true coverage.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns like "+_pred" / "+_cov" (or "-_pred" / "-_cov" for strand='-').
    strand : {'+', '-'}
        Which strand to plot.
    ncols : int, optional
        Number of subplot columns in the grid.
    figsize_per_plot : tuple(float, float)
        Width and height of each subplot.
    """

    import math
    import matplotlib.pyplot as plt
    import numpy as np

    pred_col = f"{strand}_pred"
    true_col = f"{strand}_cov"

    # Filter out rows where either prediction or truth is missing
    valid_df = df[df[pred_col].notnull() & df[true_col].notnull()].reset_index(drop=True)

    n_samples = len(valid_df)
    if n_samples == 0:
        raise ValueError("No samples with both prediction and truth available to plot.")

    nrows = math.ceil(n_samples / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
        sharex=True,
        sharey=False,  # allow individual y-scales per subplot
    )

    # Flatten axes array for easy iteration (works for 1-D, 2-D, etc.)
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    elif isinstance(axes, (list, tuple)):
        axes_flat = list(axes)
    else:
        axes_flat = [axes]

    for idx, ax in enumerate(axes_flat):
        if idx >= n_samples:
            ax.axis("off")
            continue

        row = valid_df.iloc[idx]
        pred_array = row[pred_col]
        true_array = row[true_col]

        # Align true coverage in the centre of the 3000-bp window represented by predictions
        pred_len = len(pred_array)
        true_len = len(true_array)
        pad_left = (pred_len - true_len) // 2
        true_x = range(pad_left, pad_left + true_len)

        ax.plot(range(pred_len), pred_array, label="Pred", alpha=0.7)
        ax.plot(true_x, true_array, label="True", alpha=0.7)

        # Compute Pearson correlation on the overlapping region
        pred_slice = pred_array[pad_left : pad_left + true_len]
        try:
            if np.std(pred_slice) > 0 and np.std(true_array) > 0:
                corr = float(np.corrcoef(pred_slice, true_array)[0, 1])
            else:
                corr = float("nan")
        except Exception:
            corr = float("nan")

        strain_name = row.get("strain", "?")
        ax.set_title(f"{strain_name}\nidx {row.name}  r={corr:.2f}", fontsize=8)
        ax.tick_params(labelsize=6)

    # Only one legend for the whole figure
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()

# ----------------------------------------------------------------------
# Correlation computation
# ----------------------------------------------------------------------


def _pearson_corr_aligned(pred: np.ndarray, truth: np.ndarray) -> float:
    """Return Pearson *r* between *pred* and *truth* after centre-alignment.

    The function assumes that *pred* corresponds to a fixed-size window
    (e.g. 3000 bp) while *truth* may be shorter.  *truth* is placed in the
    centre of *pred* (the same logic used for the overview plot), and the
    overlapping slice of *pred* is taken for the correlation.
    """

    if pred is None or truth is None:
        return float("nan")

    pred_len = len(pred)
    truth_len = len(truth)

    if truth_len == 0 or pred_len == 0 or truth_len > pred_len:
        return float("nan")

    pad_left = (pred_len - truth_len) // 2
    pred_slice = pred[pad_left : pad_left + truth_len]

    if pred_slice.size == 0 or truth.size == 0:
        return float("nan")

    # guard against zero variance
    if np.std(pred_slice) == 0 or np.std(truth) == 0:
        return float("nan")

    return float(np.corrcoef(pred_slice, truth)[0, 1])


def save_correlations_csv(
    df: "pd.DataFrame",
    out_csv: str = "sample_correlations.csv",
) -> "pd.DataFrame":
    """Compute per-sample Pearson correlations and save them to *out_csv*.

    The resulting CSV contains one row per sample with columns:
        row_idx, strain, plus_corr, minus_corr

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns *+_pred*, *+_cov*, *-_pred*, *-_cov*.
    out_csv : str, optional
        Destination path.  Default = ``sample_correlations.csv`` in CWD.

    Returns
    -------
    pandas.DataFrame
        The table that was written out.
    """

    records = []
    for row_idx, row in df.iterrows():
        plus_corr = _pearson_corr_aligned(row["+_pred"], row["+_cov"])
        minus_corr = _pearson_corr_aligned(row["-_pred"], row["-_cov"])

        records.append(
            {
                "row_idx": row_idx,
                "strain": row.get("strain", None),
                "plus_corr": plus_corr,
                "minus_corr": minus_corr,
            }
        )

    df_corr = pd.DataFrame.from_records(records)
    df_corr.to_csv(out_csv, index=False)
    print(f"Saved correlations for {len(df_corr)} samples -> {out_csv}")
    return df_corr

def _extract_centered_coverage(
    bw: "pyBigWig.pyBigWig",
    chrom: str,
    start: int,
    end: int,
    window: int = 3000,
) -> np.ndarray:
    """Return a *window*-long coverage vector centred on *(start, end)*.

    The function obeys chromosome boundaries via ``bw.chroms()`` and pads any
    out-of-range positions with zeros so that the returned NumPy array always
    has length *window*.
    """

    chrom_sizes = bw.chroms()
    if chrom not in chrom_sizes:
        raise ValueError(f"Chromosome '{chrom}' not found in bigWig file.")

    chrom_len: int = chrom_sizes[chrom]

    import warnings

    chunk_len = end - start
    if chunk_len > window:
        # Crop the interval to the central *window* bp.
        excess = chunk_len - window
        trim_left = excess // 2
        trim_right = excess - trim_left

        warnings.warn(
            f"Requested interval length ({chunk_len}) exceeds window size ({window}). "
            f"Cropping {trim_left} bp from the left and {trim_right} bp from the right "
            f"to keep the central {window} bp window.",
            RuntimeWarning,
        )

        # Update start/end to the cropped coordinates
        start += trim_left
        end -= trim_right
        chunk_len = end - start  # should now equal *window*

    flank_total = window - chunk_len
    left_pad = flank_total // 2
    right_pad = flank_total - left_pad

    desired_left = start - left_pad
    desired_right = end + right_pad

    # Clamp to chromosome boundaries
    query_left = max(desired_left, 0)
    query_right = min(desired_right, chrom_len)

    # Retrieve coverage and replace NaNs with zeros
    try:
        values = np.array(bw.values(chrom, query_left, query_right), dtype=float)
    except RuntimeError as err:
        raise RuntimeError(
            f"pyBigWig failed for {chrom}:{query_left}-{query_right}\n{err}"
        )

    values = np.nan_to_num(values, nan=0.0)

    # Determine how much padding is required due to boundaries
    missing_left = query_left - desired_left  # >0 when desired_left < 0
    missing_right = desired_right - query_right  # >0 when desired_right > chrom_len

    if missing_left > 0:
        values = np.concatenate([np.zeros(missing_left, dtype=float), values])
    if missing_right > 0:
        values = np.concatenate([values, np.zeros(missing_right, dtype=float)])

    # Sanity check
    if values.size != window:
        raise AssertionError(
            f"Centered coverage length mismatch: expected {window}, got {values.size}"
        )

    return values


def _save_track_values_to_npz(
    cov_matrix: np.ndarray,
    out_dir: str,
    sample_id: str,
) -> str:
    """Write *cov_matrix* to ``out_dir/{sample_id}.npz`` and return the path."""

    import os

    os.makedirs(out_dir, exist_ok=True)

    safe_id = (
        sample_id.replace("/", "_").replace(" ", "_").replace(":", "_")[:200]
    )  # guard against ridiculously long names
    out_path = os.path.join(out_dir, f"{safe_id}.npz")

    # Use key 'a' to stay compatible with existing loader
    np.savez_compressed(out_path, a=cov_matrix.astype(np.float32))

    return out_path


def _assign_fold(strain: str) -> str:
    """Return *"val"* if *strain* is in the predefined validation list, else *"train"*."""

    val_strains = {
        "JS613",
        "JS710",
        "JS715",
        "JS610",
        "JS716",
        "JS614",
        "JS626",
    }

    return "val" if strain in val_strains else "train"


def create_train_df(
    brooks_eval_df_path: str = "brooks_eval_df.pkl",
    npz_output_dir: str = "data/track_values",
    track_annotation_path: str = "data/track_annotation.json",
    samples_append_path: Optional[str] = None,
) -> pd.DataFrame:
    """Build a *training-ready* sample DataFrame from *brooks_eval_df.pkl*.

    Overview of the planned workflow (to be implemented):
    1.  Load *brooks_eval_df.pkl* (produced by :pyfunc:`create_eval_df`).
    2.  Parse *track_annotation_path* – a JSON file whose ``"+"`` and ``"-"``
        keys list the bigWig basenames that form the canonical **track order**.
        ``n_tracks = len(annot["+"]) + len(annot["-"])``.
    3.  For every row in the Brooks eval table:
        a. Determine the indices of the row-specific plus / minus bigWig files
           within the annotation lists **after stripping the directory part**
           (``os.path.basename``).
        b. Use :pyfunc:`_extract_centered_coverage` to grab the 3000-bp window
           from each bigWig.  This helper uses ``bw.chroms()`` to respect
           chromosome boundaries and pads with zeros when necessary.
        c. Initialise a *(n_tracks, 3000)* matrix **filled with −1** (the
           sentinel value used elsewhere for "masked") and write the two
           coverage vectors into their respective rows.
        d. Persist the matrix via :pyfunc:`_save_track_values_to_npz` (which
           writes a compressed ``.npz`` with key ``'a'``) and store the path
           in the new DataFrame under ``track_values``.
    4.  Assemble a new DataFrame with **exactly** the columns required by the
        existing training pipeline so that it can be concatenated with the
        canonical ``samples_060525.pkl``:
            - chr_loss
            - strand_loss
            - start_sample
            - end_sample
            - start_loss
            - end_loss
            - loss_sequence           (centre 3000 bp of *sample_sequence*)
            - sample_sequence         (padded to exactly 5000 bp)
            - fold                    ("train" / "val" as per strain)
            - track_values            (path to the ``.npz`` file)
            - hom_graph_id            (left empty / NaN)
    5.  If *samples_append_path* is provided, the function will append the new
        rows to the existing pickle and write the combined DataFrame back to a new path (add date to path).

    The function returns the newly created DataFrame, regardless of whether it
    was appended to an existing file.
    """
    import os
    import json
    from datetime import datetime

    # ------------------------------------------------------------------
    # 1.  Load Brooks eval DataFrame
    # ------------------------------------------------------------------
    if not pathlib.Path(brooks_eval_df_path).exists():
        raise FileNotFoundError(brooks_eval_df_path)

    brooks_df = pd.read_pickle(brooks_eval_df_path).copy()

    # ------------------------------------------------------------------
    # 2.  Track annotation ➜ determine mapping and total track count
    # ------------------------------------------------------------------
    annot_path = pathlib.Path(track_annotation_path)
    if not annot_path.exists():
        raise FileNotFoundError(
            f"track_annotation.json not found at expected location '{track_annotation_path}'."
        )

    with open(annot_path, "r") as fh:
        annot: Dict[str, list] = json.load(fh)

    plus_tracks = annot.get("+", [])
    minus_tracks = annot.get("-", [])

    if not plus_tracks or not minus_tracks:
        raise ValueError("track_annotation.json must define non-empty '+' and '-' lists")

    n_plus = len(plus_tracks)
    n_minus = len(minus_tracks)
    n_tracks = n_plus + n_minus

    plus_map = {name: idx for idx, name in enumerate(plus_tracks)}
    minus_map = {name: idx for idx, name in enumerate(minus_tracks)}

    # ------------------------------------------------------------------
    # 3.  Prepare bigWig handle cache
    # ------------------------------------------------------------------
    bw_cache: Dict[str, pyBigWig.pyBigWig] = {}

    def _get_bw(path: str) -> pyBigWig.pyBigWig:
        if path not in bw_cache:
            bw_cache[path] = pyBigWig.open(path)
        return bw_cache[path]

    # ------------------------------------------------------------------
    # 4.  Iterate rows and build new samples list
    # ------------------------------------------------------------------
    new_rows = []

    for idx, row in brooks_df.iterrows():
        chrom = row["chrom"]
        start = int(row["start"])
        end = int(row["end"])

        plus_path = row["+_file"]
        minus_path = row["-_file"]

        if not plus_path or not minus_path:
            # Skip if any track missing (should already be filtered but double-check)
            continue

        plus_name = os.path.basename(plus_path)
        minus_name = os.path.basename(minus_path)

        if plus_name not in plus_map or minus_name not in minus_map:
            print(f"[warning] Track mapping missing for row {idx}. Skipping.")
            continue

        try:
            bw_plus = _get_bw(plus_path)
            bw_minus = _get_bw(minus_path)
        except RuntimeError as err:
            print(f"[warning] Could not open bigWig for row {idx}: {err}. Skipping.")
            continue

        try:
            plus_cov = _extract_centered_coverage(bw_plus, chrom, start, end, window=3000)
            minus_cov = _extract_centered_coverage(bw_minus, chrom, start, end, window=3000)
        except Exception as err:
            print(f"[warning] Coverage extraction failed for row {idx}: {err}. Skipping.")
            continue

        cov_matrix = np.full((n_tracks, 3000), -1.0, dtype=np.float32)
        cov_matrix[plus_map[plus_name]] = plus_cov
        cov_matrix[n_plus + minus_map[minus_name]] = minus_cov

        sample_id = f"brooks_{idx}"
        npz_path = _save_track_values_to_npz(cov_matrix, npz_output_dir, sample_id)

        sequence: str = row["sequence"]
        # Ensure sample_sequence is exactly 5000 bp for compatibility
        if len(sequence) != 5000:
            raise AssertionError(
                f"Row {idx}: expected sequence length 5000, got {len(sequence)}"
            )

        # central 3000 bp
        loss_seq_start = (len(sequence) - 3000) // 2
        loss_seq_end = loss_seq_start + 3000
        loss_sequence = sequence[loss_seq_start:loss_seq_end]

        # Determine leading N padding (if any)
        leading_ns = len(sequence) - len(sequence.lstrip("N"))

        start_sample = leading_ns if row["seq_start"] == 0 else row["seq_start"]
        end_sample = row["seq_end"]

        # Loss coordinates relative to genome (may be negative if padding)
        chunk_len = end - start
        flank_total = 3000 - chunk_len
        left_pad = flank_total // 2
        right_pad = flank_total - left_pad

        start_loss = start - left_pad
        end_loss = end + right_pad

        new_rows.append(
            {
                "chr_loss": chrom,
                "strand_loss": "+",  # Brooks orientation unknown – default '+'
                "start_sample": int(start_sample),
                "end_sample": int(end_sample),
                "start_loss": int(start_loss),
                "end_loss": int(end_loss),
                "loss_sequence": loss_sequence,
                "sample_sequence": sequence,
                "fold": _assign_fold(row["strain"]),
                "track_values": npz_path,
                "hom_graph_id": np.nan,
            }
        )

    # Close bigWig handles
    for bw in bw_cache.values():
        try:
            bw.close()
        except Exception:
            pass

    train_df = pd.DataFrame(new_rows)

    # ------------------------------------------------------------------
    # 5.  Optionally append to existing samples file
    # ------------------------------------------------------------------
    if samples_append_path is not None:
        if not pathlib.Path(samples_append_path).exists():
            raise FileNotFoundError(samples_append_path)

        base_df = pd.read_pickle(samples_append_path)
        combined_df = pd.concat([base_df, train_df], ignore_index=True)

        ts = datetime.now().strftime("%y%m%d_%H%M%S")
        out_path = pathlib.Path(samples_append_path).with_stem(
            pathlib.Path(samples_append_path).stem + f"_with_brooks_{ts}"
        )
        combined_df.to_pickle(out_path)
        print(f"Combined DataFrame written to {out_path}")

    return train_df