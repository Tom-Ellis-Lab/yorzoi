"""
This script is run after every training run and evaluates models.

Evaluations included:
1. Correlation of predicted and true RNA‑seq coverage (untransformed, unbinned)
2. Mean Relative Absolute Error (MRAE)
3. Subset of DREAM challenge test data
4. Transcription termination prediction accuracy

Usage
-----
python autoeval.py --base-path /path/to/run [--eval EVAL_TYPES] [--device DEVICE]

Example: python clex/eval/autoeval.py --base-path /home/tds122/clex/runs/2025-05-11_16-26-05_ablation_100_rep3 --eval shalem,dream --device cuda:1

If ``--base-path`` is omitted, the current working directory is used (maintaining
backward compatibility with earlier versions).

Base path must contain the model.pth file, by default we search for "model_best.pth"

Available evaluations:
- default: Run all evaluations
- baseline: Run correlation and MRAE evaluation
- shalem: Run only Shalem Terminator evaluation
- dream: Run only DREAM challenge evaluation
- brooks: Run only Brooks challenge evaluation

Multiple evaluations can be specified with comma-separation: --eval shalem,dream
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List

import typer

from clex.eval.utils import evaluate_model, load_model_for_testing
from clex.eval.promoter.utils import evaluate as eval_DREAM
from clex.eval.terminator.utils import evaluate as eval_Shalem
from clex.eval.scramble.utils import evaluate as eval_Brooks

# ---------------------------------------------------------------------------
# CLI setup with Typer
# ---------------------------------------------------------------------------

app = typer.Typer(
    add_help_option=True,
    help="Evaluate a trained model given its checkpoint directory.",
)

VALID_EVALS = ["default", "baseline", "shalem", "dream", "brooks"]

def parse_eval_types(eval_str: str) -> List[str]:
    """Parse comma-separated evaluation types and validate them."""
    if eval_str == "default":
        return ["default"]
    
    eval_types = [e.strip().lower() for e in eval_str.split(",")]
    invalid_types = [e for e in eval_types if e not in VALID_EVALS]
    if invalid_types:
        raise typer.BadParameter(f"Invalid evaluation types: {invalid_types}. Valid types are: {VALID_EVALS}")
    return eval_types

@app.command()
def evaluate(
    base_path: Optional[Path] = typer.Option(
        None,
        "--base-path",
        "-b",
        dir_okay=True,
        exists=False,  # Allow non‑existent until utils try to resolve
        resolve_path=True,
        help=(
            "Base path to the training‑run directory that holds checkpoints "
            "and config. Defaults to the current working directory if not "
            "provided."
        ),
    ),
    eval_types: str = typer.Option(
        "default",
        "--eval",
        "-e",
        help="Comma-separated list of evaluations to run. Use 'default' to run all evaluations.",
        callback=parse_eval_types,
    ),
    device: str = typer.Option(
        "cuda:0",
        "--device",
        "-d",
        help="Device to run evaluation on (e.g., 'cuda:0', 'cuda:1', 'cpu').",
    ),
):
    """Run the evaluation pipeline on a trained model."""

    bp: str | Path = base_path or ""  # Preserve original default behaviour

    print(f"Evaluating model found in {base_path}")
    print(f"Using device: {device}")

    print("Loading model and data...")
    model, test_loader, track_order, model_resolution = load_model_for_testing(base_path=str(bp), device=device)
    print("\tDone.")

    # Run selected evaluation(s)
    if "default" in eval_types or "baseline" in eval_types:
        print("Running baseline eval...")
        assert test_loader is not None, "test_loader should not be None for baseline evaluation"
        metrics, track_corr_df, true_values, pred_values = evaluate_model(
            model=model,
            test_loader=test_loader,
            track_name_order=track_order or [],
            output_dir=f"{bp}/eval",
            model_resolution=model_resolution,
            device=device,
        )
        print("\tDone.")

    if "default" in eval_types or "dream" in eval_types:
        print("Running DREAM challenge eval...")
        DREAM_path = f"{base_path}/DREAM"
        os.makedirs(DREAM_path, exist_ok=True)
        eval_DREAM(model=model, output_dir_base=DREAM_path, device=device, model_resolution=model_resolution)
        print("\tDone.")
    
    if "default" in eval_types or "shalem" in eval_types:
        print("Running Shalem Terminator eval...")
        Shalem_path = f"{base_path}/Shalem"
        os.makedirs(Shalem_path, exist_ok=True)
        eval_Shalem(model=model, output_dir_base=Shalem_path, device=device, model_resolution=model_resolution)
        print("\tDone.")
    
    if "default" in eval_types or "brooks" in eval_types:
        print("Running Brooks challenge eval...")
        Brooks_path = f"{base_path}/Brooks"
        os.makedirs(Brooks_path, exist_ok=True)
        eval_Brooks(
            model=model,
            device=device,
            track_annotation_path=f"{base_path}/track_annotation.json",
            model_resolution=model_resolution,
            brooks_eval_df_path=f"/home/tds122/clex/clex/eval/brooks/brooks_eval_df.pkl",
            output_dir=Brooks_path,
        )
        print("\tDone.")


if __name__ == "__main__":
    app()
