"""
Script to train the model in all its different configurations using a json config file.

Usage:
python -m clex.train -c [path/to/config] -g [gpu name] -m [clex|base] -d [1-2 word description]

E.g.: python -m clex.train -c /home/tds122/clex/train_configs/template.json -g cuda:0 -m clex -d template_description
"""

import wandb
import os
import json
from time import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    ConstantLR,
    CosineAnnealingWarmRestarts,
)
from yorzoi.dataset import GenomicDataset, custom_collate_factory
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from yorzoi.loss import poisson_multinomial
from yorzoi.model.baseline import DNAConvNet
from yorzoi.config import BorzoiConfig, TrainConfig


# Helper ---------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Freeze logic for fine-tuning
# ---------------------------------------------------------------------------


def _freeze_backbone(model: nn.Module):
    """Freeze all layers except the prediction head(s).

    Works for the Borzoi model wrapped as ``nn.Sequential(TargetLengthCrop, Borzoi)``
    and for the simple ``DNAConvNet`` baseline.
    """

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


# Training loop --------------------------------------------------------------


def train_model(
    run_path: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=10,
    device="cuda:0",
    patience: int = 10,
    finetune_epochs: int = 10,
    randomize_track_order: bool = False,
    freeze_backbone: bool = True,
    finetune_lr_factor: float = 0.1,
    run_config=None,
):
    os.makedirs(f"{run_path}/predictions")

    model.to(device)

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")

    print("Model parameters dtype:", next(model.parameters()).dtype)

    # Track the early-stopping & phase-switch logic
    randomisation_active: bool = randomize_track_order
    epochs_without_improve: int = (
        0  # counts epochs w/o val-loss improvement (pre-train)
    )
    finetune_counter: int = 0  # counts epochs in fixed-order fine-tune phase
    backbone_frozen: bool = False  # ensure we freeze only once

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_poisson_loss = 0
        total_multinomial_loss = 0
        for i, batch in enumerate(train_loader):
            sequences, targets = batch[0], batch[1]

            # Convert inputs to float
            sequences = sequences.to(device).requires_grad_(True)

            targets = targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            if epoch == 0 and i == 0:
                visualize_input(sequences)

            if run_config.borzoi_cfg["flashed"]:
                with torch.autocast(device_type="cuda"):
                    outputs = model(sequences)
                    outputs = outputs.squeeze(
                        1
                    )  # NOTE: may be removed once output shapes are fixed
            else:
                outputs = model(sequences)

            # -----------------------------
            # Track-order randomisation (pre-training)
            # -----------------------------
            if randomisation_active:
                outputs, targets = _randomise_tracks(outputs, targets)

            # Calculate loss
            loss, _, multinomial_term, poisson_term = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            total_multinomial_loss += multinomial_term.item()
            total_poisson_loss += poisson_term.item()

            # Print progress
            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Batch [{i + 1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}"
                )

        avg_train_loss = total_loss / len(train_loader)
        avg_train_poisson_loss = total_poisson_loss / len(train_loader)
        avg_train_multinomial_loss = total_multinomial_loss / len(train_loader)
        wandb.log({"avg_train_loss": avg_train_loss, "epoch": epoch + 1})
        wandb.log(
            {"avg_train_poisson_loss": avg_train_poisson_loss, "epoch": epoch + 1}
        )
        wandb.log(
            {
                "avg_train_multinomial_loss": avg_train_multinomial_loss,
                "epoch": epoch + 1,
            }
        )
        print("Total loss:", total_loss)

        # Evaluate performance on validation set
        model.eval()
        val_loss = 0
        val_poisson_loss = 0
        val_multinomial_loss = 0
        with torch.no_grad():
            if run_config.borzoi_cfg["flashed"]:
                with torch.autocast(device_type="cuda"):
                    for i, batch in enumerate(val_loader):
                        sequences, targets = batch[0], batch[1]

                        sequences = sequences.to(device)
                        targets = targets.to(device)

                        outputs = model(sequences)
                        outputs = outputs.squeeze(1)  # TODO fix

                        # Apply track randomisation in the *same* way as for training.
                        if randomisation_active:
                            outputs, targets = _randomise_tracks(outputs, targets)

                        loss, _, multinomial_term, poisson_term = criterion(
                            outputs, targets
                        )
                        val_loss += loss.item()
                        val_multinomial_loss += multinomial_term.item()
                        val_poisson_loss += poisson_term.item()
            else:
                for i, batch in enumerate(val_loader):
                    sequences, targets = batch[0], batch[1]

                    sequences = sequences.to(device)
                    targets = targets.to(device)

                    outputs = model(sequences)
                    outputs = outputs.squeeze(1)  # TODO fix

                    if randomisation_active:
                        outputs, targets = _randomise_tracks(outputs, targets)

                    loss, _, multinomial_term, poisson_term = criterion(
                        outputs, targets
                    )
                    val_loss += loss.item()
                    val_multinomial_loss += multinomial_term.item()
                    val_poisson_loss += poisson_term.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_poisson_loss = val_poisson_loss / len(val_loader)
        avg_val_multinomial_loss = val_multinomial_loss / len(val_loader)

        wandb.log({"avg_val_loss": avg_val_loss, "epoch": epoch + 1})
        wandb.log(
            {"avg_val_poisson_loss": avg_val_poisson_loss, "epoch": epoch + 1}
        )  # TODO
        wandb.log(
            {"avg_val_multinomial_loss": avg_val_multinomial_loss, "epoch": epoch + 1}
        )  # TODO

        # Plot the last batch of validation predictions

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Average Train Loss: {avg_train_loss:.4f}, Average Validation Loss: {avg_val_loss:.4f}"
        )
        val_losses.append(avg_val_loss)
        train_losses.append(avg_train_loss)

        # Plot training and validation losses
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epoch + 2), train_losses, label="Training Loss")
        plt.plot(range(1, epoch + 2), val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        wandb.log({"loss_curve": wandb.Image(plt)})
        plt.savefig(f"{run_path}/training_validation_loss.png")
        plt.close()

        scheduler.step()

        # Save checkpoint if validation improved in current phase ------------------
        improved = avg_val_loss < best_val_loss
        if improved:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{run_path}/model_best.pth")
            wandb.save(f"{run_path}/model_best.pth")
            print(
                f"Saved best model at epoch {epoch + 1} (val_loss={avg_val_loss:.4f})"
            )

        # -----------------------------
        # Switch from pre-training to fine-tuning & decide when to stop
        # -----------------------------
        if randomisation_active:
            # we're in *pre-training*
            if improved:
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            if epochs_without_improve >= patience:
                print(
                    f"Validation loss has not improved for {patience} epochs. "
                    "Switching to fixed track-order fine-tuning."
                )
                randomisation_active = False  # disable permutation from next epoch
                epochs_without_improve = 0
                best_val_loss = float("inf")  # reset the running best

                if freeze_backbone and not backbone_frozen:
                    print(
                        "Freezing backbone layers and resetting optimizer for finetuning …"
                    )
                    _freeze_backbone(model)

                    # Re-instantiate optimizer & scheduler for head params only
                    prev_lr = (
                        optimizer.param_groups[0]["lr"]
                        if len(optimizer.param_groups)
                        else 1e-4
                    )
                    head_lr = finetune_lr_factor * prev_lr

                    optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=head_lr,
                    )

                    scheduler = CosineAnnealingLR(
                        optimizer, T_max=finetune_epochs, eta_min=1e-6
                    )

                    backbone_frozen = True
        else:
            # we're in *fine-tuning*
            finetune_counter += 1
            if finetune_counter >= finetune_epochs:
                print(
                    f"Completed {finetune_epochs} fine-tuning epochs. Training finished."
                )
                break

    wandb.save(json.dumps(run_config.__dict__))


def test_model(
    base_folder: str,
    test_loader: DataLoader,
    model: nn.Module,
    criterion,
    device="cuda:2",
):
    os.makedirs(f"{base_folder}/evaluations")
    print("\n=== MODEL EVALUATION ===")

    with torch.no_grad():
        # with torch.autocast(device_type="cuda"):
        test_loss = 0
        for i, batch in enumerate(test_loader):
            sequences, targets = batch[0], batch[1]

            if i % 10 == 0:
                print(f"\t[Batch {i}/{len(test_loader)}]")

            sequences = sequences.to(device)
            targets = targets.to(device)

            outputs = model(sequences)
            outputs = outputs.squeeze(1)  # TODO uglyy
            loss, _, _, _ = criterion(outputs, targets)
            test_loss += loss.item()

    print(f"\tMean batch loss: {test_loss / len(test_loader)}")

    test_loss /= len(test_loader)

    print(f"\Batch-wise Mean Test Loss: {test_loss:.4f}")

    # Save the test loss in json
    with open(f"{base_folder}/evaluations/test_loss.json", "w") as f:
        f.write(json.dumps({"test_loss": test_loss}))


def main(cfg_path: str, device: str, model_name: str, run_id: str):
    import os

    cfg = Config.read_from_json(cfg_path)

    torch.manual_seed(seed=cfg.seed)

    base_folder = f"runs/{run_id}"

    os.makedirs(base_folder)

    # Copy the config into the basefolder
    import shutil
    from pathlib import Path

    shutil.copy(cfg_path, Path(base_folder))
    shutil.move(
        f"{base_folder}/{cfg_path.split('/')[-1]}",
        dst=f"{base_folder}/train_config.json",
    )

    wandb.init(project="clex", config=cfg.__dict__, name=run_id)

    import pandas as pd

    print("Loading data...")

    samples = pd.read_pickle(cfg.path_to_samples)
    if cfg.subset_data:
        for col in cfg.subset_data:
            samples = samples[samples[col].isin(cfg.subset_data[col])]

    print("\tDone.")

    train_samples = samples[samples["fold"] == "train"]
    val_samples = samples[samples["fold"] == "val"]
    test_samples = samples[samples["fold"] == "test"]

    print("Creating train-val-test datasets...")
    train_dataset = GenomicDataset(
        train_samples,
        resolution=cfg.resolution,
        split_name="train",
        rc_aug=cfg.augmentation["rc_aug"],
        noise_tracks=cfg.augmentation["noise"],
    )
    val_dataset = GenomicDataset(
        val_samples, resolution=cfg.resolution, split_name="val"
    )
    test_dataset = GenomicDataset(
        test_samples, resolution=cfg.resolution, split_name="test"
    )
    print("\tDone.")

    # Print mean and std dev of train, val and test datasets
    print("== Approx. Dataset Statistics ==")
    print("\tTrain dataset mean:", train_dataset.mean_track_values)
    print("\tTrain dataset std dev:", train_dataset.std_track_values)
    print("\tValidation dataset mean:", val_dataset.mean_track_values)
    print("\tValidation dataset std dev:", val_dataset.std_track_values)

    print("\n")

    print(
        "Train dataset length:",
        len(train_dataset),
        f"({100 * (len(train_dataset) / len(samples)):.2f}%)",
        "| Validation dataset length:",
        len(val_dataset),
        f"({100 * (len(test_dataset) / len(samples)):.2f}%)",
        "| Test dataset length:",
        len(test_dataset),
        f"({100 * (len(test_dataset) / len(samples)):.2f}%)",
    )

    # Create the dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_train_loader,
        collate_fn=custom_collate_factory(resolution=cfg.resolution),
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )  # TODO: set shuffle back to true later when you have confirmed that the model is training correctly
    # or find a different way to plot the predictions for a fixed set of samples; since it's SGD,
    # the batches should be shuffled to prevent being stuck with a particularly unrepresentative
    # set of batches.
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_val_loader,
        collate_fn=custom_collate_factory(resolution=cfg.resolution),
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
    )  # Keep shuffle as False to check the progression of the model during training
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_test_loader,
        collate_fn=custom_collate_factory(resolution=cfg.resolution),
    )

    # Initialize the model
    model = None
    if model_name == "baseline":
        model = DNAConvNet()
    if model_name == "clex":
        from yorzoi.model.borzoi import Borzoi
        from yorzoi.model.utils import TargetLengthCrop

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

        model = Borzoi(config)

        if cfg.checkpoint_path:
            model.load_state_dict(torch.load(cfg.checkpoint_path, map_location="cpu"))

        model.to(device)
    else:
        raise ValueError("Unknown model! Valid options are 'DNAConvNet' and 'clex'")

    # Define loss function and optimizer
    def criterion(output, targets):
        return poisson_multinomial(
            output,
            targets,
            poisson_weight=cfg.loss["poisson_weight"],
            epsilon=cfg.loss["epsilon"],
            rescale=False,
            reduction=cfg.loss["reduction"],
        )

    optimizer = None
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
        raise ValueError("Unknown optimizer!")

    scheduler = None
    if cfg.scheduler == "steplr":
        scheduler = StepLR(optimizer, step_size=18, gamma=0.1)
    elif cfg.scheduler == "cosineannealinglr":
        scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    elif cfg.scheduler == "cosineannealingwr":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    elif cfg.scheduler == "constant":
        scheduler = ConstantLR(optimizer, factor=1, total_iters=1)
    else:
        raise ValueError("Unknown scheduler value")

    wandb.watch(model, log="all", log_freq=100)

    # summary(model, input_size=(1, 4992, 4), device=device)

    # Train the model
    train_model(
        run_path=base_folder,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=cfg.num_epochs,
        device=device,
        patience=cfg.patience,
        finetune_epochs=cfg.finetune_epochs,
        randomize_track_order=(cfg.randomize_track_order and model_name == "clex"),
        freeze_backbone=True,
        finetune_lr_factor=0.1,
        run_config=cfg,  # TODO: refactor such that config is only passed once
    )

    print(f"Trained for {(time() - t0) / 60} min")

    # Evaluate on the test set
    model.eval()

    test_model(  # TODO uncomment when real training runs start again
        base_folder=base_folder,
        test_loader=test_loader,
        model=model,
        criterion=criterion,
        device=device,
    )

    torch.save(model.state_dict(), f"{base_folder}/model_final.pth")
    model.save_pretrained(base_folder)
    wandb.save(f"{base_folder}/model_final.pth")
    repo_id = f"tom-ellis-lab/clex-{run_id}"
    model.push_to_hub(repo_id, private=False)
    wandb.finish()


if __name__ == "__main__":
    from datetime import datetime
    import argparse
    from time import time

    t0 = time()

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--description", type=str, default="")
    parser.add_argument("-m", "--model", type=str, default="")
    parser.add_argument("-g", "--gpu", type=str, default="")
    parser.add_argument("-c", "--cfg", type=str, default="")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.description:
        run_id += f"_{args.description}"

    print("=== Starting training run with args ===")
    print(f"\tDescription: {args.description}")
    print(f"\tModel: {args.model}")
    print(f"\tGPU: {args.gpu}")
    print(f"\Config: {args.cfg}")

    main(cfg_path=args.cfg, device=args.gpu, model_name=args.model, run_id=run_id)
