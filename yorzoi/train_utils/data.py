import pandas as pd
from torch.utils.data import DataLoader
from yorzoi.config import TrainConfig
from yorzoi.dataset import GenomicDataset, custom_collate_factory


def create_datasets(
    cfg: TrainConfig,
) -> tuple[GenomicDataset, GenomicDataset, GenomicDataset]:
    samples = pd.read_pickle(cfg.path_to_samples)
    if cfg.subset_data:
        for col in cfg.subset_data:
            samples = samples[samples[col].isin(cfg.subset_data[col])]

    train_samples = samples[samples["fold"] == "train"]
    val_samples = samples[samples["fold"] == "val"]
    test_samples = samples[samples["fold"] == "test"]

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

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    cfg: TrainConfig,
    train_dataset: GenomicDataset,
    val_dataset: GenomicDataset,
    test_dataset: GenomicDataset,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create the dataloaders for the training, validation, and test sets."""

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_train_loader,
        collate_fn=custom_collate_factory(resolution=cfg.resolution),
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_val_loader,
        collate_fn=custom_collate_factory(resolution=cfg.resolution),
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_test_loader,
        collate_fn=custom_collate_factory(resolution=cfg.resolution),
    )

    return train_loader, val_loader, test_loader
