"""
Copied from https://github.com/johahi/borzoi-pytorch
"""

import json
from transformers import PretrainedConfig


class BorzoiConfig(PretrainedConfig):
    model_type = "borzoi"

    def __init__(
        self,
        dim=1536,
        depth=8,
        heads=8,
        # output_heads = dict(human = 5313, mouse= 1643),
        return_center_bins_only=True,
        attn_dim_key=64,
        attn_dim_value=192,
        dropout_rate=0.2,
        attn_dropout=0.05,
        pos_dropout=0.01,
        enable_mouse_head=False,
        **kwargs,
    ):
        self.dim = dim
        self.depth = depth
        self.heads = heads
        # self.output_heads = output_heads
        self.attn_dim_key = attn_dim_key
        self.attn_dim_value = attn_dim_value
        self.dropout_rate = dropout_rate
        self.attn_dropout = attn_dropout
        self.pos_dropout = pos_dropout
        self.return_center_bins_only = return_center_bins_only
        self.enable_mouse_head = enable_mouse_head
        super().__init__(**kwargs)


class TrainConfig:
    def __init__(
        self,
        batch_size=10,
        num_epochs=100,
        scheduler="cosineannealinglr",
        patience=30,
        optimizer="adam",
        lr=1e-4,
        subset_data=None,
        path_to_samples=None,
        fwd_track_only="",
        shuffle_train_loader="",
        shuffle_val_loader="",
        shuffle_test_loader="",
        borzoi_cfg=None,
        checkpoint_path=None,
        seed=42,
        finetune_epochs=10,
        randomize_track_order=True,
    ):
        self.config_path = ""
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.scheduler = scheduler
        self.patience = patience
        self.optimizer = optimizer
        self.lr = lr
        self.subset_data = subset_data
        self.path_to_samples = path_to_samples
        self.fwd_track_only = fwd_track_only
        self.shuffle_train_loader = shuffle_train_loader
        self.shuffle_val_loader = shuffle_val_loader
        self.shuffle_test_loader = shuffle_test_loader
        self.borzoi_cfg = borzoi_cfg if borzoi_cfg else {}
        self.checkpoint_path = checkpoint_path
        self.resolution = borzoi_cfg["resolution"] if borzoi_cfg else 10
        self.seed = seed
        self.finetune_epochs = finetune_epochs
        self.randomize_track_order = randomize_track_order

    @classmethod
    def read_from_json(cls, path: str):
        with open(path, "r") as f:
            json_cfg = json.load(f)
        # Create a default instance
        instance = cls()
        # Overwrite attributes with those in the json file
        instance.__dict__.update(json_cfg)
        # Optionally store the config file path
        instance.config_path = path
        return instance
