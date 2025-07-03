# Copyright 2023 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


from yorzoi.config import BorzoiConfig
from transformers import PreTrainedModel
import torch.nn as nn
import torch
import numpy as np
import math
import copy
from pathlib import Path

from .utils import Residual, TargetLengthCrop, undo_squashed_scale
from .attn_modules import Attention, FlashAttention

import pandas as pd

# torch.backends.cudnn.deterministic = True

# torch.set_float32_matmul_precision('high')


class ConvDna(nn.Module):
    def __init__(self, out_channels, resolution):
        super(ConvDna, self).__init__()
        self.conv_layer = nn.Conv1d(
            in_channels=4, out_channels=out_channels, kernel_size=15, padding="same"
        )
        self.max_pool = (
            nn.MaxPool1d(kernel_size=2, padding=0) if resolution > 1 else nn.Identity()
        )

    def forward(self, x):
        return self.max_pool(self.conv_layer(x))


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels=None, kernel_size=1, conv_type="standard"
    ):
        super(ConvBlock, self).__init__()
        if conv_type == "separable":
            self.norm = nn.Identity()
            depthwise_conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                groups=in_channels,
                padding="same",
                bias=False,
            )
            pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            self.conv_layer = nn.Sequential(depthwise_conv, pointwise_conv)
            self.activation = nn.Identity()
        else:
            self.norm = nn.BatchNorm1d(in_channels, eps=0.001)
            self.activation = nn.GELU(approximate="tanh")
            self.conv_layer = nn.Conv1d(
                in_channels, out_channels, kernel_size=kernel_size, padding="same"
            )

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv_layer(x)
        return x


class Borzoi(PreTrainedModel):
    config_class = BorzoiConfig
    base_model_prefix = "borzoi"

    @staticmethod
    def from_hparams(**kwargs):
        return Borzoi(BorzoiConfig(**kwargs))

    def __init__(self, config):
        super(Borzoi, self).__init__(config)
        self.flashed = config.flashed if "flashed" in config.__dict__.keys() else False
        self.enable_human_head = (
            config.enable_human_head
            if "enable_human_head" in config.__dict__.keys()
            else True
        )
        self.enable_mouse_head = config.enable_mouse_head
        self.conv_dna = ConvDna(out_channels=256, resolution=config.resolution)
        self._max_pool = nn.MaxPool1d(kernel_size=2, padding=0)
        self.resolution = config.resolution
        print("resolution:", self.resolution)
        if self.resolution > 1:
            self.res_tower = nn.Sequential(
                # EDIT: reduced number of ConvBlocks in tower as sequence length is much smaller
                ConvBlock(in_channels=256, out_channels=320, kernel_size=5),
                self._max_pool,
                ConvBlock(in_channels=320, out_channels=384, kernel_size=5),
                self._max_pool,
                ConvBlock(in_channels=384, out_channels=448, kernel_size=5),
            )
        else:
            self.res_tower = nn.Sequential(
                ConvBlock(in_channels=256, out_channels=320, kernel_size=5),
                ConvBlock(in_channels=320, out_channels=384, kernel_size=5),
                ConvBlock(in_channels=384, out_channels=448, kernel_size=5),
            )

        self.unet1 = nn.Sequential(
            self._max_pool,
            ConvBlock(
                in_channels=448, out_channels=512, kernel_size=5
            ),  # EDIT: changed in_channels
        )
        transformer = []
        for _ in range(config.depth):
            transformer.append(
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.LayerNorm(config.dim, eps=0.001),
                            (
                                Attention(
                                    config.dim,
                                    heads=config.heads,
                                    dim_key=config.attn_dim_key,
                                    dim_value=config.attn_dim_value,
                                    dropout=config.attn_dropout,
                                    pos_dropout=config.pos_dropout,
                                    num_rel_pos_features=32,
                                )
                                if not self.flashed
                                else FlashAttention(
                                    config.dim,
                                    heads=config.heads,
                                    dropout=config.attn_dropout,
                                    pos_dropout=config.pos_dropout,
                                )
                            ),
                            nn.Dropout(0.2),
                        )
                    ),
                    Residual(
                        nn.Sequential(
                            nn.LayerNorm(config.dim, eps=0.001),
                            nn.Linear(config.dim, config.dim * 2),
                            nn.Dropout(config.dropout_rate),
                            nn.ReLU(),
                            nn.Linear(config.dim * 2, config.dim),
                            nn.Dropout(config.dropout_rate),
                        )
                    ),
                )
            )

        self.horizontal_conv0, self.horizontal_conv1 = (
            ConvBlock(
                in_channels=config.horizontal_conv0["in_channels"],
                out_channels=config.horizontal_conv0["out_channels"],
                kernel_size=1,
            ),  # EDIT: changed in_channels and out_channels
            ConvBlock(in_channels=config.dim, out_channels=config.dim, kernel_size=1),
        )
        self.upsample = torch.nn.Upsample(scale_factor=2)
        self.transformer = nn.Sequential(*transformer)
        self.upsampling_unet1 = nn.Sequential(
            ConvBlock(in_channels=config.dim, out_channels=config.dim, kernel_size=1),
            self.upsample,
        )
        self.separable1 = nn.Sequential(
            ConvBlock(
                in_channels=config.separable1["separ_conv"]["in_channels"],
                out_channels=config.separable1["separ_conv"]["out_channels"],
                kernel_size=3,
                conv_type="separable",
            ),
            nn.Conv1d(  # TS: I added this to reduce channel numbers as borzoi predicts many more channels than we initially do.
                # TS: we might remove this once we increase the channel numbers
                in_channels=config.separable1["separ_conv"]["out_channels"],
                out_channels=config.separable1["conv1d"]["out_channels"],
                kernel_size=1,
            ),
        )
        self.upsampling_unet0 = nn.Sequential(
            ConvBlock(
                in_channels=config.upsampling_unet0["in_channels"],
                out_channels=config.upsampling_unet0["out_channels"],
                kernel_size=1,
            ),
            self.upsample,
        )
        self.separable0 = nn.Sequential(
            ConvBlock(
                in_channels=config.separable0["in_channels"],
                out_channels=config.separable0["in_channels"],
                kernel_size=3,
                conv_type="separable",
            ),
            nn.Conv1d(
                in_channels=config.separable0["in_channels"],
                out_channels=config.separable0["out_channels"],
                kernel_size=1,
            ),  # TS: added this. Check out separable1 for details.
        )
        if config.return_center_bins_only:
            self.crop = TargetLengthCrop(3000 // config.resolution)
        else:
            self.crop = TargetLengthCrop(
                499
            )  # as in Borzoi # TODO this is untested and not thought through
        self.final_joined_convs = nn.Sequential(
            ConvBlock(
                in_channels=config.final_joined_convs["in_channels"],
                out_channels=config.final_joined_convs["out_channels"],
                kernel_size=1,
            ),
            nn.Dropout(0.1),
            nn.GELU(approximate="tanh"),
        )
        if self.enable_human_head:
            self.human_head = nn.Conv1d(
                in_channels=config.head["in_channels"],
                out_channels=config.head["out_channels"],
                kernel_size=1,
            )
        if self.enable_mouse_head:
            self.mouse_head = nn.Conv1d(
                in_channels=1920, out_channels=2608, kernel_size=1
            )
        self.final_softplus = nn.Softplus()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # module.weight.data.normal_(mean=0.0, std=0.02)
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
            module.bias.data.zero_()

    def set_track_subset(self, track_subset):
        """
        Creates a subset of tracks by reassigning weights in the human head.

        Args:
           track_subset: Indices of the tracks to keep.

        Returns:
            None
        """
        if not hasattr(self, "human_head_bak"):
            self.human_head_bak = copy.deepcopy(self.human_head)
        else:
            self.reset_track_subset()
        self.human_head = nn.Conv1d(1920, len(track_subset), 1)
        self.human_head.weight = nn.Parameter(
            self.human_head_bak.weight[track_subset].clone()
        )
        self.human_head.bias = nn.Parameter(
            self.human_head_bak.bias[track_subset].clone()
        )

    def reset_track_subset(self):
        """
        Resets the human head to the original weights.

        Returns:
            None
        """
        self.human_head = copy.deepcopy(self.human_head_bak)

    def get_embs_after_crop(self, x):
        """
        Performs the forward pass of the model until right before the final conv layers, and includes a cropping layer.

        Args:
            x (torch.Tensor): Input DNA sequence tensor of shape (N, 4, L).

        Returns:
             torch.Tensor: Output of the model up to the cropping layer with shape (N, dim, crop_length)
        """
        x = x.permute(0, 2, 1)
        x = self.conv_dna(x)
        x_unet0 = self.res_tower(x)
        x_unet1 = self.unet1(x_unet0)

        x = self._max_pool(x_unet1)

        x_unet1 = self.horizontal_conv1(x_unet1)

        x_unet0 = self.horizontal_conv0(x_unet0)

        x = self.transformer(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.upsampling_unet1(x)
        x += x_unet1
        x = self.separable1(x)
        x = self.upsampling_unet0(x)
        x += x_unet0
        x = self.separable0(x)
        x = self.crop(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)

    def predict(self, seqs, gene_slices, remove_squashed_scale=False):
        """
        Predicts only for bins of interest in a batched fashion
        Args:
            seqs (torch.tensor): Nx4xL tensor of one-hot sequences
            gene_slices List[torch.Tensor]: tensors indicating bins of interest
            removed_squashed_scale (bool, optional): whether to undo the squashed scale

        Returns:
            Tuple[torch.Tensor, list[int]]: 1xCxB tensor of bin predictions, as well as offsets that indicate where sequences begin/end
        """
        # Calculate slice offsets
        slice_list = []
        slice_length = []
        offset = self.crop.target_length
        for i, gene_slice in enumerate(gene_slices):
            slice_list.append(gene_slice + i * offset)
            slice_length.append(gene_slice.shape[0])
        slice_list = torch.concatenate(slice_list)
        # Get embedding after cropped
        seq_embs = self.get_embs_after_crop(seqs)
        # Reshape to flatten the batch dimension (i.e. concatenate sequences)
        seq_embs = seq_embs.permute(1, 0, 2).flatten(start_dim=1).unsqueeze(0)
        # Extract the bins of interest
        seq_embs = seq_embs[:, :, slice_list]
        # Run the model head
        seq_embs = self.final_joined_convs(seq_embs)
        with torch.cuda.amp.autocast(enabled=False):
            conved_slices = self.final_softplus(self.human_head(seq_embs.float()))
        if remove_squashed_scale:
            conved_slices = undo_squashed_scale(conved_slices)
        return conved_slices, slice_length

    def forward(self, x, is_human=True, data_parallel_training=False):
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): Input DNA sequence tensor of shape (N, 4, L).
            is_human (bool, optional): If True, use the human head; otherwise, use the mouse head. Defaults to True.
            data_parallel_training (bool, optional): If True, perform forward pass specific to DDP. Defaults to False.

        Returns:
            torch.Tensor: Output tensor with shape (N, C, L), where C is the number of tracks.
        """
        x = self.get_embs_after_crop(x)
        x = self.final_joined_convs(x)
        # disable autocast for more precision in final layer
        with torch.cuda.amp.autocast(enabled=False):
            if data_parallel_training:
                # we need this to get gradients for both heads if doing DDP training
                if is_human:
                    human_out = (
                        self.final_softplus(self.human_head(x.float()))
                        + 0 * self.mouse_head(x.float()).sum()
                    )
                    return human_out
                else:
                    mouse_out = (
                        self.final_softplus(self.mouse_head(x.float()))
                        + 0 * self.human_head(x.float()).sum()
                    )
                    return mouse_out
            else:
                if is_human:
                    return self.final_softplus(self.human_head(x.float()))
                else:
                    return self.final_softplus(self.mouse_head(x.float()))
