import torch
import pandas as pd
from torch.utils.data import Dataset
import pathlib
from typing import Union, Optional, Any, Dict
import numpy as np
import pyBigWig
from torch.utils.data import DataLoader
import os, json
from yorzoi.utils import untransform_then_unbin


class BrooksDataset(Dataset):
    """Dataset wrapping *brooks_eval_df.pkl* for inference.

    Each item is a tuple ``(seq_tensor, meta_dict)`` where:
        seq_tensor : torch.FloatTensor of shape (L, 4) – one-hot encoded DNA
        meta_dict  : {"row_idx", "+_file", "-_file"}
    """

    def __init__(self, pkl_path: str, resolution: int = 1):
        if not pathlib.Path(pkl_path).exists():
            raise FileNotFoundError(pkl_path)

        self.df = pd.read_pickle(pkl_path)
        self.resolution = resolution  # For potential downsampling

    def __len__(self):
        return len(self.df)

    def _one_hot(self, seq: str) -> torch.Tensor:
        from yorzoi.constants import nucleotide2onehot

        arr = [nucleotide2onehot.get(base.upper(), [0, 0, 0, 0]) for base in seq]
        return torch.tensor(arr, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row["sequence"]  # already 5000 bp (or longer)
        seq_tensor = self._one_hot(seq)

        meta = {
            "row_idx": int(self.df.index[idx]),
            "+_file": row["+_file"],
            "-_file": row["-_file"],
        }

        return seq_tensor, meta
