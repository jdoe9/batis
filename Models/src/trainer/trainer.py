"""
general trainer: supports training Resnet18, Satlas, and SATMAE
"""
import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset.dataloader import EbirdVisionDataset
from src.transforms.transforms import get_transforms

class EbirdDataModule(pl.LightningDataModule):
    def __init__(self, opts) -> None:
        super().__init__()
        self.opts = opts

        self.seed = self.opts.program.seed
        self.batch_size = self.opts.data.loaders.batch_size
        self.num_workers = self.opts.data.loaders.num_workers
        self.data_base_dir = self.opts.data.files.base
        self.targets_folder = self.opts.data.files.targets_folder
        self.env_data_folder = self.opts.data.files.env_data_folder
        self.images_folder = self.opts.data.files.images_folder
        self.df_train = pd.read_csv(os.path.join(self.data_base_dir, self.opts.data.files.train))
        self.df_val = pd.read_csv(os.path.join(self.data_base_dir, self.opts.data.files.val))
        self.df_test = pd.read_csv(os.path.join(self.data_base_dir, self.opts.data.files.test))
        self.bands = self.opts.data.bands
        self.env = self.opts.data.env
        self.env_var_sizes = self.opts.data.env_var_sizes
        self.datatype = self.opts.data.datatype
        self.target = self.opts.data.target.type
        self.subset = self.opts.data.target.subset
        self.res = self.opts.data.multiscale
        self.use_loc = self.opts.loc.use
        self.num_species = self.opts.data.total_species

    def setup(self, stage: Optional[str] = None) -> None:
        """create the train/test/val splits and prepare the transforms for the multires"""

        self.all_train_dataset = EbirdVisionDataset(df_paths=self.df_train, data_base_dir=self.data_base_dir,
                                                    bands=self.bands, env=self.env, env_var_sizes=self.env_var_sizes,
                                                    transforms=get_transforms(self.opts, "train"), mode="train",
                                                    datatype=self.datatype, target=self.target,
                                                    targets_folder=self.targets_folder,
                                                    env_data_folder=self.env_data_folder,
                                                    images_folder=self.images_folder, subset=self.subset, res=self.res,
                                                    use_loc=self.use_loc, num_species=self.num_species)

        self.all_test_dataset = EbirdVisionDataset(df_paths=self.df_test, data_base_dir=self.data_base_dir,
                                                   bands=self.bands, env=self.env, env_var_sizes=self.env_var_sizes,
                                                   transforms=get_transforms(self.opts, "val"), mode="test",
                                                   datatype=self.datatype, target=self.target,
                                                   targets_folder=self.targets_folder,
                                                   env_data_folder=self.env_data_folder,
                                                   images_folder=self.images_folder, subset=self.subset, res=self.res,
                                                   use_loc=self.use_loc, num_species=self.num_species)

        self.all_val_dataset = EbirdVisionDataset(df_paths=self.df_val, data_base_dir=self.data_base_dir,
                                                  bands=self.bands, env=self.env, env_var_sizes=self.env_var_sizes,
                                                  transforms=get_transforms(self.opts, "val"), mode="val",
                                                  datatype=self.datatype, target=self.target,
                                                  targets_folder=self.targets_folder,
                                                  env_data_folder=self.env_data_folder,
                                                  images_folder=self.images_folder, subset=self.subset, res=self.res,
                                                  use_loc=self.use_loc, num_species=self.num_species)

        # TODO: Create subsets of the data

        self.train_dataset = self.all_train_dataset

        self.test_dataset = self.all_test_dataset

        self.val_dataset = self.all_val_dataset

    def train_dataloader(self) -> DataLoader[Any]:
        """Returns the actual dataloader"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, )

    def val_dataloader(self) -> DataLoader[Any]:
        """Returns the validation dataloader"""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, )

    def test_dataloader(self) -> DataLoader[Any]:
        """Returns the test dataloader"""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, )
