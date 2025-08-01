"""
main training script
To run: python train.py args.config=$CONFIG_FILE_PATH
"""

import os
import hydra
import sys
from omegaconf import OmegaConf, DictConfig
from typing import Any, Dict, cast
import pytorch_lightning as pl
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from src.dataset.dataloader import EbirdVisionDataset
from src.transforms.transforms import get_transforms

#from pytorch_lightning.callbacks import ModelCheckpoint

from src.utils.config_utils import load_opts
from src.dataset.dataloader import get_subset
from src.trainer.utils import get_target_size, get_nb_bands, get_scheduler, init_first_layer_weights, \
    load_from_checkpoint
from src.losses.metrics import get_metrics

from src.trainer.trainer import EbirdDataModule
from src.utils.compute_normalization_stats import *

#train_hetreg_mac2.py
# Define your Mean-Variance ResNet18 Model.
class ResNet18Dropout(nn.Module):
    def __init__(self, output_dim, opts, use_sigmoid_mean=True):
        """
        Args:
            output_dim (int): Number of output components (N).
            use_sigmoid_mean (bool): Whether to apply a sigmoid activation on the mean predictions.
        """
        super(ResNet18Dropout, self).__init__()
        self.use_sigmoid_mean = use_sigmoid_mean
        self.resnet = models.resnet18(pretrained=True)
        self.opts = opts 

        if len(self.opts.data.bands) != 3 or len(self.opts.data.env) > 0:
            self.bands = self.opts.data.bands + self.opts.data.env
            orig_channels = self.resnet.conv1.in_channels
            weights = self.resnet.conv1.weight.data.clone()
            self.resnet.conv1 = nn.Conv2d(get_nb_bands(self.bands), 64, kernel_size=(7, 7), stride=(2, 2),
                                             padding=(3, 3), bias=False, )
            # assume first three channels are rgb
            if self.opts.experiment.module.pretrained:
                # self.model.conv1.weight.data[:, :orig_channels, :, :] = weights
                self.resnet.conv1.weight.data = init_first_layer_weights(get_nb_bands(self.bands), weights)
        
        self.feature_extractor = nn.Sequential(
            *list(self.resnet.children())[:-2],       # keep conv layers
            nn.Dropout2d(p=0.2)         # MC‑Dropout (epistemic)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        d = self.resnet.fc.in_features

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d, output_dim)
        )

    
    def forward(self, x):

        feat = self.feature_extractor(x)
        h = self.pool(feat)
        out = self.head(h)
        
        mean = out
        
        if self.use_sigmoid_mean:
            mean = torch.sigmoid(mean)

        return mean


def custom_cross_entropy_loss(pred, target, reduction='mean'):
        """
        target: ground truth
        pred: prediction
        reduction: mean, sum, none
        """
        loss = (-1 * target * torch.log(pred + 1e-7) - 1 * (1 - target) * torch.log(
            1 - pred + 1e-7))
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        else:  # reduction = None
            loss = loss

        return loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    config_path = parser.parse_args().config
    
    base_dir = os.getcwd()

    config_path = os.path.join(base_dir, config_path)
    default_config = os.path.join(base_dir, "configs/defaults.yaml")

    config = load_opts(config_path, default=default_config)
    global_seed = 877
    print(global_seed)
    print("SUP")

    config.variables.bioclim_means, config.variables.bioclim_std, config.variables.ped_means, config.variables.ped_std = compute_means_stds_env_vars(
            root_dir=config.data.files.base,
            train_csv=config.data.files.train,
            env=config.data.env,
            env_data_folder=config.data.files.env_data_folder,
            output_file_means=config.data.files.env_means,
            output_file_std=config.data.files.env_stds
        )
    
    config.variables.rgbnir_means, config.variables.rgbnir_std = compute_means_stds_images(
            root_dir=config.data.files.base,
            train_csv=config.data.files.train,
            output_file_means=config.data.files.rgbnir_means,
            output_file_std=config.data.files.rgbnir_stds)
    
    pl.seed_everything(global_seed)

    os.makedirs(config.save_path, exist_ok=True)
    os.makedirs(config.save_preds_path, exist_ok=True)

    with open(os.path.join(config.save_path, "config.yaml"), "w") as fp:
        OmegaConf.save(config=config, f=fp)
    fp.close()

    trainer_args = cast(Dict[str, Any], OmegaConf.to_object(config.trainer))
    print(trainer_args)

    print(config.experiment.module.freeze)

    freeze_backbone = config.experiment.module.freeze
    subset = get_subset(config.data.target.subset, config.data.total_species)
    target_size = get_target_size(config, subset)
    print(target_size)
    print("Predicting ", target_size, "species")

    target_type = config.data.target.type
    learning_rate = config.experiment.module.lr
    sigmoid_activation = nn.Sigmoid()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    print(config.experiment.module.lr)
    model = ResNet18Dropout(output_dim=755, opts=config)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.experiment.module.lr, weight_decay=0.00001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                     factor=config.scheduler.reduce_lr_plateau.factor,
                                                     patience=config.scheduler.reduce_lr_plateau.lr_schedule_patience
                                                     )

    print(config.scheduler.name)

    opts = config

    
    dm = EbirdDataModule(config)
    dm.prepare_data()
    dm.setup()

    train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()
    val_loader = dm.val_dataloader()

    num_epochs = config.max_epochs

    opts = config

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} / {num_epochs}")

        model.train()
        sys.stdout.flush()

        running_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            sys.stdout.flush()
            optimizer.zero_grad()

            x = batch['sat'].to(device).squeeze(1)
            y = batch['target'].to(device)

            hotspot_id = batch['hotspot_id']  

            weighted_loss_operations = {
                "sqrt": torch.sqrt,
                "log": torch.log,
                "nchklists": lambda x: x
            }

            weight_type = opts.experiment.module.loss_weight
            
            new_weights_val = weighted_loss_operations[weight_type](batch["num_complete_checklists"].to(device))
            new_weights = torch.ones_like(y, device=device) * new_weights_val.view(-1, 1)

            mean = model(x)

            pred = mean.type_as(y)

            loss = custom_cross_entropy_loss(pred, y)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        

        avg_loss = running_loss / (batch_idx + 1)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                x = batch['sat'].to(device).squeeze(1)
                y = batch['target'].to(device)
                hotspot_id = batch['hotspot_id']

                mean = model(x)

                pred = mean.type_as(y)
                
                loss = custom_cross_entropy_loss(pred, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                x = batch['sat'].to(device).squeeze(1)
                y = batch['target'].to(device)
                hotspot_id = batch['hotspot_id']

                mean = model(x)

                pred = mean.type_as(y)
                
                loss = custom_cross_entropy_loss(pred, y)
                test_loss += loss

                if opts.save_preds_path != "":
                    preds_path = opts.save_preds_path
                    for i, elem in enumerate(pred):
                        np.save(os.path.join(preds_path, batch["hotspot_id"][i] + ".npy"),
                            elem.cpu().detach().numpy())

        test_loss /= len(test_loader)
        print(f"Epoch {epoch+1} - Test Loss: {test_loss:.4f}")
        
        torch.save(model.state_dict(), opts.save_path + "/last.ckpt")
            
    var = 1
    print(device)

   