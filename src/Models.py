import torch
import torch.nn as nn
from diffusers.models.vae import Decoder

class Voxel2StableDiffusionModel(torch.nn.Module):
    # define the prototype of the module
    def __init__(self, in_dim=39548, h=2048, n_blocks=4):
        super().__init__()

        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h, bias=False),
            nn.LayerNorm(h),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h, bias=False),
                nn.LayerNorm(h),
                nn.SiLU(inplace=True),
                nn.Dropout(0.3),
            ) for _ in range(n_blocks)
        ])
        
        self.lin1 = nn.Linear(h, 16384, bias=False)
        self.norm = nn.GroupNorm(1, 64)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

        init_weights(self.lin0)
        self.mlp.apply(init_weights)
        init_weights(self.lin1)

        self.upsampler = Decoder(
            in_channels=64,
            out_channels=4,
            up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
            block_out_channels=[64, 128, 256],
            layers_per_block=1,
        )
        for parm in self.upsampler.parameters():
            parm.require_grad = False
        self.upsampler.eval()

    # define how it forward, using the module defined above
    def forward(self, x):
        x = self.lin0(x)
        residual = x
        for res_block in self.mlp:
            x = res_block(x)
            x = x + residual
            residual = x
        x = x.reshape(len(x), -1)
        x = self.lin1(x)
        x = self.norm(x.reshape(x.shape[0], -1, 16, 16).contiguous())
        return self.upsampler(x)

import os
from utils import load_image
from torch.utils.data import Dataset

class MyDataset(Dataset):
  def __init__(self, fmri_data, images_folder, transform=None):
    self.fmri_data = fmri_data
    self.images_folder = images_folder
    self.image_paths = [f"{images_folder}/{filename}" for filename in os.listdir(images_folder)]
    self.transform = transform

  def __len__(self):
    return len(self.fmri_data)

  def __getitem__(self, idx):
    fmri = self.fmri_data[idx]
    image_path = self.image_paths[idx]
    image = load_image(image_path)

    if(self.transform):
      image = self.transform(image)

    return fmri, image