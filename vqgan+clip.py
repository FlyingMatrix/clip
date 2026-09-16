"""
    VQGAN (Vector Quantized Generative Adversarial Network) + CLIP (Contrastive Language-Image Pre-training)

    The script downloads a PRETRAINED VQGAN checkpoint automatically on first run (ImageNet f16, 16384-codebook — the classic VQGAN+CLIP model), ~1 GB, cached locally afterwards. 
    Alternative checkpoints are listed in CHECKPOINTS below.
"""

import os
import sys
import math
import urllib.request

import torch
import torch.nn.functional as F
from torch import nn, optim
import torchvision.transforms.functional as TF
from PIL import Image

import open_clip
from omegaconf import OmegaConf


# Config
device = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT      = "A watercolor painting of a frog on a lily pad"
IMAGE_SIZE  = (384, 384)     # output resolution (multiples of 16 for the f16 model)
NUM_CUTS    = 32
TOTAL_STEPS = 600
LR          = 0.1            # Adam lr on the latent (latent space likes a big lr)
SEED        = 42

# Optional: start from a real image instead of random latent.
# Set to a path, or leave None for pure text-to-image.
INIT_IMAGE  = None           # e.g. "/path/to/watercolor_pool.png"

CHECKPOINTS = {
    # name: (config_url, checkpoint_url, is_gumbel)
    "imagenet_f16_16384": (
        "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1",
        "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1",
        False,
    ),
    # A smaller/faster option (1024 codebook):
    "imagenet_f16_1024": (
        "https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1",
        "https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fckpts%2Flast.ckpt&dl=1",
        False,
    ),
}
MODEL_NAME = "imagenet_f16_16384"
CKPT_DIR   = "vqgan_checkpoints"


# Straight-through helpers (let gradients flow through argmin/clamp)
class ReplaceGrad(torch.autograd.Function):
    """
        
    """
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)

replace_grad = ReplaceGrad.apply

