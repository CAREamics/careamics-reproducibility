#!/usr/bin/env python
# coding: utf-8

# CAREamics - 2D Example for BSD68 Data
# --------------------------------------
import pprint
from pathlib import Path

import numpy as np

import tifffile
from careamics_portfolio import PortfolioManager

from careamics_restoration.engine import Engine
from careamics_restoration.metrics import psnr

#### Import Dataset Portfolio
# Explore portfolio
portfolio = PortfolioManager()
print(portfolio.denoising)

# Download and unzip the files
root_path = Path("data")
files = portfolio.denoising.N2V_BSD68.download(root_path)
print(f"List of downloaded files: {files}")

data_path = Path(root_path / "denoising-N2V_BSD68.unzip/BSD68_reproducibility_data")
train_path = data_path / "train"
val_path = data_path / "val"
test_path = data_path / "test" / "images"
gt_path = data_path / "test" / "gt"

train_path.mkdir(parents=True, exist_ok=True)
val_path.mkdir(parents=True, exist_ok=True)
test_path.mkdir(parents=True, exist_ok=True)
gt_path.mkdir(parents=True, exist_ok=True)

#### training and validation data
train_image = tifffile.imread(next(iter(train_path.rglob("*.tiff"))))[0]
val_image = tifffile.imread(next(iter(val_path.rglob("*.tiff"))))[0]

#### Initialize the Engine
engine = Engine(config_path="n2v_2D_BSD.yml")
pprint.PrettyPrinter(indent=2).pprint(engine.cfg.model_dump(exclude_optionals=False))

#### Run training
train_stats, val_stats = engine.train(train_path=train_path, val_path=val_path)


#############################################
#############   Prediction   ################
#############################################
def PSNR(gt, img):
    """PSNR calculation between ground truth and noisy image"""
    mse = np.mean(np.square(gt - img))
    return 20 * np.log10(255) - 10 * np.log10(mse)


preds = engine.predict(
    input=test_path, tile_shape=[64, 64], overlaps=[48, 48], axes="YX"
)

# Create a list of ground truth images
gts = [tifffile.imread(f) for f in sorted(gt_path.glob("*.tiff"))]

psnr_total = 0
for pred, gt in zip(preds, gts):
    psnr_total += psnr(gt, pred)

print(f"PSNR total: {psnr_total / len(preds)}")
