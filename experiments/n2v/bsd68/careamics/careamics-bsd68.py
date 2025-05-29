#!/usr/bin/env python
# coding: utf-8

# CAREamics - 2D Example for BSD68 Data
# --------------------------------------
from pathlib import Path
import numpy as np
import tifffile
from careamics_portfolio import PortfolioManager
from careamics import CAREamist
from careamics.config import create_n2v_configuration
from careamics.utils.metrics import scale_invariant_psnr
from microssim import micro_structural_similarity

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

#### configuration
config = create_n2v_configuration(
    experiment_name="n2v_BSD",
    data_type="array",
    axes="SYX",
    patch_size=(64, 64),
    batch_size=128,
    num_epochs=50,
    masked_pixel_percentage=0.2,
    struct_n2v_axis="none",
)

#### Initialize CAREamist
careamist = CAREamist(source=config)

#### Run training
careamist.train(
    train_source=train_path,
    val_source=val_path
)

#############################################
#############   Prediction   ################
#############################################

# Load test images and ground truth
test_images = [tifffile.imread(f) for f in sorted(test_path.glob("*.tiff"))]
gt_images = [tifffile.imread(f) for f in sorted(gt_path.glob("*.tiff"))]

# Predict on test images
predictions = []
for test_img in test_images:
    pred = careamist.predict(
        source=test_img,
        tile_size=(128, 128),
        tile_overlap=(48, 48)
    )
    predictions.append(pred[0].squeeze())

# Calculate metrics
psnr_total = 0
microssim_total = 0
for pred, gt in zip(predictions, gt_images):
    psnr_total += scale_invariant_psnr(gt, pred)
    microssim_total += micro_structural_similarity(pred, gt)

print(f"Average PSNR: {psnr_total / len(predictions):.2f}")
print(f"Average MicroSSIM: {microssim_total / len(predictions):.2f}")