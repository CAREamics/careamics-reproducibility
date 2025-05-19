#!/usr/bin/env python
# coding: utf-8
# CAREamics - N2V Example for Convallaria Data
# -------------------------------------------
import os
from pathlib import Path
import numpy as np
import tifffile
from careamics import CAREamist
from careamics.config import create_n2v_configuration
from careamics.utils.metrics import scale_invariant_psnr

# Create data directories
root_path = Path("data")
data_path = root_path / "convallaria"
data_path.mkdir(parents=True, exist_ok=True)

# Paths to data files
conv_path = "/group/jug/datasets/Convallaria_diaphragm/20190520_tl_25um_50msec_05pc_488_130EM_Conv.tif"
fd_path = "/group/jug/datasets/Convallaria_diaphragm/20190726_tl_50um_500msec_wf_130EM_FD.tif"

# Results directory
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Load the stacks
print("Loading Convallaria stack...")
conv_stack = tifffile.imread(conv_path)
fd_stack = tifffile.imread(fd_path)
print("Convallaria stack shape:", conv_stack.shape)
print("Field diaphragm stack shape:", fd_stack.shape)

# Create ground truth by averaging all frames
gt_image = np.mean(conv_stack, axis=0)

# Select a single noisy image for training (frame 50)
train_image = conv_stack[50]
print(f"Ground truth shape: {gt_image.shape}")
print(f"Training image shape: {train_image.shape}")

# Configure Noise2Void model
print("Configuring Noise2Void model...")
config = create_n2v_configuration(
    experiment_name="convallaria_n2v",
    data_type="array",
    axes="YX",
    patch_size=(64, 64),
    batch_size=16,
    num_epochs=100,
    masked_pixel_percentage=0.2,
    struct_n2v_axis="none",
    model_checkpoint={},
)

# Initialize and train model
print("Initializing and training model...")
working_dir = os.path.join(results_dir, "checkpoints")
os.makedirs(working_dir, exist_ok=True)

careamist = CAREamist(source=config, working_dir=working_dir)
careamist.train(
    train_source=train_image,
    val_percentage=0.1,
)

# Predict
print("Making predictions...")
prediction = careamist.predict(
    source=train_image,
    tile_size=(256, 256),
    tile_overlap=(64, 64)
)

# Save the prediction
tifffile.imwrite(str(results_dir / "prediction.tif"), prediction[0].squeeze().astype(np.float32))

# Calculate PSNR
psnr_noisy = scale_invariant_psnr(gt_image, train_image)
psnr_denoised = scale_invariant_psnr(gt_image, prediction[0].squeeze())

# Save metrics to a text file
with open(str(results_dir / "metrics.txt"), "w") as f:
    f.write(f"PSNR of noisy input: {psnr_noisy:.2f}\n")
    f.write(f"PSNR of N2V denoised: {psnr_denoised:.2f}\n")

print(f"PSNR of noisy input: {psnr_noisy:.2f}")
print(f"PSNR of N2V denoised: {psnr_denoised:.2f}")