#!/usr/bin/env python
# coding: utf-8

# CAREamics - 2D StructN2V for StructConvallaria Dataset
#-----------------------------------------------------------------

from pathlib import Path
import numpy as np
import tifffile
from careamics import CAREamist
from careamics.config import create_n2v_configuration
from careamics.utils.metrics import scale_invariant_psnr
from microssim import micro_structural_similarity

#### Load Dataset via Careamics Portfolio
from careamics_portfolio import PortfolioManager

# Download StructConvallaria dataset
portfolio = PortfolioManager()
files = portfolio.denoising.StructConvallaria.download(Path("data"))
conv_stack = tifffile.imread(files[0])

# Create ground truth by averaging all frames
gt_image = np.mean(conv_stack, axis=0)

# Select a single noisy image for training
train_image = conv_stack[50]
print(f"Ground truth shape: {gt_image.shape}")
print(f"Training image shape: {train_image.shape}")
print(f"StructConvallaria stack shape: {conv_stack.shape}")

#### StructN2V Configuration
config = create_n2v_configuration(
    experiment_name="structconvallaria_structn2v",
    data_type="array",
    axes="YX",
    patch_size=(64, 64),
    batch_size=16,
    num_epochs=50,
    masked_pixel_percentage=0.2,
    # StructN2V parameters for handling structured noise
    struct_n2v_axis="horizontal",
    struct_n2v_span=11,
    model_params={
        "num_channels_init": 32
    },
    optimizer_params={
        "lr": 0.0001
    },
    lr_scheduler_params={
        "factor": 0.5,
        "patience": 10
    }
)

#### Initialize CAREamist
careamist = CAREamist(source=config)

#### Run training
careamist.train(
    train_source=train_image,
    val_percentage=0.1
)

#############################################
############# Prediction ###################
#############################################

# Predict on the training image
prediction = careamist.predict(
    source=train_image,
    tile_size=(256, 256),
    tile_overlap=(64, 64)
)

# Test on additional frames
test_frames = [10, 25, 75, 90]
predictions = []
test_images = []

for frame_idx in test_frames:
    test_img = conv_stack[frame_idx]
    
    pred = careamist.predict(
        source=test_img,
        data_type="array",
        axes="YX",
        tile_size=(256, 256),
        tile_overlap=(64, 64)
    )
    
    predictions.append(pred[0].squeeze())
    test_images.append(test_img)

# Calculate metrics against ground truth
psnr_total = 0
microssim_total = 0
for pred, test_img in zip(predictions, test_images):
    psnr_total += scale_invariant_psnr(gt_image, pred)
    microssim_total += micro_structural_similarity(pred, gt_image)

print(f"Average PSNR: {psnr_total / len(predictions):.2f}")
print(f"Average MicroSSIM: {microssim_total / len(predictions):.2f}")
print("StructN2V training and evaluation completed successfully!")