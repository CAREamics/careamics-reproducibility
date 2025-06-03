#!/usr/bin/env python
# coding: utf-8

# CAREamics - 2D Example for Convallaria Data
# ---------------------------------------------------
from pathlib import Path
import numpy as np
import tifffile
from careamics import CAREamist
from careamics.config import create_n2v_configuration
from careamics.utils.metrics import scale_invariant_psnr
from microssim import micro_structural_similarity

# Path to dataset
conv_path = "/group/jug/datasets/Convallaria_diaphragm/20190520_tl_25um_50msec_05pc_488_130EM_Conv.tif"

# Load the stack and create ground truth
conv_stack = tifffile.imread(conv_path)
print(f"Convallaria stack shape: {conv_stack.shape}")

# Create ground truth by averaging all frames
gt_image = np.mean(conv_stack, axis=0)

# Select single noisy image for training (frame 50)
train_image = conv_stack[50]

#### configuration
config = create_n2v_configuration(
    experiment_name="n2v_convallaria",
    data_type="array",
    axes="YX",
    patch_size=(64, 64),
    batch_size=16,
    num_epochs=100,
    masked_pixel_percentage=0.2,
    struct_n2v_axis="none",
)

#### Initialize CAREamist
careamist = CAREamist(source=config)

#### Run training
careamist.train(
    train_source=train_image,
    val_percentage=0.1,
)

#############################################
############# Prediction ################### 
#############################################

# Create test images from different frames
test_frames = [10, 25, 75, 90]
predictions = []
gt_images = []

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
    gt_images.append(gt_image)

# Calculate metrics
psnr_total = 0
microssim_total = 0
for pred, gt in zip(predictions, gt_images):
    psnr_total += scale_invariant_psnr(gt, pred)
    microssim_total += micro_structural_similarity(pred, gt)

print(f"Average PSNR: {psnr_total / len(predictions):.2f}")
print(f"Average MicroSSIM: {microssim_total / len(predictions):.2f}")