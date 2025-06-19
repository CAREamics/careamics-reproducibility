#!/usr/bin/env python
# coding: utf-8

# CAREamics - 3D CARE for Tribolium Dataset
#----------------------------------------------------------------

from pathlib import Path
import numpy as np
import requests
import zipfile
import io
import tifffile
from careamics import CAREamist
from careamics.config import create_care_configuration
from careamics.utils.metrics import scale_invariant_psnr
from microssim import micro_structural_similarity

#### Download and Load Dataset

# Download and extract the dataset
url = 'http://csbdeep.bioimagecomputing.com/example_data/tribolium.zip'
print("Downloading and extracting Tribolium dataset...")
response = requests.get(url)
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    z.extractall('data')
print("Dataset successfully downloaded and extracted to 'data'.")

# Define paths to training and test data
train_low_path = Path('data/tribolium/train/low/nGFP_0.1_0.2_0.5_20_13_late.tif')
train_gt_path = Path('data/tribolium/train/GT/nGFP_0.1_0.2_0.5_20_13_late.tif')
test_low_path = Path('data/tribolium/test/low/nGFP_0.1_0.2_0.5_20_14_late.tif')
test_gt_path = Path('data/tribolium/test/GT/nGFP_0.1_0.2_0.5_20_14_late.tif')

# Load training images
train_low_images = tifffile.imread(train_low_path)
train_gt_images = tifffile.imread(train_gt_path)

print(f"Training Data (Low SNR): Shape: {train_low_images.shape}")
print(f"Training Data (Ground Truth): Shape: {train_gt_images.shape}")

#### Configuration
config = create_care_configuration(
    experiment_name="CARE_Tribolium",
    data_type="array",
    axes="ZYX",
    patch_size=(16, 64, 64),
    batch_size=1,
    num_epochs=100,
    model_params={
        "num_channels_init": 32
    },
    optimizer_params={
        "lr": 0.0004
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
    train_source=train_low_images,
    train_target=train_gt_images,
    val_percentage=0.05,
    val_minimum_split=20
)

#############################################
############# Prediction ###################
#############################################

# Load test images
test_low_images = tifffile.imread(test_low_path)
test_gt_images = tifffile.imread(test_gt_path)

print(f"Test Data (Low SNR): Shape: {test_low_images.shape}")
print(f"Test Data (Ground Truth): Shape: {test_gt_images.shape}")

# Perform prediction with the test data
predictions = careamist.predict(
    source=test_low_images,
    data_type="array",
    axes="ZYX",
    tile_size=(16, 64, 64),
    tile_overlap=(8, 32, 32),
    batch_size=1,
    tta=False
)

# Convert predictions to array and remove unnecessary dimensions
test_predictions = np.squeeze(np.array(predictions))

print(f"Test predictions shape: {test_predictions.shape}")

# Calculate metrics against ground truth
psnr_total = 0
microssim_total = 0

# Calculate metrics on subset of slices
num_slices = test_low_images.shape[0]
slice_indices = [num_slices // 4, num_slices // 2, 3 * num_slices // 4]

for slice_idx in slice_indices:
    pred_slice = test_predictions[slice_idx]
    gt_slice = test_gt_images[slice_idx]
    
    psnr_total += scale_invariant_psnr(gt_slice, pred_slice)
    microssim_total += micro_structural_similarity(pred_slice, gt_slice)

print(f"Average PSNR (subset): {psnr_total / len(slice_indices):.2f}")
print(f"Average MicroSSIM (subset): {microssim_total / len(slice_indices):.2f}")
print("Training and evaluation completed successfully!")
