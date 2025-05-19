#!/usr/bin/env python
# coding: utf-8
# CAREamics - 2D Example for BSD68 Data 
# ------------------------------------------------------
import os
from pathlib import Path
import numpy as np
import tifffile
from careamics import CAREamist
from careamics.config import create_n2v_configuration
from careamics.utils.metrics import scale_invariant_psnr
from careamics_portfolio import PortfolioManager

# Create directories
root_path = Path("data")
root_path.mkdir(exist_ok=True)

# Results directory
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Explore portfolio
print("Loading BSD68 dataset...")
portfolio = PortfolioManager()

# Download and unzip the files
files = portfolio.denoising.N2V_BSD68.download(root_path)
print(f"List of downloaded files: {files}")

data_path = Path(root_path / "denoising-N2V_BSD68.unzip/BSD68_reproducibility_data")
train_path = data_path / "train"
val_path = data_path / "val"
test_path = data_path / "test" / "images"
gt_path = data_path / "test" / "gt"

# Load training image for N2V
train_files = list(train_path.glob("*.tiff"))
if len(train_files) == 0:
    raise FileNotFoundError(f"No training files found in {train_path}")

print(f"Loading training file: {train_files[0]}")
train_image = tifffile.imread(train_files[0])[0]
print(f"Training image shape: {train_image.shape}")

# Configure Noise2Void model
print("Configuring Noise2Void model...")
config = create_n2v_configuration(
    experiment_name="n2v_BSD",
    data_type="array",
    axes="YX",
    patch_size=(64, 64),
    batch_size=128,
    num_epochs=50,
    masked_pixel_percentage=0.2,
    struct_n2v_axis="none",
    model_checkpoint={"save_top_k": 3, "monitor": "val_loss"},
)

# Update the model parameters to match original script
config["algorithm_config"]["model"]["num_channels_init"] = 32
config["algorithm_config"]["optimizer"]["parameters"]["lr"] = 0.0004
config["algorithm_config"]["lr_scheduler"]["parameters"]["factor"] = 0.5

# Initialize and train model
print("Initializing and training model...")
working_dir = os.path.join(results_dir, "checkpoints")
os.makedirs(working_dir, exist_ok=True)

careamist = CAREamist(
    source=config,
    working_dir=working_dir
)

careamist.train(
    train_source=train_image,
    val_percentage=0.1,
)

# Prediction and evaluation
print("Making predictions and evaluating results...")
test_files = sorted(test_path.glob("*.tiff"))
gt_files = sorted(gt_path.glob("*.tiff"))

if len(test_files) == 0 or len(gt_files) == 0:
    raise FileNotFoundError(f"No test or ground truth files found")

print(f"Found {len(test_files)} test images and {len(gt_files)} ground truth images")

# Create predictions
psnr_noisy_values = []
psnr_denoised_values = []

for i, (test_file, gt_file) in enumerate(zip(test_files, gt_files)):
    print(f"Processing image {i+1}/{len(test_files)}: {test_file.name}")
    
    # Load test and ground truth images
    test_img = tifffile.imread(test_file)
    gt_img = tifffile.imread(gt_file)
    
    # Make prediction
    prediction = careamist.predict(
        source=test_img,
        tile_size=(64, 64),
        tile_overlap=(48, 48)
    )
    
    # Get prediction image
    pred_img = prediction[0].squeeze()
    
    # Calculate PSNR
    psnr_noisy = scale_invariant_psnr(gt_img, test_img)
    psnr_denoised = scale_invariant_psnr(gt_img, pred_img)
    
    psnr_noisy_values.append(psnr_noisy)
    psnr_denoised_values.append(psnr_denoised)
    
    # Save prediction
    output_file = results_dir / f"pred_{test_file.name}"
    tifffile.imwrite(str(output_file), pred_img.astype(np.float32))
    
    print(f" PSNR noisy: {psnr_noisy:.2f}, PSNR denoised: {psnr_denoised:.2f}")

# Calculate average PSNR
avg_psnr_noisy = np.mean(psnr_noisy_values)
avg_psnr_denoised = np.mean(psnr_denoised_values)

# Save metrics 
with open(str(results_dir / "metrics.txt"), "w") as f:
    f.write(f"Average PSNR of noisy input: {avg_psnr_noisy:.2f}\n")
    f.write(f"Average PSNR of N2V denoised: {avg_psnr_denoised:.2f}\n")
    f.write("\nIndividual PSNR values:\n")
    for i, (noisy, denoised) in enumerate(zip(psnr_noisy_values, psnr_denoised_values)):
        f.write(f"Image {i+1}: Noisy {noisy:.2f}, Denoised {denoised:.2f}\n")

print("\nSummary:")
print(f"Average PSNR of noisy input: {avg_psnr_noisy:.2f}")
print(f"Average PSNR of N2V denoised: {avg_psnr_denoised:.2f}")
print(f"PSNR improvement: {avg_psnr_denoised - avg_psnr_noisy:.2f}")