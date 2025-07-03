#!/usr/bin/env python
# coding: utf-8

# CAREamics - 2D Example for BSD68 Dataset
# ----------------------------------------
from pathlib import Path
import numpy as np
import tifffile
from careamics_portfolio import PortfolioManager
from careamics import CAREamist
from careamics.config import create_n2v_configuration
from careamics.utils.metrics import scale_invariant_psnr
from microssim import micro_structural_similarity

#### Load Dataset via Careamics Portfolio
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

#### Load all training and validation data
def load_images_from_path(path):
    """Load all TIFF images from a directory into a numpy array."""
    image_files = sorted(path.glob("*.tiff"))
    images = []
    
    for file in image_files:
        img = tifffile.imread(file)
        if img.ndim == 3: 
            for frame in img:
                images.append(frame)
        else:  
            images.append(img)
    
    return np.array(images)

# Load all training and validation images
train_images = load_images_from_path(train_path)
val_images = load_images_from_path(val_path)

print(f"Loaded {len(train_images)} training images")
print(f"Loaded {len(val_images)} validation images")

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
    logger="wandb", 
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
    train_source=train_images, 
    val_source=val_images
)

#############################################
#############   Prediction   ################
#############################################

# Load test images and ground truth
test_files = sorted(test_path.glob("*.tiff"))
gt_files = sorted(gt_path.glob("*.tiff"))

# Predict on test images 
predictions = []
gt_images = []

for test_file, gt_file in zip(test_files, gt_files):
    test_img = tifffile.imread(test_file)

    pred = careamist.predict(
        source=test_img, 
        data_type="array",
        axes="YX",
        tile_size=(128, 128),
        tile_overlap=(48, 48)
    )
    
    predictions.append(pred[0].squeeze())
    gt_images.append(tifffile.imread(gt_file))

# Calculate metrics
psnr_values = []
microssim_values = []
for pred, gt in zip(predictions, gt_images):
    psnr_values.append(scale_invariant_psnr(gt, pred))
    microssim_values.append(micro_structural_similarity(pred, gt))

print(f"Average PSNR: {np.mean(psnr_values):.2f} ± {np.std(psnr_values):.2f}")
print(f"Average MicroSSIM: {np.mean(microssim_values):.2f} ± {np.std(microssim_values):.2f}")