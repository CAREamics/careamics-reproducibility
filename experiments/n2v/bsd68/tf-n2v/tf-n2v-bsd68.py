#!/usr/bin/env python
# coding: utf-8

# Noise2Void - 2D Example for BSD68 Data
# TensorFlow implementation
# --------------------------------------
# The data used in this notebook is the same as presented in the paper, and this
# script follows the jupyter notebook.

import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from careamics_portfolio import PortfolioManager
from n2v.models import N2VConfig, N2V
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Check GPU availability
print(tf.config.list_physical_devices("GPU"))

# Root to save data and model
user = os.environ.get("USER")
root = Path("/scratch") / user / "reproducibility"
if not root.exists():
    root.mkdir(parents=True)

# Experiment folder
experiment = Path(root, "n2v", "tf-n2v-bsd68")
if not experiment.exists():
    experiment.mkdir(parents=True)

#############################################
###############   Data   ####################
#############################################

# Data Preparation
data_path = root / "data"
if not data_path.exists():
    data_path.mkdir()
print(f"Path to data: {data_path}")

portfolio = PortfolioManager()
paths = portfolio.denoising.N2V_BSD68.download(data_path)
print(f"Files downloaded to: {paths}")

# Load training data
X_train = np.load(
    data_path
    / "denoising-N2V_BSD68.unzip"
    / "BSD68_reproducibility_data"
    / "train"
    / "DCNN400_train_gaussian25.npy"
)
X_val = np.load(
    data_path
    / "denoising-N2V_BSD68.unzip"
    / "BSD68_reproducibility_data"
    / "val"
    / "DCNN400_validation_gaussian25.npy"
)

# Add channel dimension
X_train = X_train[..., np.newaxis]
print(f"Train data dimension {X_train.shape}")
X_val = X_val[..., np.newaxis]
print(f"Validation data dimension {X_val.shape}")

#############################################
###########   Configuration   ###############
#############################################

config = N2VConfig(
    X_train,
    unet_kern_size=3,
    train_epochs=100,
    train_steps_per_epoch=400,
    train_loss="mse",
    batch_norm=True,
    train_batch_size=128,
    n2v_perc_pix=0.198,
    n2v_patch_shape=(64, 64),
    unet_n_first=96,
    unet_residual=True,
    n2v_manipulator="uniform_withCP",
    n2v_neighborhood_radius=2,
    single_net_per_channel=False,
)

# Let's look at the parameters stored in the config-object.
print("Configuration:")
print(vars(config))

#############################################
###############   Model   ###################
#############################################

# A name used to identify the model
model_name = "BSD68_reproducibility_5x5"

# Base directory in which our model will live
model_path = experiment / "model"
if not model_path.exists():
    model_path.mkdir()

# We are now creating our network model.
model = N2V(config, model_name, basedir=model_path)
model.prepare_for_training(metrics=())

#############################################
##############   Training   #################
#############################################
history = model.train(X_train, X_val)


#############################################
#############   Prediction   ################
#############################################
def PSNR(gt, img):
    """PSNR calculation between ground truth and noisy image"""
    mse = np.mean(np.square(gt - img))
    return 20 * np.log10(255) - 10 * np.log10(mse)


# Load ground test ground truth
groundtruth_data = np.load(
    data_path
    / "denoising-N2V_BSD68.unzip"
    / "BSD68_reproducibility_data"
    / "test"
    / "bsd68_groundtruth.npy",
    allow_pickle=True,
)
test_data = np.load(
    data_path
    / "denoising-N2V_BSD68.unzip"
    / "BSD68_reproducibility_data"
    / "test"
    / "bsd68_gaussian25.npy",
    allow_pickle=True,
)

# Load weights corresponding to the smallest validation loss
# Smallest validation loss does not necessarily correspond to best performance,
# because the loss is computed to noisy target pixels.
model.load_weights("weights_best.h5")

# Predict
pred = []
psnrs = []
psnrs_noisy = []
for gt, img in zip(groundtruth_data, test_data):
    prediction = model.predict(img.astype(np.float32), "YX", tta=False)
    pred.append(prediction)
    psnrs.append(PSNR(gt, prediction))
    psnrs_noisy.append(PSNR(gt, img))

# Compute mean and std
psnrs = np.array(psnrs)
psnrs_noisy = np.array(psnrs_noisy)
avg = np.round(np.mean(psnrs), 2)
avg_noisy = np.round(np.mean(psnrs_noisy), 2)
std = np.round(np.std(psnrs), 2)
std_noisy = np.round(np.std(psnrs_noisy), 2)
print(f"PSNR (noisy): {avg_noisy} ± {std_noisy}")
print(f"PSNR (pred without tta): {avg} ± {std}")

# Save results
path_results = experiment / "results"
if not path_results.exists():
    path_results.mkdir()

for i, img in enumerate(pred):  # note: test images have inhomogeneous XY dims
    name = f"prediction_tf_n2v_bsd68_gaussian25_{i}.npy"
    np.save(path_results / name, img)
