#!/usr/bin/env python3

# Noise2Void - 2D Example for BSD68 Data
# torch implementation (PPN2V)
# --------------------------------------
# The data used in this notebook is the same as presented in the PPN2V paper.

import os
from pathlib import Path

from careamics_portfolio import PortfolioManager
import numpy as np
import torch

from ppn2v.unet import UNet
from ppn2v.pn2v import utils, training, prediction


# TODO use the paths given by the portfolio manager

# See if we can use a GPU
device = utils.getDevice()
print(f"Device {device} found.")

# Root to save data and model
user = os.environ.get("USER")
root = Path("/scratch", user, "reproducibility")
if not root.exists():
    root.mkdir(parents=True)

# Experiment folder
experiment = Path(root, "n2v", "torch-ppn2v-bsd68")
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

# Adding channel dimension
X_train = X_train[..., np.newaxis]
print(f"Train data dimension {X_train.shape}")
X_val = X_val[..., np.newaxis]
print(f"Validation data dimension {X_val.shape}")


#############################################
###########   Configuration   ###############
#############################################
configuration = {
    "unet_n_depth": 2,
    "unet_n_first": 96,
    "train_epochs": 100,
    "train_steps_per_epoch": 400,
    "train_batch_size": 128,
    "train_learning_rate": 0.0004,
    "numMaskedPixels": 9,
    "patchSize": 64,
    "valSize": 4,
}
print(f"Configuration: {configuration}")


#############################################
###############   Model   ###################
#############################################
net = UNet(
    1, depth=configuration["unet_n_depth"], start_filts=configuration["unet_n_first"]
)

#############################################
##############   Training   #################
#############################################
model_path = experiment / "model"
if not model_path.exists():
    model_path.mkdir()

trainHist, valHist = training.trainNetwork(
    net=net,
    trainData=X_train,
    valData=X_val,
    postfix="N2V",
    directory=model_path,
    device=device,
    numOfEpochs=configuration["train_epochs"],
    stepsPerEpoch=configuration["train_steps_per_epoch"],
    batchSize=configuration["train_batch_size"],
    learningRate=configuration["train_learning_rate"],
    numMaskedPixels=configuration["numMaskedPixels"],
    patchSize=configuration["patchSize"],
    valSize=configuration["valSize"],
    noiseModel=None,
    augment=False,
)


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

# Load weights
trained_model_path = model_path / "last_N2V.net"
net = torch.load(trained_model_path)

# Predict
pred = []
psnrs = []
psnrs_noisy = []
for gt, img in zip(groundtruth_data, test_data):
    pred_arr = prediction.tiledPredict(
        img, net, ps=128, overlap=32, noiseModel=None, device=device, outScaling=10
    )
    pred.append(pred_arr)
    psnrs.append(PSNR(gt, pred_arr))
    psnrs_noisy.append(PSNR(gt, img))

# Compute mean and std
psnrs = np.array(psnrs)
psnrs_noisy = np.array(psnrs_noisy)
avg = np.round(np.mean(psnrs), 2)
avg_noisy = np.round(np.mean(psnrs_noisy), 2)
std = np.round(np.std(psnrs), 2)
std_noisy = np.round(np.std(psnrs_noisy), 2)
print(f"PSNR (noisy): {avg_noisy} ± {std_noisy}")
print(f"PSNR (prediction): {avg} ± {std}")

# Save results
path_results = experiment / "results"
if not path_results.exists():
    path_results.mkdir()

for i, img in enumerate(pred):  # note: test images have inhomogeneous XY
    name = f"prediction_tf_n2v_bsd68_gaussian25_{i}.npy"
    np.save(path_results / name, img)
