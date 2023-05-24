#!/usr/bin/env python3

# Noise2Void - 2D Example for BSD68 Data
# torch implementation (PPN2V)
# --------------------------------------
# The data used in this notebook is the same as presented in the PPN2V paper.


from pathlib import Path

from microscopy_portfolio import Portfolio
import numpy as np
import torch

from unet.model import UNet
from pn2v import utils
from pn2v import training
from pn2v import prediction


# See if we can use a GPU
device = utils.getDevice()
print(f"Device {device} found.")


#############################################
###############   Data   ####################
#############################################

# Data Preparation
data_path = Path(__file__).parent.parent / "data"
if not data_path.exists():
    data_path.mkdir()

portfolio = Portfolio()
portfolio.denoising.N2V_BSD68.download(data_path)

# Load training data
X_train = np.load(
    data_path / "BSD68_reproducibility_data" / "train" / "DCNN400_train_gaussian25.npy"
)
X_val = np.load(
    data_path
    / "BSD68_reproducibility_data"
    / "train"
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
    "train_epochs": 10,
    "train_steps_per_epoch": 400,
    "train_batch_size": 128,
    "train_learning_rate": 0.0004,
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
trainHist, valHist = training.trainNetwork(
    net=net,
    trainData=X_train,
    valData=X_val,
    postfix="N2V",
    directory="model/",
    device=device,
    numOfEpochs=configuration["train_epochs"],
    stepsPerEpoch=configuration["train_steps_per_epoch"],
    batchSize=configuration["train_batch_size"],
    learningRate=configuration["train_learning_rate"],
    augment=True,
    patchSize=64,
    noiseModel=None,
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
    data_path / "BSD68_reproducibility_data" / "train" / "bsd68_groundtruth.npy",
    allow_pickle=True,
)
test_data = np.load(
    data_path / "BSD68_reproducibility_data" / "train" / "bsd68_gaussian25.npy",
    allow_pickle=True,
)

# Load weights
model_path = Path("model", "last_N2V.net")
net = torch.load(model_path)

# Predict
pred = []
psnrs = []
psnrs_noisy = []
for gt, img in zip(groundtruth_data, test_data):
    prediction, _ = prediction.predict(img, net, None, device, outScaling=10)
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
print(f"PSNR (without tta): {avg} ± {std}")
