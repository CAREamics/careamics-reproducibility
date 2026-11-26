from pathlib import Path

import numpy as np
from tifffile import imread

from datasets.data import Data

FOLDER_NAME = "Denoising_Planaria"
CONDITION = "condition_1"
TRAIN_SPLIT = 0.9
CALIBRATION_PATCHES = 500
PCT_LOW = 1.0
PCT_HIGH = 99.8


def planaria(data_dir: Path):
    # TODO: download the Denoising_Planaria archive into `data_dir` and extract it.
    root = data_dir / FOLDER_NAME
    with np.load(root / "train_data" / "data_label.npz") as archive:
        noisy, clean = archive["X"][:, 0], archive["Y"][:, 0]
    split = int(TRAIN_SPLIT * noisy.shape[0])

    test, test_target = [], []
    for path in sorted((root / "test_data" / CONDITION).glob("*.tif")):
        volume = imread(path).astype(np.float32)
        low, high = np.percentile(volume, PCT_LOW), np.percentile(volume, PCT_HIGH)
        test.append((volume - low) / (high - low))

        volume = imread(root / "test_data" / "GT" / path.name).astype(np.float32)
        low, high = np.percentile(volume, PCT_LOW), np.percentile(volume, PCT_HIGH)
        test_target.append((volume - low) / (high - low))

    return Data(
        train=noisy[:split],
        val=noisy[split:],
        test=test,
        test_target=test_target,
        calibration_signal=clean[:CALIBRATION_PATCHES],
        calibration_observation=noisy[:CALIBRATION_PATCHES],
    )
