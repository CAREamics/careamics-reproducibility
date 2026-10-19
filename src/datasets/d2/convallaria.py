"""2D timelapse, setup comes from the HDN repo."""

from pathlib import Path

import numpy as np
from careamics_portfolio import PortfolioManager
from tifffile import imread

from datasets.data import Data

STACK_NAME = "20190520_tl_25um_50msec_05pc_488_130EM_Conv.tif"
CALIBRATION_NAME = "20190726_tl_50um_500msec_wf_130EM_FD.tif"
TRAIN_SPLIT = 0.85
EVAL_CROP = 512


def convallaria(data_dir: Path):
    downloaded = PortfolioManager().denoising.Convallaria.download(data_dir)
    stack = imread(data_dir / STACK_NAME)
    stack = stack.astype(np.float32)
    observation = imread(data_dir / CALIBRATION_NAME)
    observation = observation.astype(np.float32)

    split = int(TRAIN_SPLIT * stack.shape[0])
    test = stack[:, :EVAL_CROP, :EVAL_CROP]
    signal = np.broadcast_to(
        observation.mean(axis=0)[np.newaxis], observation.shape
    ).copy()
    return Data(
        train=stack[:split],
        val=stack[split:],
        test=test,
        test_target=test.mean(axis=0),
        calibration_signal=signal,
        calibration_observation=observation,
    )


if __name__ == "__main__":
    convallaria(Path("data"))
