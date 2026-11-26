"""2D timelapse, setup comes from the HDN repo."""

from pathlib import Path

import numpy as np
import pooch
from tifffile import imread

from datasets.data import Data

URL = "https://zenodo.org/records/5156913/files/Convallaria_diaphragm.zip"
KNOWN_HASH = "sha256:d2d60c12b2331fc69f708139361cc2ee0665ceaa7291af98a68d9964f5f1490b"
STACK_NAME = "20190520_tl_25um_50msec_05pc_488_130EM_Conv.tif"
CALIBRATION_NAME = "20190726_tl_50um_500msec_wf_130EM_FD.tif"
TRAIN_SPLIT = 0.85
EVAL_CROP = 512


def convallaria(data_dir: Path):
    pooch.retrieve(
        url=URL,
        known_hash=KNOWN_HASH,
        fname="Convallaria_diaphragm.zip",
        path=data_dir,
        processor=pooch.Unzip(extract_dir="Convallaria"),
    )
    data_path = data_dir / "Convallaria" / "Convallaria_diaphragm"
    stack = imread(data_path / STACK_NAME)
    stack = stack.astype(np.float32)
    observation = imread(data_path / CALIBRATION_NAME)
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
