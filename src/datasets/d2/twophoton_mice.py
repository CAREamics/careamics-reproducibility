"""
NOISE_LEVEL: from HDN
TEST_FOV: from the FMD paper
VAL_FOVS: assumption
CALIBRATION_CAPTURES_PER_FOV: assumption
"""

import tarfile
from pathlib import Path

import gdown
import imageio.v3 as imageio
import numpy as np

from datasets.data import Data

FOLDER_NAME = "TwoPhoton_MICE"
GDRIVE_ID = "1lhsFAlXsXk26yqHzT0_-3R8MUb7G0NVa"
NOISE_LEVEL = "raw"
NUM_FOVS = 20
TEST_FOV = 19
VAL_FOVS = (17, 18)
TRAIN_FOVS = [f for f in range(1, NUM_FOVS + 1) if f != TEST_FOV and f not in VAL_FOVS]
CALIBRATION_CAPTURES_PER_FOV = 10


def twophoton_mice(data_dir: Path):
    root = data_dir / FOLDER_NAME
    if not root.is_dir():
        data_dir.mkdir(parents=True, exist_ok=True)
        archive = data_dir / f"{FOLDER_NAME}.tar"
        gdown.download(id=GDRIVE_ID, output=str(archive))
        with tarfile.open(archive) as tar:
            tar.extractall(data_dir)
        archive.unlink()

    train, signal, observation = [], [], []
    for fov in TRAIN_FOVS:
        paths = sorted((root / NOISE_LEVEL / str(fov)).glob("*.png"))
        captures = np.stack([imageio.imread(p).astype(np.float32) for p in paths])
        ground_truth = imageio.imread(root / "gt" / str(fov) / "avg50.png").astype(
            np.float32
        )
        train.append(captures)
        observation.append(captures[:CALIBRATION_CAPTURES_PER_FOV])
        signal.append(
            np.broadcast_to(
                ground_truth[np.newaxis],
                (CALIBRATION_CAPTURES_PER_FOV, *captures.shape[1:]),
            )
        )

    val = []
    for fov in VAL_FOVS:
        paths = sorted((root / NOISE_LEVEL / str(fov)).glob("*.png"))
        val.append(np.stack([imageio.imread(p).astype(np.float32) for p in paths]))

    paths = sorted((root / NOISE_LEVEL / str(TEST_FOV)).glob("*.png"))
    test = np.stack([imageio.imread(p).astype(np.float32) for p in paths])
    test_target = imageio.imread(root / "gt" / str(TEST_FOV) / "avg50.png").astype(
        np.float32
    )

    return Data(
        train=np.concatenate(train),
        val=np.concatenate(val),
        test=test,
        test_target=test_target,
        calibration_signal=np.concatenate(signal).copy(),
        calibration_observation=np.concatenate(observation),
    )


if __name__ == "__main__":
    twophoton_mice(Path("data"))
