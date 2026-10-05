"""
DATA_FILE, CALIBRATION_FILE: from PPN2V (examples/MouseSkullNuclei)
TEST_CROP: from PPN2V, matches HDN "200 images of size 512 x 256"
SPLIT_FRACTION: from the HDN Convallaria notebook, HDN has no Mouse nuclei notebook
"""

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import tifffile

from datasets.data import Data

FOLDER_NAME = "Mouse skull nuclei"
URL = "https://zenodo.org/records/5156960/files/Mouse%20skull%20nuclei.zip?download=1"
DATA_FILE = "example2_digital_offset300.tif"
CALIBRATION_FILE = "edgeoftheslide_300offset.tif"
SPLIT_FRACTION = 0.85
TEST_CROP = 256


def mouse_nuclei(data_dir: Path):
    root = data_dir / FOLDER_NAME
    if not root.is_dir():
        data_dir.mkdir(parents=True, exist_ok=True)
        archive = data_dir / f"{FOLDER_NAME}.zip"
        urllib.request.urlretrieve(URL, archive)
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.namelist() if not m.startswith("__MACOSX")]
            zf.extractall(data_dir, members=members)
        archive.unlink()

    stack = tifffile.imread(root / DATA_FILE).astype(np.float32)
    split = int(SPLIT_FRACTION * stack.shape[0])
    test = stack[:, :, :TEST_CROP]

    calibration = tifffile.imread(root / CALIBRATION_FILE).astype(np.float32)
    calibration_signal = np.broadcast_to(
        calibration.mean(axis=0)[np.newaxis], calibration.shape
    )

    return Data(
        train=stack[:split],
        val=stack[split:],
        test=test,
        test_target=test.mean(axis=0),
        calibration_signal=calibration_signal.copy(),
        calibration_observation=calibration,
    )


if __name__ == "__main__":
    mouse_nuclei(Path("data"))
