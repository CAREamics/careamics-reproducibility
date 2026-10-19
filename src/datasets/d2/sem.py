from pathlib import Path

import numpy as np
import tifffile
from careamics_portfolio import PortfolioManager

from datasets.data import Data


def sem(data_dir: Path):
    files = sorted(PortfolioManager().denoising.N2V_SEM.download(data_dir))
    train_path, val_path = files

    train = tifffile.imread(train_path).astype(np.float32)
    val = tifffile.imread(val_path).astype(np.float32)

    return Data(train=train, val=val)


if __name__ == "__main__":
    sem(Path("data"))
