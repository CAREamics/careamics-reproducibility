"""2D crops, settings come from DenoiseSeg."""

from pathlib import Path

import numpy as np
from careamics_portfolio import PortfolioManager

from datasets.data import Data

TRAIN_NPZ_NAME = "train_data.npz"
TEST_NPZ_NAME = "test_data.npz"
NOISE_STD = 70.0
SEED = 42


def flywing(data_dir: Path, noise_std=NOISE_STD, seed=SEED):
    files = PortfolioManager().denoiseg.Flywing_n0.download(data_dir)

    with np.load(data_dir / TRAIN_NPZ_NAME) as archive:
        clean_train = archive["X_train"].astype(np.float32)
        clean_val = archive["X_val"].astype(np.float32)

    with np.load(data_dir / TEST_NPZ_NAME) as archive:
        test_target = archive["X_test"].astype(np.float32)

    rng = np.random.default_rng(seed)
    train = clean_train + rng.normal(0.0, noise_std, clean_train.shape)
    train = train.astype(np.float32)
    val = clean_val + rng.normal(0.0, noise_std, clean_val.shape)
    val = val.astype(np.float32)
    test = test_target + rng.normal(0.0, noise_std, test_target.shape)
    test = test.astype(np.float32)
    return Data(train=train, val=val, test=test, test_target=test_target)


if __name__ == "__main__":
    flywing(Path("data"))
