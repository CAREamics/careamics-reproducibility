"""
N2V, HDN and CAREamics use different test images processing:
- N2V mirror pads in the normalized space
- HDN crops to even and zero pads in normalized space
- CAREmics uses tiled inference

Experimentally, for N2V the N2V and CAREamics processing have similar results, HDN is
slightly worse.
"""

from pathlib import Path

from careamics_portfolio import PortfolioManager

from datasets.data import Data


def bsd68(data_dir: Path):
    data = PortfolioManager().denoising.N2V_BSD68.download(data_dir)
    data_path = data_dir / "denoising-N2V_BSD68.unzip/BSD68_reproducibility_data"
    train_path = data_path / "train"
    val_path = data_path / "val"
    test_path = data_path / "test" / "images"
    gt_path = data_path / "test" / "gt"

    return Data(train=train_path, val=val_path, test=test_path, test_target=gt_path)


if __name__ == "__main__":
    bsd68(Path("data"))
