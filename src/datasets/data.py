from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np

DataSource: TypeAlias = np.ndarray | list[np.ndarray] | Path | list[Path]


@dataclass(kw_only=True)
class Data:
    train: DataSource | None = None
    train_target: DataSource | None = None
    val: DataSource | None = None
    val_target: DataSource | None = None
    test: DataSource | None = None
    test_target: DataSource | None = None
    test_slices: list[tuple[slice, ...]] | None = None
    calibration_signal: DataSource | None = None
    calibration_observation: DataSource | None = None
