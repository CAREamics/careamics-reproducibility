import platform
from importlib.metadata import distribution

import torch

from common import IMPL


def get_run_meta(algo, dataset, seed):
    return {
        "impl": IMPL,
        "algo": algo,
        "dataset": dataset,
        "seed": seed,
        "careamics_version": distribution("careamics").version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "hostname": platform.node(),
    }
