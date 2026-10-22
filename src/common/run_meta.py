import json
import platform
from importlib.metadata import distribution

import torch

from common import IMPL


def get_run_meta(algo, dataset, seed, mode):
    careamics = distribution("careamics")
    return {
        "impl": IMPL,
        "algo": algo,
        "dataset": dataset,
        "seed": seed,
        "mode": mode,
        "careamics_version": careamics.version,
        "careamics_commit": json.loads(careamics.read_text("direct_url.json"))[
            "vcs_info"
        ]["commit_id"],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "hostname": platform.node(),
    }
