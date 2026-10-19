import time
from contextlib import contextmanager

import torch


@contextmanager
def cuda_timer():
    timer = {}
    torch.cuda.synchronize()
    start = time.perf_counter()
    yield timer
    torch.cuda.synchronize()
    timer["seconds"] = time.perf_counter() - start
