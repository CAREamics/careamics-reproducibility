import time
from contextlib import contextmanager

import torch


@contextmanager
def cuda_timer():
    timer = {}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    yield timer
    torch.cuda.synchronize()
    timer["seconds"] = time.perf_counter() - start
    timer["peak_memory_mib"] = torch.cuda.max_memory_allocated() / 2**20
