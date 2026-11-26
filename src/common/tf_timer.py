import time
from contextlib import contextmanager

import tensorflow as tf


@contextmanager
def tf_timer():
    timer = {}
    tf.config.experimental.reset_memory_stats("GPU:0")
    start = time.perf_counter()
    yield timer
    timer["seconds"] = time.perf_counter() - start
    timer["peak_memory_mib"] = (
        tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**20
    )
