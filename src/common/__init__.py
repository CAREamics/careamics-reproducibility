import os
from pathlib import Path

# deterministic cuBLAS needs a fixed workspace, set before the first CUDA matmul
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

DATA_DIR = Path("data")
RUNS_DIR = Path("/project/careamics/repro")
RESULTS_DIR = Path("results")
WANDB_PROJECT = "careamics-reproducibility"
