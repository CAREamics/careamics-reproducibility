import json
import platform
import time
from importlib.metadata import version

import numpy as np
import tensorflow as tf
import tifffile
import typer
import wandb
from n2v.models import N2V, N2VConfig
from wandb.integration.keras import WandbMetricsLogger

from common import DATA_DIR, RESULTS_DIR, RUNS_DIR, WANDB_PROJECT
from datasets.d2.bsd68 import bsd68

IMPL = "original"
ALGO = "n2v"
DATASET = "bsd68"

PSNR_RANGE = 255.0


def main(seed: int = 42):
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    run_name = f"{ALGO}-{DATASET}-s{seed}"

    data = bsd68(DATA_DIR)
    train_data = tifffile.imread(data.train / "DCNN400_train_gaussian25.tiff")
    train_data = train_data[..., np.newaxis]
    val_data = tifffile.imread(data.val / "DCNN400_validation_gaussian25.tiff")
    val_data = val_data[..., np.newaxis]

    config = N2VConfig(
        train_data,
        unet_kern_size=3,
        train_steps_per_epoch=400,
        train_epochs=200,
        train_loss="mse",
        batch_norm=True,
        train_batch_size=128,
        n2v_perc_pix=0.198,
        n2v_patch_shape=(64, 64),
        unet_n_first=96,
        unet_residual=True,
        n2v_manipulator="uniform_withCP",
        n2v_neighborhood_radius=2,
        single_net_per_channel=False,
    )
    model = N2V(config, run_name, basedir=RUNS_DIR / IMPL)
    model.prepare_for_training(metrics=())

    gpu = tf.config.list_physical_devices("GPU")[0]
    build = tf.sysconfig.get_build_info()
    meta = {
        "impl": IMPL,
        "algo": ALGO,
        "dataset": DATASET,
        "seed": seed,
        "n2v_version": version("n2v"),
        "tensorflow_version": tf.__version__,
        "cuda_version": build.get("cuda_version"),
        "cudnn_version": build.get("cudnn_version"),
        "gpu": tf.config.experimental.get_device_details(gpu)["device_name"],
        "hostname": platform.node(),
    }

    run = wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        group=DATASET,
        tags=[f"impl:{IMPL}", f"algo:{ALGO}", f"dataset:{DATASET}"],
        config={**vars(config), **meta},
        dir=model.logdir,
        settings=wandb.Settings(save_code=True),
    )
    model.callbacks.append(WandbMetricsLogger())

    start = time.perf_counter()
    history = model.train(train_data, val_data)
    train_seconds = time.perf_counter() - start
    epochs = len(history.history["loss"])

    result = {
        **meta,
        "wandb_project": WANDB_PROJECT,
        "wandb_run_id": run.id,
        "n2v_config": vars(config),
        "epochs": epochs,
        "train_seconds": train_seconds,
        "seconds_per_epoch": train_seconds / epochs,
    }

    for weights in ["best", "last"]:
        model.load_weights(f"weights_{weights}.h5")
        psnrs = []
        start = time.perf_counter()
        for noisy_path, gt_path in zip(
            sorted(data.test.glob("*.tif*")),
            sorted(data.test_target.glob("*.tif*")),
        ):
            noisy = tifffile.imread(noisy_path)
            gt = tifffile.imread(gt_path)
            prediction = model.predict(noisy, "YX", tta=False)
            mse = np.mean(np.square(gt - prediction), dtype=np.float64)
            psnrs.append(float(20 * np.log10(PSNR_RANGE) - 10 * np.log10(mse)))
        result[f"psnr_{weights}_mean"] = float(np.mean(psnrs))
        result[f"psnr_{weights}_per_image"] = psnrs
        result[f"predict_{weights}_seconds"] = time.perf_counter() - start
        result[f"{weights}_checkpoint"] = str(model.logdir / f"weights_{weights}.h5")

    path = RESULTS_DIR / IMPL / ALGO / f"{DATASET}_s{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))

    run.summary.update(result)
    run.finish()

    print(
        f"{run_name}  PSNR best {result['psnr_best_mean']:.4f}"
        f"  last {result['psnr_last_mean']:.4f}  -> {path}"
    )


if __name__ == "__main__":
    typer.run(main)
