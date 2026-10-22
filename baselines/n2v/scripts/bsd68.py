import json
import platform
import time
from datetime import datetime
from importlib.metadata import version

import numpy as np
import tensorflow as tf
import tifffile
import typer
import wandb
from n2v.models import N2V, N2VConfig
from skimage.metrics import structural_similarity
from wandb.integration.keras import WandbMetricsLogger

from common import DATA_DIR, RESULTS_DIR, RUNS_DIR, WANDB_PROJECT
from common.metrics import psnr, scale_invariant_psnr
from common.run_files import save_run_files
from datasets.d2.bsd68 import bsd68

IMPL = "original"
ALGO = "n2v"
DATASET = "bsd68"

DATA_RANGE = 255.0
N_IMAGES = 3


def main(seed: int = 42):
    tf.keras.utils.set_random_seed(seed)
    mode = "benchmark"
    run_name = f"{IMPL}-{ALGO}-{DATASET}-s{seed}-{mode}"

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
    model = N2V(
        config,
        f"{run_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        basedir=RUNS_DIR / DATASET / ALGO,
    )
    model.prepare_for_training(metrics=())
    run_dir = model.logdir
    git = save_run_files(run_dir, __file__)

    gpu = tf.config.list_physical_devices("GPU")[0]
    build = tf.sysconfig.get_build_info()
    meta = {
        "impl": IMPL,
        "algo": ALGO,
        "dataset": DATASET,
        "seed": seed,
        "mode": mode,
        "n2v_version": version("n2v"),
        "tensorflow_version": tf.__version__,
        "cuda_version": build.get("cuda_version"),
        "cudnn_version": build.get("cudnn_version"),
        "gpu": tf.config.experimental.get_device_details(gpu)["device_name"],
        "hostname": platform.node(),
        **git,
    }

    run = wandb.init(
        project=WANDB_PROJECT,
        id=run_dir.name,
        name=run_name,
        group=DATASET,
        tags=[
            f"impl:{IMPL}",
            f"algo:{ALGO}",
            f"dataset:{DATASET}",
            f"mode:{mode}",
        ],
        config={**vars(config), **meta},
        dir=run_dir,
        settings=wandb.Settings(save_code=True, code_dir="src"),
    )
    model.callbacks.append(WandbMetricsLogger())

    tf.config.experimental.reset_memory_stats("GPU:0")
    start = time.perf_counter()
    history = model.train(train_data, val_data)
    train_seconds = time.perf_counter() - start
    train_peak_memory_mib = (
        tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**20
    )
    epochs = len(history.history["loss"])
    (run_dir / "history.json").write_text(
        json.dumps(
            {
                key: [float(value) for value in values]
                for key, values in history.history.items()
            },
            indent=2,
        )
    )

    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir()

    result = {
        "train/epochs": epochs,
        "train/seconds": train_seconds,
        "train/seconds_per_epoch": train_seconds / epochs,
        "train/peak_memory_mib": train_peak_memory_mib,
    }
    per_image = {}
    images = {}
    for weights in ["best", "last"]:
        model.load_weights(f"weights_{weights}.h5")
        psnrs, si_psnrs, ssims, images[weights] = [], [], [], []
        tf.config.experimental.reset_memory_stats("GPU:0")
        start = time.perf_counter()
        for i, (noisy_path, gt_path) in enumerate(
            zip(
                sorted(data.test.glob("*.tif*")),
                sorted(data.test_target.glob("*.tif*")),
            )
        ):
            noisy = tifffile.imread(noisy_path)
            gt = tifffile.imread(gt_path)
            gt = gt.astype(np.float32)
            prediction = model.predict(noisy, "YX", tta=False)

            psnrs.append(float(psnr(gt, prediction, DATA_RANGE)))
            si_psnrs.append(float(scale_invariant_psnr(gt, prediction)))
            ssims.append(
                float(structural_similarity(gt, prediction, data_range=DATA_RANGE))
            )

            if i < N_IMAGES:
                tifffile.imwrite(
                    predictions_dir / f"{weights}_{noisy_path.stem}.tif", prediction
                )
                images[weights].append(
                    wandb.Image(
                        np.clip(
                            np.concatenate([noisy, prediction, gt], axis=1),
                            0,
                            DATA_RANGE,
                        ).astype(np.uint8),
                        caption=f"{noisy_path.name}: noisy | prediction | gt",
                    )
                )
        result[f"test/{weights}/psnr"] = float(np.mean(psnrs))
        result[f"test/{weights}/si_psnr"] = float(np.mean(si_psnrs))
        result[f"test/{weights}/ssim"] = float(np.mean(ssims))
        result[f"test/{weights}/seconds"] = time.perf_counter() - start
        result[f"test/{weights}/peak_memory_mib"] = (
            tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**20
        )
        result[f"checkpoint/{weights}"] = str(run_dir / f"weights_{weights}.h5")
        per_image[f"test/{weights}/psnr_per_image"] = psnrs
        per_image[f"test/{weights}/si_psnr_per_image"] = si_psnrs
        per_image[f"test/{weights}/ssim_per_image"] = ssims

    summary = {
        **meta,
        "wandb_project": WANDB_PROJECT,
        "wandb_run_id": run.id,
        **result,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "per_image.json").write_text(json.dumps(per_image, indent=2))

    path = RESULTS_DIR / DATASET / ALGO / f"{run_dir.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({**summary, "n2v_config": vars(config), **per_image}, indent=2)
    )

    run.summary.update(result)
    run.log({f"test/{weights}/images": images[weights] for weights in images})
    run.finish()

    print(
        f"{run_name}  PSNR best {result['test/best/psnr']:.4f}"
        f"  last {result['test/last/psnr']:.4f}  -> {run_dir}"
    )


if __name__ == "__main__":
    typer.run(main)
