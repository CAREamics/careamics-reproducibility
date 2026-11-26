import json
import platform
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
from common.tf_timer import tf_timer
from datasets.d2.bsd68 import bsd68

DATA_RANGE = 255.0
N_IMAGES = 3


def main(seed: int = 42):
    # Intialization
    tf.keras.utils.set_random_seed(seed)
    mode = "bench"
    run_name = f"orig-n2v-bsd68-s{seed}-{mode}"

    data = bsd68(DATA_DIR)
    train_data = tifffile.imread(data.train / "DCNN400_train_gaussian25.tiff")
    train_data = train_data[..., np.newaxis]
    val_data = tifffile.imread(data.val / "DCNN400_validation_gaussian25.tiff")
    val_data = val_data[..., np.newaxis]

    # Configuration
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
        basedir=RUNS_DIR / "bsd68" / "n2v",
    )
    model.prepare_for_training(metrics=())

    # Save metadata and run files
    run_dir = model.logdir
    git = save_run_files(run_dir, __file__)
    gpu = tf.config.list_physical_devices("GPU")[0]
    build = tf.sysconfig.get_build_info()
    run_metadata = {
        "impl": "orig",
        "algo": "n2v",
        "dataset": "bsd68",
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

    # Add logger for wandb
    run = wandb.init(
        project=WANDB_PROJECT,
        id=run_dir.name,
        name=run_name,
        group="bsd68",
        tags=[
            "impl:orig",
            "algo:n2v",
            "dataset:bsd68",
            f"mode:{mode}",
        ],
        config={**vars(config), **run_metadata},
        dir=run_dir,
        settings=wandb.Settings(save_code=True, code_dir="src"),
    )
    model.callbacks.append(WandbMetricsLogger())
    run_metadata["wandb_run_id"] = run.id

    # Train
    with tf_timer() as train_timer:
        history = model.train(train_data, val_data)
    epochs = len(history.history["loss"])
    run_metadata.update(
        {
            "train/epochs": epochs,
            "train/seconds": train_timer["seconds"],
            "train/seconds_per_epoch": train_timer["seconds"] / epochs,
            "train/peak_memory_mib": train_timer["peak_memory_mib"],
            "test/best/epoch": int(np.argmin(history.history["val_loss"])) + 1,
            "test/last/epoch": epochs,
        }
    )

    # Predict
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir()

    images = {}
    for checkpoint in ["best", "last"]:
        model.load_weights(f"weights_{checkpoint}.h5")
        psnrs, si_psnrs, ssims, images[checkpoint] = [], [], [], []
        with tf_timer() as predict_timer:
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

                psnrs.append(psnr(gt, prediction, DATA_RANGE))
                si_psnrs.append(scale_invariant_psnr(gt, prediction))
                ssims.append(
                    structural_similarity(gt, prediction, data_range=DATA_RANGE)
                )

                if i < N_IMAGES:
                    tifffile.imwrite(
                        predictions_dir / f"{checkpoint}_{noisy_path.stem}.tif",
                        prediction,
                    )
                    images[checkpoint].append(
                        wandb.Image(
                            np.clip(
                                np.concatenate([noisy, prediction, gt], axis=1),
                                0,
                                DATA_RANGE,
                            ).astype(np.uint8),
                            caption=f"{noisy_path.name}: noisy | prediction | gt",
                        )
                    )
        run_metadata[f"test/{checkpoint}/psnr"] = float(np.mean(psnrs))
        run_metadata[f"test/{checkpoint}/si_psnr"] = float(np.mean(si_psnrs))
        run_metadata[f"test/{checkpoint}/ssim"] = float(np.mean(ssims))
        run_metadata[f"test/{checkpoint}/seconds"] = predict_timer["seconds"]
        run_metadata[f"test/{checkpoint}/peak_memory_mib"] = predict_timer[
            "peak_memory_mib"
        ]
        run_metadata[f"checkpoint/{checkpoint}"] = str(
            run_dir / f"weights_{checkpoint}.h5"
        )
        run_metadata[f"test/{checkpoint}/psnr_per_image"] = psnrs
        run_metadata[f"test/{checkpoint}/si_psnr_per_image"] = si_psnrs
        run_metadata[f"test/{checkpoint}/ssim_per_image"] = ssims

    # Save results
    (run_dir / "history.json").write_text(
        json.dumps(
            {
                key: [float(value) for value in values]
                for key, values in history.history.items()
            },
            indent=2,
        )
    )
    (run_dir / "summary.json").write_text(json.dumps(run_metadata, indent=2))

    path = RESULTS_DIR / "bsd68" / "n2v" / f"{run_dir.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**run_metadata, "config": vars(config)}, indent=2))

    run.summary.update(
        {
            key: value
            for key, value in run_metadata.items()
            if key.startswith(("train/", "test/", "checkpoint/"))
            and not key.endswith("_per_image")
        }
    )
    run.log({f"test/{weights}/images": images[weights] for weights in images})
    run.finish()


if __name__ == "__main__":
    typer.run(main)
