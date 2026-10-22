import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
import torch
import typer
import wandb
from careamics import CAREamist
from careamics.config.factories import create_advanced_n2v_config
from lightning.pytorch import seed_everything
from skimage.metrics import structural_similarity

from common import DATA_DIR, RESULTS_DIR, RUNS_DIR, WANDB_PROJECT
from common.cuda_timer import cuda_timer
from common.metrics import psnr, scale_invariant_psnr
from common.run_files import save_run_files
from common.run_meta import get_run_meta
from common.wandb_utils import detach_wandb, setup_wandb
from datasets.d2.bsd68 import bsd68

DATA_RANGE = 255.0
NUM_WORKERS = 4
TILE_SIZE = (128, 128)
TILE_OVERLAP = (48, 48)
PREDICT_BATCH = 8
N_IMAGES = 3


def main(seed: int = 42, benchmark: bool = False):
    # Setup
    seed_everything(seed, workers=True)
    mode = "bench" if benchmark else "det"
    run_name = f"careamics-n2v-bsd68-s{seed}-{mode}"
    run_dir = (
        RUNS_DIR
        / "bsd68"
        / "n2v"
        / f"{run_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    data = bsd68(DATA_DIR)
    train_data = tifffile.imread(data.train / "DCNN400_train_gaussian25.tiff")
    train_data = train_data.astype(np.float32)
    val_data = tifffile.imread(data.val / "DCNN400_validation_gaussian25.tiff")
    val_data = val_data.astype(np.float32)

    # Configuration
    config = create_advanced_n2v_config(
        experiment_name=run_name,
        data_type="array",
        axes="SYX",
        patch_size=[64, 64],
        batch_size=64,
        num_epochs=100,
        augmentations=["x_flip", "y_flip", "rotate_90"],
        use_n2v2=False,
        trainer_params={
            "precision": "32-true",
            "accelerator": "gpu",
            "devices": 1,
            "deterministic": not benchmark,
            "benchmark": benchmark,
        },
        checkpoint_params={
            "monitor": "val_loss",
            "mode": "min",
            "save_top_k": 1,
            "save_last": False,
            "auto_insert_metric_name": False,
        },
        num_workers=NUM_WORKERS,
        seed=seed,
        logger="wandb",
    )
    careamist = CAREamist(config, work_dir=run_dir)

    # Save metadata and run files
    git = save_run_files(run_dir, __file__)
    full_config = config.model_dump(mode="json")
    (run_dir / "config.json").write_text(json.dumps(full_config, indent=2))
    run_metadata = {**get_run_meta("n2v", "bsd68", seed, mode), **git}

    # Add logger for wandb
    run_metadata["wandb_run_id"] = setup_wandb(
        careamist,
        {
            **run_metadata,
            **{k: v for k, v in full_config.items() if k != "algorithm_config"},
        },
    )

    # Train
    with cuda_timer() as train_timer:
        careamist.train(train_data=train_data, val_data=val_data)
    detach_wandb(careamist)
    best_checkpoint = careamist.trainer.checkpoint_callback.best_model_path
    careamist.trainer.save_checkpoint(Path(best_checkpoint).with_name("last.ckpt"))
    epochs = careamist.trainer.current_epoch
    run_metadata.update(
        {
            "train/epochs": epochs,
            "train/seconds": train_timer["seconds"],
            "train/seconds_per_epoch": train_timer["seconds"] / epochs,
            "train/peak_memory_mib": train_timer["peak_memory_mib"],
            "test/best/epoch": torch.load(best_checkpoint, weights_only=False)["epoch"]
            + 1,
            "test/last/epoch": epochs,
            "checkpoint/best": best_checkpoint,
            "checkpoint/last": str(Path(best_checkpoint).with_name("last.ckpt")),
        }
    )

    # Predict
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir()

    images = {}
    for checkpoint in ["best", "last"]:
        psnrs, si_psnrs, ssims, images[checkpoint] = [], [], [], []
        with cuda_timer() as predict_timer:
            for i, (noisy_path, gt_path) in enumerate(
                zip(
                    sorted(data.test.glob("*.tif*")),
                    sorted(data.test_target.glob("*.tif*")),
                )
            ):
                noisy = tifffile.imread(noisy_path)
                noisy = noisy.astype(np.float32)
                gt = tifffile.imread(gt_path)
                gt = gt.astype(np.float32)
                predictions, _sources = careamist.predict(
                    noisy,
                    axes="YX",
                    tile_size=TILE_SIZE,
                    tile_overlap=TILE_OVERLAP,
                    batch_size=PREDICT_BATCH,
                    num_workers=NUM_WORKERS,
                    checkpoint=run_metadata[f"checkpoint/{checkpoint}"],
                )
                prediction = np.asarray(predictions[0]).squeeze()

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
        run_metadata[f"test/{checkpoint}/psnr_per_image"] = psnrs
        run_metadata[f"test/{checkpoint}/si_psnr_per_image"] = si_psnrs
        run_metadata[f"test/{checkpoint}/ssim_per_image"] = ssims

    # Save results
    (run_dir / "summary.json").write_text(json.dumps(run_metadata, indent=2))

    path = RESULTS_DIR / "bsd68" / "n2v" / f"{run_dir.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**run_metadata, "config": full_config}, indent=2))

    with wandb.init(
        project=WANDB_PROJECT,
        id=run_metadata["wandb_run_id"],
        resume="must",
        dir=run_dir / "wandb_logs",
    ) as run:
        run.summary.update(
            {
                key: value
                for key, value in run_metadata.items()
                if key.startswith(("train/", "test/", "checkpoint/"))
                and not key.endswith("_per_image")
            }
        )
        run.log({f"test/{weights}/images": images[weights] for weights in images})


if __name__ == "__main__":
    typer.run(main)
