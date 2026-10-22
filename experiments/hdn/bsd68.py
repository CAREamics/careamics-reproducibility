import json
import math
from datetime import datetime

import numpy as np
import tifffile
import typer
import wandb
from careamics import CAREamist
from careamics.config.factories import create_advanced_hdn_config
from careamics.config.noise_model import GaussianMixtureNMConfig
from careamics.config.noise_model.noise_model_config import MultiChannelNMConfig
from lightning.pytorch import seed_everything
from skimage.metrics import structural_similarity

from common import DATA_DIR, IMPL, RESULTS_DIR, RUNS_DIR, WANDB_PROJECT
from common.cuda_timer import cuda_timer
from common.metrics import psnr, scale_invariant_psnr
from common.run_files import save_run_files
from common.run_meta import get_run_meta
from common.wandb_utils import detach_wandb, setup_wandb
from datasets.d2.bsd68 import bsd68

ALGO = "hdn"
DATASET = "bsd68"

DATA_RANGE = 255.0
NUM_WORKERS = 8
MMSE_SAMPLES = 100
PREDICT_BATCH = 1
PAD_MULTIPLE = 128
N_IMAGES = 3


def main(seed: int = 42, benchmark: bool = False):
    seed_everything(seed, workers=True)
    mode = "benchmark" if benchmark else "deterministic"
    run_name = f"{IMPL}-{ALGO}-{DATASET}-s{seed}-{mode}"
    run_dir = (
        RUNS_DIR
        / DATASET
        / ALGO
        / f"{run_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    git = save_run_files(run_dir, __file__)

    data = bsd68(DATA_DIR)
    train_data = tifffile.imread(data.train / "DCNN400_train_gaussian25.tiff")
    train_data = train_data.astype(np.float32)
    val_data = tifffile.imread(data.val / "DCNN400_validation_gaussian25.tiff")
    val_data = val_data.astype(np.float32)

    weight = np.zeros((3, 2), dtype=np.float32)
    weight[1, 0] = math.log(25.0**2)
    weight[1, 1] = -60.0
    noise_model = MultiChannelNMConfig(
        noise_models=[
            GaussianMixtureNMConfig(
                weight=weight,
                n_gaussian=1,
                n_coeff=2,
                min_signal=0.0,
                max_signal=DATA_RANGE,
                min_sigma=1.0,
            )
        ]
    )

    config = create_advanced_hdn_config(
        experiment_name=run_name,
        data_type="array",
        axes="SYX",
        patch_size=[128, 128],
        batch_size=8,
        num_epochs=400,
        num_steps=1000,
        augmentations=["x_flip", "y_flip", "rotate_90"],
        noise_model=noise_model,
        model_params={
            "z_dims": [32] * 6,
            "n_filters": 64,
            "encoder_blocks_per_layer": 5,
            "decoder_blocks_per_layer": 5,
            "encoder_dropout": 0.2,
            "decoder_dropout": 0.2,
            "nonlinearity": "ELU",
        },
        optimizer="Adamax",
        optimizer_params={"lr": 3e-4},
        lr_scheduler="ReduceLROnPlateau",
        lr_scheduler_params={
            "mode": "min",
            "patience": 30,
            "factor": 0.5,
            "min_lr": 1e-12,
        },
        trainer_params={
            "accumulate_grad_batches": 2,
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
            "save_last": True,
            "auto_insert_metric_name": False,
        },
        early_stopping_params={
            "monitor": "val_loss",
            "mode": "min",
            "patience": 100,
            "min_delta": 0.0,
        },
        num_workers=NUM_WORKERS,
        seed=seed,
        logger="wandb",
    )

    careamist = CAREamist(config, work_dir=run_dir)
    meta = {
        **get_run_meta(ALGO, DATASET, seed, mode),
        **git,
        "mmse_samples": MMSE_SAMPLES,
    }
    full_config = config.model_dump(mode="json")
    (run_dir / "config.json").write_text(json.dumps(full_config, indent=2))
    run_id = setup_wandb(
        careamist,
        {
            **meta,
            **{k: v for k, v in full_config.items() if k != "algorithm_config"},
        },
    )

    with cuda_timer() as train_timer:
        careamist.train(train_data=train_data, val_data=val_data)
    epochs = careamist.trainer.current_epoch
    best_checkpoint = careamist.trainer.checkpoint_callback.best_model_path
    last_checkpoint = careamist.trainer.checkpoint_callback.last_model_path

    detach_wandb(careamist)

    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir()

    fill = float(train_data.mean())
    result = {
        "train/epochs": epochs,
        "train/seconds": train_timer["seconds"],
        "train/seconds_per_epoch": train_timer["seconds"] / epochs,
        "train/peak_memory_mib": train_timer["peak_memory_mib"],
        "checkpoint/best": best_checkpoint,
        "checkpoint/last": last_checkpoint,
    }
    per_image = {}
    images = {}
    for weights, checkpoint in [("best", best_checkpoint), ("last", last_checkpoint)]:
        seed_everything(seed, workers=True)
        psnrs, si_psnrs, ssims, images[weights] = [], [], [], []
        with cuda_timer() as predict_timer:
            for i, (noisy_path, gt_path) in enumerate(
                zip(
                    sorted(data.test.glob("*.tif*")),
                    sorted(data.test_target.glob("*.tif*")),
                )
            ):
                noisy = tifffile.imread(noisy_path)
                noisy = noisy.astype(np.float32)[:-1, :-1]
                gt = tifffile.imread(gt_path)
                gt = gt.astype(np.float32)[:-1, :-1]
                pad = tuple(
                    (
                        (-s) % PAD_MULTIPLE // 2,
                        (-s) % PAD_MULTIPLE - (-s) % PAD_MULTIPLE // 2,
                    )
                    for s in noisy.shape
                )
                padded = np.pad(noisy, pad, mode="constant", constant_values=fill)

                predictions, _uncertainty, _sources = careamist.predict(
                    padded[np.newaxis, ...],
                    n_predictions=MMSE_SAMPLES,
                    batch_size=PREDICT_BATCH,
                    num_workers=NUM_WORKERS,
                    checkpoint=checkpoint,
                )
                frame = np.asarray(predictions[0]).reshape(-1, *padded.shape)[0]

                (top, bottom), (left, right) = pad
                frame = frame[
                    top : frame.shape[0] - bottom if bottom else None,
                    left : frame.shape[1] - right if right else None,
                ]
                psnrs.append(float(psnr(gt, frame, DATA_RANGE)))
                si_psnrs.append(float(scale_invariant_psnr(gt, frame)))
                ssims.append(
                    float(structural_similarity(gt, frame, data_range=DATA_RANGE))
                )

                if i < N_IMAGES:
                    tifffile.imwrite(
                        predictions_dir / f"{weights}_{noisy_path.stem}.tif", frame
                    )
                    images[weights].append(
                        wandb.Image(
                            np.clip(
                                np.concatenate([noisy, frame, gt], axis=1),
                                0,
                                DATA_RANGE,
                            ).astype(np.uint8),
                            caption=f"{noisy_path.name}: noisy | prediction | gt",
                        )
                    )
        result[f"test/{weights}/psnr"] = float(np.mean(psnrs))
        result[f"test/{weights}/si_psnr"] = float(np.mean(si_psnrs))
        result[f"test/{weights}/ssim"] = float(np.mean(ssims))
        result[f"test/{weights}/seconds"] = predict_timer["seconds"]
        result[f"test/{weights}/peak_memory_mib"] = predict_timer["peak_memory_mib"]
        per_image[f"test/{weights}/psnr_per_image"] = psnrs
        per_image[f"test/{weights}/si_psnr_per_image"] = si_psnrs
        per_image[f"test/{weights}/ssim_per_image"] = ssims

    summary = {
        **meta,
        "wandb_project": WANDB_PROJECT,
        "wandb_run_id": run_id,
        **result,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "per_image.json").write_text(json.dumps(per_image, indent=2))

    path = RESULTS_DIR / DATASET / ALGO / f"{run_dir.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({**summary, "careamics_config": full_config, **per_image}, indent=2)
    )

    with wandb.init(
        project=WANDB_PROJECT,
        id=run_id,
        resume="must",
        dir=run_dir / "wandb_logs",
    ) as run:
        run.summary.update(result)
        run.log({f"test/{weights}/images": images[weights] for weights in images})

    print(
        f"{run_name}  PSNR best {result['test/best/psnr']:.4f}"
        f"  last {result['test/last/psnr']:.4f}  -> {run_dir}"
    )


if __name__ == "__main__":
    typer.run(main)
