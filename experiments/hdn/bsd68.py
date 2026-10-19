import json
import math

import numpy as np
import typer
import wandb
from careamics import CAREamist
from careamics.config.factories import create_advanced_hdn_config
from careamics.config.noise_model import GaussianMixtureNMConfig
from careamics.config.noise_model.noise_model_config import MultiChannelNMConfig
from careamics.metrics.metrics import psnr as psnr_fn
from careamics.metrics.metrics import scale_invariant_psnr
from lightning.pytorch import seed_everything
from tifffile import imread

from common import DATA_DIR, IMPL, RESULTS_DIR, RUNS_DIR, WANDB_PROJECT
from common.cuda_timer import cuda_timer
from common.run_meta import get_run_meta
from common.wandb_utils import detach_wandb, setup_wandb
from datasets.d2.bsd68 import bsd68

ALGO = "hdn"
DATASET = "bsd68"

PSNR_RANGE = 255.0
NUM_WORKERS = 8
MMSE_SAMPLES = 100
PREDICT_BATCH = 1
PAD_MULTIPLE = 128


def main(seed: int = 42):
    seed_everything(seed, workers=True)
    run_name = f"{ALGO}-{DATASET}"

    data = bsd68(DATA_DIR)
    train_data = imread(data.train / "DCNN400_train_gaussian25.tiff")
    train_data = train_data.astype(np.float32)
    val_data = imread(data.val / "DCNN400_validation_gaussian25.tiff")
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
                max_signal=PSNR_RANGE,
                min_sigma=1.0,
            )
        ]
    )

    config = create_advanced_hdn_config(
        experiment_name=run_name,
        data_type="array",
        axes="SYX",
        patch_size=[64, 64],
        batch_size=32,
        num_epochs=1011,
        num_steps=396,
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
            "deterministic": True,
            "benchmark": False,
        },
        checkpoint_params={
            "monitor": "val_loss",
            "mode": "min",
            "save_top_k": 1,
            "save_last": True,
            "auto_insert_metric_name": False,
        },
        num_workers=NUM_WORKERS,
        seed=seed,
        logger="wandb",
    )

    careamist = CAREamist(config, work_dir=RUNS_DIR / run_name)
    run_id = setup_wandb(careamist, ALGO, DATASET)

    with cuda_timer() as train_timer:
        careamist.train(train_data=train_data, val_data=val_data)
    train_seconds = train_timer["seconds"]
    epochs = careamist.trainer.current_epoch

    detach_wandb(careamist)

    fill = float(train_data.mean())
    psnrs, si_psnrs = [], []
    with cuda_timer() as predict_timer:
        for noisy_path, gt_path in zip(
            sorted(data.test.glob("*.tif*")),
            sorted(data.test_target.glob("*.tif*")),
        ):
            noisy = imread(noisy_path)
            noisy = noisy.astype(np.float32)[:-1, :-1]
            gt = imread(gt_path)
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
            )
            frame = np.asarray(predictions[0]).reshape(-1, *padded.shape)[0]

            (top, bottom), (left, right) = pad
            frame = frame[
                top : frame.shape[0] - bottom if bottom else None,
                left : frame.shape[1] - right if right else None,
            ]
            psnrs.append(float(psnr_fn(gt, frame, PSNR_RANGE)))
            si_psnrs.append(float(scale_invariant_psnr(gt, frame)))
    predict_seconds = predict_timer["seconds"]

    result = {
        **get_run_meta(ALGO, DATASET, seed),
        "wandb_project": WANDB_PROJECT,
        "wandb_run_id": run_id,
        "careamics_config": config.model_dump(mode="json"),
        "mmse_samples": MMSE_SAMPLES,
        "psnr_mean": float(np.mean(psnrs)),
        "si_psnr_mean": float(np.mean(si_psnrs)),
        "psnr_per_image": psnrs,
        "si_psnr_per_image": si_psnrs,
        "epochs": epochs,
        "train_seconds": train_seconds,
        "seconds_per_epoch": train_seconds / epochs,
        "predict_seconds": predict_seconds,
        "best_checkpoint": careamist.trainer.checkpoint_callback.best_model_path,
        "last_checkpoint": careamist.trainer.checkpoint_callback.last_model_path,
    }

    path = RESULTS_DIR / IMPL / ALGO / f"{DATASET}_s{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))

    with wandb.init(
        project=WANDB_PROJECT,
        id=run_id,
        resume="must",
        dir=careamist.work_dir / "wandb_logs",
    ) as run:
        run.summary.update(result)

    print(
        f"{run_name}  PSNR {result['psnr_mean']:.4f}"
        f"  SI-PSNR {result['si_psnr_mean']:.4f}  -> {path}"
    )


if __name__ == "__main__":
    typer.run(main)
