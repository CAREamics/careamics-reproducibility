import json

import numpy as np
import typer
import wandb
from careamics import CAREamist
from careamics.config.factories import create_advanced_hdn_config
from careamics.metrics.metrics import psnr as psnr_fn
from careamics.metrics.metrics import scale_invariant_psnr
from careamics.noise_model import NoiseModelTrainer
from lightning.pytorch import seed_everything

from common import DATA_DIR, IMPL, RESULTS_DIR, RUNS_DIR, WANDB_PROJECT
from common.cuda_timer import cuda_timer
from common.run_meta import get_run_meta
from common.wandb_utils import detach_wandb, setup_wandb
from datasets.d2.convallaria import convallaria

ALGO = "hdn"
DATASET = "convallaria"

NUM_WORKERS = 8
MMSE_SAMPLES = 100
PREDICT_BATCH = 1


def main(seed: int = 42):
    seed_everything(seed, workers=True)
    run_name = f"{ALGO}-{DATASET}"

    data = convallaria(DATA_DIR)

    noise_model_trainer = NoiseModelTrainer(n_gaussian=3, n_coeff=2, min_sigma=50.0)
    noise_model_trainer.train_from_pairs(
        signal=data.calibration_signal,
        observation=data.calibration_observation,
        axes="SYX",
        n_epochs=2000,
        learning_rate=0.1,
        batch_size=250_000,
    )
    noise_model = noise_model_trainer.get_config()

    config = create_advanced_hdn_config(
        experiment_name=run_name,
        data_type="array",
        axes="SYX",
        patch_size=[64, 64],
        batch_size=32,
        num_epochs=236,
        num_steps=680,
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
            "patience": 10,
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
        careamist.train(train_data=data.train, val_data=data.val)
    train_seconds = train_timer["seconds"]
    epochs = careamist.trainer.current_epoch
    best_checkpoint = careamist.trainer.checkpoint_callback.best_model_path
    last_checkpoint = careamist.trainer.checkpoint_callback.last_model_path

    detach_wandb(careamist)

    gt = data.test_target
    data_range = float(gt.max() - gt.min())
    psnrs, si_psnrs = [], []
    with cuda_timer() as predict_timer:
        predictions, _uncertainty, _sources = careamist.predict(
            data.test,
            n_predictions=MMSE_SAMPLES,
            batch_size=PREDICT_BATCH,
            num_workers=NUM_WORKERS,
        )
        frames = np.asarray(predictions[0]).reshape(-1, *gt.shape)
        for frame in frames:
            psnrs.append(float(psnr_fn(gt, frame, data_range)))
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
        "best_checkpoint": best_checkpoint,
        "last_checkpoint": last_checkpoint,
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
