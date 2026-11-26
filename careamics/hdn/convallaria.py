import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
import torch
import typer
import wandb
from careamics import CAREamist
from careamics.config.factories import create_advanced_hdn_config
from careamics.noise_model import NoiseModelTrainer
from lightning.pytorch import seed_everything
from skimage.metrics import structural_similarity

from common import DATA_DIR, RESULTS_DIR, RUNS_DIR, WANDB_PROJECT
from common.cuda_timer import cuda_timer
from common.metrics import psnr, scale_invariant_psnr
from common.run_files import save_run_files
from common.run_meta import get_run_meta
from common.wandb_utils import detach_wandb, setup_wandb
from datasets.d2.convallaria import convallaria

N_IMAGES = 3


def main(seed: int = 42, benchmark: bool = False):
    # Setup
    seed_everything(seed, workers=True)
    mode = "bench" if benchmark else "det"
    run_name = f"careamics-hdn-convallaria-s{seed}-{mode}"
    run_dir = (
        RUNS_DIR
        / "convallaria"
        / "hdn"
        / f"{run_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    data = convallaria(DATA_DIR)

    # Configuration
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
        num_epochs=500,
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
        early_stopping_params={
            "monitor": "val_loss",
            "mode": "min",
            "patience": 30,
            "min_delta": 0.0,
        },
        num_workers=8,
        seed=seed,
        logger="wandb",
    )
    careamist = CAREamist(config, work_dir=run_dir)

    # Save metadata and run files
    git = save_run_files(run_dir, __file__)
    full_config = config.model_dump(mode="json")
    (run_dir / "config.json").write_text(json.dumps(full_config, indent=2))
    run_metadata = {
        **get_run_meta("hdn", "convallaria", seed, mode),
        **git,
        "mmse_samples": 100,
    }

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
        careamist.train(train_data=data.train, val_data=data.val)
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

    gt = data.test_target
    data_range = float(gt.max() - gt.min())
    images = {}
    for checkpoint in ["best", "last"]:
        seed_everything(seed, workers=True)
        psnrs, si_psnrs, ssims, images[checkpoint] = [], [], [], []
        with cuda_timer() as predict_timer:
            for i, noisy in enumerate(data.test):
                predictions, _uncertainty, _sources = careamist.predict(
                    noisy,
                    axes="YX",
                    tile_size=(128, 128),
                    tile_overlap=(48, 48),
                    batch_size=8,
                    num_workers=8,
                    n_predictions=100,
                    checkpoint=run_metadata[f"checkpoint/{checkpoint}"],
                )
                prediction = np.asarray(predictions[0]).squeeze()

                psnrs.append(psnr(gt, prediction, data_range))
                si_psnrs.append(scale_invariant_psnr(gt, prediction))
                ssims.append(
                    structural_similarity(gt, prediction, data_range=data_range)
                )

                if i < N_IMAGES:
                    tifffile.imwrite(
                        predictions_dir / f"{checkpoint}_{i:03d}.tif",
                        prediction,
                    )
                    images[checkpoint].append(
                        wandb.Image(
                            np.clip(
                                (
                                    np.concatenate([noisy, prediction, gt], axis=1)
                                    - gt.min()
                                )
                                / data_range
                                * 255,
                                0,
                                255,
                            ).astype(np.uint8),
                            caption=f"frame {i}: noisy | prediction | gt",
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

    path = RESULTS_DIR / "convallaria" / "hdn" / f"{run_dir.name}.json"
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
