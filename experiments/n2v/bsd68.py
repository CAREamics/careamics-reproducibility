import json

import numpy as np
import tifffile
import typer
import wandb
from careamics import CAREamist
from careamics.config.factories import create_advanced_n2v_config
from careamics.metrics.metrics import psnr as psnr_fn
from careamics.metrics.metrics import scale_invariant_psnr
from lightning.pytorch import seed_everything

from common import DATA_DIR, IMPL, RESULTS_DIR, RUNS_DIR, WANDB_PROJECT
from common.cuda_timer import cuda_timer
from common.run_meta import get_run_meta
from common.wandb_utils import detach_wandb, setup_wandb
from datasets.d2.bsd68 import bsd68

ALGO = "n2v"
DATASET = "bsd68"

PSNR_RANGE = 255.0
NUM_WORKERS = 8
TILE_SIZE = (128, 128)
TILE_OVERLAP = (48, 48)
PREDICT_BATCH = 8


def main(seed: int = 42):
    seed_everything(seed, workers=True)
    run_name = f"{ALGO}-{DATASET}"

    data = bsd68(DATA_DIR)
    train_data = tifffile.imread(data.train / "DCNN400_train_gaussian25.tiff")
    train_data = train_data.astype(np.float32)
    val_data = tifffile.imread(data.val / "DCNN400_validation_gaussian25.tiff")
    val_data = val_data.astype(np.float32)

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
            "deterministic": True,
            "benchmark": False,
        },
        checkpoint_params={
            "save_top_k": 0,
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

    psnrs, si_psnrs = [], []
    with cuda_timer() as predict_timer:
        for noisy_path, gt_path in zip(
            sorted(data.test.glob("*.tif*")),
            sorted(data.test_target.glob("*.tif*")),
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
            )
            frame = np.asarray(predictions[0]).squeeze()

            psnrs.append(float(psnr_fn(gt, frame, PSNR_RANGE)))
            si_psnrs.append(float(scale_invariant_psnr(gt, frame)))
    predict_seconds = predict_timer["seconds"]

    result = {
        **get_run_meta(ALGO, DATASET, seed),
        "wandb_project": WANDB_PROJECT,
        "wandb_run_id": run_id,
        "careamics_config": config.model_dump(mode="json"),
        "tile_size": TILE_SIZE,
        "tile_overlap": TILE_OVERLAP,
        "psnr_mean": float(np.mean(psnrs)),
        "si_psnr_mean": float(np.mean(si_psnrs)),
        "psnr_per_image": psnrs,
        "si_psnr_per_image": si_psnrs,
        "epochs": epochs,
        "train_seconds": train_seconds,
        "seconds_per_epoch": train_seconds / epochs,
        "predict_seconds": predict_seconds,
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
