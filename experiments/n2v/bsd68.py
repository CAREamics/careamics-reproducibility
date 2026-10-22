import json
from datetime import datetime

import numpy as np
import tifffile
import typer
import wandb
from careamics import CAREamist
from careamics.config.factories import create_advanced_n2v_config
from lightning.pytorch import seed_everything
from skimage.metrics import structural_similarity

from common import DATA_DIR, IMPL, RESULTS_DIR, RUNS_DIR, WANDB_PROJECT
from common.cuda_timer import cuda_timer
from common.metrics import psnr, scale_invariant_psnr
from common.run_files import save_run_files
from common.run_meta import get_run_meta
from common.wandb_utils import detach_wandb, setup_wandb
from datasets.d2.bsd68 import bsd68

ALGO = "n2v"
DATASET = "bsd68"

DATA_RANGE = 255.0
NUM_WORKERS = 8
TILE_SIZE = (128, 128)
TILE_OVERLAP = (48, 48)
PREDICT_BATCH = 8
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
            "save_last": True,
            "auto_insert_metric_name": False,
        },
        num_workers=NUM_WORKERS,
        seed=seed,
        logger="wandb",
    )

    careamist = CAREamist(config, work_dir=run_dir)
    meta = {**get_run_meta(ALGO, DATASET, seed, mode), **git}
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
        psnrs, si_psnrs, ssims, images[weights] = [], [], [], []
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
                    checkpoint=checkpoint,
                )
                frame = np.asarray(predictions[0]).squeeze()

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
