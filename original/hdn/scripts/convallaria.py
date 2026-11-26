import json
import platform
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
import torch
import training
import typer
import wandb
from boilerplate import boilerplate
from lib import utils
from lib.gaussianMixtureNoiseModel import GaussianMixtureNoiseModel
from models.lvae import LadderVAE
from skimage.metrics import structural_similarity
from sklearn.feature_extraction import image

from common import DATA_DIR, RESULTS_DIR, RUNS_DIR, WANDB_PROJECT
from common.cuda_timer import cuda_timer
from common.metrics import psnr, scale_invariant_psnr
from common.run_files import save_run_files
from datasets.d2.convallaria import convallaria

N_IMAGES = 3

np.float = float
np.asscalar = lambda a: a.item()


class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, *args, verbose=False, **kwargs):
        super().__init__(*args, **kwargs)


torch.optim.lr_scheduler.ReduceLROnPlateau = ReduceLROnPlateau


def main(seed: int = 42, benchmark: bool = False):
    # Setup
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(not benchmark)
    torch.backends.cudnn.deterministic = not benchmark
    torch.backends.cudnn.benchmark = benchmark
    mode = "bench" if benchmark else "det"
    run_name = f"orig-hdn-convallaria-s{seed}-{mode}"
    run_dir = (
        RUNS_DIR
        / "convallaria"
        / "hdn"
        / f"{run_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir.mkdir(parents=True)

    data = convallaria(DATA_DIR)

    # Configuration
    config = {
        "gaussian_noise_std": None,
        "n_gaussian": 3,
        "n_coeff": 2,
        "min_sigma": 50,
        "noise_model_epochs": 2000,
        "noise_model_batch_size": 250000,
        "noise_model_lr": 0.1,
        "patch_size": 64,
        "num_val_patches": 1000,
        "model_name": "convallaria",
        "batch_size": 64,
        "virtual_batch": 8,
        "lr": 3e-4,
        "max_epochs": 500,
        "steps_per_epoch": 400,
        "val_loss_patience": 30,
        "test_batch_size": 100,
        "z_dims": [32] * 6,
        "blocks_per_layer": 5,
        "batchnorm": True,
        "free_bits": 1.0,
        "mmse_samples": 100,
        "tta": False,
    }
    config["num_patches"] = int(
        float(data.train.shape[2] * data.train.shape[1])
        / float(config["patch_size"] ** 2)
    )

    device = torch.device("cuda")
    noise_model_name = (
        f"GMMNoiseModel_convallaria_{config['n_gaussian']}_{config['n_coeff']}"
        "_calibration"
    )
    GaussianMixtureNoiseModel(
        min_signal=np.min(data.calibration_signal),
        max_signal=np.max(data.calibration_signal),
        path=f"{run_dir}/",
        weight=None,
        n_gaussian=config["n_gaussian"],
        n_coeff=config["n_coeff"],
        min_sigma=config["min_sigma"],
        device=device,
    ).train(
        data.calibration_signal,
        data.calibration_observation,
        batchSize=config["noise_model_batch_size"],
        n_epochs=config["noise_model_epochs"],
        learning_rate=config["noise_model_lr"],
        name=noise_model_name,
    )
    noise_model = GaussianMixtureNoiseModel(
        params=np.load(run_dir / f"{noise_model_name}.npz"), device=device
    )

    patch_shape = (config["patch_size"], config["patch_size"])
    train_data = utils.augment_data(data.train)
    train_images = np.zeros(
        shape=(train_data.shape[0] * config["num_patches"], *patch_shape)
    )
    for i in range(train_data.shape[0]):
        train_images[i * config["num_patches"] : (i + 1) * config["num_patches"]] = (
            image.extract_patches_2d(
                train_data[i],
                patch_shape,
                max_patches=config["num_patches"],
                random_state=i,
            )
        )
    val_images = np.zeros(
        shape=(data.val.shape[0] * config["num_patches"], *patch_shape)
    )
    for i in range(data.val.shape[0]):
        val_images[i * config["num_patches"] : (i + 1) * config["num_patches"]] = (
            image.extract_patches_2d(
                data.val[i],
                patch_shape,
                max_patches=config["num_patches"],
                random_state=i,
            )
        )
    val_images = val_images[: config["num_val_patches"]]

    train_loader, val_loader, test_loader, data_mean, data_std = (
        boilerplate._make_datamanager(
            train_images,
            val_images,
            val_images[:100],
            config["batch_size"],
            config["test_batch_size"],
        )
    )
    model = LadderVAE(
        z_dims=config["z_dims"],
        blocks_per_layer=config["blocks_per_layer"],
        data_mean=data_mean,
        data_std=data_std,
        noiseModel=noise_model,
        device=device,
        batchnorm=config["batchnorm"],
        free_bits=config["free_bits"],
        img_shape=(train_images.shape[1], train_images.shape[2]),
    ).cuda()
    model.train()

    # Save metadata and run files
    git = save_run_files(run_dir, __file__)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    run_metadata = {
        "impl": "orig",
        "algo": "hdn",
        "dataset": "convallaria",
        "seed": seed,
        "mode": mode,
        "hdn_commit": subprocess.run(
            ["git", "-C", str(Path(training.__file__).parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "hostname": platform.node(),
        **git,
    }

    # Add logger for wandb
    run = wandb.init(
        project=WANDB_PROJECT,
        id=run_dir.name,
        name=run_name,
        group="convallaria",
        tags=[
            "impl:orig",
            "algo:hdn",
            "dataset:convallaria",
            f"mode:{mode}",
        ],
        config={**config, **run_metadata},
        dir=run_dir,
        settings=wandb.Settings(save_code=True, code_dir="src"),
    )
    run_metadata["wandb_run_id"] = run.id

    # Train
    with cuda_timer() as train_timer:
        training.train_network(
            model=model,
            lr=config["lr"],
            max_epochs=config["max_epochs"],
            steps_per_epoch=config["steps_per_epoch"],
            directory_path=f"{run_dir}/",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            virtual_batch=config["virtual_batch"],
            gaussian_noise_std=config["gaussian_noise_std"],
            model_name=config["model_name"],
            val_loss_patience=config["val_loss_patience"],
        )
    history = {
        "loss": np.load(run_dir / "model" / "train_loss.npy").tolist(),
        "reconstruction_loss": np.load(
            run_dir / "model" / "train_reco_loss.npy"
        ).tolist(),
        "kl_loss": np.load(run_dir / "model" / "train_kl_loss.npy").tolist(),
        "val_loss": np.load(run_dir / "model" / "val_loss.npy").tolist(),
    }
    epochs = len(history["loss"])
    run.define_metric("epoch/epoch")
    run.define_metric("epoch/*", step_metric="epoch/epoch")
    for epoch in range(epochs):
        run.log(
            {
                "epoch/epoch": epoch,
                **{f"epoch/{key}": values[epoch] for key, values in history.items()},
            }
        )
    run_metadata.update(
        {
            "train/epochs": epochs,
            "train/seconds": train_timer["seconds"],
            "train/seconds_per_epoch": train_timer["seconds"] / epochs,
            "train/peak_memory_mib": train_timer["peak_memory_mib"],
            "test/best/epoch": max(
                i + 1
                for i, value in enumerate(history["val_loss"])
                if value < 1e-6 + min(history["val_loss"][: i + 1])
            ),
            "test/last/epoch": epochs,
            "checkpoint/best": str(
                run_dir / "model" / f"{config['model_name']}_best_vae.net"
            ),
            "checkpoint/last": str(
                run_dir / "model" / f"{config['model_name']}_last_vae.net"
            ),
        }
    )

    # Predict
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir()

    gt = data.test_target
    data_range = float(gt.max() - gt.min())
    images = {}
    for checkpoint in ["best", "last"]:
        model = torch.load(run_metadata[f"checkpoint/{checkpoint}"], weights_only=False)
        model.mode_pred = True
        model.eval()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        psnrs, si_psnrs, ssims, images[checkpoint] = [], [], [], []
        with cuda_timer() as predict_timer:
            for i, noisy in enumerate(data.test):
                prediction, _samples = boilerplate.predict(
                    noisy,
                    config["mmse_samples"],
                    model,
                    config["gaussian_noise_std"],
                    device,
                    config["tta"],
                )

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
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "summary.json").write_text(json.dumps(run_metadata, indent=2))

    path = RESULTS_DIR / "convallaria" / "hdn" / f"{run_dir.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**run_metadata, "config": config}, indent=2))

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
