import wandb
from lightning.pytorch.loggers import WandbLogger

from common import IMPL, WANDB_PROJECT


def setup_wandb(careamist, algo, dataset):
    run_id = f"{careamist.config.experiment_name}_{careamist.run_version}"
    for logger in careamist.trainer.loggers:
        if isinstance(logger, WandbLogger):
            logger._wandb_init["project"] = WANDB_PROJECT
            logger._wandb_init["id"] = run_id
            logger._wandb_init["group"] = dataset
            logger._wandb_init["tags"] = [
                f"impl:{IMPL}",
                f"algo:{algo}",
                f"dataset:{dataset}",
            ]
            logger._wandb_init["settings"] = wandb.Settings(save_code=True)
    return run_id


def detach_wandb(careamist):
    careamist.trainer.loggers = [
        logger
        for logger in careamist.trainer.loggers
        if not isinstance(logger, WandbLogger)
    ]
