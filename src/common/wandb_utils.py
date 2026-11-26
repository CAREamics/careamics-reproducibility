import wandb
from lightning.pytorch.loggers import WandbLogger

from common import WANDB_PROJECT


def setup_wandb(careamist, meta):
    run_id = careamist.work_dir.name
    for logger in careamist.trainer.loggers:
        if isinstance(logger, WandbLogger):
            logger._wandb_init["project"] = WANDB_PROJECT
            logger._wandb_init["id"] = run_id
            logger._wandb_init["group"] = meta["dataset"]
            logger._wandb_init["tags"] = [
                f"impl:{meta['impl']}",
                f"algo:{meta['algo']}",
                f"dataset:{meta['dataset']}",
                f"mode:{meta['mode']}",
            ]
            logger._wandb_init["config"] = meta
            logger._wandb_init["settings"] = wandb.Settings(
                save_code=True, code_dir="src"
            )
    return run_id


def detach_wandb(careamist):
    careamist.trainer.loggers = [
        logger
        for logger in careamist.trainer.loggers
        if not isinstance(logger, WandbLogger)
    ]
