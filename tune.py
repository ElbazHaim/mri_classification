"""

"""
import os
import optuna
import pytorch_lightning as pl

from icecream import ic

from datamodules import MRIAltzheimerDataModule
from models import PlLeNet
from transformations import mri_jpg_preprocessing

BATCHSIZE = 32
CLASSES = 4
EPOCHS = 20
N_TRIALS = 1000
TIMEOUT = 3600
INPUT_SHAPE = (1, 188, 156)
DATA_DIR = os.path.join(os.getcwd(), "datamodules", "data")


def objective(trial: optuna.trial.Trial) -> float:
    n_conv_blocks = trial.suggest_int("n_conv_blocks", 1, 5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-9, 1e-3)

    model = PlLeNet(
        input_shape=INPUT_SHAPE,
        num_classes=CLASSES,
        num_conv_blocks=n_conv_blocks,
        learning_rate=learning_rate,
    )
    datamodule = MRIAltzheimerDataModule(
        data_dir=DATA_DIR, batch_size=BATCHSIZE, transform=mri_jpg_preprocessing
    )

    trainer = pl.Trainer(
        fast_dev_run=False,
        logger=True,
        enable_checkpointing=True,
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=1,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=3,
                verbose=False,
                mode="min",
            )
        ],
    )
    hyperparameters = dict(num_conv_blocks=n_conv_blocks, learning_rate=learning_rate)
    trainer.logger.log_hyperparams(hyperparameters)

    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    prune = False
    pruner = optuna.pruners.MedianPruner() if prune else optuna.pruners.NopPruner()

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)

    ic(len(study.trials))
    ic(study.best_trial)
    ic(study.best_trial.value)
