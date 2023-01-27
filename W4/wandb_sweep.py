from argparse import ArgumentParser
from typing import Dict

import wandb
from wandb.keras import WandbMetricsLogger

from W4.src.dataloader import get_train_dataloader, get_val_dataloader
from W4.src.models import get_model
from W4.src.optimizers import get_lr, get_optimizer
from W4.src.utils import prepare_gpu, load_config_from_yaml

import tensorflow as tf


def nested_dict(original_dict):
    nested_dict = {}
    for key, value in original_dict.items():
        parts = key.split(".")
        d = nested_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return nested_dict


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default="config_files/fine_tuning.yaml")
    parser.add_argument(
        "-sc", "--sweep_config", type=str, default="config_files/wandb_sweep.yaml")
    return parser.parse_args()


def train_loop(config: Dict = None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # unpack config dictionaries
        config = wandb.config

        config = nested_dict(config)

        model_name = config['model_name']
        lr_schedule_config = config["lr_scheduler"]
        optimizer_config = config["optimizer"]
        batch_size = config["batch_size"]
        data_config = config["dataloaders"]
        data_augmentation = data_config["data_augmentation"]
        epochs = config['epochs']
        early_stop_config = config["early_stopping"]

        model = get_model(model_name=model_name)

        train_dataloader = get_train_dataloader(directory=DATA_DIR, patch_size=INPUT_SIZE,
                                                batch_size=batch_size, data_augmentation=data_augmentation)
        validation_dataloader = get_val_dataloader(directory=DATA_DIR, patch_size=INPUT_SIZE,
                                                   batch_size=batch_size)

        lr = get_lr(
            lr_decay_scheduler_name=lr_schedule_config["type"],
            num_iterations=len(train_dataloader),
            config=lr_schedule_config["params"]
        )

        optimizer = get_optimizer(
            optimizer_name=optimizer_config["type"],
            learning_rate=lr,
            params=optimizer_config["params"]
        )

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        callbacks = [WandbMetricsLogger()]

        if early_stop_config['use']:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stop_config['patience']
            ))

        model.fit(train_dataloader,
                  steps_per_epoch=len(train_dataloader),
                  epochs=epochs,
                  validation_data=validation_dataloader,
                  validation_steps=len(validation_dataloader),
                  callbacks=callbacks,
                  verbose=1
                  )


if __name__ == '__main__':
    prepare_gpu()
    args = parse_args()
    g_config = load_config_from_yaml(args.config)['dataloaders']
    DATA_DIR = g_config['data_path']
    INPUT_SIZE = g_config['input_size']
    sweep_config = load_config_from_yaml(args.sweep_config)
    sweep_id = wandb.sweep(sweep_config, project="task_0_sweep")
    wandb.agent(sweep_id, train_loop, count=30)
