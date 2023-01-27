import os
from argparse import ArgumentParser

from wandb.integration.keras import WandbCallback

import wandb


from W4.src.dataloader import get_train_dataloader, get_val_dataloader
from W4.src.models import get_model
from W4.src.optimizers import get_lr, get_optimizer
from W4.src.utils import prepare_gpu, plot_metrics_and_losses, load_config_from_yaml

import tensorflow as tf


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default="config_files/fine_tuning.yaml"
    )
    return parser.parse_args()


def main(params):
    config = load_config_from_yaml(params.config)

    # unpack config dictionaries
    wandb_config = config["wandb"]
    lr_schedule_config = config["lr_scheduler"]
    optimizer_config = config["optimizer"]
    data_config = config["dataloaders"]
    early_stop_config = config["early_stopping"]
    model_config = config['model']

    # unpack useful values from config dictionaries
    input_size = data_config["input_size"]
    batch_size = data_config["batch_size"]
    epochs = config['epochs']
    data_dir = data_config["data_path"]
    inference_batch_size = data_config["inference_batch_size"]
    out_dir = config["output_path"]
    os.makedirs(out_dir, exist_ok=True)

    run_config = {
        'model': model_config['name'],
        "lr_scheduler": lr_schedule_config,
        "optimizer": optimizer_config,
        "batch_size": batch_size,
        'epochs': epochs,
        "early_stopping": early_stop_config
    }

    wandb.init(project=wandb_config['project'], entity=wandb_config['entity'], config=run_config)

    model = get_model(model_name=model_config['name'], out_dir=out_dir)

    train_dataloader = get_train_dataloader(directory=data_dir, patch_size=input_size,
                                            batch_size=batch_size)
    validation_dataloader = get_val_dataloader(directory=data_dir, patch_size=input_size,
                                               batch_size=batch_size)
    test_dataloader = get_val_dataloader(directory=data_dir, patch_size=input_size, batch_size=inference_batch_size)

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

    callbacks = [WandbCallback(), ]

    if config['early_stopping']['use']:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['early_stopping']['patience']
        ))

    history = model.fit(train_dataloader,
                        steps_per_epoch=len(train_dataloader),
                        epochs=epochs,
                        validation_data=validation_dataloader,
                        validation_steps=len(validation_dataloader),
                        callbacks=callbacks,
                        verbose=1
                        )

    result = model.evaluate(test_dataloader)
    print(result)
    print(history.history.keys())

    plot_metrics_and_losses(history, path=out_dir)


if __name__ == '__main__':
    prepare_gpu()
    args = parse_args()
    main(args)
