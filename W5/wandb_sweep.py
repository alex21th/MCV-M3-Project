from argparse import ArgumentParser
from typing import Dict

import numpy as np
from sklearn.metrics import confusion_matrix

import wandb
from wandb.keras import WandbMetricsLogger

from W5.src.callbacks import get_model_checkpoint_callback
from W5.src.dataloader import get_train_dataloader, get_val_dataloader, get_test_dataloader
from W5.src.models import get_model
from W5.src.optimizers import get_lr, get_optimizer
from W5.src.utils import prepare_gpu, load_config_from_yaml, plot_metrics_and_losses

import tensorflow as tf
from tensorflow.keras.metrics import TopKCategoricalAccuracy


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
        "-c", "--config", type=str, default="config_files/train_net.yaml")
    parser.add_argument(
        "-sc", "--sweep_config", type=str, default="config_files/sweep_mobilenetv3.yaml")
    return parser.parse_args()


def train_loop(config: Dict = None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # unpack config dictionaries
        config = wandb.config

        config = nested_dict(config)

        lr_schedule_config = config["lr_scheduler"]
        optimizer_config = config["optimizer"]
        batch_size = config["batch_size"]
        data_config = config["dataloaders"]
        data_augmentation = data_config["data_augmentation"]
        epochs = config['epochs']
        early_stop_config = config["early_stopping"]
        model_config = config['model']

        model = get_model(model_name=model_config['name'], pops=model_config['pops'])

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

        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      metrics=['accuracy', TopKCategoricalAccuracy(name='top_2_accuracy', k=2)])

        callbacks = [WandbMetricsLogger()]

        if early_stop_config['use']:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stop_config['patience']
            ))
        callbacks.append(get_model_checkpoint_callback(log_dir=wandb.run.dir + '/best_model.h5'))
        history = model.fit(train_dataloader,
                            steps_per_epoch=len(train_dataloader),
                            epochs=epochs,
                            validation_data=validation_dataloader,
                            validation_steps=len(validation_dataloader),
                            callbacks=callbacks,
                            verbose=1
                            )

        test_dataloader = get_test_dataloader(directory=DATA_DIR, patch_size=INPUT_SIZE)

        model.load_weights(wandb.run.dir + '/best_model.h5')
        result = model.evaluate(test_dataloader)
        wandb.log({"test_loss": result[0], "test_accuracy": result[1]})
        print(result)

        test_predictions = model.predict(test_dataloader)
        top_pred_ids = test_predictions.argmax(axis=1)
        wandb.log({"Test Confusion Matrix": wandb.plot.confusion_matrix(
            preds=top_pred_ids, y_true=test_dataloader.labels,
            class_names=list(test_dataloader.class_indices.keys()))})

        wandb.log({"pr": wandb.plot.pr_curve(test_dataloader.labels, test_predictions,
                                             labels=list(test_dataloader.class_indices.keys()))})
        wandb.log({"roc": wandb.plot.roc_curve(test_dataloader.labels, test_predictions,
                                               labels=list(test_dataloader.class_indices.keys()))})


if __name__ == '__main__':
    prepare_gpu()
    args = parse_args()
    g_config = load_config_from_yaml(args.config)['dataloaders']
    DATA_DIR = g_config['data_path']
    INPUT_SIZE = g_config['input_size']
    sweep_config = load_config_from_yaml(args.sweep_config)
    sweep_id = wandb.sweep(sweep_config, project="sweep_mobilenetv3_mixconvs")
    wandb.agent(sweep_id, train_loop, count=50)
