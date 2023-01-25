import os
from argparse import ArgumentParser

from keras.layers import Dropout
from tensorflow.keras.layers import Flatten

import wandb
from wandb.keras import WandbMetricsLogger
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from W4.src.dataloader import get_train_dataloader, get_val_dataloader
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
    lr_schedule_config = config["lr_scheduler"]
    out_dir = config["output_path"]
    optimizer_config = config["optimizer"]
    wandb_config = config["wandb"]
    data_config = config["dataloaders"]
    input_size = data_config["input_size"]
    batch_size = data_config["batch_size"]
    epochs = lr_schedule_config["params"]["num_epochs"]
    base_lr = lr_schedule_config["params"]["base_lr"]
    data_dir = data_config["data_path"]
    inference_batch_size = data_config["inference_batch_size"]

    wandb.init(project=wandb_config['project'], entity=wandb_config['entity'])
    wandb.config = {
        "learning_rate": base_lr,
        "lr_scheduler": lr_schedule_config["type"],
        "epochs": epochs,
        "batch_size": batch_size
    }

    os.makedirs(out_dir, exist_ok=True)

    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(input_size, input_size, 3))
    base_model.trainable = False
    base_model.summary()
    plot_model(base_model, to_file=out_dir + 'base_mobilenet.png', show_shapes=True, show_layer_names=True)

    head_model = Flatten(name="flatten")(base_model.output)
    head_model = Dense(8, activation='softmax', name='predictions')(head_model)

    model = Model(inputs=base_model.input, outputs=head_model)
    plot_model(model, to_file=out_dir + 'finetuning_mobilenet.png', show_shapes=True, show_layer_names=True)

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

    callbacks = [WandbMetricsLogger()]

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
