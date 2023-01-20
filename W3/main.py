from argparse import ArgumentParser

from tensorflow.keras.utils import plot_model

from src.mlp import get_mlp
from src.dataloader import get_train_dataloader, get_val_dataloader
from src.plotting import plot_metrics_and_losses
from src.callbacks import get_callbacks
import os
import tensorflow as tf


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="../MIT_split")  # static
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-e", "--epochs", type=int, default=40)
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("-out", "--output_dir", type=str, default="results/")
    parser.add_argument("-m", "--model", type=str, default="mlp_baseline")
    parser.add_argument("-in", "--input_size", type=int, default=16)
    parser.add_argument("-opt", "--optimizer", type=str, default='sgd')
    return parser.parse_args()


def prepare_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], "GPU")
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


def main():
    input_size = args.input_size
    lr = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    output_dir = args.output_dir
    model_name = args.model
    data_dir = args.data_dir
    optimizer_name = args.optimizer

    experiment_path = f"{output_dir}{model_name}-{input_size}-{batch_size}-{lr}-{optimizer_name}"
    best_model_path = experiment_path + "/weights.h5"
    plots_folder = experiment_path + '/plots/'
    os.makedirs(plots_folder, exist_ok=True)

    model = get_mlp(
        model_name=model_name,
        input_shape=(input_size, input_size, 3),
        output_shape=8
    )

    train_dataloader = get_train_dataloader(patch_size=input_size, batch_size=batch_size, directory=data_dir)
    val_dataloader = get_val_dataloader(patch_size=input_size, batch_size=batch_size, directory=data_dir)

    metrics = 'accuracy'
    loss = 'categorical_crossentropy'

    # TODO: add optimizer with a lr scheduler (first the lr_scheduler, then the optimizer)
    model.compile(optimizer=optimizer_name, loss=loss, metrics=metrics)
    plot_model(model, to_file=plots_folder + 'modelMLP.png', show_shapes=True, show_layer_names=True)

    callbacks = get_callbacks(best_model_path, experiment_path, es_use=True, es_patience=15)

    # Train model
    model.fit(
        x=train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_dataloader,
        validation_steps=0 if val_dataloader is None else len(val_dataloader),
        verbose=1,
    )

    print('\nFinished :)')
    plot_metrics_and_losses(history=model.history, path=plots_folder)


if __name__ == "__main__":
    prepare_gpu()
    args = parse_args()

    for i_size in [16, 32, 64]:
        args.input_size = i_size
        for lr in [0.001, 0.005]:
            args.learning_rate = lr
            for opt in ['sgd', 'adam']:
                args.optimizer = opt
                for batch_size in [8, 16, 32]:
                    args.batch_size = batch_size
                    main()
