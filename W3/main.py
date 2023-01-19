from argparse import ArgumentParser
from src.mlp import get_mlp
from src.dataloader import get_train_dataloader, get_val_dataloader
from src.plotting import plot_metrics_and_losses
from src.callbacks import get_callbacks
import os


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="../MIT_split")  # static
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-e", "--epochs", type=int, default=50)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-out", "--output_dir", type=str, default="results/")
    parser.add_argument("-m", "--model", type=str, default="mlp_baseline")
    parser.add_argument("-in", "--input_size", type=int, default=32)
    parser.add_argument("-opt", "--optimizer", type=str, default='adam')
    return parser.parse_args()


def main():
    args = parse_args()
    input_size = args.input_size
    lr = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    output_dir = args.output_dir
    model_name = args.model
    data_dir = args.data_dir
    optimizer_name = args.optimizer

    experiment_path = f"{output_dir}{model_name}-{input_size}-{batch_size}-{lr}-{optimizer_name}"
    model_path = experiment_path + "/weights.h5"
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
    callbacks = get_callbacks(model_path, es_use=True, es_patience=15)

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
    main()
