import os
from argparse import ArgumentParser

from keras.layers import Dropout
from tensorflow.keras.layers import AveragePooling2D, Flatten

import wandb
from wandb.keras import WandbMetricsLogger
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from W4.src.dataloader import get_train_dataloader, get_val_dataloader
from W4.src.utils import prepare_gpu, plot_metrics_and_losses


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="../MIT_split/")  # static
    parser.add_argument("-mp", "--best_model_path", type=str, default="best_ft_model.h5")
    parser.add_argument("-in", "--input_size", type=int, default=224)
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument("-ft_e", "--ft_epochs", type=int, default=20)
    parser.add_argument("-ft_lr", "--ft_lr", type=int, default=0.001)
    parser.add_argument("-out", "--output_dir", type=str, default="results_ft/")
    return parser.parse_args()


def main(params):
    ft_lr = params.ft_lr
    ft_epochs = params.ft_epochs
    batch_size = params.batch_size
    out_dir = params.output_dir
    data_dir = params.data_dir
    input_size = params.input_size

    wandb.init(project="test-project", entity="mcv-m3-g6")
    wandb.config = {
        "learning_rate": ft_lr,
        "epochs": ft_epochs,
        "batch_size": batch_size
    }

    os.makedirs(out_dir, exist_ok=True)

    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(input_size, input_size, 3))
    base_model.trainable = False
    base_model.summary()
    plot_model(base_model, to_file=out_dir + 'base_mobilenet.png', show_shapes=True, show_layer_names=True)

    head_model = Flatten(name="flatten")(base_model.output)
    head_model = Dropout(0.2)(head_model)
    head_model = Dense(2048, activation='relu', name='dense_head_1')(head_model)
    head_model = Dropout(0.2)(head_model)
    head_model = Dense(1024, activation='relu', name='dense_head_2')(head_model)
    head_model = Dropout(0.2)(head_model)
    head_model = Dense(8, activation='softmax', name='predictions')(head_model)

    model = Model(inputs=base_model.input, outputs=head_model)
    plot_model(model, to_file=out_dir + 'finetuning_mobilenet.png', show_shapes=True, show_layer_names=True)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    train_dataloader = get_train_dataloader(directory=data_dir, patch_size=input_size,
                                            batch_size=batch_size)
    validation_dataloader = get_val_dataloader(directory=data_dir, patch_size=input_size,
                                               batch_size=params.batch_size)
    test_dataloader = get_val_dataloader(directory=data_dir, patch_size=input_size, batch_size=1)

    history = model.fit(train_dataloader,
                        steps_per_epoch=len(train_dataloader),
                        epochs=ft_epochs,
                        validation_data=validation_dataloader,
                        validation_steps=len(validation_dataloader), callbacks=[WandbMetricsLogger()])

    result = model.evaluate(test_dataloader)
    print(result)
    print(history.history.keys())

    plot_metrics_and_losses(history, path=out_dir)


if __name__ == '__main__':
    prepare_gpu()
    args = parse_args()
    main(args)
