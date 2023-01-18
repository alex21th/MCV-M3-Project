from argparse import ArgumentParser
from mlp import get_mlp
from keras.callbacks import EarlyStopping
from dataloader import get_train_dataloader, get_val_dataloader

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="MIT_split")  # static
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-out", "--output_dir", type=str, default="results/")
    parser.add_argument("-m", "--model", type=str, default="mlp_five_layers")
    parser.add_argument("-in", "--input_size", type=str, default=32)
    return parser.parse_args()

def __main__():
    args = parse_args()
    input_size = args.input_size
    lr = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    output_dir = args.output_dir
    model_args = args.model
    
    model = get_mlp(
        model_name=model_args,
        #input_shape=(32, 32, 3)
        #output_shape = #nr of classes
    )
    experiment_path = f"{output_dir}/{model}-{input_size}-{batch_size}-{lr}"
    model_name = experiment_path + "/weights.h5"
    plots_folder = experiment_path + '/plots'


    
    train_dataloader = get_train_dataloader(PATCH_SIZE=64, BATCH_SIZE=batch_size ,directory = args.data_dir)
    val_dataloader = get_val_dataloader(PATCH_SIZE=64, BATCH_SIZE=batch_size ,directory = args.data_dir)
    
    metrics = 'accuracy'
    loss = 'categorical_crossentropy'
    model.compile(optimizer='sgd', loss=loss, metrics=metrics)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

    # Train model
    model.fit(
        x=train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=epochs,
        #callbacks=es,
        validation_data=val_dataloader,
        validation_steps=0 if val_dataloader is None else len(val_dataloader),
        verbose=1,
    )


    print('\nFinished :)')



if __name__ == "__main__":
    __main__()