from argparse import ArgumentParser
from mlp import get_mlp


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="data/")  # static
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-out", "--output_dir", type=str, default="results/")
    parser.add_argument("-m", "--model", type=str, default="mlp1")
    parser.add_argument("-in", "--in", type=str, default="mlp1")
    return parser.parse_args()


def __main__():
    args = parse_args()
    lr = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    output_dir = args.output_dir

    model = get_mlp(
        model=args.model,
        input_shape=input_shape,
        output_shape=output_shape

    )



    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

