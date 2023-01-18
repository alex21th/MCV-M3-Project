from typing import Tuple
from keras.layers import Reshape, Dense, Dropout, Activation, Flatten
from tensorflow.python.keras import Sequential


def get_mlp(input_shape: Tuple[int, int, int] = (32, 32, 3), output_shape: int = 8,
            model_name: str = 'mlp_baseline') -> Sequential:
    """
    This function returns a Keras Sequential model based on the model name.

    :param model_name: String with the model name
    :param input_shape: Tuple with the input shape
    :param output_shape: Integer with the output neurons
    :return: Keras Sequential model
    """
    model = None
    if model_name == 'mlp_baseline':
        model = mlp_baseline
    elif model_name == 'mlp_five_layers':
        model = mlp_five_layers

    return model(input_shape=input_shape, output_shape=output_shape)


def mlp_baseline(input_shape: Tuple[int, int, int] = (32, 32, 3), output_shape: int = 8) -> Sequential:
    model = Sequential()
    model.add(
        Reshape(
            (input_shape[0] * input_shape[1] * input_shape[2],),
            input_shape=input_shape,
            name='first')
    )
    model.add(Dense(units=2048, activation='relu', name='first'))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dense(units=output_shape, activation='softmax'))

    return model


def mlp_five_layers(input_shape: Tuple[int, int, int], output_shape: int, model: str = 'mlp_baseline') -> Sequential:
    """
    This function returns a Keras Sequential model based on the model name.

    :param model: String with the model name
    :param input_shape: Tuple with the input shape
    :param output_shape: Integer with the output neurons
    :return:
    """

    model = Sequential()
    model.add(Flatten(input_shape=(input_shape[0], input_shape[1], input_shape[2])))
    model.add(Dense(units=4092, activation='relu', name='first'))
    model.add(Dense(units=2048, activation='relu', name='second'))
    model.add(Dropout (0.2))
    model.add(Dense(units=1024, activation='relu', name='third'))
    model.add(Dense(units=256, activation='relu', name='fourth'))
    model.add(Dense(units=64, activation='relu', name='fifth'))
    model.add(Dense(units=output_shape, activation='softmax'))

    return model
