from typing import Tuple

from keras.layers import Reshape, Dense
from tensorflow.python.keras import Sequential


def get_mlp(input_shape: Tuple[int, int, int], output_shape: int, model: str = 'mlp_og') -> Sequential:
    """
    This function returns a Keras Sequential model based on the model name.

    :param model: String with the model name
    :param input_shape: Tuple with the input shape
    :param output_shape: Integer with the output neurons
    :return:
    """


    model = Sequential()
    model.add(
        Reshape(
            (input_shape[0] * input_shape[1] * input_shape[2],),
            input_shape=input_shape,
            name='first')
    )
    model.add(Dense(units=4092, activation='relu'))
    model.add(Dense(units=2048, activation='relu', name='second'))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=output_shape, activation='softmax'))
    model.compile(loss='categorical_crossentropy')

    return model

