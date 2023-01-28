from keras.layers import Flatten, AveragePooling2D
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dropout
import tensorflow as tf


def get_model(model_name: str, out_dir: str = None, input_size: int = 224, pops: int = 6):
    """
    Get a mobilenet model
    :param model_name: model name
    :param out_dir: output directory
    :param input_size: input size
    :return: model
    """
    if model_name == 'ft_baseline':
        model = ft_baseline(out_dir, input_size)
    elif model_name == 'ft_dense_head':
        model = ft_dense_head(out_dir, input_size)
    elif model_name == 'ft_dropout':
        model = ft_dense_head_dropout(out_dir, input_size)
    elif model_name == 'modified_mobilenet':
        model = modified_mobilenet(out_dir, input_size, pops)
    else:
        raise ValueError(f'Unknown model name: {model_name}')
    return model


# TASK 0: Create a baseline model for fine-tuning
def ft_baseline(out_dir: str = None, input_size: int = 224):
    """
    Create a baseline model for fine-tuning
    :param out_dir: output directory
    :param input_size: input size
    :return: model
    """
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(input_size, input_size, 3))
    base_model.trainable = False
    if out_dir is not None:
        base_model.summary()
        plot_model(base_model, to_file=out_dir + 'base_mobilenet.png', show_shapes=True, show_layer_names=True)

    head_model = AveragePooling2D(pool_size=(7, 7), name='average_pool_last')(base_model.output)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(8, activation='softmax', name='predictions')(head_model)

    model = Model(inputs=base_model.input, outputs=head_model)
    if out_dir is not None:
        plot_model(model, to_file=out_dir + 'finetuning_mobilenet.png', show_shapes=True, show_layer_names=True)
    return model


# TASK 1: Improved dense head
def ft_dense_head(out_dir: str = None, input_size: int = 224):
    """
    Create a baseline model for fine-tuning
    :param out_dir: output directory
    :param input_size: input size
    :return: model
    """
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(input_size, input_size, 3))
    base_model.trainable = False
    if out_dir is not None:
        base_model.summary()
        plot_model(base_model, to_file=out_dir + 'base_mobilenet.png', show_shapes=True, show_layer_names=True)

    head_model = AveragePooling2D(pool_size=(7, 7), name='average_pool_last')(base_model.output)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(2048, activation='relu')(head_model)
    head_model = Dense(1024, activation='relu')(head_model)
    head_model = Dense(8, activation='softmax', name='predictions')(head_model)

    model = Model(inputs=base_model.input, outputs=head_model)
    if out_dir is not None:
        plot_model(model, to_file=out_dir + 'finetuning_mobilenet.png', show_shapes=True, show_layer_names=True)
    return model


# TASK 1: Improved dense head with dropout
def ft_dense_head_dropout(out_dir: str = None, input_size: int = 224):
    """
    Create a baseline model for fine-tuning
    :param out_dir: output directory
    :param input_size: input size
    :return: model
    """
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(input_size, input_size, 3))
    base_model.trainable = False
    if out_dir is not None:
        base_model.summary()
        plot_model(base_model, to_file=out_dir + 'base_mobilenet.png', show_shapes=True, show_layer_names=True)

    head_model = Flatten(name="flatten")(base_model.output)
    head_model = Dense(2048, activation='relu')(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(1024, activation='relu')(head_model)
    head_model = Dense(8, activation='softmax', name='predictions')(head_model)

    model = Model(inputs=base_model.input, outputs=head_model)
    if out_dir is not None:
        plot_model(model, to_file=out_dir + 'finetuning_mobilenet.png', show_shapes=True, show_layer_names=True)
    return model


# TODO: Implement a model that include architectural changes, not only modifying the head
def modified_mobilenet(out_dir: str = None, input_size: int = 224, pops: int = 6):
    """
    Create a baseline model for fine-tuning
    :param out_dir: output directory
    :param input_size: input size
    :param pops: number of layers to pop
    :return: model
    """
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(input_size, input_size, 3))
    base_model.trainable = False

    if pops > 0:
        base_model = tf.keras.models.Sequential(base_model.layers[:-pops])

    if out_dir is not None:
        base_model.summary()
        plot_model(base_model, to_file=out_dir + 'base_mobilenet.png', show_shapes=True, show_layer_names=True)

    head_model = AveragePooling2D(pool_size=(7, 7), name='average_pool_last')(base_model.output)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(2048, activation='relu')(head_model)
    head_model = Dropout(0.2)(head_model)
    head_model = Dense(1024, activation='relu')(head_model)
    head_model = Dense(8, activation='softmax', name='predictions')(head_model)

    model = Model(inputs=base_model.input, outputs=head_model)
    if out_dir is not None:
        plot_model(model, to_file=out_dir + 'finetuning_mobilenet.png', show_shapes=True, show_layer_names=True)
    return model
