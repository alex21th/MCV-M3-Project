from typing import Union

from keras.layers import Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D
from keras.layers import Activation, BatchNormalization, Add, Multiply, Reshape
import tensorflow as tf
from keras import backend as K, Input, Model


def hard_swish(x: tf.Tensor) -> tf.Tensor:
    """
    This function defines a hard swish activation.
    """
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0


def relu6(x: tf.Tensor) -> tf.Tensor:
    """
    This function defines a relu6 activation.
    """
    return K.relu(x, max_value=6.0)


def return_activation(x: tf.Tensor, nl: str) -> tf.Tensor:
    """Convolution Block
    This function defines an activation choice.

    # Returns
        Output tensor.
    """
    if nl == 'HS':
        x = Activation(hard_swish)(x)
    if nl == 'RE':
        x = Activation(relu6)(x)

    return x


def conv_block(inputs: tf.Tensor, filters: int, kernel: Union[tuple[int, int], int],
               strides: Union[tuple[int, int], int], nl: str) -> tf.Tensor:
    """
    Convolution Block
    This function defines a 2D convolution operation with BN and activation.
    :param inputs: input tensor
    :param filters: filters
    :param kernel: kernel size
    :param strides: stride size
    :param nl: non-linearity
    :return: output tensor
    """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = Conv2D(filters, kernel, padding='same', use_bias=False, strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return return_activation(x, nl)


def se_module(inputs: tf.Tensor) -> tf.Tensor:
    """
    Squeeze and Excitation.
    This function defines a squeeze structure.
    :param inputs: input tensor
    :return: output tensor
    """
    input_channels = int(inputs.shape[-1])

    x = GlobalAveragePooling2D()(inputs)
    x = Dense(input_channels, activation='relu')(x)
    x = Dense(input_channels, activation='hard_sigmoid')(x)
    x = Reshape((1, 1, input_channels))(x)
    x = Multiply()([inputs, x])

    return x


def bottleneck(inputs, filters, kernel, e, s, squeeze, nl, alpha):
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        e: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        squeeze: Boolean, Whether to use the squeeze.
        nl: String, non-linearity activation type.
    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    input_shape = K.int_shape(inputs)

    tchannel = int(e)
    cchannel = int(alpha * filters)

    r = s == 1 and input_shape[3] == filters

    x = conv_block(inputs, tchannel, (1, 1), (1, 1), nl)

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = return_activation(x, nl)

    if squeeze:
        x = se_module(x)

    x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = Add()([x, inputs])

    return x


def mobile_net_v3_small(input_shape: int, output_shape: int = 8, alpha: float = 1.0) -> Model:
    """
    MobileNetV3 small model for Keras.

    :param input_shape: shape of input image
    :param output_shape: number of classes
    :param alpha: width multiplier
    :return: Keras functional model
    """

    inputs = Input(shape=(input_shape, input_shape, 3))

    x = conv_block(inputs, 16, (3, 3), strides=(2, 2), nl='HS')

    x = bottleneck(x, 16, (3, 3), e=16, s=2, squeeze=True, nl='RE', alpha=alpha)
    x = bottleneck(x, 24, (3, 3), e=72, s=2, squeeze=False, nl='RE', alpha=alpha)
    x = bottleneck(x, 24, (3, 3), e=88, s=1, squeeze=False, nl='RE', alpha=alpha)
    x = bottleneck(x, 40, (5, 5), e=96, s=2, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 48, (5, 5), e=120, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 48, (5, 5), e=144, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 96, (5, 5), e=288, s=2, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS', alpha=alpha)

    x = conv_block(x, 576, (1, 1), strides=(1, 1), nl='HS')
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 576))(x)

    x = Conv2D(1280, (1, 1), padding='same')(x)
    x = return_activation(x, 'HS')

    x = Conv2D(output_shape, (1, 1), padding='same', activation='softmax')(x)
    x = Reshape((output_shape,))(x)

    model = Model(inputs, x)

    model.summary()
    return model
