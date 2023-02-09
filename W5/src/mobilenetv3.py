from typing import Union, Tuple

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


def conv_block(inputs: tf.Tensor, filters: int, kernel: Union[Tuple[int, int], int],
               strides: Union[Tuple[int, int], int], nl: str) -> tf.Tensor:
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


def _get_shape_value(maybe_v2_shape):
    """
    Returns the value of a shape tensor or a scalar.
    """
    if maybe_v2_shape is None:
        return None
    elif isinstance(maybe_v2_shape, int):
        return maybe_v2_shape
    else:
        return maybe_v2_shape.value


def _split_channels(total_filters, num_groups):
    """
    Splits the filters into groups.
    """
    split = [total_filters // num_groups for _ in range(num_groups)]
    split[0] += total_filters - sum(split)
    return split


def mix_conv(
        inputs: tf.Tensor,
        kernel_size: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]],
        dilated: bool = False,
        **kwargs
) -> tf.Tensor:
    """Initialize the layer.
    Most of args are the same as tf.keras.layers.DepthwiseConv2D except it has
    an extra parameter "dilated" to indicate whether to use dilated conv to
    simulate large kernel size. If dilated=True, then dilation_rate is ignored.
    Args:
      inputs: A 4d tensor with shape [batch, heigh, width, channels].
      kernel_size: An integer or a list. If it is a single integer, then it is
        same as the original tf.keras.layers.DepthwiseConv2D. If it is a list,
        then we split the channels and perform different kernel for each group.
      strides: An integer or tuple/list of 2 integers, specifying the strides of
        the convolution along the height and width.
      dilated: Bool. indicate whether to use dilated conv to simulate large
        kernel size.
      **kwargs: other parameters passed to the original depthwise_conv layer.
    """
    _channel_axis = -1
    filters = _get_shape_value(inputs.shape[_channel_axis])
    splits = _split_channels(filters, len(kernel_size))
    x_splits = tf.split(inputs, splits, _channel_axis)
    x_res = []
    for split, ks in zip(x_splits, kernel_size):
        d = 1
        if strides[0] == 1 and dilated:
            # Only apply dilated conv for stride 1 if needed.
            d, s = (ks - 1) // 2, 3
        x_res.append(tf.keras.layers.DepthwiseConv2D(ks, strides=strides, dilation_rate=d, **kwargs)(split))
    x = tf.concat(x_res, _channel_axis)

    return x


def mix_conv_bottleneck(inputs, filters, kernel, e, s, squeeze, nl, alpha):
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

    x = mix_conv(x, kernel, strides=(s, s), dilated=False, depth_multiplier=1, padding='same')
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


def mobile_net_v3_small_mix_convs(input_shape: int, output_shape: int = 8, alpha: float = 1.0) -> Model:
    """
    MobileNetV3 small model for Keras.

    :param input_shape: shape of input image
    :param output_shape: number of classes
    :param alpha: width multiplier
    :return: Keras functional model
    """

    inputs = Input(shape=(input_shape, input_shape, 3))

    x = conv_block(inputs, 16, (3, 3), strides=(2, 2), nl='HS')

    kernels = [(3, 3), (5, 5), (7, 7), (9, 9)]

    x = mix_conv_bottleneck(x, 16, kernels, e=16, s=2, squeeze=True, nl='RE', alpha=alpha)
    x = mix_conv_bottleneck(x, 24, kernels, e=72, s=2, squeeze=False, nl='RE', alpha=alpha)
    x = mix_conv_bottleneck(x, 24, kernels, e=88, s=1, squeeze=False, nl='RE', alpha=alpha)
    x = mix_conv_bottleneck(x, 40, kernels, e=96, s=2, squeeze=True, nl='HS', alpha=alpha)
    x = mix_conv_bottleneck(x, 40, kernels, e=240, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = mix_conv_bottleneck(x, 40, kernels, e=240, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = mix_conv_bottleneck(x, 48, kernels, e=120, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = mix_conv_bottleneck(x, 48, kernels, e=144, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = mix_conv_bottleneck(x, 96, kernels, e=288, s=2, squeeze=True, nl='HS', alpha=alpha)
    x = mix_conv_bottleneck(x, 96, kernels, e=576, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = mix_conv_bottleneck(x, 96, kernels, e=576, s=1, squeeze=True, nl='HS', alpha=alpha)

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
