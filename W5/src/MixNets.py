from keras.layers import ReLU, Add
from tensorflow.keras.layers import BatchNormalization, Conv2D, Lambda, Activation, Multiply
from keras import backend as K


class GroupConvolution(layers.Layer):

    def __init__(self, filters, kernels, groups,
                 type='conv', conv_kwargs=None,
                 **kwargs):
        super(GroupConvolution, self).__init__(**kwargs)

        if conv_kwargs is None:
            conv_kwargs = {
                'strides': (1, 1),
                'padding': 'same',
                'dilation_rate': (1, 1),
                'use_bias': False,
            }

        self.filters = filters
        self.kernels = kernels
        self.groups = groups
        self.type = type
        self.strides = conv_kwargs.get('strides', (1, 1))
        self.padding = conv_kwargs.get('padding', 'same')
        self.dilation_rate = conv_kwargs.get('dilation_rate', (1, 1))
        self.use_bias = conv_kwargs.get('use_bias', False)
        self.conv_kwargs = conv_kwargs or {}

        assert type in ['conv', 'depthwise_conv']
        if type == 'conv':
            splits = self._split_channels(filters, self.groups)
            self._layers = [layers.Conv2D(splits[i], kernels[i],
                                          strides=self.strides,
                                          padding=self.padding,
                                          dilation_rate=self.dilation_rate,
                                          use_bias=self.use_bias,
                                          kernel_initializer=MixNetConvInitializer())
                            for i in range(groups)]

        else:
            self._layers = [layers.DepthwiseConv2D(kernels[i],
                                                   strides=self.strides,
                                                   padding=self.padding,
                                                   dilation_rate=self.dilation_rate,
                                                   use_bias=self.use_bias,
                                                   kernel_initializer=MixNetConvInitializer())
                            for i in range(groups)]

        self.data_format = 'channels_last'
        self._channel_axis = -1

    def call(self, inputs, **kwargs):
        if len(self._layers) == 1:
            return self._layers[0](inputs)

        filters = K.int_shape(inputs)[self._channel_axis]
        splits = self._split_channels(filters, self.groups)
        x_splits = tf.split(inputs, splits, self._channel_axis)
        x_outputs = [c(x) for x, c in zip(x_splits, self._layers)]
        x = layers.concatenate(x_outputs, axis=self._channel_axis)
        return x

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                filter_size=1,
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        return (input_shape[0],) + tuple(new_space) + (self.filters,)

    def _split_channels(self, total_filters, num_groups):
        split = [total_filters // num_groups for _ in range(num_groups)]
        split[0] += total_filters - sum(split)
        return split

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernels': self.kernels,
            'groups': self.groups,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'type': self.type,
            'conv_kwargs': self.conv_kwargs,
        }
        base_config = super(GroupConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/mixnet_model.py
def SEBlock(input_filters, se_ratio, expand_ratio, activation_fn, data_format=None):
    if data_format is None:
        data_format = K.image_data_format()

    num_reduced_filters = max(
        1, int(input_filters * se_ratio))
    filters = input_filters * expand_ratio

    if data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    def block(inputs):
        x = inputs
        x = Lambda(lambda a: K.mean(a, axis=spatial_dims, keepdims=True))(x)
        x = GroupedConv2D(
            num_reduced_filters,
            kernel_size=[1],
            strides=[1, 1],
            kernel_initializer=MixNetConvInitializer(),
            padding='same',
            use_bias=True)(x)

        x = activation_fn()(x)

        # Excite
        x = GroupedConv2D(
            filters,
            kernel_size=[1],
            strides=[1, 1],
            kernel_initializer=MixNetConvInitializer(),
            padding='same',
            use_bias=True)(x)
        x = Activation('sigmoid')(x)
        out = Multiply()([x, inputs])
        return out

    return block


# Obtained from
class MDConv(object):
    """MDConv with mixed depth-wise convolutional kernels.
    MDConv is an improved depth-wise convolution that mixes multiple kernels (e.g.
    3x3, 5x5, etc). Right now, we use a naive implementation that split channels
    into multiple groups and perform different kernels for each group.
    See Mixnet paper for more details.
    """

    def __init__(self, kernel_size, strides, dilated=False, **kwargs):
        """Initialize the layer.
        Most of args are the same as tf.keras.layers.DepthwiseConv2D except it has
        an extra parameter "dilated" to indicate whether to use dilated conv to
        simulate large kernel_size size. If dilated=True, then dilation_rate is ignored.
        Args:
          kernel_size: An integer or a list. If it is a single integer, then it is
            same as the original tf.keras.layers.DepthwiseConv2D. If it is a list,
            then we split the channels and perform different kernel_size for each group.
          strides: An integer or tuple/list of 2 integers, specifying the strides of
            the convolution along the height and width.
          dilated: Bool. indicate whether to use dilated conv to simulate large
            kernel_size size.
          **kwargs: other parameters passed to the original depthwise_conv layer.
        """
        self._channel_axis = -1
        self._dilated = dilated
        self.kernels = kernel_size

        self._conv_kwargs = {
            'strides': strides,
            'dilation_rate': kwargs.get('dilation_rate', (1, 1)),
            'kernel_initializer': kwargs.get('kernel_initializer', MixNetConvInitializer()),
            'padding': 'same',
            'use_bias': kwargs.get('use_bias', False),
        }

    def __call__(self, inputs):
        filters = K.int_shape(inputs)[self._channel_axis]
        grouped_op = GroupConvolution(filters, self.kernels, groups=len(self.kernels),
                                      type='depthwise_conv', conv_kwargs=self._conv_kwargs)
        x = grouped_op(inputs)
        return x


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/mixnet_model.py
def MixNetBlock(input_filters, output_filters,
                dw_kernel_size, expand_kernel_size,
                project_kernel_size, strides,
                expand_ratio, se_ratio,
                id_skip,
                batch_norm_momentum=0.99,
                batch_norm_epsilon=1e-3,
                dilated=None,
                data_format=None):

    if data_format is None:
        data_format = K.image_data_format()

    if data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    has_se = (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1)
    filters = input_filters * expand_ratio
    relu_activation = ReLU

    def block(inputs):
        # Expand part
        if expand_ratio != 1:
            x = GroupedConv2D(
                filters,
                kernel_size=expand_kernel_size,
                strides=[1, 1],
                kernel_initializer=MixNetConvInitializer(),
                padding='same',
                use_bias=False)(inputs)

            x = BatchNormalization(
                axis=channel_axis,
                momentum=batch_norm_momentum,
                epsilon=batch_norm_epsilon)(x)

            x = relu_activation()(x)
        else:
            x = inputs

        kernel_size = dw_kernel_size
        # Depthwise Convolutional Phase
        x = MDConv(
            kernel_size,
            strides=strides,
            dilated=dilated,
            depthwise_initializer=MixNetConvInitializer(),
            padding='same',
            use_bias=False)(x)
        x = BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)(x)
        x = relu_activation()(x)

        if has_se:
            x = SEBlock(input_filters, se_ratio, expand_ratio,
                        relu_activation,
                        data_format)(x)

        # output phase
        x = GroupedConv2D(
            output_filters,
            kernel_size=project_kernel_size,
            strides=[1, 1],
            kernel_initializer=MixNetConvInitializer(),
            padding='same',
            use_bias=False)(x)
        x = BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)(x)

        if id_skip:
            if all(s == 1 for s in strides) and (
                    input_filters == output_filters):

                # only apply drop_connect if skip presents.
                # if drop_connect_rate:
                #     x = DropConnect(drop_connect_rate)(x)

                x = Add()([x, inputs])

        return x

    return block
