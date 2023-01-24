from tensorflow.keras import backend as K


def preprocess_input(in_x, dim_ordering='default'):
    """
    Preprocesses a tensor encoding a batch of images.
    :param in_x: input Numpy tensor, 4D.
    :param dim_ordering: image data format, either "channels_first" or "channels_last".
    """
    if dim_ordering == 'default':
        dim_ordering = K.image_data_format()
    assert dim_ordering in {'channels_first', 'channels_last'}

    if dim_ordering == 'channels_first':
        # 'RGB'->'BGR'
        in_x = in_x[::-1, :, :]
        # Zero-center by mean pixel
        in_x[0, :, :] -= 103.939
        in_x[1, :, :] -= 116.779
        in_x[2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        in_x = in_x[:, :, ::-1]
        # Zero-center by mean pixel
        in_x[:, :, 0] -= 103.939
        in_x[:, :, 1] -= 116.779
        in_x[:, :, 2] -= 123.68
    return in_x
