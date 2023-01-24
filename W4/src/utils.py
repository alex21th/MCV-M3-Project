import tensorflow as tf

from matplotlib import pyplot as plt


def prepare_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], "GPU")
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


def plot_metrics_and_losses(history, path: str):
    """
    Plots the metrics and losses of the model
    :param history: history of the model
    :param path: path to save the plots
    """

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(path + 'accuracy.jpg')
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(path + 'loss.jpg')
    plt.close()
