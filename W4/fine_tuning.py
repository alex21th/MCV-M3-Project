import os
from argparse import ArgumentParser

import matplotlib
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from W4.src.preprocessing import preprocess_input
from W4.src.utils import prepare_gpu

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="../MIT_split/")  # static
    parser.add_argument("-mp", "--best_model_path", type=str, default="best_ft_model.h5")
    parser.add_argument("-in", "--input_size", type=int, default=224)
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument("-e", "--ft_epochs", type=int, default=20)
    parser.add_argument("-out", "--output_dir", type=str, default="results_ft/")
    return parser.parse_args()


def main(params):

    os.makedirs(params.output_dir, exist_ok=True)

    base_model = MobileNet(weights='imagenet')

    plot_model(base_model, to_file=params.output_dir + 'base_mobilenet.png', show_shapes=True, show_layer_names=True)

    x = base_model.layers[-2].output
    x = Dense(8, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=x)
    plot_model(model, to_file=params.output_dir + 'finetuning_mobilenet.png', show_shapes=True, show_layer_names=True)
    # freeze all layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    for layer in model.layers:
        print(layer.name, layer.trainable)

    datagen = ImageDataGenerator(featurewise_center=False,
                                 samplewise_center=False,
                                 featurewise_std_normalization=False,
                                 samplewise_std_normalization=False,
                                 preprocessing_function=preprocess_input,
                                 rotation_range=0.,
                                 width_shift_range=0.,
                                 height_shift_range=0.,
                                 shear_range=0.,
                                 zoom_range=0.,
                                 channel_shift_range=0.,
                                 fill_mode='nearest',
                                 cval=0.,
                                 horizontal_flip=False,
                                 vertical_flip=False,
                                 rescale=None)

    train_generator = datagen.flow_from_directory(params.data_dir + 'train',
                                                  target_size=(params.input_size, params.input_size),
                                                  batch_size=params.batch_size,
                                                  class_mode='categorical')

    test_generator = datagen.flow_from_directory(params.data_dir + 'test',
                                                 target_size=(params.input_size, params.input_size),
                                                 batch_size=1,
                                                 class_mode='categorical')

    validation_generator = datagen.flow_from_directory(params.data_dir + 'test',
                                                       target_size=(params.input_size, params.input_size),
                                                       batch_size=params.batch_size,
                                                       class_mode='categorical')

    history = model.fit(train_generator,
                        steps_per_epoch=(int(400 // params.batch_size) + 1),
                        epochs=params.ft_epochs,
                        validation_data=validation_generator,
                        validation_steps=(int(len(validation_generator) // params.batch_size) + 1), callbacks=[])

    result = model.evaluate(test_generator)
    print(result)
    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(params.output_dir + 'accuracy.jpg')
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(params.output_dir + 'loss.jpg')


if __name__ == '__main__':
    prepare_gpu()
    args = parse_args()
    main(args)
