from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import DirectoryIterator

from src.preprocessing import preprocess_input

from typing import Dict


def get_train_dataloader(directory: str, patch_size: int, batch_size: int, data_augmentation: Dict) -> DirectoryIterator:
    """
    Returns a dataloader for the training set
    :param directory: path directory of the dataset
    :param patch_size: resize the images to patch_size x patch_size
    :param batch_size: batch size
    :param data_augmentation: config dictionary for data augmentation parameters
    :return:
    """
    use_augmentation = data_augmentation["use"]
    if use_augmentation:

        rotation_range = data_augmentation["rotation_range"]
        width_shift_range = data_augmentation["width_shift_range"]
        height_shift_range = data_augmentation["height_shift_range"]
        shear_range = data_augmentation["shear_range"]
        zoom_range = data_augmentation["zoom_range"]
        channel_shift_range = data_augmentation["channel_shift_range"]

        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            channel_shift_range=channel_shift_range,
            fill_mode='reflect',
            horizontal_flip=True,
        )

    else:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            horizontal_flip=True
        )

    train_generator = train_datagen.flow_from_directory(
        directory + '/train',  # this is the target directory
        target_size=(patch_size, patch_size),  # all images will be resized to path_size x path_size
        batch_size=batch_size,
        classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels
    return train_generator


def get_val_dataloader(directory: str, patch_size: int, batch_size: int) -> DirectoryIterator:
    """
    Returns a dataloader for the validation set
    :param directory: path directory of the dataset
    :param patch_size: resize the images to patch_size x patch_size
    :param batch_size: batch size
    :return: ImageDataGenerator
    """
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    validation_generator = test_datagen.flow_from_directory(
        directory + '/test',  # this is the target directory
        target_size=(patch_size, patch_size),  # all images will be resized to PATCH_SIZExPATCH_SIZE
        batch_size=batch_size,
        classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels
    return validation_generator

