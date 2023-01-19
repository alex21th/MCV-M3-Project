from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import DirectoryIterator


def get_train_dataloader(directory: str, patch_size: int, batch_size: int) -> DirectoryIterator:
    """
    Returns a dataloader for the training set
    :param directory: path directory of the dataset
    :param patch_size: resize the images to patch_size x patch_size
    :param batch_size: batch size
    :return:
    """
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
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
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        directory + '/test',  # this is the target directory
        target_size=(patch_size, patch_size),  # all images will be resized to PATCH_SIZExPATCH_SIZE
        batch_size=batch_size,
        classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels
    return train_generator
