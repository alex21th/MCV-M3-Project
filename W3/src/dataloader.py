import os
#from utils import *
from keras.models import Sequential
from keras.layers import Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator




def get_train_dataloader(directory, PATCH_SIZE, BATCH_SIZE):
    train_datagen = ImageDataGenerator(
          rescale=1./255,
          horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
          directory+'/train',  # this is the target directory
          target_size=(PATCH_SIZE, PATCH_SIZE),  # all images will be resized to PATCH_SIZExPATCH_SIZE
          batch_size=BATCH_SIZE,
          classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
          class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels
    return train_generator
    



def get_val_dataloader(directory, PATCH_SIZE, BATCH_SIZE):
    train_datagen = ImageDataGenerator(
          rescale=1./255,
          horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
          directory+'/test',  # this is the target directory
          target_size=(PATCH_SIZE, PATCH_SIZE),  # all images will be resized to PATCH_SIZExPATCH_SIZE
          batch_size=BATCH_SIZE,
          classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
          class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels
    return train_generator