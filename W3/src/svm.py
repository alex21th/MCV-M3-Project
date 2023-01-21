from src.mlp import get_mlp
from src.utils import *
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn import svm
import pickle as pkl

def get_svm(svm_model, layer, IMG_SIZE = 64):
    """
    This function returns the accuracy of intermediate feature into classification with an SVM

    :param svm_model: The MLP model used
    :param layer: String with the name of the layer
    :param IMG_SIZE: Integer resize the images to IMG_SIZE
    :return: Accyracy score
    """
    train_images_filenames = pkl.load(open('MIT_split/train_images_filenames.dat','rb'))
    test_images_filenames = pkl.load(open('MIT_split/test_images_filenames.dat','rb'))
    train_images_filenames = [n[3:] for n in train_images_filenames] #comment if it doesn't work. result -> MIT_split instead of '../MIT_split'
    test_images_filenames  = [n[3:] for n in test_images_filenames]
    train_labels = pkl.load(open('MIT_split/train_labels.dat','rb'))
    test_labels = pkl.load(open('MIT_split/test_labels.dat','rb'))
    
    train_features = []
    model_layer = Model(inputs=svm_model.input, outputs=svm_model.get_layer(layer).output)

    for i in range(len(train_images_filenames)):
        filename = train_images_filenames[i]
        x = np.asarray(Image.open(filename))
        x = np.expand_dims(np.resize(x, (IMG_SIZE, IMG_SIZE, 3)), axis=0)
        x_features = model_layer.predict(x/255.0)
        train_features.append(np.asarray(x_features).reshape(-1))

    train_features = np.asarray(train_features)
    test_features = []

    for i in range(len(test_images_filenames)):
        filename = test_images_filenames[i]
        x = np.asarray(Image.open(filename))
        x = np.expand_dims(np.resize(x, (IMG_SIZE, IMG_SIZE, 3)), axis=0)
        x_features = model_layer.predict(x/255.0)
        test_features.append(np.asarray(x_features).reshape(-1))

    test_features = np.asarray(test_features)

    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)

    parameters = {'kernel': ('rbf', 'linear', 'sigmoid')}
    grid = GridSearchCV(svm.SVC(), parameters, n_jobs=3, cv=8)
    grid.fit(train_features, train_labels)

    best_kernel = grid.best_params_['kernel']

    classifier = svm.SVC(kernel=best_kernel)
    classifier.fit(train_features,train_labels)

    accuracy = classifier.score(test_features, test_labels)

    #compute_roc(train_features, test_features,train_labels,test_labels, classifier,RESULTS+'ROC_svm.png')

    print('Test accuracy: ', accuracy)
    return accuracy