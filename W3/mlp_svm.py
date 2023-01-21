from argparse import ArgumentParser

import pandas as pd
from keras import Model
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import plot_model
import pickle
import numpy as np
from tqdm import tqdm
import os

from W3.src.mlp import get_mlp
from W3.src.utils import prepare_gpu, compute_roc
from W3.src.utils import load_images


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="../MIT_split/")  # static
    parser.add_argument("-m", "--model_name", type=str, default='mlp_baseline')
    parser.add_argument("-mp", "--best_model_path", type=str, default="results/mlp_baseline-16-8-0.001-sgd/weights.h5")
    parser.add_argument("-in", "--input_size", type=int, default=16)
    parser.add_argument("-out", "--output_dir", type=str, default="results/")
    parser.add_argument("-ly", "--layer", type=str, default='second')
    return parser.parse_args()


def main():
    input_size = args.input_size
    model_name = args.model_name
    best_model_path = args.best_model_path
    output_dir = args.output_dir
    data_dir = args.data_dir
    layer_name = args.layer

    train_images_filenames = pickle.load(open(data_dir + 'train_images_filenames.dat', 'rb'))
    test_images_filenames = pickle.load(open(data_dir + 'test_images_filenames.dat', 'rb'))
    train_labels = pickle.load(open(data_dir + 'train_labels.dat', 'rb'))
    test_labels = pickle.load(open(data_dir + 'test_labels.dat', 'rb'))

    # Load images
    train_images = load_images(train_images_filenames, desc='Loading TRAIN images...')
    test_images = load_images(test_images_filenames, desc='Loading TEST images...')

    model = get_mlp(
        model_name=model_name,
        input_shape=(input_size, input_size, 3),
        output_shape=8
    )

    print(model.summary())
    plot_model(model, to_file=output_dir + 'svm_model.png', show_shapes=True, show_layer_names=True)

    model.load_weights(best_model_path)

    train_features = []
    model_layer = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    for x in tqdm(train_images):
        x = np.expand_dims(np.resize(x, (input_size, input_size, 3)), axis=0)
        x_features = model_layer.predict(x / 255.0)
        train_features.append(np.asarray(x_features).reshape(-1))

    train_features = np.asarray(train_features)
    test_features = []

    for x in tqdm(test_images):
        x = np.expand_dims(np.resize(x, (input_size, input_size, 3)), axis=0)
        x_features = model_layer.predict(x / 255.0)
        test_features.append(np.asarray(x_features).reshape(-1))

    test_features = np.asarray(test_features)

    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)

    parameters = {'kernel': ('rbf', 'linear', 'sigmoid')}
    grid = GridSearchCV(svm.SVC(), parameters, n_jobs=3, cv=8, verbose=1)
    grid.fit(train_features, train_labels)

    best_kernel = grid.best_params_['kernel']

    classifier = svm.SVC(kernel=best_kernel)
    classifier.fit(train_features, train_labels)

    accuracy = classifier.score(test_features, test_labels)

    compute_roc(train_features, test_features, train_labels, test_labels, classifier, output_dir + 'ROC_svm.png')

    print('Test accuracy: ', accuracy)

    svm_kernel_pd = pd.DataFrame.from_dict(grid.cv_results_)

    df_svm = svm_kernel_pd[['param_kernel', 'mean_test_score', 'std_test_score', 'rank_test_score']]
    df_svm.rename(
        columns={'param_kernel': 'Kernel', 'mean_test_score': 'Mean Accuracy', 'std_test_score': 'Std Accuracy',
                 'rank_test_score': 'Rank best accuracy'}, inplace=True)
    df_svm.to_csv(os.path.join(args.output_dir, 'svm_gridsearch.csv'))


if __name__ == "__main__":
    prepare_gpu()
    args = parse_args()
    main()
