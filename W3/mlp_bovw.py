from __future__ import print_function

from argparse import ArgumentParser

from src.utils import *
from keras.models import Model, load_model

import pickle
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import GridSearchCV


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="../MIT_split/")  # static
    parser.add_argument("-m", "--model_name", type=str, default='mlp_baseline')
    parser.add_argument("-mp", "--best_model_path", type=str, default="results/mlp_baseline-16-8-0.001-sgd/weights.h5")
    parser.add_argument("-in", "--input_size", type=int, default=16)
    parser.add_argument("-out", "--output_dir", type=str, default="results/bovw/")
    parser.add_argument("-ly", "--layer", type=str, default='second')
    return parser.parse_args()


def get_visual_words(descriptors, codebook, codebook_size):
    visual_words = np.empty((len(descriptors), codebook_size), dtype=np.float32)
    for i, des in enumerate(descriptors):
        words = codebook.predict(des)
        visual_words[i, :] = np.bincount(words, minlength=codebook_size)

    return StandardScaler().fit_transform(visual_words)


def main(argum):
    model_filename = argum.output_dir + 'model_mlp_patches.h5'
    os.makedirs(argum.output_dir, exist_ok=True)

    train_images_filenames = pickle.load(open(argum.data_dir + 'train_images_filenames.dat', 'rb'))
    test_images_filenames = pickle.load(open(argum.data_dir + 'test_images_filenames.dat', 'rb'))
    train_labels = pickle.load(open(argum.data_dir + 'train_labels.dat', 'rb'))
    test_labels = pickle.load(open(argum.data_dir + 'test_labels.dat', 'rb'))

    model = load_model(model_filename)
    model = Model(inputs=model.input, outputs=model.layers[-2].output)
    model.summary()

    patch_size = model.layers[0].input.shape[1:3]
    num_patches = (256 // patch_size.as_list()[0]) ** 2

    def get_descriptors(mdel, images_filenames: str):
        descriptors = np.empty((len(images_filenames), num_patches, model.layers[-1].output_shape[1]))
        for i, filename in enumerate(tqdm(images_filenames)):
            img = Image.open(filename)
            patches = image.extract_patches_2d(np.array(img), patch_size, max_patches=num_patches)
            descriptors[i, :, :] = mdel.predict(patches / 255.)
        return descriptors

    codebook_size = 512  # 760

    train_descriptors = get_descriptors(model, train_images_filenames)

    print("Performing KMeans...")
    codebook = MiniBatchKMeans(n_clusters=codebook_size,
                               verbose=False,
                               batch_size=codebook_size * 20,
                               compute_labels=False,
                               reassignment_ratio=10 ** -4,
                               random_state=42)

    codebook.fit(np.vstack(train_descriptors))

    train_visual_words = get_visual_words(train_descriptors, codebook, codebook_size)

    # gridsearch SVM
    parameters = {'kernel': ('rbf', 'linear', 'sigmoid')}
    grid = GridSearchCV(svm.SVC(), parameters, n_jobs=3, cv=8, verbose=1)
    grid.fit(train_visual_words, train_labels)

    best_kernel = grid.best_params_['kernel']
    print(f'Best SVM kernel: {best_kernel}')

    # test
    test_descriptors = get_descriptors(model, test_images_filenames)
    test_visual_words = get_visual_words(test_descriptors, codebook, codebook_size)

    classifier = svm.SVC(kernel=best_kernel)
    classifier.fit(train_visual_words, train_labels)

    compute_roc(train_visual_words, test_visual_words, train_labels, test_labels, classifier,
                argum.output_dir + 'ROC_bow.png')

    accuracy = classifier.score(test_visual_words, test_labels)

    print(f'Test accuracy: {accuracy}')

    save_confusion_matrix(test_labels, classifier.predict(test_visual_words),
                          argum.output_dir + 'confusion_matrix_box.png')


if __name__ == "__main__":
    prepare_gpu()
    args = parse_args()
    main(args)
