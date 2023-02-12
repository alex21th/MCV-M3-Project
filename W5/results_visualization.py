from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import wandb
from tensorflow.keras.utils import plot_model
from PIL import Image

from W5.src.dataloader import get_test_dataloader
from W5.src.models import get_model
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.optimizers import Adam


def visualize_wrong_predictions(image_filenames, labels, predictions, samples_per_class, mapping=None):
    print(f'Number of samples: {len(predictions)}')
    print(f'Number of wrongly classified samples: {sum(predictions != labels)}')

    # get unique classses
    classes = np.unique(np.array(labels))
    num_classes = len(classes)
    # set size for plot
    plt.figure(figsize=(20, 8))

    def get_index_fp(idxs):
        items = []
        for idx in idxs:
            if predictions[idx] != labels[idx]:
                items.append(idx)
        return items

    for y, cls in enumerate(classes):
        idxs_all = np.flatnonzero(np.array(labels) == cls)
        idxs_fp = get_index_fp(idxs_all)
        idxs = np.random.choice(idxs_fp, samples_per_class, replace=True)
        class_accuracy = 100 * (1 - (len(idxs_fp) / len(idxs_all)))
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(Image.open(image_filenames[idx]))
            plt.axis('off')
            label = mapping[labels[idx]]
            preds = mapping[predictions[idx]]
            if i == 0:
                plt.title(
                    f'{cls} \n ClassAccuracy: {class_accuracy:.2f}  \n GT: {label},\n Pred: {preds}')
            else:
                plt.title(f'GT: {label},\n Pred: {preds}')
    plt.show()


def visualize_correct_predictions(image_filenames, labels, predictions, samples_per_class=3, mapping=None):
    print(f'Number of samples: {len(predictions)}')
    print(f'Number of correctly classified samples: {sum(predictions == labels)}')

    # get unique classses
    classes = np.unique(np.array(labels))
    num_classes = len(classes)
    # set size for plot
    plt.figure(figsize=(20, 8))

    def get_index_tp(idxs):
        items = []
        for idx in idxs:
            if predictions[idx] == labels[idx]:
                items.append(idx)
        return items

    for y, cls in enumerate(classes):
        idxs_all = np.flatnonzero(np.array(labels) == cls)
        idxs_tp = get_index_tp(idxs_all)
        idxs = np.random.choice(idxs_tp, samples_per_class, replace=False)
        class_accuracy = 100 * (len(idxs_tp) / len(idxs_all))
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(Image.open(image_filenames[idx]))
            plt.axis('off')
            label = mapping[labels[idx]]
            preds = mapping[predictions[idx]]
            if i == 0:
                plt.title(
                    f'{cls} \n ClassAccuracy: {class_accuracy:.2f}  \n GT: {label},\n Pred: {preds}')
            else:
                plt.title(f'GT: {label},\n Pred: {preds}')
    plt.show()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-m", "--model_name", type=str, default="mobilenet_v3_mixconvs")
    parser.add_argument(
        "-w", "--weights", type=str, default="best_tiny.h5")
    parser.add_argument(
        "-d", "--data_dir", type=str, default="../data/MIT_small_train_1")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    model = get_model(model_name=args.model_name, pops=0)
    test_dataloader = get_test_dataloader(directory=args.data_dir, patch_size=256)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001),
                  metrics=['accuracy', TopKCategoricalAccuracy(name='top_2_accuracy', k=2)])
    plot_model(model, to_file='./' + 'tiny.png', show_shapes=True, show_layer_names=True)
    model.load_weights(args.weights)
    result = model.evaluate(test_dataloader)
    print(result)

    test_predictions = model.predict(test_dataloader)
    top_pred_ids = test_predictions.argmax(axis=1)


    def invert_dict(d):
        return {v: k for k, v in d.items()}
    invereted_map = invert_dict(test_dataloader.class_indices)
    labels = test_dataloader.labels
    # visualize
    visualize_wrong_predictions(test_dataloader.filepaths, labels, top_pred_ids, 3, invereted_map)
    visualize_correct_predictions(test_dataloader.filepaths, labels, top_pred_ids, 3, invereted_map)
