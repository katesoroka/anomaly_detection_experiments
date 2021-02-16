"""
This file consists of utility functions, embedding model and function for model testing
"""

import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, Input,  Cropping2D, Add
from keras.layers import Flatten, Dense
from keras.models import Model
import glob
import cv2
import itertools
from utils import silent

from pyod.models.xgbod import XGBOD
from pyod.models.ocsvm import OCSVM
from pyod.models.cof import COF
from pyod.utils.data import evaluate_print



INPUT_DIM = 64



def preprocess_image(img):
    """
    Helper function for image preprocessing with such steps:
     - reading from disk
     - resizing
     - normalization
     - adding dimension for tensor-like shape
    """
    img = cv2.imread(img, 0)
    img = cv2.resize(img, (INPUT_DIM, INPUT_DIM))
    img = np.array(img, np.float32)
    img = img / 255.
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 3)
    return img


def image_generator(img_list):
    """
    Image generator function from image list. Yields image and label. Label is taken from containing folder name
    """
    while True:
        img = random.choice(img_list)
        label = os.path.basename(os.path.dirname(img))  # add label function according to the dataset tree
        img = preprocess_image(img)
        yield img, label

def image_batch_generator(img_paths, model, batch_size, features=True):
    """
    Generates batches of images from image_generator. Yields batches of images, image embeddings and class labels
    """
    while True:
        ig = image_generator(img_paths)
        batch_img, batch_features, batch_labels = [], [], []

        for img, lab in ig:
            # Add the image and mask to the batch
            if features:
                img = np.expand_dims(img, 0)
                img_embedding = model.predict(img)
                batch_features.append(img_embedding)
            batch_img.append(img)
            batch_labels.append(lab)
            # If we've reached our batchsize, yield the batch and reset
            if len(batch_img) == batch_size:
                yield batch_img, batch_features, batch_labels
                batch_img, batch_features, batch_labels = [], [], []

        # If we have an nonempty batch left, yield it out and reset
        if len(batch_img) != 0:
            yield np.stack(batch_img, axis=1), np.array(batch_features), batch_labels
            batch_img, batch_features, batch_labels = [], [], []

def residual(n_filters, input):
    """
    residual block for convolutional encoder
    """
    shape = input.shape
    _, h, w, d = shape
    l1 = Conv2D(n_filters, (5, 5), padding='valid', activation='elu')(input)
    l2 = Conv2D(n_filters, (1, 1), padding='valid', activation='linear')(l1)
    l3 = Cropping2D(cropping=2)(input)
    added = Add()([l2, l3])
    return added


def embeddings(input_dim, h=16, n_embeddings=64):
    """
    Convolutional residual model for embeddings creations. Idea of this model is taken from this paper:
    https://www.sciencedirect.com/science/article/pii/S2212827119302409
    """
    input_shape = (input_dim, input_dim, 1)
    inputs = Input(shape=input_shape)
    x = Conv2D(h, (7, 7), input_shape=input_shape, padding='valid', activation="elu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = residual(h, x)
    x = residual(h, x)
    x = residual(h, x)
    x = Conv2D(h, (1, 1), activation='linear')(x)
    x = Flatten()(x)
    embeddings = Dense(n_embeddings, name='embeddings')(x)
    model = Model(inputs=inputs, outputs=embeddings)
    print(model.summary())
    return model


def get_train_test_lists(dataset_path, classes=('glare_small', 'normal'), test_size=0.25):
    """
    Function for gathering lists of image paths on the disk and splitting them to train and test data
    classes - list of classes that should be gathered in dataset_path. They are names of folders, and then names of folders are considered as labels
    Labels are then encoded for using pyod module, as this: anomaly class == 1, common class == 0
    """
    image_set = []
    label_set = []
    for cls in classes:
        dir = os.path.join(dataset_path, cls)
        img_list = glob.glob(dir + '/*.png')
        img_list.extend(glob.glob(dir + '/*.jpg'))
        label = None
        if cls == 'glare_small' or cls == 'glare':
            label = 1
        if cls == 'normal':
            label = 0

        labels = list(itertools.repeat(label, len(img_list)))
        image_set.extend(img_list)
        label_set.extend(labels)
    X_train, X_test, y_train, y_test = train_test_split(image_set, label_set, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def generate_embeddings_gen(dataset_path, classes):
    """
    Function for creating train and test generators with image embeddigns
    """
    model = embeddings(INPUT_DIM)
    X_train, X_test, y_train, y_test = get_train_test_lists(dataset_path, classes)
    # create data generators
    batch_size = 16
    train_batch_generator = image_batch_generator(X_train, model, batch_size=batch_size)
    test_batch_generator = image_batch_generator(X_test, model, batch_size=batch_size)

    return train_batch_generator, test_batch_generator

def generate_embeddings_memory(dataset_path, classes):
    """
    function for creating embeddings in-memory
    """
    # get embeddings not in generators
    model = embeddings(INPUT_DIM)
    X_train, X_test, y_train, y_test = get_train_test_lists(dataset_path, classes)
    X_train_em = []
    X_test_em = []

    for im in X_train:
        img = preprocess_image(im)
        embeds = model.predict(img)
        embeds = np.squeeze(embeds)
        X_train_em.append(embeds)
    for im in X_test:
        img = preprocess_image(im)
        embeds = model.predict(img)
        embeds = np.squeeze(embeds)
        X_test_em.append(embeds)
    return np.array(X_train_em), np.array(X_test_em), np.array(y_train), np.array(y_test)

def test_gens():
    """
    test of generate_embeddings_gen fucntions
    """
    dataset_path = "/home/kateryna/Documents"
    train_gen, test_gen = generate_embeddings_gen(dataset_path)
    img, feature, labels = next(train_gen)
    print(len(img), len(feature), labels)

def read_images(img_path, labels_list, test_size=0.25):
    '''
    Function for reading images in-memory
    :param img_path: img path to folders with  images
    :return: array of images and labels
    '''
    image_set = []
    label_set = []
    images = []
    for cls in labels_list:
        dir = os.path.join(img_path, cls)
        img_list = glob.glob(dir + '/*.png')
        img_list.extend(glob.glob(dir + '/*.jpg'))
        label = None
        if cls == 'glare_small' or cls == 'glare':
            label = 1
        if cls == 'normal':
            label = 0
        labels = list(itertools.repeat(label, len(img_list)))
        image_set.extend(img_list)
        label_set.extend(labels)
    X_train, X_test, y_train, y_test = train_test_split(image_set, label_set, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score
from pyod.models.iforest import IForest
from pyod.models.copod import COPOD
from pyod.models.cblof import CBLOF
from pyod.models.vae import VAE


def test_autoencoder():
    """
    function for testing VAE autoencoder module
    """
    dataset_path = "/home/kateryna/Documents"
    X_train, X_test, y_train, y_test = read_images(dataset_path, labels_list=['normal', 'glare_small'], test_size=0.25)
    X_train_im = []
    for im in X_train:
        img = preprocess_image(im)
        img = np.array(img)
        img = img.flatten()
        X_train_im.append(img)
    X_train_im = np.array(X_train_im)

    X_test_im = []
    for im in X_test:
        img = preprocess_image(im)
        img = np.array(img)
        img = img.flatten()
        X_test_im.append(img)
    X_test_im = np.array(X_test_im)

    autoenc = VAE(encoder_neurons=[16, 32], decoder_neurons=[32, 16], latent_dim=32, epochs=50)
    autoenc.fit(X_train_im, y_train)
    y_pred = autoenc.predict(X_test_im)
    y_test_scores = autoenc.decision_function(X_test_im)
    conf_mtx_test = confusion_matrix(y_test, y_pred, labels=[0, 1])
    evaluate_print('vae', y_test, y_test_scores)
    print(conf_mtx_test)




def classic_model_testing():
    """
    function for classic models' testing, that take embeddigns as inputs
    """
    dataset_path = "/home/kateryna/Documents"
    X_train, X_test, y_train, y_test = generate_embeddings_memory(dataset_path, classes=['normal', 'glare_small'])
    contam = 0.08
    models = [XGBOD(), OCSVM(contamination=contam), IForest(contamination=contam, n_estimators=150), XGBOD(learning_rate=0.01, n_estimators=150),
              COPOD(contamination=contam)]
    for model in models:
        model_name = model.__str__().split('(')[0]
        clf = model
        clf.fit(X_train, y_train)

        y_train_pred = clf.labels_
        y_train_scores = clf.decision_scores_

        # get the prediction on the test data
        # 0 stands for inliers and 1 for outliers.
        y_test_pred = clf.predict(X_test)
        y_test_scores = clf.decision_function(X_test)
        # y_probabilities = clf.predict_proba(X_test)
        print("\nOn Training Data:")
        evaluate_print(model_name, y_train, y_train_scores)
        print("\nOn Test Data:")
        evaluate_print(model_name, y_test, y_test_scores)
        print('roc auc', roc_auc_score(y_test, y_test_scores))

        conf_mtx_test = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
        print(conf_mtx_test)
        conf_mtx_train = confusion_matrix(y_train, y_train_pred, labels=[0, 1])
        print(conf_mtx_train)
        print('~~~')

from sklearn.model_selection import ParameterGrid

def param_grid_search():
    # for model = XGBOD()
    param_dict = {'learning_rate': [0.01, 0.1, 0.001], 'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 10]}
    grid = ParameterGrid(param_dict)
    dataset_path = "/home/kateryna/Documents"
    X_train, X_test, y_train, y_test = generate_embeddings_memory(dataset_path, classes=['normal', 'glare_small'])

    max_precision = 0
    max_recall = 0
    max_precision_id = None
    max_recall_id = None

    for i, params in enumerate(grid):
        clf = XGBOD(**params)
        clf.fit(X_train, y_train)

        # get the prediction on the test data
        # 0 stands for inliers and 1 for outliers.
        y_test_pred = clf.predict(X_test)
        # y_test_scores = clf.decision_function(X_test)
        # precision and recall regrading positive class
        precision = precision_score(y_test, y_test_pred, pos_label=0)
        recall = recall_score(y_test, y_test_pred, pos_label=0)
        if max_precision < precision:
            max_precision = precision
            max_precision_id = i
        if max_recall < recall:
            max_recall = recall
            max_recall_id = i
    print('max recall', max_recall)
    print('max precision', max_precision)
    best_params_precision = grid[max_precision_id]
    best_params_recall = grid[max_recall_id]
    print('best parameters set for precision', best_params_precision)
    print('best parameters set for recall', best_params_recall)




if __name__=='__main__':
    # classic_model_testing()
    # test_autoencoder()
    param_grid_search()




