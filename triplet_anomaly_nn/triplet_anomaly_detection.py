import numpy as np
import random
import os
from itertools import permutations
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.utils.random import sample_without_replacement
from keras.layers import Conv2D, MaxPooling2D,  Input, Cropping2D, Add, concatenate
from keras.layers import  Dense
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
import glob
import cv2
import itertools
import matplotlib.pyplot as plt

from tensorflow.keras.activations import softmax

INPUT_DIM = 64
MAX_INT = np.iinfo(np.int32).max

def euclidean_distance_loss(vec1, vec2):
    """
    Euclidean distance loss
    :param y_true: TensorFlow tensor
    :param y_pred: TensorFlow tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(vec1 - vec2), axis=-1))



def residual(n_filters, input):
    """
    residual block for convolutional embedder
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
    the embedding model, encoder for images
    """
    input_shape = (input_dim, input_dim, 1)
    inputs = Input(shape=input_shape)
    x = Conv2D(h, (7, 7), input_shape=input_shape, padding='valid', activation="elu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = residual(h, x)
    x = residual(h, x)
    x = residual(h, x)
    embeddings = Conv2D(h, (1, 1), activation='linear')(x)
    embeddings = Dense(n_embeddings, name='embeddings')(embeddings)
    model = Model(inputs=inputs, outputs=embeddings)
    print(model.summary())
    return model



def triplet_loss(y_true, y_pred, alpha=0.4):
    """
    Implementation of the triplet loss function. MAE loss
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    print('y_pred.shape = ', y_pred)

    total_lenght = y_pred.shape.as_list()[-1]
    print('total_lenght=',  total_lenght)
    #     total_lenght =12

    anchor = y_pred[:, :, :, 0:int(total_lenght * 1 / 3)]
    positive = y_pred[:, :, :, int(total_lenght * 1 / 3):int(total_lenght * 2 / 3)]
    negative = y_pred[:, :, :, int(total_lenght * 2 / 3):int(total_lenght * 3 / 3)]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)


    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0.0)

    return loss




def tripletModel(input_dim):
    """
    the learning model. There are 3 inputs, anchor, positive and negative
    They are flown through embeddings network, concatenated, and then triplet loss calculaeted, which is based on vector distances
    """

    adam_optim = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)

    anchor_input = Input((input_dim, input_dim, 1,), name='anchor_input')
    positive_input = Input((input_dim, input_dim, 1,), name='positive_input')
    negative_input = Input((input_dim, input_dim, 1,), name='negative_input')
    # Shared embedding layer for positive and negative items
    Shared_DNN = embeddings(input_dim, h=16)

    encoded_anchor = Shared_DNN(anchor_input)
    encoded_positive = Shared_DNN(positive_input)
    encoded_negative = Shared_DNN(negative_input)

    merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')

    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_vector)
    model.compile(loss=triplet_loss, optimizer=adam_optim)


    print(model.summary())
    return model



def read_data(dataset_path):
    """
    in-memory data reader
    """
    classes = ['normal', 'glare']
    im_list_normal = glob.glob(dataset_path + '/normal/*.png')
    im_list_normal.extend(glob.glob(dataset_path + '/normal/*.jpg'))
    im_list_anomaly = glob.glob(dataset_path + '/glare/*.png')
    im_list_anomaly.extend(glob.glob(dataset_path + '/glare/*.jpg'))


    normal_im = []
    anom_im = []

    for im_path in im_list_normal:
        img = cv2.imread(im_path, 0)
        img = cv2.resize(img, (INPUT_DIM, INPUT_DIM))
        img = np.array(img, np.float32)
        img = img / 255.
        img = np.expand_dims(img, 2)
        normal_im.append(img)

    for im_path in im_list_anomaly:
        img = cv2.imread(im_path, 0)
        img = cv2.resize(img, (INPUT_DIM, INPUT_DIM))
        img = np.array(img, np.float32)
        img = img / 255.
        img = np.expand_dims(img, 2)
        anom_im.append(img)

    normal_labels = list(itertools.repeat('normal', len(im_list_normal)))
    anom_labels = list(itertools.repeat('glare', len(im_list_anomaly)))
    normal_im.extend(anom_im)
    normal_labels.extend(anom_labels)
    return normal_im, normal_labels


def image_gen(img_list):
    """
    image generator
    """
    while True:
        img = random.choice(img_list)
        label = os.path.basename(os.path.dirname(img))  # add label function according to the dataset tree
        img = cv2.imread(img, 0)
        img = cv2.resize(img, (INPUT_DIM, INPUT_DIM))
        img = np.array(img, np.float32)
        img = img / 255.
        img = np.expand_dims(img, 2)
        yield img, label

def get_train_test(dataset_path, test_size=0.2):
    '''
    image generator for triplets
    '''
    im_list_normal = glob.glob(dataset_path + '/normal/*.png')
    im_list_normal.extend(glob.glob(dataset_path + '/normal/*.jpg'))
    im_list_anomaly = glob.glob(dataset_path + '/glare/*.png')
    im_list_anomaly.extend(glob.glob(dataset_path + '/glare/*.jpg'))
    random.shuffle(im_list_normal)
    random.shuffle(im_list_anomaly)
    train_normal, test_normal = train_test_split(im_list_normal, test_size=test_size)
    train_anomaly, test_anomaly = train_test_split(im_list_anomaly, test_size=test_size)

    ig_normal_train, ig_normal_test = image_gen(train_normal), image_gen(test_normal)
    ig_anchor_train, ig_anchor_test = image_gen(train_normal), image_gen(test_normal)
    ig_anomaly_train, ig_anomaly_test = image_gen(train_anomaly), image_gen(test_anomaly)
    return ig_normal_train, ig_normal_test, ig_anchor_train, ig_anchor_test, ig_anomaly_train, ig_anomaly_test


def image_batch_generator(img_lists, batchsize, pair_quantity):
    """
    image batch generator - generating triplet pairs
    """
    ig_normal, ig_anchor, ig_anomaly = img_lists

    triplet_pairs = []
    y = 0
    while pair_quantity > 0:
        im_normal, label_norm = next(ig_normal)
        im_anchor, label_anch = next(ig_anchor)
        im_anomaly, label_anomaly = next(ig_anomaly)

        triplet_pairs.append([im_anchor, im_normal, im_anomaly])

        # If we've reached our batchsize, yield the batch and reset
        if len(triplet_pairs) == batchsize:
            pair_quantity = pair_quantity - batchsize
            triplet_pairs = np.array(triplet_pairs)
            yield ([np.array(triplet_pairs)[:,0,:,:,:], np.array(triplet_pairs)[:,1,:,:,:], np.array(triplet_pairs)[:,2,:,:,:]], np.zeros((0, batchsize)))
            # , np.zeros((1, batch_size)
            triplet_pairs = []







def generate_triplet(x, y, testsize=0.3, ap_pairs=10, an_pairs=10):
    """
    x: list of train samples
    y: list of train labels
    testsize: part of test set size
    ap_pairs: quantity of anchor and positive pairs
    an_pairs: quantity of anchor and negative pairs
    """

    data_xy = tuple([x, y])
    data_xy = np.array(data_xy)
    trainsize = 1 - testsize

    triplet_train_pairs = []
    triplet_test_pairs = []
    for data_class in sorted(set(data_xy[1])):
        same_class_idx = np.where((data_xy[1] == data_class))[0]
        diff_class_idx = np.where(data_xy[1] != data_class)[0]
        A_P_pairs = random.sample(list(permutations(same_class_idx, 2)), k=ap_pairs)  # Generating Anchor-Positive pairs
        Neg_idx = random.sample(list(diff_class_idx), k=an_pairs)

        # train
        A_P_len = len(A_P_pairs)
        Neg_len = len(Neg_idx)
        for ap in A_P_pairs[:int(A_P_len * trainsize)]:
            Anchor = data_xy[0][ap[0]]
            Positive = data_xy[0][ap[1]]
            for n in Neg_idx:
                Negative = data_xy[0][n]
                triplet_train_pairs.append([Anchor, Positive, Negative])
                # test
        for ap in A_P_pairs[int(A_P_len * trainsize):]:
            Anchor = data_xy[0][ap[0]]
            Positive = data_xy[0][ap[1]]
            for n in Neg_idx:
                Negative = data_xy[0][n]
                triplet_test_pairs.append([Anchor, Positive, Negative])

    return np.array(triplet_train_pairs), np.array(triplet_test_pairs)

def evaluate_model():
    pass

def train_with_memory_data(dataset_path):
    """
    in-memory network training
    """
    x, y = read_data(dataset_path)
    X_train, X_test = generate_triplet(x, y, ap_pairs=300, an_pairs=150)

    anchor = X_train[:, 0, :]
    positive = X_train[:, 1, :]
    negative = X_train[:, 2, :]
    anchor_test = X_test[:, 0, :]
    positive_test = X_test[:, 1, :]
    negative_test = X_test[:, 2, :]


    model = tripletModel(input_dim=INPUT_DIM)
    y_dummy = np.empty((anchor.shape[0], 300))
    y_dummy2 = np.empty((anchor_test.shape[0], 1))
    history = model.fit([anchor, positive, negative], y=y_dummy, validation_data=([anchor_test, positive_test, negative_test], y_dummy2), batch_size=1, epochs=10)
    plt.figure(figsize=(5, 5))
    plt.plot(history.history['loss'])
    plt.grid()
    plt.title('model loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

    # save model
    PR_PATH = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(PR_PATH, 'model')
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model.save(model_path)

def lesinn(x_train):
    """the outlier scoring method, a bagging ensemble of Sp. See the following reference for detail.
    Pang, Guansong, Kai Ming Ting, and David Albrecht.
    "LeSiNN: Detecting anomalies by identifying least similar nearest neighbours."
    In Data Mining Workshop (ICDMW), 2015 IEEE International Conference on, pp. 623-630. IEEE, 2015.
    """
    rng = np.random.RandomState(42)
    ensemble_size = 50
    subsample_size = 8
    scores = np.zeros([x_train.shape[0], 1])
    # for reproductibility purpose
    seeds = rng.randint(MAX_INT, size = ensemble_size)
    for i in range(0, ensemble_size):
        rs = np.random.RandomState(seeds[i])
#        sid = np.random.choice(x_train.shape[0], subsample_size)
        sid = sample_without_replacement(n_population = x_train.shape[0], n_samples = subsample_size, random_state = rs)
        subsample = x_train[sid]
        kdt = KDTree(subsample, metric='euclidean')
        dists, indices = kdt.query(x_train, k = 1)
        scores += dists
    scores = scores / ensemble_size
    return scores


# def load_model_predict(model_name, X, labels, embedding_size, filename):
#     """load the representation learning model and do the mappings.
#     LeSiNN, the Sp ensemble, is applied to perform outlier scoring
#     in the representation space.
#     """
#     rankModel, representation = tripletModel(X.shape[1], embedding_size=20)
#     rankModel.load_weights(model_name)
#     representation = Model(inputs=rankModel.input[0],
#                            outputs=rankModel.get_layer('hidden_layer').get_output_at(0))
#
#     new_X = representation.predict(X)
#     #    writeRepresentation(new_X, labels, embedding_size, filename + str(embedding_size) + "D_RankOD")
#     scores = lesinn(new_X)
#     rauc = aucPerformance(scores, labels)
#     writeResults(filename, embedding_size, rauc)
#     #    writeOutlierScores(scores, labels, str(embedding_size) + "D_"+filename)
#     return rauc

def train_with_data_generator(dataset_path):
    """
    training function with data generator
    """
    ig_normal_train, ig_normal_test, ig_anchor_train, ig_anchor_test, ig_anomaly_train, ig_anomaly_test = get_train_test(dataset_path, 0.2)
    batch_size = 16

    train_generator = image_batch_generator([ig_normal_train, ig_anchor_train, ig_anomaly_train], batch_size, pair_quantity=10000)
    test_generator = image_batch_generator([ig_normal_test, ig_anchor_test, ig_anomaly_test], batch_size, pair_quantity=20000)


    model = tripletModel(input_dim=INPUT_DIM)
    # y_dummy_train = np.empty((500, 1))
    # y_dummy_test = np.empty((300, 1))

    history = model.fit_generator(generator=train_generator, validation_data=test_generator,  epochs=25)
    plt.figure(figsize=(5, 5))
    plt.plot(history.history['loss'])
    plt.grid()
    plt.title('model loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

    # save model
    PR_PATH = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(PR_PATH, 'model_gen')
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model.save(model_path)


if __name__ == '__main__':
    dataset_path = ""
    train_with_data_generator(dataset_path)





