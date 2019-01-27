"""Dummy classification system.

Skeleton code for a assignment solution.

To make a working solution you will need to rewrite parts
of the code below. In particular, the functions
reduce_dimensions and classify_page currently have
dummy implementations that do not do anything useful.

version: v1.0
"""
import math
import numpy as np
import utils.utils as utils
import operator
import scipy.linalg
import scipy.misc
from scipy import stats
import copy
from collections import Counter
import utils.utils as utils


pca_size = 20

def reduce_dimensions(feature_vectors_full, model):
    """Reduce dimensions using feature selection to the 10 best features

    Params:
    feature_vectors_full - feature vectors stored as rows in a matrix
    model - a dictionary storing the outputs of the model training stage
    """

    """ Initially, the mean was subtracted but this was removed as it gives higher results
     pca_mean = np.mean(feature_vectors_full)
     pcatrain_data = np.dot((feature_vectors_full - pca_mean), model['pca_axes'])
    """
    pcatrain_data = np.dot(feature_vectors_full, model['pca_axes'])
    best_features = model['best_features']
    return np.array(pcatrain_data[:, best_features])

def divergence(class1, class2):
    """
    NOTICE: Code for divergence obtained from lab classes

    compute a vector of 1-D divergences
    class1 - data matrix for class 1, each row is a sample
    class2 - data matrix for class 2
    returns: d12 - a vector of 1-D divergence scores
    """
    # Compute the mean and variance of each feature vector element
    m1 = np.mean(class1, axis=0)
    m2 = np.mean(class2, axis=0)
    v1 = np.var(class1, axis=0)
    v2 = np.var(class2, axis=0)

    # Plug mean and variances into the formula for 1-D divergence.
    # (Note that / and * are being used to compute multiple 1-D
    #  divergences without the need for a loop)
    d12 = 0.5 * (v1/v2 + v2/v1 - 2) + 0.5 * ( m1-m2 ) * (m1-m2) * (1.0/v1+1.0/v2)
    return d12

def multidivergence(class1, class2, features):
    """
    NOTICE: Code for multidivergence obtained from lab classes

    compute divergence between class1 and class2
    class1 - data matrix for class 1, each row is a sample
    class2 - data matrix for class 2
    features - the subset of features to use
    returns: d12 - a scalar divergence score
    """

    ndim=len(features);

    # compute mean vectors
    mu1 = np.mean(class1[:,features], axis=0)
    mu2 = np.mean(class2[:,features], axis=0)

    # compute distance between means
    dmu = mu1 - mu2

    # compute covariance and inverse covariance matrices
    cov1 = np.cov(class1[:,features], rowvar=0)
    cov2 = np.cov(class2[:,features], rowvar=0)
    icov1 = np.linalg.inv(cov1)
    icov2 = np.linalg.inv(cov2)

    # plug everything into the formula for multivariate gaussian divergence
    d12 = (0.5 * np.trace(np.dot(icov1,cov2) + np.dot(icov2,cov1) - 2*np.eye(ndim)) + 0.5 * np.dot(np.dot(dmu,(icov1 +icov2)), dmu))

    return d12

def get_pca_axes(fvectors_train_full):
    """
    Get the principal component axis
    Paramaters:
    fvectors_train_full: feature vectors from train data stored as rows in a matrix
    """
    covx = np.cov(fvectors_train_full, rowvar=0)
    n = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(n - 40, n - 1))
    v = np.fliplr(v)
    return v

def feature_selection(fvectors_train, model):
    """
    Selecting the top 10 features through multidivergence

    Paramaters:
    fvectors_train_full: feature vectors from train data stored in a row matrix
    model: dictionary
    """
    # Getting the training data
    fvectors_train_mean = np.mean(fvectors_train)
    pcatrain_data = np.dot((fvectors_train - fvectors_train_mean), model['pca_axes'])
    labels_train = np.array(model['labels_train'])

    # Getting all possible labels in the training data
    unique_labels = (list(set(labels_train)))
    unique_labels.sort()
    char_range = len(unique_labels)
    pca_range = list(range(0,pca_size))

    # Creating an empty list to add test features & selected features
    total_features = []

    """
    Carefully looping one character against another, making sure it doesn't loop 
    characters which have been paired before again.
    """
    print('Getting multidivergences for train data')
    for firstChar in range(char_range):
        firstChar_sample = labels_train == unique_labels[firstChar]
        for secondChar in range(firstChar + 1, char_range):
            secondChar_sample = labels_train == unique_labels[secondChar]
            if (np.sum(firstChar_sample) > 1) and (np.sum(secondChar_sample) > 1):
                firstChar_data = pcatrain_data[firstChar_sample, :]
                secondChar_data = pcatrain_data[secondChar_sample, :]
                """
                Using divergence to find the best feature
                The value gotten is 1, and returns a very poor result

                d12 = divergence(firstChar_data, secondChar_data)
                first_feature = np.argmax(d12)
                """

                # Best feature obtained using brute force / trial & error
                best_feature = 3
                print(best_feature)
                result_features = [best_feature]
                nfeatures = [(i)
                            for i in pca_range
                            if i not in result_features]

                """
                Finding the 10 best features using multidivergence
                """
                for _ in range(9):
                    combinedFeatures = []
                    multidivergence_list = [] #A list of multidivergences
                    for j in nfeatures:
                        """
                        Copying the selected features from result features,
                        and then adding the test features into the same list
                        """
                        combinedFeatures = copy.deepcopy(result_features)
                        combinedFeatures.append(j)

                        """
                        Getting the new multidivergences between the test features
                        and the selected features, then append them into a new list
                        """
                        multidivergence_list.append(multidivergence(firstChar_data, secondChar_data, combinedFeatures))

                    """
                    Selecting features with the highest multidivergence,
                    Removing those features from the next set of test features
                    to prevent testing the same features over again
                    """
                    top_multidivergence_list = nfeatures[np.argmax(multidivergence_list)]
                    result_features.append(top_multidivergence_list)
                    nfeatures.remove(top_multidivergence_list) # To prevent testing the same feature

                # Append the selected features into the list of total features
                total_features.append(sorted(result_features))

            """
            Putting all the featuers into a 1-D list,
            then getting the best 10 features.

            The best 10 features are the ones that appear the most
            """
            count = Counter(np.ravel(np.array(total_features)))
            common_features = count.most_common(10)
            result_features = [t[0] for t in common_features]
            return np.array(list(result_features))

def get_pca_axes(fvectors_train_full):
    """
    NOTICE: Method obtained from lab classes

    Calculates the principal component axes
    Params:
    fvectors_train_full - feature vectors from train data stored as rows in a matrix
    """
    covx = np.cov(fvectors_train_full, rowvar=0)
    n_orig = covx.shape[0]
    [d,v] = scipy.linalg.eigh(covx, eigvals = (n_orig-pca_size, n_orig-1))
    v = np.fliplr(v)
    return v

def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width

def images_to_feature_vectors(images, bbox_size=None):
    """Reformat characters into feature vectors.
    Takes a list of images stored as 2D-arrays and returns
    a matrix in which each row is a fixed length feature vector
    corresponding to the image.abs
    Params:
    images - a list of images stored as arrays
    bbox_size - an optional fixed bounding box size for each image
    """
    # If no bounding box size is supplied then compute a suitable
    # bounding box by examining sizes of the supplied images.
    if bbox_size is None:
        bbox_size = get_bounding_box_size(images)

    bbox_h, bbox_w = bbox_size
    nfeatures = bbox_h * bbox_w
    fvectors = np.empty((len(images), nfeatures))
    for i, image in enumerate(images):
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        h = min(h, bbox_h)
        w = min(w, bbox_w)
        padded_image[0:h, 0:w] = image[0:h, 0:w]
        fvectors[i, :] = padded_image.reshape(1, nfeatures)
    return fvectors

# The three functions below this point are called by train.py
# and evaluate.py and need to be provided.

def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.
    Params:
    train_page_names - list of training page names
    """
    print('Reading data')
    images_train = []
    labels_train = []
    for page_name in train_page_names:
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)
    labels_train = np.array(labels_train)

    print('Extracting features from training data')
    bbox_size = get_bounding_box_size(images_train)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)

    model_data = dict()
    model_data['labels_train'] = labels_train.tolist()
    model_data['bbox_size'] = bbox_size
    # Storing PCA Axis into a model
    model_data['pca_axes'] = get_pca_axes(fvectors_train_full).tolist()
    # Storing the top 10 features from feature selection into a model
    model_data['best_features'] = feature_selection(fvectors_train_full, model_data).tolist()

    print('Reducing to 10 dimensions')
    model_data['fvectors_train'] = reduce_dimensions(fvectors_train_full, model_data).tolist()
    return model_data

def load_test_page(page_name, model):
    """Load test data page.
    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.
    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    bbox_size = model['bbox_size']
    images_test = utils.load_char_images(page_name)
    fvectors_test = images_to_feature_vectors(images_test, bbox_size)
    # Perform the dimensionality reduction.
    fvectors_test_reduced = reduce_dimensions(fvectors_test, model)
    return fvectors_test_reduced

def classify_page(page, model):
    """
    Using K-Nearest NEighbour to classify the data

    Parameters:
    page - a matrix where each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """

    # Selecting desired features from the train data and test data
    fvectors_train = np.array(model['fvectors_train'])
    labels_train = np.array(model['labels_train'])
    fvectors_test = np.array(page)
    test = np.array(page)

    # Calculations obtained from lab classes
    x = np.dot(page, fvectors_train.transpose())
    modtest = np.sqrt(np.sum(page*page, axis=1))
    modtrain = np.sqrt(np.sum(fvectors_train*fvectors_train, axis=1))
    dist = x/np.outer(modtest, modtrain.transpose())  # cosine distance

    # K-Nearest Neighbour constant
    k = 11

    # Nearest Neighbour Classification
    if k == 1:
        nearest = np.argmax(dist, axis=1) # Gets the maximum cosine distance
        labels = labels_train[nearest]
    else:
        # NOTICE: Implementation of K-Nearest Neighbour obtained from lecture on 3rd Dec 2018
        knearest = np.argsort(-dist, axis = 1)[:, 0:k]
        klabels = labels_train[knearest]
        label, count = stats.mode(klabels, axis = 1)
        label = np.reshape(label, len(test))
    return  label

def correct_errors(page, labels, bboxes, model):
    """Dummy error correction. Returns labels unchanged.
    parameters:

    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage
    """
    return labels

def error_correction(page, labels, bboxes, model):
    word_list = model['word_list']

    # Variables for bboxes word
    bboxes_split = 6
    bboxes_label = bboxes.shape[0]
    firstChar_border = bboxes[1:, 0]
    secondChar_border = bboxes[0:(bboxes_label-1), 2]
    word_gap  = np.absolute(firstChar_border - secondChar_border)

    word = []
    word_list = []
    updated_word_list = []

    """
    Looping through all characters in the data
    Splits the words if the gap between the first
    and second character is more than 6.

    The words are then added into a word list to
    compare with the test data and correct that data.
    """
    for i in range(bboxes_label - 1):
        word = labels[i]
        if (word_gap > bboxes_split):
            word_list.append(word)
        else:
            i += 1
            word_list.append(word)
    return np.array(word_list)