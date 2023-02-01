# Loads and Processes the data that will be used in QCNN and Hierarchical Classifier Training
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses


def data_load_and_process(dataset, classes=[0, 1], feature_reduction="resize1024", binary=True):
    if dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)


    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0  # normalize the data

    
    x_train_filter_01 = np.where((y_train == classes[0]) | (y_train == classes[1]))
    x_test_filter_01 = np.where((y_test == classes[0]) | (y_test == classes[1]))

    X_train, X_test = x_train[x_train_filter_01], x_test[x_test_filter_01]
    Y_train, Y_test = y_train[x_train_filter_01], y_test[x_test_filter_01]

    if binary == False:
        Y_train = [1 if y == classes[0] else 0 for y in Y_train]
        Y_test = [1 if y == classes[0] else 0 for y in Y_test]
    elif binary == True:
        Y_train = [1 if y == classes[0] else -1 for y in Y_train]
        Y_test = [1 if y == classes[0] else -1 for y in Y_test]
        
    
    
    if feature_reduction == 'resize256':
        if dataset=="cifar10":
            X_train, X_test = tf.squeeze(X_train).numpy(), tf.squeeze(X_test).numpy()
            X_train = tf.image.rgb_to_grayscale(X_train[:])
            X_test = tf.image.rgb_to_grayscale(X_test[:])
            
        X_train = tf.image.resize(X_train[:], (16, 16)).numpy()
        X_test = tf.image.resize(X_test[:], (16, 16)).numpy()
        X_train = tf.reshape(X_train[:], (np.shape(X_train)[0], 256, 1))
        X_test = tf.reshape(X_test[:], (np.shape(X_test)[0], 256, 1))
        X_train, X_test = tf.squeeze(X_train).numpy(), tf.squeeze(X_test).numpy()
        
    return X_train, X_test, Y_train, Y_test
        
    
     
    
