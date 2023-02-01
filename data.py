# Loads and Processes the data that will be used in QCNN and Hierarchical Classifier Training
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
pca32 = ['pca32-1', 'pca32-2', 'pca32-3', 'pca32-4']
autoencoder32 = ['autoencoder32-1', 'autoencoder32-2', 'autoencoder32-3', 'autoencoder32-4']
pca30 = ['pca30-1', 'pca30-2', 'pca30-3', 'pca30-4']
autoencoder30 = ['autoencoder30-1', 'autoencoder30-2', 'autoencoder30-3', 'autoencoder30-4']
pca16 = ['pca16-1', 'pca16-2', 'pca16-3', 'pca16-4', 'pca16-compact']
autoencoder16 = ['autoencoder16-1', 'autoencoder16-2', 'autoencoder16-3', 'autoencoder16-4', 'autoencoder16-compact']
pca12 = ['pca12-1', 'pca12-2', 'pca12-3', 'pca12-4']
autoencoder12 = ['autoencoder12-1', 'autoencoder12-2', 'autoencoder12-3', 'autoencoder12-4']

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
    #x_train, x_test = x_train / 255.0, x_test / 255.0  # normalize the data

    
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
        
    #X_train_shape = np.array(np.shape(X_train))
    #X_test_shape = np.array(np.shape(X_test))
    
    
    if feature_reduction == 'resize256':
        if dataset=="cifar10":
            X_train, X_test = tf.squeeze(X_train).numpy(), tf.squeeze(X_test).numpy()
            X_train = tf.image.rgb_to_grayscale(X_train[:])
            X_test = tf.image.rgb_to_grayscale(X_test[:])
            #X_train, X_test = tf.squeeze(X_train).numpy(), tf.squeeze(X_test).numpy()
            
        #X_train = tf.image.resize(X_train[:], (256, 1)).numpy()
        #X_test = tf.image.resize(X_test[:], (256, 1)).numpy()
        X_train = tf.image.resize(X_train[:], (16, 16)).numpy()
        X_test = tf.image.resize(X_test[:], (16, 16)).numpy()
        X_train = tf.reshape(X_train[:], (np.shape(X_train)[0], 256, 1))
        X_test = tf.reshape(X_test[:], (np.shape(X_test)[0], 256, 1))
        X_train, X_test = tf.squeeze(X_train).numpy(), tf.squeeze(X_test).numpy()
        
    return X_train, X_test, Y_train, Y_test
        
    
    #X_train_final = []
    #X_test_final = []
   # for i in range(0, X_train_shape[0]):
        ##image1=np.c_[np.zeros(28), np.zeros(28), X_train[i], np.zeros(28), np.zeros(28)]
        #image2=np.r_[ [np.zeros(32)], [np.zeros(32)], image1, [np.zeros(32)], [np.zeros(32)]]
        #if feature_reduction == "resize1024":
            #image2 = np.reshape(image2, (1024, 1))
        #X_train_final.append(image2)
        #if feature_reduction== "resize256":
            #image2 = tf.image.resize(X_train[i], (16, 16))
    #X_Train_final = np.squeeze(X_train_final)
        
    #for j in range(0, X_test_shape[0]):
        #Image1=np.c_[np.zeros(28), np.zeros(28), X_test[j], np.zeros(28), np.zeros(28)]
        #Image2=np.r_[ [np.zeros(32)], [np.zeros(32)], Image1, [np.zeros(32)], [np.zeros(32)]]
        #if feature_reduction == "resize1024":
            #Image2 = np.reshape(Image2, (1024, 1))
        #X_test_final.append(Image2)
    #X_Test_final = np.squeeze(X_test_final)

     
    #return X_Train_final, X_Test_final, Y_train, Y_test
     
    
