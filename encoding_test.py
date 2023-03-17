# Loads and Processes the data that will be used in QCNN and Hierarchical Classifier Training
import h5py
import numpy as np
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding, AngleEmbedding

import medmnist
from medmnist import INFO, Evaluator
import torch.utils.data as data
import torchvision.transforms as transforms

#from keras.utils.io_utils import HDF5Matrix
#from keras.preprocessing.image import ImageDataGenerator




def data_load_and_process(dataset, classes=[0, 1], feature_reduction="resize256", binary=True):
    if dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0  # normalize the data
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0  # normalize the data
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0  # normalize the data
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
    elif dataset == "pneumoniamnist":
        data_flag = "pneumoniamnist"
        download = True
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
        # preprocessing
        data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
        train_dataset = DataClass(split='train', transform=data_transform, download=download)
        test_dataset = DataClass(split='test', transform=data_transform, download=download)
        train_loader = data.DataLoader(dataset=train_dataset, batch_size = len(train_dataset))
        test_loader = data.DataLoader(dataset=test_dataset, batch_size = len(test_dataset))
        for x, y in train_loader:
            x1_train=x.numpy()
            y1_train=y.numpy()
            X_train = x1_train.squeeze()
            Y_train = y1_train.squeeze()
            X_train = X_train[..., np.newaxis]

        for x, y in test_loader:
            x1_test=x.numpy()
            y1_test=y.numpy()
            X_test = x1_test.squeeze()
            Y_test = y1_test.squeeze()
            
    elif dataset == "bloodmnist":
        data_flag = "bloodmnist"
        download = True
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
        # preprocessing
        data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
        train_dataset = DataClass(split='train', transform=data_transform, download=download)
        test_dataset = DataClass(split='test', transform=data_transform, download=download)
        train_loader = data.DataLoader(dataset=train_dataset, batch_size = len(train_dataset))
        test_loader = data.DataLoader(dataset=test_dataset, batch_size = len(test_dataset))
        for x, y in train_loader:
            x1_train=x.numpy()
            y1_train=y.numpy()
            x_train = x1_train.squeeze()
            y_train = y1_train.squeeze()
            #x_train = x_train[..., np.newaxis]
        for x, y in test_loader:
            x1_test=x.numpy()
            y1_test=y.numpy()
            x_test = x1_test.squeeze()
            y_test = y1_test.squeeze()
            #x_test = x_test[..., np.newaxis]
            
    elif dataset == "patchCamelyon":
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        f1 = h5py.File('camelyonpatch_level_2_split_train_x.h5', 'r')
        f2 = h5py.File('camelyonpatch_level_2_split_train_y.h5', 'r')
        f3 = h5py.File('camelyonpatch_level_2_split_test_x.h5', 'r')
        f4 = h5py.File('camelyonpatch_level_2_split_test_y.h5', 'r')
        a_group_key = list(f1.keys())[0]
        b_group_key = list(f2.keys())[0]
        c_group_key = list(f3.keys())[0]
        d_group_key = list(f4.keys())[0]
        data1 = np.array(f1.get(a_group_key))
        data2 = np.array(f2.get(b_group_key))
        data3 = np.array(f3.get(c_group_key))
        data4 = np.array(f4.get(d_group_key))
        for i in range(10):
            x_train.append(tf.image.resize_with_crop_or_pad(np.array(data1[i]), 32, 32))
            y_train.append(np.squeeze(np.asarray(data2[i])))
            
        for i in range(4):
            x_test.append(tf.image.resize_with_crop_or_pad(np.array(data3[i]), 32, 32))
            y_test.append(np.squeeze(np.asarray(data4[i])))
    
#              
        
    return x_train

        

    
    

    
#    x_train_filter_01 = np.where((y_train == classes[0]) | (y_train == classes[1]))
#    x_test_filter_01 = np.where((y_test == classes[0]) | (y_test == classes[1]))

#    X_train, X_test = x_train[x_train_filter_01], x_test[x_test_filter_01]
#    Y_train, Y_test = y_train[x_train_filter_01], y_test[x_test_filter_01]

#    if binary == False:
#        Y_train = [1 if y == classes[0] else 0 for y in Y_train]
#        Y_test = [1 if y == classes[0] else 0 for y in Y_test]
#    elif binary == True:
#        Y_train = [1 if y == classes[0] else -1 for y in Y_train]
#        Y_test = [1 if y == classes[0] else -1 for y in Y_test]
        
        
    #X_train_final = []
    #X_test_final = []
    
#    if feature_reduction == 'resize256':
#        if dataset=="cifar10":
#            #X1_train = np.zeros((10000, 1024))
#            #X1_test = np.zeros((np.array(np.shape(X_test))[0], 1024))
#            X_train = tf.image.resize(X_train[:], (32, 32)).numpy()
#            X_test = tf.image.resize(X_test[:], (32, 32)).numpy()
#            pad1 = np.full((10000, 32, 32, 1), 0.4)
#            pad2 = np.full((np.array(np.shape(X_test))[0], 32, 32, 1), 0.4)
#            X_train = np.append(X_train, pad1, axis=3)
#            X_test = np.append(X_test, pad2, axis=3)
#            X1_train = tf.reshape(X_train[:], (np.shape(X_train)[0], 4096, 1, 1))
#            X1_test = tf.reshape(X_test[:], (np.shape(X_test)[0], 4096, 1, 1))
#            #X_train = tf.image.rgb_to_grayscale(X_train[:])
#            #X_test = tf.image.rgb_to_grayscale(X_test[:])
#            X1_train, X1_test = tf.squeeze(X1_train).numpy(), tf.squeeze(X1_test).numpy()
#            
#        else:   
#            #X_train = tf.image.resize(X_train[:], (256, 1)).numpy()
#	    #X_test = tf.image.resize(X_test[:], (256, 1)).numpy()
#            X_train = tf.image.resize(X_train[:], (32, 32)).numpy()
#            X_test = tf.image.resize(X_test[:], (32, 32)).numpy()
#            X1_train = tf.reshape(X_train[:], (np.shape(X_train)[0], 1024, 1))
#            X1_test = tf.reshape(X_test[:], (np.shape(X_test)[0], 1024, 1))
#            X1_train, X1_test = tf.squeeze(X1_train).numpy(), tf.squeeze(X1_test).numpy()
        
    
        
        #return X_train, X_test, Y_train, Y_test
        
#    X_train_shape = np.array(np.shape(X_train))
#    X_test_shape = np.array(np.shape(X_test))
#    Y_train_shape = np.array(np.shape(Y_train))
#    Y_test_shape = np.array(np.shape(Y_test))
    
    #return X_train, X_test, Y_train, Y_test
        
    #for i in range(0, X_train_shape[0]):
        #image1=np.c_[np.zeros(28), np.zeros(28), X_train[i], np.zeros(28), np.zeros(28)]
        #image2=np.r_[ [np.zeros(32)], [np.zeros(32)], image1, [np.zeros(32)], [np.zeros(32)]]
        #if feature_reduction == "resize1024":
            #image2 = np.reshape(image2, (1024, 1))
        #if feature_reduction== "resize256":
            #image2 = tf.image.resize(image2, (16, 16))
        #X_train_final.append(image2)
    #X_Train_final = np.squeeze(X_train_final)
        
    #for j in range(0, X_test_shape[0]):
        #Image1=np.c_[np.zeros(28), np.zeros(28), X_test[j], np.zeros(28), np.zeros(28)]
        #Image2=np.r_[ [np.zeros(32)], [np.zeros(32)], Image1, [np.zeros(32)], [np.zeros(32)]]
        #if feature_reduction == "resize1024":
            #Image2 = np.reshape(Image2, (1024, 1))
        #if feature_reduction== "resize256":
            #Image2 = tf.image.resize(image2, (16, 16))
        #X_test_final.append(Image2)
    #X_Test_final = np.squeeze(X_test_final)

     
    #return X_Train_final, X_Test_final, Y_train, Y_test
    #return X_train[3]
    


        
data = data_load_and_process("bloodmnist", classes=[0, 1], feature_reduction="resize256", binary=True)
data1 = np.transpose(data, (0, 2, 3, 1))
print(np.shape(data1))
#X_train = tf.image.resize(data[:], (32, 32)).numpy()
#print(np.shape(X_train))
#X1_train = tf.reshape(X_train, (np.shape(X_train)[0], 1024, 1))
#X = data.flatten()
#print(np.max(np.array(data[7])), np.min(np.array(data[3])))

s1 = np.matrix([[0,1],[1,0]])
ID = np.identity(2)
#A = np.kron(np.kron(np.kron(ID, s1), 
#B = np.kron(np.kron(np.kron(np.kron(ID, s1), np.kron(ID, s1)), np.kron(np.kron(ID, ID), np.kron(s1, ID))), np.kron(np.kron(s1, ID), np.kron(ID, s1)))
#B = np.kron(ID, np.kron(ID, A))
#B = np.kron(np.kron(np.kron(np.kron(ID, s1), np.kron(ID, s1)), np.kron(np.kron(ID, ID), np.kron(s1, ID))), np.kron(s1, ID))
#Y = np.dot(B, data)
#Y = np.array(Y)
#print(Y.shape)
#Z = np.squeeze(Y, axis=0)
#print(Z.shape)

#C =  np.kron(np.kron(np.kron(np.kron(ID, s1), np.kron(ID, s1)), np.kron(np.kron(ID, s1), np.kron(ID, s1))), np.kron(ID, s1))
#C = np.kron(np.kron(np.kron(np.kron(ID, s1), np.kron(ID, s1)), np.kron(np.kron(ID, s1), np.kron(ID, s1))), np.kron(np.kron(ID, s1), np.kron(ID, s1)))
#Z = np.dot(C, np.transpose(Y))
#Z = np.array(Z)
#Z1 = np.reshape(Z, (32, 32))

#X = np.array([[0, 1], [1, 0]])
#ID = np.array([[1, 0], [0, 1]])#
#herm = np.kron(np.kron(X, ID), X)
#h=qml.pauli_decompose(herm)
#print(h)

#Y = np.zeros([6, 2])

#Z1 = np.array([[[0, 0, 0.5],[0, 0, 0]], [[0.5, 0, 0],[0, 0.5, 0]]])
#Y = np.array([[[0], [0]], [[0], [0]]])
#Z = np.append(Z1, Y, axis=2)
#for i in range(2):
    #for j in range(2):
        #for k in range(3):
            #Y[i+2*k, j] = Z1[i, j, k]

#print(Z)
#Y1 = tf.reshape(Z1, (12, 1))
#Y = np.reshape(Z1, (12, 1))
#print(Y1)
#print(Y)
#S = np.reshape(Y1, (2, 2, 3))
#print(S)


#print(np.shape(data[0]))
#print(np.shape(data[1]))
#print(np.shape(data[2]))
#print(np.shape(data[3]))
#image = np.reshape(data[0][1], (32, 32))
#plt.plot(np.asarray(data[0]), cmap="Greys")
#plt.show()
#for x in X_test:
    #print(x)
    #print("\n")
