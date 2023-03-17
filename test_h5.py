import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

f1 = h5py.File('camelyonpatch_level_2_split_train_x.h5', 'r')
f2 = h5py.File('camelyonpatch_level_2_split_train_y.h5', 'r')
f3 = h5py.File('camelyonpatch_level_2_split_test_x.h5', 'r')
f4 = h5py.File('camelyonpatch_level_2_split_test_y.h5', 'r')
#print(list(f1.keys()))
a_group_key = list(f1.keys())[0]
b_group_key = list(f2.keys())[0]
c_group_key = list(f3.keys())[0]
d_group_key = list(f4.keys())[0]
data1 = np.array(f1.get(a_group_key))
data2 = np.array(f2.get(b_group_key))
data3 = np.array(f3.get(c_group_key))
data4 = np.array(f4.get(d_group_key))
x_train = []
y_train = []
x_test = []
y_test = []


#F1 = open("patchCamelyon/train_data_x.pkl", "wb")
#for i in range(1000):
#    image = tf.image.resize_with_crop_or_pad(np.array(data1[i]), 32, 32)
#    x_train.append(np.array(image))
#    #F1.write(str(np.array(image)))
#    #F1.write("\n")
#pickle.dump(x_train, F1)
#F1.close()
#    
#F2 = open("patchCamelyon/train_data_y.pkl", "wb")
#for i in range(1000):
#    y_train.append(np.squeeze(data2[i]))
#    #F2.write(str(data2[i]))
#    #F2.write("\n")
#pickle.dump(y_train, F2)
#F2.close()
##np.savetxt("patchCamelyon/train_data_y.out" ,y_train)
#    
#F3 = open("patchCamelyon/test_data_x.pkl", "wb")
#for i in range(100):
#    image = tf.image.resize_with_crop_or_pad(np.array(data3[i]), 32, 32)
#    x_test.append(np.array(image))
#    #F3.write(str(np.array(image)))
#    #F3.write("\n")
#pickle.dump(x_test, F3)
#F3.close()
##np.savetxt("patchCamelyon/test_data_x.out", x_test)
#    
#F4 = open("patchCamelyon/test_data_y.pkl", "wb")
#for i in range(100):
#    y_test.append(np.squeeze(data4[i]))
#    #F4.write(str(data4[i]))
#    #F4.write("\n")
#pickle.dump(y_test, F4)
#F4.close()
#np.savetxt("patchCamelyon/test_data_y.out" ,y_test)

#print(np.shape(x_train))
#print(np.shape(y_train)) 
#print(np.shape(x_test))
#print(np.shape(y_test))

#pkl_file = open('patchCamelyon/train_data_x.pkl', 'rb')

#data = np.array(pickle.load(pkl_file))
#data = data / 255.0 # normalize the data
#print(type(data))
#print(np.shape(data))
#print(data)
#print(np.shape(data1[0]))
#print(image[:,:,0])
#plt.imshow(image)
#plt.show()
#ds_arr = f1[a_group_key][()]  
