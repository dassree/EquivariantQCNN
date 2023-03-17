import numpy as np
import matplotlib.pyplot as plt

#f1 = open("fashion_mnist1.txt", "r")
#f2 = open("fashion_mnist2.txt", "r")
#f3 = open("fashion_mnist3.txt", "r")
#f4 = open("fashion_mnist4.txt", "r")
#f5 = open("fashion_mnist5.txt", "r")

#f6 = open("Result/results-0.txt", "r")
#f7 = open("Result/results-1.txt", "r")
#f8 = open("Result/results-2.txt", "r")
#f9 = open("Result/results-3.txt", "r")
#f10 = open("Result/results-4.txt", "r")

f1 = open("cifar10/stepsize=0.01/cifar10-0_eqv.txt", "r")
f2 = open("cifar10/stepsize=0.01/cifar10-1_eqv.txt", "r")
f3 = open("cifar10/stepsize=0.01/cifar10-2_eqv.txt", "r")
f4 = open("cifar10/stepsize=0.01/cifar10-3_eqv.txt", "r")
f5 = open("cifar10/stepsize=0.01/cifar10-4_eqv.txt", "r")
f6 = open("cifar10/stepsize=0.01/cifar10-5_eqv.txt", "r")
f7 = open("cifar10/stepsize=0.01/cifar10-6_eqv.txt", "r")
f8 = open("cifar10/stepsize=0.01/cifar10-7_eqv.txt", "r")
f9 = open("cifar10/stepsize=0.01/cifar10-8_eqv.txt", "r")
f10 = open("cifar10/stepsize=0.01/cifar10-9_eqv.txt", "r")

accuracy = np.zeros((10, 101))
#print(np.shape(accuracy[1, :]))

#iterations = [x.split('	')[0] for x in f1.readlines()]

accuracy[0, :]= [x.split('	')[1] for x in f1.readlines()]
accuracy[1, :]= [x.split('	')[1] for x in f2.readlines()]
accuracy[2, :]= [x.split('	')[1] for x in f3.readlines()]
accuracy[3, :]= [x.split('	')[1] for x in f4.readlines()]
accuracy[4, :]= [x.split('	')[1] for x in f5.readlines()]
accuracy[5, :]= [x.split('	')[1] for x in f6.readlines()]
accuracy[6, :]= [x.split('	')[1] for x in f7.readlines()]
accuracy[7, :]= [x.split('	')[1] for x in f8.readlines()]
accuracy[8, :]= [x.split('	')[1] for x in f9.readlines()]
accuracy[9, :]= [x.split('	')[1] for x in f10.readlines()]

f = open("cifar10/mean_sd_eqv_0.01.txt", "w")
for i in range(101):
    f.write(str(i*10) + "	" + str(np.mean(accuracy[:, i])) + "	" + str(np.std(accuracy[:, i])))
    f.write("\n")

