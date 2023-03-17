import numpy as np
import matplotlib.pyplot as plt

f1 = open("fashion_mnist1_eqv.txt", "r")
f2 = open("fashion_mnist2_eqv.txt", "r")
f3 = open("fashion_mnist3_eqv.txt", "r")
f4 = open("fashion_mnist4_eqv.txt", "r")
f5 = open("fashion_mnist5_eqv.txt", "r")


accuracy = np.zeros((5, 101))
#print(np.shape(accuracy[1, :]))

#iterations = [x.split('	')[0] for x in f1.readlines()]

accuracy[0, :]= [x.split('	')[1] for x in f1.readlines()]
accuracy[1, :]= [x.split('	')[1] for x in f2.readlines()]
accuracy[2, :]= [x.split('	')[1] for x in f3.readlines()]
accuracy[3, :]= [x.split('	')[1] for x in f4.readlines()]
accuracy[4, :]= [x.split('	')[1] for x in f5.readlines()]

f = open("mean_sd.txt", "w")
for i in range(101):
    f.write(str(i*10) + "	" + str(np.mean(accuracy[:, i])) + "	" + str(np.std(accuracy[:, i])))
    f.write("\n")

