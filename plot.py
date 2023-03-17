import numpy as np
import matplotlib.pyplot as plt

f1 = open("cifar10/mean_sd_eqv.txt", "r")
f2 = open("cifar10/mean_sd.txt", "r")

x = [(i*10) for i in range(101)]
#print(x)

Y1 = np.zeros((2, 101))
for i, line1 in enumerate(f1):
    Y1[0, i] = line1.split("	")[1]
    Y1[1, i] = line1.split("	")[2]

Y2 = np.zeros((2, 101))
for i, line2 in enumerate(f2):
    Y2[0, i] = line2.split("	")[1]
    Y2[1, i] = line2.split("	")[2]


#plt.errorbar(x, Y1[0, :], Y1[1, :], linestyle='solid', marker='^', label="equivariant QCNN")
#plt.errorbar(x, Y2[0, :], Y2[1, :], linestyle='dashed', marker='p', label="Non-equivariant QCNN")
plt.plot(x, Y1[0, :], "-", color="blue", label="equivariant QCNN")
plt.fill_between(x, Y1[0, :] - Y1[1, :], Y1[0, :] + Y1[1, :],
                 color='blue', alpha=0.2)
plt.plot(x, Y2[0, :], "-", color="gray", label="Non-equivariant QCNN")
plt.fill_between(x, Y2[0, :] - Y2[1, :], Y2[0, :] + Y2[1, :],
                 color='gray', alpha=0.2)
plt.legend(loc="lower right")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("No. of iterations", fontsize=14, labelpad = 10)
plt.ylabel("Mean test accuracy with errors", fontsize=14, labelpad = 10)
plt.show()
