# Implementation of Quantum circuit training procedure
import QCNN_circuit
import pennylane as qml
from pennylane import numpy as np
import autograd.numpy as anp

def accuracy_test(predictions, labels, cost_fn, binary = True):
    if cost_fn == 'mse':
        if binary == True:
            acc = 0
            for l, p in zip(labels, predictions):
                prediction = np.amax(p)
                if np.abs(l - prediction) < 1:
                    acc = acc + 1
            return acc / len(labels)

        else:
            acc = 0
            for l, p in zip(labels, predictions):
                if np.abs(l - p) < 0.5:
                    acc = acc + 1
            return acc / len(labels)

    elif cost_fn == 'cross_entropy':
        acc = 0
        for l,p in zip(labels, predictions):
            if p[0] > p[1]:
                P = 0
            else:
                P = 1
            if P == l:
                acc = acc + 1
        return acc / len(labels)


def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        prediction = np.amax(p)
        loss = loss + (l - prediction) ** 2
    loss = loss / len(labels)
    return loss

def cross_entropy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        c_entropy = l * (anp.log(p[l])) + (1 - l) * anp.log(1 - p[1 - l])
        loss = loss + c_entropy
    return -1 * loss

def cost(params, X, Y, U, U_params, embedding_type, circuit, cost_fn):
    if circuit == 'QCNN':
        predictions = np.array([QCNN_circuit.QCNN(x, params, U, U_params, embedding_type, cost_fn=cost_fn) for x in X])

    if cost_fn == 'mse':
        loss = square_loss(Y, predictions)
    elif cost_fn == 'cross_entropy':
        loss = cross_entropy(Y, predictions)
    return loss

# Circuit training parameters
#steps = 400
learning_rate = 0.005
batch_size = 25
def circuit_training(X_train, Y_train, X_test, Y_test, U, U_params, embedding_type, circuit, cost_fn, steps, binary, processNumber):
    if circuit == 'QCNN':
        if U == 'U_SU4_no_pooling' or U == 'U_SU4_1D' or U == 'U_9_1D':
            total_params = U_params * 2
        else:
            total_params = U_params[0] + 3*U_params[1] + 8


    params = np.random.randn(total_params, requires_grad=True)
    print("Proc " + str(processNumber) + " params: "+ str(params))
    opt = qml.NesterovMomentumOptimizer(stepsize=learning_rate)
    #opt = qml.AdamOptimizer(stepsize=learning_rate)
    loss_history = []

    for it in range(steps):
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]
        if it % 10 ==1:
            #train_predictions = np.array([QCNN_circuit.QCNN(x, params, U, U_params, embedding_type, cost_fn = cost_fn) for x in X_train])
            #train_accuracy = accuracy_test(train_predictions, Y_train, cost_fn, binary)
            #print("Train Accuracy after step " + str(it-1) + ": " + str(accuracy))
            test_predictions = np.array([QCNN_circuit.QCNN(x, params, U, U_params, embedding_type, cost_fn = cost_fn) for x in X_test])
            test_accuracy = accuracy_test(test_predictions, Y_test, cost_fn, binary)
            print("Proc " + str(processNumber) + " " +str(it-1) + "	" + str(test_accuracy))
        params, cost_new = opt.step_and_cost(lambda v: cost(v, X_batch, Y_batch, U, U_params, embedding_type, circuit, cost_fn),
                                                     params)
        loss_history.append(cost_new)
        if it % 10 == 0:
            print("Proc " + str(processNumber) + " iteration: ", it, " cost: ", cost_new)
        
    return loss_history, params


