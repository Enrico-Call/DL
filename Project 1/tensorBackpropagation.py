from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import numpy.random as npr
from data import load_mnist
import matplotlib.pyplot as plt

npr.seed(305)

(xtrain, ytrain), (xtest, ytest), size = load_mnist(final=True)
xtrain = xtrain/255
xtest = xtest/255

ytrain_cat = np.zeros((ytrain.size, ytrain.max()+1))
ytrain_cat[np.arange(ytrain.size),ytrain] = 1
ytest_cat = np.zeros((ytest.size, ytest.max()+1))
ytest_cat[np.arange(ytest.size),ytest] = 1

m = xtrain.shape[0]

X_train, X_test = xtrain.T, xtest.T
Y_train, Y_test = ytrain_cat.T, ytest_cat.T

def sigmoid(z):
    s = 1. / (1. + np.exp(-z))
    return s

def compute_loss(Y, Y_hat):
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1./m) * L_sum

    return L

def forward_pass(X, params):
    cache = {}

    cache["K"] = np.matmul(params["W"], X) + params["b"]
    cache["H"] = sigmoid(cache["K"])
    cache["K1"] = np.matmul(params["V"], cache["H"]) + params["c"]
    cache["Y"] = np.exp(cache["K1"]) / np.sum(np.exp(cache["K1"]), axis=0)

    return cache

def back_propagate(X, Y, params, cache):
    dY = cache["Y"] - Y
    dV = (1./m_batch) * np.matmul(dY, cache["H"].T)
    dc = (1./m_batch) * np.sum(dY, axis=1, keepdims=True)

    dH = np.matmul(params["V"].T, dY)
    dK = dH * sigmoid(cache["H"]) * (1 - sigmoid(cache["H"]))
    dW = (1./m_batch) * np.matmul(dK, X.T)
    db = (1./m_batch) * np.sum(dK, axis=1, keepdims=True)

    grads = {"dW": dW, "db": db, "dV": dV, "dc": dc}

    return grads

# hyperparameters
learning_rate = 0.05
batch_size = 32
batches = -(-m // batch_size)
epochs = 5

# initialization

train_cost = []
train_acc = []
test_cost = []
test_acc = []
param = []

for iter in range(5):

    train_cost_list = []
    train_acc_list = []
    test_cost_list = []
    test_acc_list = []
    params = {"W": npr.normal(0, 1, size=(300, 784))*np.sqrt(1. / 784),
              "b": np.zeros(shape=(300, 1)),
              "V": npr.normal(0, 1, size=(10, 300))*np.sqrt(1. / 300),
              "c": np.zeros(shape=(10, 1))}

    permutation = np.random.permutation(X_train.shape[1])
    X_train_shuffled = X_train[:, permutation]
    Y_train_shuffled = Y_train[:, permutation]

    for i in range(epochs):

        for j in range(batches):

            begin = j * batch_size
            end = min(begin + batch_size, X_train.shape[1] - 1)
            X = X_train_shuffled[:, begin:end]
            Y = Y_train_shuffled[:, begin:end]
            m_batch = end - begin

            cache = forward_pass(X, params)
            grads = back_propagate(X, Y, params, cache)

            params["W"] = params["W"] - learning_rate * grads["dW"]
            params["b"] = params["b"] - learning_rate * grads["db"]
            params["V"] = params["V"] - learning_rate * grads["dV"]
            params["c"] = params["c"] - learning_rate * grads["dc"]
        

        cache = forward_pass(X_train, params)
        predictions = np.argmax(cache["Y"], axis=0)
        labels = np.argmax(Y_train, axis=0)
        train_cost_partial = compute_loss(Y_train, cache["Y"])
        train_cost_list.append(train_cost_partial)
        train_acc_list.append(accuracy_score(labels, predictions))
        cache = forward_pass(X_test, params)
        predictions = np.argmax(cache["Y"], axis=0)
        labels = np.argmax(Y_test, axis=0)
        test_cost_partial = compute_loss(Y_test, cache["Y"])
        test_cost_list.append(test_cost_partial)
        test_acc_list.append(accuracy_score(labels, predictions))

    train_cost.append(train_cost_list)
    train_acc.append(train_acc_list)
    test_cost.append(test_cost_list)
    test_acc.append(test_acc_list)

# final arrays computation

train_cost_final = np.array(train_cost).mean(axis=0)
train_acc_final = np.array(train_acc).mean(axis=0)
train_sd = np.array(train_cost).std(axis=0)
train_acc_sd = np.array(train_acc).std(axis=0)
test_cost_final = np.array(test_cost).mean(axis=0)
test_acc_final = np.array(test_acc).mean(axis=0)
test_sd = np.array(test_cost).std(axis=0)
test_acc_sd = np.array(test_acc).std(axis=0)

cache = forward_pass(X_test, params)
predictions = np.argmax(cache["Y"], axis=0)
labels = np.argmax(Y_test, axis=0)
print(classification_report(predictions, labels))

# plotting of loss

plt.plot(np.arange(1, 6, 1), train_cost_final, '-', color='black', label='Training')
plt.fill_between(np.arange(1, 6, 1), train_cost_final+train_sd, train_cost_final-train_sd, facecolor='black', alpha=0.5)
plt.plot(np.arange(1, 6, 1), test_cost_final, '--',  color='black', label='Test')
plt.fill_between(np.arange(1, 6, 1), test_cost_final+test_sd, test_cost_final-test_sd, facecolor='grey', alpha=0.5)
plt.xlabel('Epoch')
plt.xticks(ticks=np.arange(1, 6, 1))
plt.ylabel('Cross-Entropy Loss')
plt.title('Learning Rate: '+ str(learning_rate))
plt.legend()
plt.show()

# plotting of accuracy

plt.plot(np.arange(1, 6, 1), train_acc_final, '-', color='black', label='Training')
plt.fill_between(np.arange(1, 6, 1), train_acc_final+train_acc_sd, train_acc_final-train_acc_sd, facecolor='black', alpha=0.5)
plt.plot(np.arange(1, 6, 1), test_acc_final, '--',  color='black', label='Test')
plt.fill_between(np.arange(1, 6, 1), test_acc_final+test_acc_sd, test_acc_final-test_acc_sd, facecolor='grey', alpha=0.5)
plt.xlabel('Epoch')
plt.xticks(ticks=np.arange(1, 6, 1))
plt.ylabel('Accuracy')
plt.title('Learning Rate: '+ str(learning_rate))
plt.legend()
plt.show()