import math
from random import normalvariate
import matplotlib.pyplot as plt
from data import load_synth

def initialize_network(method):
    if(method == 'static'):
        w = [[1., 1., 1.], [-1., -1., -1.]]
        v = [[1., 1.], [-1., -1.],[-1., -1.]]
    elif(method == 'random'):
        w = [[normalvariate(0, 1) for _ in range(3)], [normalvariate(0, 1) for _ in range(3)]]
        v = [[normalvariate(0, 1) for _ in range(2)], [normalvariate(0, 1) for _ in range(2)], [normalvariate(0, 1) for _ in range(2)]]

    b = [0., 0., 0.]
    c = [0., 0.]

    return w, b, v, c

def initialize_network_gradients_scalar():
    dv = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    dh = [0.0, 0.0, 0.0]
    dk = [0.0, 0.0, 0.0]
    db = [0.0, 0.0, 0.0]
    dw = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    dy = [0., 0.]
    return dv, dh, dk, dw, db, dy

def softmax(val):
    a = [math.e**x for x in val]
    b = sum([math.e**x for x in val])
    c = [x/b for x in a]
    return c

def sigmoid(val):
    sig = 1 / (1 + math.exp(-val))
    return sig

def sigmoid_for_list(val):
    h = [0]*len(val)
    for i in range(len(val)):
        h[i] = sigmoid(val[i])
    return h

def total_loss(y_hat, y):
    loss = 0.0
    for i in range(len(y_hat)):
        loss += y[i] * (-math.log(y_hat[i]))
    return loss

def linear_calculation(w, b, inputs, length):
    k = [0.]*length
    for j in range(len(k)):
        for i in range(len(inputs)):
            k[j] += w[i][j] * inputs[i]
        k[j] += b[j]
    return k

def forward_pass(w, b, inputs, v, c, y):
    k1 = linear_calculation(w, b, inputs, 3)
    h = sigmoid_for_list(k1)
    k2 = linear_calculation(v, c, h, 2)
    y_hat = softmax(k2)

    return h, y_hat

def softmax_der(y_hat, y):
    dy = [0., 0.]
    for i in range(len(y_hat)):
        dy[i] = y_hat[i] - y[i]
    return dy

def backpropagate_scalar(inputs, y_hat, y, v, h):
    dv, dh, dk, dw, db, dy = initialize_network_gradients_scalar()
    dy = softmax_der(y_hat, y)

    for i in range(len(h)):
        for j in range(len(dy)):
            dv[i][j] = dy[j] * h[i]
            dh[i] += dy[j] * v[i][j]

    dc = dy.copy()

    for i in range(len(h)):
        dk[i] = dh[i]*h[i]*(1-h[i])

    for j in range(len(dk)):
        for i in range(len(inputs)):
            dw[i][j] = dk[j] * inputs[i]
        db[j] = dk[j]

    return dy, dv, dh, dk, dw, db, dc

def update_parameters_scalar(v, c, w, b, dv, dc, dw, db, learning_rate: float):
    n_x, n_h, n_y = 2, 3, 2

    for i in range(n_y):
        for j in range(n_h):
            v[j][i] -= learning_rate * dv[j][i]
        c[i] -= learning_rate * dc[i]

    for i in range(n_h):
        for j in range(n_x):
            w[j][i] -= learning_rate * dw[j][i]
        b[i] -= learning_rate * db[i]
    return v, c, w, b

def simple_model():
    w, b, v, c = initialize_network(method='static')
    inputs = [1., -1]
    y = [1., 0.]
    h, y_hat = forward_pass(w, b, inputs, v, c, y)
    loss = total_loss(y_hat, y)
    y, dv, dh, dk, dw, db, dc = backpropagate_scalar(inputs, y_hat, y, v, h)
    print('derivatives wrt w, b\n', dw, '\n', db)
    print('derivatives wrt v, c\n', dv, '\n', dc)

def scalar_model(epochs):
    (xtrain, ytrain), (xval, yval), random = load_synth()
    epoch_loss = []
    epoch_loss_val = []
    w, b, v, c = initialize_network(method = 'random')
    for i in range(epochs):
        for j in range(len(xtrain)):
            inputs = xtrain[j]
            if ytrain[j] == 1: y = [1, 0]
            else: y = [0, 1]
            h, y_hat = forward_pass(w, b, inputs, v, c, y)
            loss = total_loss(y_hat, y)
            dy, dv, dh, dk, dw, db, dc = backpropagate_scalar(inputs, y_hat, y, v, h)
            v, c, w, b = update_parameters_scalar(v, c, w, b, dv, dc, dw, db, 1e-3)
        epoch_loss.append(loss)
        print('Epoch: ', i+1,' Loss: ',loss)
        for j in range(len(xval)):
            inputs = xval[j]
            if yval[j] == 1:
                y = [1, 0]
            else:
                y = [0, 1]
            h, y_hat = forward_pass(w, b, inputs, v, c, y)
            loss = total_loss(y_hat, y)
        epoch_loss_val.append(loss)
        print('Val Loss: ', loss)

    plt.plot(range(1, len(epoch_loss)+1, 1), epoch_loss, '-', color='black', label= 'Training')
    plt.plot(range(1, len(epoch_loss_val) + 1, 1), epoch_loss_val, '--', color='black', label= 'Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Evolution of Loss')
    plt.legend()
    plt.show()

# q3
# simple_model()

# q4
scalar_model(25)