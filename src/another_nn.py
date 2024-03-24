import numpy as np
import sys

inputs = np.array([
    [0, 0, 1],  # 0
    [0, 1, 1],  # 1
    [1, 0, 1],  # 1
    [1, 1, 1],  # 0
])

outputs = np.array([[0, 1, 1, 0]]).T

w0 = 2 * np.random.random((3, 4)) - 1
w1 = 2 * np.random.random((4, 1)) - 1
epochs = 10000  # epochs = int(sys.argv[1])


for i in range(epochs + 1):
    # forward pass
    layer_0 = inputs
    layer_1 = 1 / (1 + np.exp(-np.dot(layer_0, w0)))
    layer_2 = 1 / (1 + np.exp(-np.dot(layer_1, w1)))

    loss = outputs - layer_2

    if i % 100 == 0:
        print("loss: ", np.mean(np.abs(loss)))

    # backward pass

    dsigmoid_layer2 = loss * (layer_2 * (1 - layer_2))
    dw1 = layer_1.T.dot(dsigmoid_layer2)
    dlayer_1 = dsigmoid_layer2.dot(w1.T)
    dsigmoid_layer1 = dlayer_1 * (layer_1 * (1 - layer_1))
    dw0 = layer_0.T.dot(dsigmoid_layer1)

    w1 += dw1
    w0 += dw0

# sigmoid(x) = 1 / (1+ np.exp(-x))
# dsigmoid(x)/dx = sigmoid(x) * (1 - sigmoid(x))

# Y = XW