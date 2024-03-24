from feedforward_nn import SimpleNeuralNetwork
import numpy as np

network = SimpleNeuralNetwork()

print(network.weights)

train_inputs = np.array(
    [[1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], ]
)

train_outputs = np.array([[0, 1, 0, 0, 1, 0]]).T

train_iterations = 50000

network.train(train_inputs, train_outputs, train_iterations)

print(network.weights)

print("Testing the data")
test_data = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0], ])

for data in test_data:
    print(f"Result for {data} is:")
    print(network.propagation(data))