from nnfs.datasets import spiral_data

import numpy as np
import nnfs

from activations import Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy
from layers import Layer_Dense
from optimizer import Optimizer_SGD

nnfs.init()

# import matplotlib.pyplot as plt

X, y = spiral_data(samples=100, classes=3)

# 3 neurons per input, 2 inputs per 'shot'
dense1 = Layer_Dense(2,3)

# Create ReLU activation (to be used with dense layer)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_SGD(learning_rate=0.85)

for epoch in range(10001):

    # Forward pass to calculate loss function
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    # Calculate accuracy
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
        f'accuracy: {accuracy}, ' +
        f'loss: {loss:.3f}')

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)