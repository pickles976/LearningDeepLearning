from nnfs.datasets import spiral_data

import numpy as np
import nnfs

from activations import Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy
from layers import Layer_Dense, Layer_Dropout
from optimizer import Optimizer_SGD, Optimizer_Adam

nnfs.init()

# import matplotlib.pyplot as plt

X, y = spiral_data(samples=100, classes=3)

# 2 input features, 64 output values
dense1 = Layer_Dense(2, 512, weight_regularizer_L2=5e-4, bias_regularizer_L2=5e-4)

# Create ReLU activation (to be used with dense layer)
activation1 = Activation_ReLU()

dropout1 = Layer_Dropout(0.1)

# 64 inputs, 3 outputs
dense2 = Layer_Dense(512, 3)

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-5)

for epoch in range(10001):

    # Forward pass to calculate loss function
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    dense2.forward(dropout1.output)

    data_loss = loss_activation.forward(dense2.output, y)
    regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)
    loss = data_loss + regularization_loss

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
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

# Validate model

# create test dataset
X_test, y_test = spiral_data(samples=100, classes=3)

dense1.forward(X_test)

activation1.forward(dense1.output)

dense2.forward(activation1.output)

loss = loss_activation.forward(dense2.output, y_test)

predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions==y_test)

print(f'Validation, acc: {accuracy:.3f}, loss: {loss:.3f}')