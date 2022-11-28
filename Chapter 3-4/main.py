from nnfs.datasets import spiral_data

import numpy as np
import nnfs

from layers import Activation_ReLU, Activation_Softmax, Layer_Dense

nnfs.init()

# import matplotlib.pyplot as plt

X, y = spiral_data(samples=100, classes=3)

# 3 neurons per input, 2 inputs per 'shot'
dense1 = Layer_Dense(2,3)

# Create ReLU activation (to be used with dense layer)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)

activation2 = Activation_Softmax()

dense1.forward(X)

activation1.forward(dense1.output)

dense2.forward(activation1.output)

activation2.forward(dense2.output)

print(activation2.output[:5])

