import numpy as np

# batching

# 4x3 Matrix
inputs = [[1.0, 2.0, 3.0, 2.5],
[1.0, 2.0, 3.0, 2.5],
[1.0, 2.0, 3.0, 2.5]]

#4x3 Matrix
weights1 = [[0.2, 0.8, -0.5, 1.0],
[0.5, -0.91, 0.26, -0.5],
[-0.26, -0.27, 0.17, 0.87]]

biases1 = [2.0, 3.0, 0.5]

#4x3 Matrix
weights2 = [[0.2, 0.8, -0.5],
[0.5, -0.91, 0.26],
[-0.26, 0.17, 0.87]]

biases2 = [2.0, 3.0, 0.5]


# Need to tranpose to do dot product
layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)