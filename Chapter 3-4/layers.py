import numpy as np

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weigths, and biases
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:

    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # subtract so that the largest number is zero

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims=True)

        self.output = probabilities

# Questions:

# Q: What os the 0.01 magic number called?
# A: 

# Q: why n_inputs, n_neurons?
# A: Remember how we were transposing before?
