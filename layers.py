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
        # save inputs for backprop
        self.inputs = inputs

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues) # dvalues is output gradient
        self.dbiases = np.sum(dvalues, axis=0,keepdims=True) 
        # gradient on input values
        self.dinputs = np.dot(dvalues, self.weights.T)
