import numpy as np

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons, 
    weight_regularizer_L1=0, bias_regularizer_L1=0, 
    weight_regularizer_L2=0, bias_regularizer_L2=0):

        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # Set regularization strength
        self.weight_regularizer_L1 = weight_regularizer_L1
        self.bias_regularizer_L1 = bias_regularizer_L1
        self.weight_regularizer_L2 = weight_regularizer_L2
        self.bias_regularizer_L2 = bias_regularizer_L2

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

        # Gradients on regularization

        # L1 on weights
        if self.weight_regularizer_L1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_L1 * dL1

        # L2 on weights
        if self.weight_regularizer_L2 > 0:
            self.dweights += 2 * self.weight_regularizer_L2 * self.weights

        # L1 on biases
        if self.bias_regularizer_L1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_L1 * dL1

        # L2 on biases
        if self.bias_regularizer_L2 > 0:
            self.dbiases += 2 * self.bias_regularizer_L2 * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)

class Layer_Dropout:

    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs):
        self.inputs = inputs
        # generate and save scaled binary mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # apply mask to output
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask