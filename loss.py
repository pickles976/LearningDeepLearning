import numpy as np

class Loss: 

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)

        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    
    # forward pass
    def forward(self, y_pred, y_true):

        # samples in a batch
        samples = len(y_pred)

        # clip data to prevent log(0)
        # clip both sides to not drag mean towards a specific value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # probabilities for target values
        if len(y_true.shape) == 1: # categorical labels
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        elif len(y_true.shape) == 2: # Mask values for one-hot encoded labels
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        regative_log_likelihoods = -np.log(correct_confidences)
        return regative_log_likelihoods

    # backward pass
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        # normalize gradient sum to batch size
        self.dinputs = self.dinputs / samples

# TODO: I dont really understand the shape ==2 thing?