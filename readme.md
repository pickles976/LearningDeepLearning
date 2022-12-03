# Important Concepts

## Activation Functions

### Sigmoid

The Sigmoid goes from 0 -> 1
lim = 0 as x -> -inf
lim = 1 as x -> inf
0 = 0.5

### ReLU

y = x if x > 0
y = 0 if x <= 0

ReLU is nice because it doesn't cause vanishing gradient like sigmoid, is almost free to compute, and results in sparse matrices (matrices with zeros) which is good for representation.

### Softmax

Softmax is good for classficiation/yassification

Softmax turns outputs into "confidence scores" which sum up to one. The largest score becomes the classification. 

score = output[i] / sum(e ^ output[:])


## Loss Functions

### MSE

### Categorical Cross-Entropy Loss

Cross-entropy compares two probability distributions.
Example, let's say our softmaxed output looks like:
[0.2, 0.7, 0.1] and the actual distribution is [0 , 1, 0]
The zeros cancel out and we get
loss = -(log(0.7) * 1)

this reduces to the negative log of the target class' confidence score! ezpz

## Backprop and Autodiff

This was difficult for me to understand, so I am going to try to explain it in a way that makes sense to me.

inputs = [2, 3]
weights = [[0.5, -0.5, 2.0],
           [0.3, 0.2, 0.1]]

Let's say you are given these inputs and weights to compute some output. The error function ends up being
the difference between your expected output and actual output. To change your weights, you want to know their impact on the output. You can get the gradient, which is a matrix of dE/dw, i.e. the change in error over the change in weights.

For a given inference, the slope of a weight is just going to be the input to it. So take w00 = 0.5 from out weights array. 0.5 * 2 is 1. If we change w00 to be 1, then the output is just 2. We can see that the slope is just 2. So d/dw00 = 2

Similarly d/dx will be the gradient of all the weights connected to it, so d/dx0 = [0.5, -0.5, 2.0]
This makes sense, since these are all partial derivatives the function relating x0 to the output.

To find the relationship of these partial derivatives d/dx and d/dw to the output, you have to also differentiate the Softmax and Categorical cross-entropy loss functions. This still doesn't make sense to me so I will put it off for later.

## Optimization

### Learning rate decay

learning_rate = initial_learning_rate * (1 / ( 1 + learning_rate_decay * step ))

Each step decreases the learning rate. A learning rate decay param of 0.01 will cause the learning rate to halve after 100 steps.

Pros: simple
Cons: 
- does not depend on our loss/accuracy
- easy to get trapped in a local minimum

### Momentum

The idea of gradients is particularly useful for understanding this concept. If the state of our model is the position of a ball in some R^n space, then the slope of the hill it's on would be the gradient of its params. With this analogy we can find the minimum by letting the ball roll "downhill" naturally with momentum. If it starts sufficiently high up, it should have enough momentum to escape any local minima. 

### AdaGrad

**Ada**ptive **Grad**ient Descent uses a per-parameter learning rate. The idea is this normalizes the features.

### RMSProp

Root Mean Square Propagation also calculates an adaptive learning rate per param, just with a different formula.

### Adam 

**Ada**ptive **M**omentum uses momentum, but with a per-weight adaptive learning rate.

## Data Practices

### K-Fold Cross-Validation

Assume we have limited data. Let's split it into ABCDE

For one epoch we'll train on ABCD and validate on E
The next epoch we'll train on ABDE and validate on E
so on and so forth

### Pre-processing

Neural nets work best on data valued from -1 to +1
Normalizing data to this range can be a pre-processing step.

### L1 and L2 regularization

L1 and L2 regularization are used to calculate a penalty added to the loss value to penalize for having large weights and biases.

#### Forward Pass:

L1's regularization penalty is the sum of all the absolute values for the weights and biases.

L1w = λsum(abs(w)) where w is the array of weights

L2's regularization penalty is the sum of all the squares of the weights and biases. This non-linearity penalizes larger values more heavily.

L2w = λsum(w ** 2) where w is the array of weights

Overall loss:
Loss = DataLoss + L1w + L1b + L2w + L2b

#### Backward Pass:

