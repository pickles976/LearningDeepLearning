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

