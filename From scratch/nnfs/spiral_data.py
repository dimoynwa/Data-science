from layers import Layer_Dense, Activation_Relu, Activation_Softmax_Loss_Categorical_CrossEntropy, Optimizer_SGD, Optimizer_AdaGrad
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

# Create Network
dense1 = Layer_Dense(2, 64)
activation1 = Activation_Relu()

dense2 = Layer_Dense(64, 3)
loss = Activation_Softmax_Loss_Categorical_CrossEntropy()

#optimizer = Optimizer_SGD(initial_learning_rate=1., decay=.001, momentum=0.9)

optimizer = Optimizer_AdaGrad(initial_learning_rate=1., decay=1e-5)

# Train in loop
for epoch in range(91):
    # Perform a forward pass of our training data through this layer
    dense1.forward(X)
    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer here and returns loss
    loss_val = loss.forward(dense2.output, y)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(loss.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    if epoch % 10 == 0:
        print(f'epoch: {epoch}, ' +
        f'acc: {accuracy:.3f}, ' +
        f'loss: {loss_val:.3f}, ' +
        f'optimizer learning rate: {optimizer.current_learning_rate:.3f}')

    # Backward pass
    loss.backward(loss.output, y)
    dense2.backward(loss.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update parameters
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()