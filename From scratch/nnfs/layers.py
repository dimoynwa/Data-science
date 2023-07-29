import numpy as np
import nnfs
from nnfs.datasets import spiral_data

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons) -> None:
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # A row vector 
        self.biases = np.zeros(shape=(1, n_neurons), dtype='float64')

    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalue):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalue)
        self.dweights = self.dweights.astype(np.float64)
        self.dbiases = np.sum(dvalue, axis=0, keepdims=True)
        self.dbiases = self.dbiases.astype(np.float64)

        # Gradient on values
        self.dinputs = np.dot(dvalue, self.weights.T)
        self.dinputs = self.dinputs.astype(np.float64)

class Activation_Relu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Since we need to modify the original variable,
        # let's make a copy of the values first
        self.dinputs = dvalues.astype(np.float64).copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

        pass

class Activation_Softmax:
    def forward(self, inputs):
        # Unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues, dtype='float64')

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array, create a column vector => shape: (classes, 1)
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output and => shape (classes, classes)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            # self.dinput[index]: shape: (classes, classes) * (classes, 1) = (classes, 1)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
            # Shape of self.dinputs will be (n_examples, classes)

# Common loss class
class Loss:
    def calculate(self, output, y):
        # All sample losses
        sample_losses = self.forward(output, y)

        # Calculate and return Mean of all sample losses
        return np.mean(sample_losses)

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y):
        # Clip the data to avoid 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Not one-hot, 
        if len(y.shape) == 1:
            correct_confidences = y_pred_clipped[:, y]
        elif len(y.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y, axis=1)

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)

        # Number of labels
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = - y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Both togheter
class Activation_Softmax_Loss_Categorical_CrossEntropy():
    def __init__(self) -> None:
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        # Pass the inputs to activation layer
        self.activation.forward(inputs)
        self.output = self.activation.output
        # Pass activation output to loss
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Optimizer: Stochastic gradient descent
class Optimizer_SGD:
    def __init__(self, initial_learning_rate=1., decay=0., momentum=0.) -> None:
        self.initial_learning_rate = initial_learning_rate
        self.current_learning_rate = self.initial_learning_rate
        self.decay = decay
        self.step = 0
        self.momentum = momentum

    # Call before any parameters update
    def pre_update_params(self):
        if self.decay > 0:
            self.current_learning_rate = self.initial_learning_rate / (1. + self.step * self.decay)

    def update_params(self, layer):
        if self.momentum:
            # If layer does not contain momentum arrays, create them filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights, dtype='float64')

            if not hasattr(layer, 'bias_momentums'):
                layer.bias_momentums = np.zeros_like(layer.biases, dtype='float64')
        
            weight_updates = self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            # Build bias updates
            bias_updates = self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        else:
            weight_updates = self.current_learning_rate * layer.dweights
            bias_updates = self.current_learning_rate * layer.dbiases

        layer.weights -= weight_updates
        layer.biases -= bias_updates

    # Call after any parameters update
    def post_update_params(self):
        self.step = self.step + 1

class Optimizer_AdaGrad:
    def __init__(self, initial_learning_rate=1., decay=0., epsilon=1e-7) -> None:
        self.initial_learning_rate = initial_learning_rate
        self.current_learning_rate = self.initial_learning_rate
        self.decay = decay
        self.step = 0
        self.epsilon = epsilon

    # Call before any parameters update
    def pre_update_params(self):
        if self.decay > 0:
            self.current_learning_rate = self.initial_learning_rate / (1. + self.step * self.decay)

    def update_params(self, layer):
        # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)

        if not hasattr(layer, 'bias_cache'):
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # Vanilla SGD parameter update + normalization with square rooted cache
        weight_updates = self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        bias_updates = self.current_learning_rate * layer.dbiases / (np.sqrt(layer.dbiases, dtype='float64') + self.epsilon)

        layer.weights -= weight_updates
        layer.biases -= bias_updates

    # Call after any parameters update
    def post_update_params(self):
        self.step = self.step + 1

class Optimizer_RMSProp():
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=.9) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.rho = rho
        self.step = 0

    def pre_update_params(self):
        if self.decay > 0:
            self.current_learning_rate = self.initial_learning_rate / (1. + self.step * self.decay)

    def update_params(self, layer):
        # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
        if not hasattr(layer, 'bias_cache'):
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)


    # Call after any parameters update
    def post_update_params(self):
        self.step = self.step + 1    

nnfs.init()

# Create Dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense Layer, X.shape == (100, 2) so 2 input features, 3 outputs
dence1 = Layer_Dense(X.shape[1], 3)

# Perform forward pass 
dence1.forward(X)

# Create activation layer
activation1 = Activation_Relu()

# Forward Dense 1 layer output through ReLU activation
activation1.forward(dence1.output)

dence2 = Layer_Dense(activation1.output.shape[1], 3)

dence2.forward(activation1.output)

# activation2 = Activation_Softmax()

# activation2.forward(dence2.output)
# print(f'First 5 results: {activation2.output[:5]}')

loss = Activation_Softmax_Loss_Categorical_CrossEntropy()
loss_val = loss.forward(dence2.output, y)

print(f'Loss: {loss_val}')

def calculate_accuracy(y_pred, y):
    predictions = np.argmax(y_pred, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    
    accuracy = np.mean(predictions == y)
    return accuracy

accuracy = calculate_accuracy(loss.output, y)
print(f'Accuracy: {accuracy}')

# Create optimizer
optimizer = Optimizer_SGD()

# backward pass, also known as backpropagation:
loss.backward(loss.output, y)
dence2.backward(loss.dinputs)
activation1.backward(dence2.dinputs)
dence1.backward(activation1.dinputs)

# Update weights and biases
optimizer.update_params(dence1)
optimizer.update_params(dence2)