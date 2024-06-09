Backpropagation is a method used to train neural networks by adjusting the weights based on the error rate obtained in the previous epoch (iteration). It involves a forward pass to compute the loss, and a backward pass to compute the gradients and update the weights.

Here's an example of backpropagation implemented in Python for a simple feedforward neural network with one hidden layer:

```python
import numpy as np

# Activation function: Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Mean Squared Error Loss
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Training Data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input: XOR problem
y = np.array([[0], [1], [1], [0]])              # Output: XOR problem

# Seed for reproducibility
np.random.seed(42)

# Initialize weights randomly with mean 0
input_layer_neurons = X.shape[1]  # Input layer
hidden_layer_neurons = 2          # Hidden layer
output_layer_neurons = 1          # Output layer

# Weights and bias initialization
W1 = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
b1 = np.random.uniform(size=(1, hidden_layer_neurons))
W2 = np.random.uniform(size=(hidden_layer_neurons, output_layer_neurons))
b2 = np.random.uniform(size=(1, output_layer_neurons))

# Learning rate
lr = 0.1

# Training loop
for epoch in range(10000):
    # Forward Pass
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    y_pred = sigmoid(final_input)

    # Compute Loss
    loss = mean_squared_error(y, y_pred)

    # Backward Pass
    # Calculate gradients
    d_loss_y_pred = y_pred - y
    d_y_pred_final_input = sigmoid_derivative(y_pred)
    d_loss_final_input = d_loss_y_pred * d_y_pred_final_input

    d_final_input_hidden_output = W2
    d_loss_hidden_output = d_loss_final_input.dot(d_final_input_hidden_output.T)
    d_hidden_output_hidden_input = sigmoid_derivative(hidden_output)
    d_loss_hidden_input = d_loss_hidden_output * d_hidden_output_hidden_input

    # Update weights and biases
    W2 -= hidden_output.T.dot(d_loss_final_input) * lr
    b2 -= np.sum(d_loss_final_input, axis=0, keepdims=True) * lr

    W1 -= X.T.dot(d_loss_hidden_input) * lr
    b1 -= np.sum(d_loss_hidden_input, axis=0, keepdims=True) * lr

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

print("Final loss:", loss)
print("Predictions:", y_pred)
```

### Explanation

1. **Initialization**:

    - We define the sigmoid activation function and its derivative.
    - We set up the XOR problem data (`X` and `y`).
    - We initialize the weights (`W1`, `W2`) and biases (`b1`, `b2`) with random values.
    - We define the learning rate (`lr`).

2. **Forward Pass**:

    - Compute the input to the hidden layer (`hidden_input`) by performing a dot product between `X` and `W1` and adding the bias `b1`.
    - Apply the sigmoid activation function to get the hidden layer's output (`hidden_output`).
    - Compute the input to the output layer (`final_input`) by performing a dot product between `hidden_output` and `W2` and adding the bias `b2`.
    - Apply the sigmoid activation function to get the predicted output (`y_pred`).

3. **Loss Calculation**:

    - Calculate the mean squared error loss between the predicted output (`y_pred`) and the actual output (`y`).

4. **Backward Pass**:

    - Calculate the gradient of the loss with respect to the predicted output (`d_loss_y_pred`).
    - Calculate the gradient of the predicted output with respect to the final input (`d_y_pred_final_input`).
    - Calculate the gradient of the loss with respect to the final input (`d_loss_final_input`).
    - Calculate the gradient of the final input with respect to the hidden output (`d_final_input_hidden_output`).
    - Calculate the gradient of the loss with respect to the hidden output (`d_loss_hidden_output`).
    - Calculate the gradient of the hidden output with respect to the hidden input (`d_hidden_output_hidden_input`).
    - Calculate the gradient of the loss with respect to the hidden input (`d_loss_hidden_input`).

5. **Weight and Bias Updates**:

    - Update the weights (`W2`, `W1`) and biases (`b2`, `b1`) using the calculated gradients and the learning rate.

6. **Training Loop**:
    - Repeat the forward and backward passes for a specified number of epochs (10,000 in this example).
    - Print the loss at regular intervals to monitor the training process.

This example demonstrates the core principles of backpropagation in a simple neural network.
