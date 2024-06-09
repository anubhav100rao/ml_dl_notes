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
    d_loss_hidden_output = d_loss_final_input.dot(
        d_final_input_hidden_output.T)
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
