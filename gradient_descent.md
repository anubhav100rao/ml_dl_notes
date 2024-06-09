Below is an implementation of linear regression using gradient descent in Python.

### Linear Regression using Gradient Descent

1. **Import necessary libraries**:

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    ```

2. **Define the data**:

    ```python
    # Example dataset
    X = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    y = np.array([1, 3, 2, 3, 5], dtype=np.float32)
    ```

3. **Normalize the data** (optional but recommended for better convergence):

    ```python
    X = (X - np.mean(X)) / np.std(X)
    ```

4. **Initialize parameters**:

    ```python
    m = len(X)  # Number of data points
    theta0 = 0  # Intercept
    theta1 = 0  # Slope
    learning_rate = 0.01
    num_iterations = 1000
    ```

5. **Define the cost function**:

    ```python
    def compute_cost(X, y, theta0, theta1):
        m = len(y)
        predictions = theta0 + theta1 * X
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost
    ```

6. **Gradient Descent function**:

    ```python
    def gradient_descent(X, y, theta0, theta1, learning_rate, num_iterations):
        m = len(y)
        cost_history = []

        for i in range(num_iterations):
            predictions = theta0 + theta1 * X
            theta0 -= (learning_rate / m) * np.sum(predictions - y)
            theta1 -= (learning_rate / m) * np.sum((predictions - y) * X)
            cost = compute_cost(X, y, theta0, theta1)
            cost_history.append(cost)

            if i % 100 == 0:  # Print cost every 100 iterations
                print(f"Iteration {i}: Cost {cost}")

        return theta0, theta1, cost_history
    ```

7. **Run Gradient Descent**:

    ```python
    theta0, theta1, cost_history = gradient_descent(X, y, theta0, theta1, learning_rate, num_iterations)
    ```

8. **Plot the cost history** (optional):

    ```python
    plt.plot(range(num_iterations), cost_history)
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.title("Cost Function using Gradient Descent")
    plt.show()
    ```

9. **Plot the linear regression line**:

    ```python
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.plot(X, theta0 + theta1 * X, color='red', label='Linear Regression')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    ```

10. **Full Code**:

```python
import numpy as np
import matplotlib.pyplot as plt

# Example dataset
X = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y = np.array([1, 3, 2, 3, 5], dtype=np.float32)

# Normalize the data
X = (X - np.mean(X)) / np.std(X)

# Initialize parameters
m = len(X)  # Number of data points
theta0 = 0  # Intercept
theta1 = 0  # Slope
learning_rate = 0.01
num_iterations = 1000

# Define the cost function
def compute_cost(X, y, theta0, theta1):
    m = len(y)
    predictions = theta0 + theta1 * X
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Gradient Descent function
def gradient_descent(X, y, theta0, theta1, learning_rate, num_iterations):
    m = len(y)
    cost_history = []

    for i in range(num_iterations):
        predictions = theta0 + theta1 * X
        theta0 -= (learning_rate / m) * np.sum(predictions - y)
        theta1 -= (learning_rate / m) * np.sum((predictions - y) * X)
        cost = compute_cost(X, y, theta0, theta1)
        cost_history.append(cost)

        if i % 100 == 0:  # Print cost every 100 iterations
            print(f"Iteration {i}: Cost {cost}")

    return theta0, theta1, cost_history

# Run Gradient Descent
theta0, theta1, cost_history = gradient_descent(X, y, theta0, theta1, learning_rate, num_iterations)

# Plot the cost history
plt.plot(range(num_iterations), cost_history)
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.title("Cost Function using Gradient Descent")
plt.show()

# Plot the linear regression line
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, theta0 + theta1 * X, color='red', label='Linear Regression')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
```

This code snippet demonstrates the entire process of implementing linear regression using gradient descent. The cost function is minimized iteratively, and the resulting linear regression line is plotted.
