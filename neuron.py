import numpy as np




def sigmoid(z):
    """
    Apply the sigmoid activation function.

    Parameters:
    z (float): The input value.

    Returns:
    float: The output of the sigmoid function.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    out[~pos] = np.exp(z[~pos]) / (1 + np.exp(z[~pos]))
    return out



def sigmoid_derivative(z):
    """
    Calculate the derivative of the sigmoid activation function.

    Parameters:
    z (float): The input value.

    Returns:
    float: The derivative of the sigmoid function.
    """
    s = sigmoid(z)
    return s * (1 - s)

def relu_derivative(z):
    """
    Calculate the derivative of the ReLU activation function.

    Parameters:
    z (float): The input value.

    Returns:
    float: The derivative of the ReLU function.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    out[z > 0] = 1
    out[z <= 0] = 0
    return out

def relu(z):
    """
    Apply the ReLU activation function.

    Parameters:
    z (float): The input value.

    Returns:
    float: The output of the ReLU function.
    """
    z = np.asarray(z, dtype=float)
    return np.maximum(0, z)


def neuron_calculate(inputs, weights, bias):
    """
    Calculate the output of a neuron given inputs, weights, and bias.

    Parameters:
    inputs (list): A list of input values.
    weights (list): A list of weights corresponding to the inputs.
    bias (float): The bias term for the neuron.

    Returns:
    float: The output of the neuron after applying the activation function.
    """
    weights = np.asarray(weights, dtype=float)
    inputs = np.asarray(inputs, dtype=float)
    # Calculate the weighted sum of inputs
    weighted_sum = np.dot(inputs, weights) + bias

    # Apply the activation function
    return sigmoid(weighted_sum)


if __name__ == "__main__":
    # Hand-computed check:
    # z = 1*0.5 + 2*(-1) + 0.5 = -1
    # sigmoid(-1) = 1 / (1 + e^1) ~= 0.2689414214
    result = neuron_calculate([1, 2], [0.5, -1], 0.5)
    expected = 0.2689414213699951
    assert np.isclose(result, expected), f"expected {expected}, got {result}"
    print(f"neuron_calculate basic case: {result:.10f} (expected ~{expected:.10f}) OK")

    # Numerical stability check: large negative z should not overflow/warn,
    # and should saturate to ~0.
    result_stable = neuron_calculate([1000], [-1], 0)
    assert np.isclose(result_stable, 0.0, atol=1e-6), f"expected ~0, got {result_stable}"
    print(f"neuron_calculate stability case (z=-1000): {result_stable} OK")

    # Large positive z should saturate to ~1.
    result_stable_pos = neuron_calculate([1000], [1], 0)
    assert np.isclose(result_stable_pos, 1.0, atol=1e-6), f"expected ~1, got {result_stable_pos}"
    print(f"neuron_calculate stability case (z=+1000): {result_stable_pos} OK")

    # sigmoid_derivative sanity check at z=0 -> should be exactly 0.25
    deriv = sigmoid_derivative(np.array([0.0]))
    assert np.isclose(deriv[0], 0.25), f"expected 0.25, got {deriv[0]}"
    print(f"sigmoid_derivative(0): {deriv[0]} OK")

    # relu / relu_derivative sanity checks
    relu_vals = relu(np.array([-5, 0, 5]))
    assert np.array_equal(relu_vals, [0, 0, 5]), f"got {relu_vals}"
    print(f"relu([-5, 0, 5]): {relu_vals} OK")

    relu_d_vals = relu_derivative(np.array([-5, 0, 5]))
    assert np.array_equal(relu_d_vals, [0, 0, 1]), f"got {relu_d_vals}"
    print(f"relu_derivative([-5, 0, 5]): {relu_d_vals} OK")

    print("\nAll neuron.py checks passed.")
