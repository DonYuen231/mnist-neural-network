"""
Gradient checking: verify backprop against numerical gradients.

For every weight and bias, nudge it up and down by a tiny eps, measure how the
loss changes, and compare that slope to what backward() computed. If the two
agree to a relative error below 1e-5, the backprop math is correct.

Run directly: python gradient_check.py
"""
import network
import numpy as np
from losses import cross_entropy_loss


def check_param(nn, X, y, param, grad, eps=1e-5):
    """
    Compare backprop's gradient for one parameter array against a numerical estimate.

    Nudges each entry of `param` up and down by eps, measures how the loss
    changes, and compares that slope to the matching entry of `grad`.

    Parameters:
    nn (NeuralNetwork): The network. backward() must already have been called.
    X (numpy.ndarray): Input batch used for the forward passes.
    y (numpy.ndarray): Integer class labels for X.
    param (numpy.ndarray): The array to nudge, e.g. nn.layers[0].weights.
    grad (numpy.ndarray): Backprop's gradient for it, e.g. nn.layers[0].grad_weights.
    eps (float): Size of the nudge.

    Returns:
    float: The largest relative error over every entry of param.
    """
    max_error = 0
    for i, j in np.ndindex(param.shape):
        value = param[i, j]
        param[i, j] = value + eps
        loss_plus = cross_entropy_loss(nn.forward(X), y)
        param[i, j] = value - eps
        loss_minus = cross_entropy_loss(nn.forward(X), y)
        param[i, j] = value  # restore, so the next entry is checked on the real network
        numeric = (loss_plus - loss_minus) / (2 * eps)
        analytic = grad[i, j]
        relative_error = abs(numeric - analytic) / max(abs(numeric) + abs(analytic), 1e-12)
        if max_error < relative_error:
            max_error = relative_error
    return max_error


if __name__ == "__main__":
    nn = network.NeuralNetwork([4, 5, 3])
    rng = np.random.default_rng(42)
    X = rng.random((3, 4))
    y = np.array([0, 1, 2])

    # Analytic gradients from backprop, computed once up front
    logits = nn.forward(X)
    nn.backward(logits, y)

    # Check every weight and bias in every layer against the numerical estimate
    worst = 0
    for k, layer in enumerate(nn.layers):
        for name, param, grad in [
            ("weights", layer.weights, layer.grad_weights),
            ("biases", layer.biases, layer.grad_biases),
        ]:
            error = check_param(nn, X, y, param, grad)
            print(f"layer {k} {name}: max relative error = {error:.2e}")
            worst = max(worst, error)

    # Project target from the build plan: relative error below 1e-5
    assert worst < 1e-5, f"gradient check FAILED: worst error {worst:.2e}"
    print(f"\nGradient check passed (worst error {worst:.2e} < 1e-5)")
