"""
DenseLayer: a fully connected layer with its forward pass and backpropagation.

Run directly to run the self-tests: python layers.py
"""
import numpy as np

import activations


class DenseLayer:
    """
    A fully connected layer: every input connects to every neuron.

    forward() computes Z = inputs @ weights + biases for the whole batch at
    once, then applies the activation (or returns Z unchanged for the output
    layer). It caches the inputs and Z, which backward() needs to compute
    grad_weights, grad_biases, and the gradient passed to the previous layer.
    """

    def __init__(self, input_dim, output_dim, rng, activation=None):
        """
        Initialize a layer with weights, biases, and an optional activation function.

        Parameters:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        activation (callable, optional): Activation function to apply. Defaults to None.
        rng (np.random.Generator): Shared random generator used to initialize weights.
        """

        self.weights = rng.standard_normal((input_dim, output_dim)) * np.sqrt(2. / input_dim)  # He initialization
        self.biases = np.zeros((1, output_dim))
        self.activation = activation

    def forward(self, inputs):
        """
        Perform the forward pass through the layer.

        Parameters:
        inputs (numpy.ndarray): Input data.

        Returns:
        numpy.ndarray: The output of the layer after applying weights, biases, and activation function.
        """
        z = np.dot(inputs, self.weights) + self.biases
        self.A_inputs = inputs  # Store the linear input for potential use in backpropagation
        self.Z = z  # Store the linear output for potential use in backpropagation
        if self.activation is not None:
            return self.activation(z)
        else:
            return z

    def backward(self, dA):
        """
        Perform the backward pass through the layer.

        Parameters:
        dA (numpy.ndarray): Gradient of the loss with respect to the output of this layer.

        Returns:
        numpy.ndarray: Gradient of the loss with respect to the input of this layer.
        """
        if self.activation is None:
            dZ = dA
        elif self.activation == activations.relu:
            dZ = np.multiply(activations.relu_derivative(self.Z),dA)
        else:
            # Fail loudly instead of a confusing UnboundLocalError on dZ below
            raise ValueError(f"no derivative known for activation {self.activation}")
        self.grad_weights = np.dot(self.A_inputs.T,dZ)
        self.grad_biases = np.sum(dZ, axis=0, keepdims=True)
        self.grad_A_inputs = np.dot(dZ,self.weights.T)
        return self.grad_A_inputs


if __name__ == "__main__":
    # One shared generator for the whole test: reproducible, but each layer
    # draws different numbers from it.
    rng = np.random.default_rng(seed=42)
    X = rng.random((5, 784))  # fake batch: 5 "images", 784 pixels each

    l1 = DenseLayer(784, 128, rng, activations.relu)
    l2 = DenseLayer(128, 128, rng, activations.relu)

    # Shape check: (5, 784) @ (784, 128) -> (5, 128)
    out1 = l1.forward(X)
    assert out1.shape == (5, 128), f"expected (5, 128), got {out1.shape}"
    print(f"l1 output shape: {out1.shape} OK")

    # ReLU never outputs negatives
    assert (out1 >= 0).all(), "ReLU layer produced negative outputs"
    print("l1 outputs all >= 0 OK")

    # Chaining: l1's output is l2's input, same as NeuralNetwork.forward will do
    out2 = l2.forward(out1)
    assert out2.shape == (5, 128), f"expected (5, 128), got {out2.shape}"
    print(f"l2 output shape (chained): {out2.shape} OK")

    # Shared rng -> layers get different weights (the hardcoded-seed bug)
    assert not np.array_equal(l1.weights[:128], l2.weights), "l1 and l2 got identical weights"
    print("l1 and l2 weights differ OK")

    # forward() cached what backprop will need
    assert l1.A_inputs is X, "l1 did not cache its input"
    assert l1.Z.shape == (5, 128), f"cached Z has shape {l1.Z.shape}"
    print("forward() cached A_inputs and Z OK")

    # No activation -> identity: output is exactly Z (raw logits)
    l3 = DenseLayer(128, 10, rng)
    logits = l3.forward(out2)
    assert np.array_equal(logits, l3.Z), "activation=None should return Z unchanged"
    print(f"no-activation layer returns Z unchanged, shape {logits.shape} OK")

    # backward(): every gradient has the same shape as the thing it belongs to
    dA_in = l1.backward(rng.random((5, 128)))
    assert l1.grad_weights.shape == l1.weights.shape, f"got {l1.grad_weights.shape}"
    assert l1.grad_biases.shape == l1.biases.shape, f"got {l1.grad_biases.shape}"
    assert dA_in.shape == X.shape, f"got {dA_in.shape}"
    print("backward() gradient shapes match weights, biases, inputs OK")

    # backward() hand-computed: 2 inputs -> 1 neuron, no activation, 1 image
    # grad_weights = A_inputs.T @ dZ = [[1], [4]] * 0.5 = [[0.5], [2.0]]
    # grad_biases  = sum of dZ over batch          = [[0.5]]
    # grad_inputs  = dZ @ weights.T = 0.5 * [2, 3]   = [[1.0, 1.5]]
    hand = DenseLayer(2, 1, rng)
    hand.weights = np.array([[2.0], [3.0]])
    hand.forward(np.array([[1.0, 4.0]]))
    grad_in = hand.backward(np.array([[0.5]]))
    assert np.allclose(hand.grad_weights, [[0.5], [2.0]]), f"got {hand.grad_weights}"
    assert np.allclose(hand.grad_biases, [[0.5]]), f"got {hand.grad_biases}"
    assert np.allclose(grad_in, [[1.0, 1.5]]), f"got {grad_in}"
    print("backward() hand-computed example OK")

    # ReLU blocks gradient for "off" neurons: inputs [1, 2] give Z = [3, -3],
    # so neuron 1 (Z < 0) has slope 0 and gets an all-zero grad_weights column
    gate = DenseLayer(2, 2, rng, activations.relu)
    gate.weights = np.array([[1.0, -1.0], [1.0, -1.0]])
    gate.forward(np.array([[1.0, 2.0]]))
    gate.backward(np.array([[1.0, 1.0]]))
    assert np.allclose(gate.grad_weights, [[1.0, 0.0], [2.0, 0.0]]), f"got {gate.grad_weights}"
    assert np.allclose(gate.grad_biases, [[1.0, 0.0]]), f"got {gate.grad_biases}"
    print("backward() ReLU passes no gradient through neurons with Z <= 0 OK")

    print("\nAll layers.py checks passed.")
