import numpy as np
from neuron import relu

class DenseLayer:
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
        # TODO (Part 5): DRAFT ONLY - not yet correct or tested. Do not call.
        #  - self.activation(self.Z, derivative=True) fails: relu/sigmoid take no
        #    `derivative` argument (derivatives are separate functions in neuron.py)
        #  - dividing dW/db by m double-counts the 1/B already applied in the
        #    output layer's dZ = (y_hat - y) / B
        #  - verify against gradient checking (Part 6) once rewritten
        if self.activation is not None:
            dZ = dA * self.activation(self.Z, derivative=True)
        else:
            dZ = dA

        m = self.A_inputs.shape[0]  # Number of examples
        self.dW = np.dot(self.A_inputs.T, dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        dA_prev = np.dot(dZ, self.weights.T)

        return dA_prev

if __name__ == "__main__":
    # One shared generator for the whole test: reproducible, but each layer
    # draws different numbers from it.
    rng = np.random.default_rng(seed=42)
    X = rng.random((5, 784))  # fake batch: 5 "images", 784 pixels each

    l1 = DenseLayer(784, 128, rng, relu)
    l2 = DenseLayer(128, 128, rng, relu)

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

    print("\nAll layers.py checks passed.")
