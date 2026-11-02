from layers import DenseLayer
import neuron
import numpy as np
from losses import cross_entropy_grad
class NeuralNetwork:
    def __init__(self, layer_sizes, seed=42):
        rng = np.random.default_rng(seed)
        layers = []
        for i in range(len(layer_sizes) - 1):
            is_last_layer = (i == len(layer_sizes) - 2)
            activation_function = None if is_last_layer else neuron.relu
            layers.append(DenseLayer(layer_sizes[i], layer_sizes[i + 1], rng, activation_function))
        self.layers = layers
    def forward(self, X):
        """
        Perform the forward pass through the entire network.

        Parameters:
        X (numpy.ndarray): Input data.

        Returns:
        numpy.ndarray: The output of the network after passing through all layers.
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X
    def backward(self, logits, Y):
        """
        Perform the backward pass (backpropagation) through the entire network.

        Starts from the cross-entropy gradient at the output and passes it back
        through each layer in reverse order. Each layer stores its own
        grad_weights and grad_biases for the update step.

        Must be called right after forward() on the same batch: each layer's
        backward() uses the A_inputs and Z cached by the most recent forward().

        Parameters:
        logits (numpy.ndarray): Output of forward(), shape (B, C).
        Y (numpy.ndarray): Integer class labels, shape (B,).

        Returns:
        None. Gradients are stored on each layer.
        """
        grads = cross_entropy_grad(logits, Y)
        for layer in reversed(self.layers):
            grads = layer.backward(grads)




if __name__ == "__main__":
    # Example usage
    layer_sizes = [784, 128, 64, 10]  # Example architecture for MNIST
    nn = NeuralNetwork(layer_sizes)

    rng = np.random.default_rng(seed=42)
    X = rng.random((5, 784))  # Fake batch: 5 "images", 784 pixels each
    output = nn.forward(X)

    # n sizes -> n - 1 layers (catches off-by-one errors in the range)
    assert len(nn.layers) == 3, f"expected 3 layers, got {len(nn.layers)}"
    print(f"layer count: {len(nn.layers)} OK")

    # Each layer's dims chain correctly: 784->128, 128->64, 64->10
    for layer, (n_in, n_out) in zip(nn.layers, [(784, 128), (128, 64), (64, 10)]):
        assert layer.weights.shape == (n_in, n_out), f"got {layer.weights.shape}"
    print("layer weight shapes OK")

    # ReLU on hidden layers, no activation (raw logits) on the output layer
    assert all(layer.activation is neuron.relu for layer in nn.layers[:-1]), "hidden layer missing relu"
    assert nn.layers[-1].activation is None, "output layer should have no activation"
    print("activations OK (relu hidden, None output)")

    # Full forward pass: (5, 784) -> (5, 10), one score per digit per image
    assert output.shape == (5, 10), f"expected (5, 10), got {output.shape}"
    print(f"output shape: {output.shape} OK")

    # backward(): every layer gets gradients shaped like its own weights/biases
    y = np.array([3, 1, 4, 1, 5])  # fake labels, one per image
    nn.backward(output, y)
    for i, layer in enumerate(nn.layers):
        assert layer.grad_weights.shape == layer.weights.shape, f"layer {i}: {layer.grad_weights.shape}"
        assert layer.grad_biases.shape == layer.biases.shape, f"layer {i}: {layer.grad_biases.shape}"
        assert np.all(np.isfinite(layer.grad_weights)), f"layer {i} has non-finite gradients"
    print("backward() gradient shapes match every layer OK")

    # The output layer has no activation, so its bias gradient is just the
    # loss gradient summed over the batch (checks the chain starts correctly)
    from losses import cross_entropy_loss
    expected_db = cross_entropy_grad(output, y).sum(axis=0, keepdims=True)
    assert np.allclose(nn.layers[-1].grad_biases, expected_db), "output layer grad_biases wrong"
    print("output layer receives cross_entropy_grad OK")

    # End-to-end direction check: one small step against the gradient should
    # lower the loss. (Part 6 gradient checking will verify the exact values.)
    loss_before = cross_entropy_loss(output, y)
    for layer in nn.layers:
        layer.weights -= 0.01 * layer.grad_weights
        layer.biases -= 0.01 * layer.grad_biases
    loss_after = cross_entropy_loss(nn.forward(X), y)
    assert loss_after < loss_before, f"loss went up: {loss_before:.4f} -> {loss_after:.4f}"
    print(f"one gradient step lowers loss: {loss_before:.4f} -> {loss_after:.4f} OK")

    print("\nAll network.py checks passed.")
