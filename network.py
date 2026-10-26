from layers import DenseLayer
import neuron
import numpy as np
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

    print("\nAll network.py checks passed.")
