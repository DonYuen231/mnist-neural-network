"""
Mini-batch SGD training: one epoch of updates, accuracy, and the full
training loop with per-epoch shuffling and train/validation tracking.

Run directly to train with the chosen settings: python train.py
"""
import numpy as np

import data
import network
from losses import cross_entropy_loss


def train_one_epoch(nn, X, Y, batch_size, learning_rate):
    """
    Run one pass over the training data with mini-batch SGD.

    For each batch: forward, loss, backward, then update every layer's
    weights and biases. backward() already averages the gradient over the
    batch, so no extra summing or dividing is needed here.

    Parameters:
    nn (NeuralNetwork): The network to train (updated in place).
    X (numpy.ndarray): Training inputs, shape (N, 784).
    Y (numpy.ndarray): Integer labels, shape (N,).
    batch_size (int): Images per batch. The last batch may be smaller.
    learning_rate (float): Step size for each update.

    Returns:
    float: The average batch loss over the epoch.
    """
    losses = []
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]
        logits = nn.forward(X_batch)
        losses.append(cross_entropy_loss(logits, Y_batch))
        nn.backward(logits, Y_batch)
        for layer in nn.layers:
            layer.weights -= learning_rate * layer.grad_weights
            layer.biases -= learning_rate * layer.grad_biases
    return np.mean(losses)


def accuracy(nn, X, Y):
    """
    Fraction of images whose predicted digit matches the label.

    The prediction for each image is the digit with the highest score
    (argmax of the logits). No softmax needed: softmax keeps the order of
    the scores, so the highest logit is also the highest probability.

    Parameters:
    nn (NeuralNetwork): The network to evaluate.
    X (numpy.ndarray): Inputs, shape (N, 784).
    Y (numpy.ndarray): Integer labels, shape (N,).

    Returns:
    float: Accuracy between 0 and 1 (e.g. 0.97 = 97% correct).
    """
    predictions = np.argmax(nn.forward(X), axis=1)
    return np.mean(predictions == Y)


def train(nn, X_train, Y_train, X_val, Y_val, epochs, batch_size, learning_rate, seed=0,
          verbose=True):
    """
    Train the network for several epochs, reshuffling the data each epoch.

    After each epoch, measures loss and accuracy on both the training set
    and the validation set. The validation set is never trained on, so it
    shows how well the network does on new images. Overfitting shows up as
    val loss rising while train loss keeps falling.

    Parameters:
    nn (NeuralNetwork): The network to train (updated in place).
    X_train, Y_train (numpy.ndarray): Training inputs (N, 784) and labels (N,).
    X_val, Y_val (numpy.ndarray): Validation inputs and labels.
    epochs (int): Number of full passes over the training data.
    batch_size (int): Images per batch.
    learning_rate (float): Step size for each update.
    seed (int): Seed for the shuffling. The rng is created once, so each
        epoch gets a different order but the whole run is reproducible.
    verbose (bool): If True, print a line after every epoch.

    Returns:
    tuple of lists, one entry per epoch:
        (train_losses, val_losses, train_accs, val_accs)
    """
    rng = np.random.default_rng(seed)
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    for epoch in range(epochs):
        # One shared order for X and Y, so every image keeps its own label
        indices = rng.permutation(len(X_train))
        X_shuffle, Y_shuffle = X_train[indices], Y_train[indices]

        train_one_epoch(nn, X_shuffle, Y_shuffle, batch_size, learning_rate)

        # Measure both sets with the finished network for this epoch, so train
        # and val numbers are directly comparable
        train_losses.append(cross_entropy_loss(nn.forward(X_train), Y_train))
        val_losses.append(cross_entropy_loss(nn.forward(X_val), Y_val))
        train_accs.append(accuracy(nn, X_train, Y_train))
        val_accs.append(accuracy(nn, X_val, Y_val))
        if verbose:
            print(f"Epoch {epoch + 1}: "
                  f"train loss {train_losses[-1]:.4f}, val loss {val_losses[-1]:.4f}, "
                  f"train acc {train_accs[-1]:.2%}, val acc {val_accs[-1]:.2%}")
    return train_losses, val_losses, train_accs, val_accs


if __name__ == "__main__":
    # Same pipeline as data.py: load, flatten + scale to [0, 1], hold out a validation set
    X_train, Y_train, X_test, Y_test = data.load_mnist()
    X_train, X_test = data.preprocess_data(X_train, X_test)
    X_train, Y_train, X_val, Y_val = data.split_data(X_train, Y_train)

    nn = network.NeuralNetwork([784, 128, 64, 10])

    # Untrained baseline: should be about 2.30 (= -log(1/10), a uniform guess)
    print(f"loss before training: {cross_entropy_loss(nn.forward(X_train), Y_train):.4f}")

    train_losses, val_losses, train_accs, val_accs = train(
        nn, X_train, Y_train, X_val, Y_val, epochs=20, batch_size=16, learning_rate=0.1
    )
