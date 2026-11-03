import time

import data
import network
from train import train


def run_experiment(X_train, Y_train, X_val, Y_val, layer_sizes, learning_rate,
                   batch_size, epochs):
    """
    Train a brand-new network with one set of hyperparameters.

    A fresh network is built every call, so no run gets a head start from
    weights an earlier run already trained. The network seed (42) and the
    shuffle seed (0) stay fixed, so the hyperparameters passed in are the
    only thing that differs between runs.

    Parameters:
    X_train, Y_train (numpy.ndarray): Training inputs (N, 784) and labels (N,).
    X_val, Y_val (numpy.ndarray): Validation inputs and labels.
    layer_sizes (list of int): e.g. [784, 128, 64, 10].
    learning_rate (float): Step size for each update.
    batch_size (int): Images per batch.
    epochs (int): Number of full passes over the training data.

    Returns:
    dict: the settings, per-epoch losses/train_accs/val_accs, and seconds taken.
    """
    nn = network.NeuralNetwork(layer_sizes, seed=42)
    start = time.perf_counter()
    losses, train_accs, val_accs = train(
        nn, X_train, Y_train, X_val, Y_val,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        seed=0, verbose=False,
    )
    seconds = time.perf_counter() - start
    return {
        "layer_sizes": layer_sizes,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "losses": losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "seconds": seconds,
    }


if __name__ == "__main__":
    # Load once, outside the loop: every run uses exactly the same data split
    X_train, Y_train, X_test, Y_test = data.load_mnist()
    X_train, X_test = data.preprocess_data(X_train, X_test)
    X_train, Y_train, X_val, Y_val = data.split_data(X_train, Y_train)

    # Baseline settings from Part 7. Each sweep changes one of these at a time.
    BASE_SIZES = [784, 128, 64, 10]
    BASE_BATCH = 64
    BASE_EPOCHS = 5

    print("Learning rate sweep (sizes [784, 128, 64, 10], batch 64, 5 epochs)")
    print(f"{'lr':>8} | {'final loss':>10} | {'train acc':>9} | {'val acc':>8} | {'time':>6}")
    for lr in [0.001, 0.01, 0.1, 0.5, 1.0]:
        result = run_experiment(X_train, Y_train, X_val, Y_val,
                                BASE_SIZES, lr, BASE_BATCH, BASE_EPOCHS)
        print(f"{lr:>8} | {result['losses'][-1]:>10.4f} | "
              f"{result['train_accs'][-1]:>9.2%} | {result['val_accs'][-1]:>8.2%} | "
              f"{result['seconds']:>5.1f}s")
