"""
Part 8: hyperparameter experiments.

Each sweep changes one hyperparameter at a time, keeping the rest at the
Part 7 baseline, and compares the settings on validation accuracy. The test
set is never used here; it is held back for the final evaluation.

Every setting is trained once per seed. The seed changes the starting
weights and the shuffle order, so the spread across seeds shows how much a
score moves by chance alone. A difference smaller than that spread is not
evidence that one setting is better.
"""
import time

import numpy as np

import data
import network
from train import train

# Part 7 baseline. Each sweep changes one of these and keeps the others.
# The batch size sweep found batch 16 works better, so train.py now uses 16.
# These stay at 64 so the documented sweep results can be reproduced exactly.
BASE_LR = 0.1
BASE_SIZES = [784, 128, 64, 10]
BASE_BATCH = 64
BASE_EPOCHS = 5

SEEDS = [0, 1, 2]


def run_experiment(X_train, Y_train, X_val, Y_val, layer_sizes, learning_rate,
                   batch_size, epochs, seed=0):
    """
    Train a brand-new network with one set of hyperparameters.

    A fresh network is built every call, so no run gets a head start from
    weights an earlier run already trained.

    Parameters:
    X_train, Y_train (numpy.ndarray): Training inputs (N, 784) and labels (N,).
    X_val, Y_val (numpy.ndarray): Validation inputs and labels.
    layer_sizes (list of int): e.g. [784, 128, 64, 10].
    learning_rate (float): Step size for each update.
    batch_size (int): Images per batch.
    epochs (int): Number of full passes over the training data.
    seed (int): Used for both the starting weights and the shuffle order.
        Same seed + same settings = the exact same result.

    Returns:
    dict: per-epoch train_losses/val_losses/train_accs/val_accs, and seconds taken.
    """
    nn = network.NeuralNetwork(layer_sizes, seed=seed)
    start = time.perf_counter()
    train_losses, val_losses, train_accs, val_accs = train(
        nn, X_train, Y_train, X_val, Y_val,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        seed=seed, verbose=False,
    )
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "seconds": time.perf_counter() - start,
    }


def run_sweep(title, column, settings, datasets):
    """
    Run each setting once per seed and print one summary row per setting.

    Parameters:
    title (str): Heading printed above the table.
    column (str): Header for the first column (the thing being varied).
    settings (list of (label, dict)): A row label plus the keyword arguments
        for run_experiment, e.g. ("0.1", {"learning_rate": 0.1, ...}).
    datasets (tuple): (X_train, Y_train, X_val, Y_val).

    Columns: mean train acc, mean val acc, the lowest-highest val acc across
    seeds, the train-val gap (an overfitting signal), and mean seconds.
    """
    print(f"\n{title} ({len(SEEDS)} seeds each)")
    print(f"{column:>20} | {'train acc':>9} | {'val acc':>8} | {'val range':>15} | "
          f"{'gap':>6} | {'time':>6}")
    for label, kwargs in settings:
        runs = [run_experiment(*datasets, **kwargs, seed=s) for s in SEEDS]
        train_acc = np.mean([r["train_accs"][-1] for r in runs])
        vals = [r["val_accs"][-1] for r in runs]
        seconds = np.mean([r["seconds"] for r in runs])
        print(f"{label:>20} | {train_acc:>9.2%} | {np.mean(vals):>8.2%} | "
              f"{min(vals):>6.2%} - {max(vals):>6.2%} | "
              f"{train_acc - np.mean(vals):>6.2%} | {seconds:>5.1f}s")


def baseline(**changes):
    """The baseline settings as keyword arguments, with some values swapped in."""
    settings = {"layer_sizes": BASE_SIZES, "learning_rate": BASE_LR,
                "batch_size": BASE_BATCH, "epochs": BASE_EPOCHS}
    settings.update(changes)
    return settings


if __name__ == "__main__":
    # Load once, outside the sweeps: every run uses exactly the same data split
    X_train, Y_train, X_test, Y_test = data.load_mnist()
    X_train, X_test = data.preprocess_data(X_train, X_test)
    X_train, Y_train, X_val, Y_val = data.split_data(X_train, Y_train)
    datasets = (X_train, Y_train, X_val, Y_val)

    run_sweep(
        "Learning rate sweep (sizes [784, 128, 64, 10], batch 64, 5 epochs)", "lr",
        [(str(lr), baseline(learning_rate=lr)) for lr in [0.001, 0.01, 0.1, 0.5, 1.0]],
        datasets,
    )

    run_sweep(
        "Hidden width sweep (lr 0.1, batch 64, 5 epochs)", "sizes",
        [(str(sizes), baseline(layer_sizes=sizes))
         for sizes in [[784, 32, 10], [784, 128, 64, 10], [784, 256, 128, 10]]],
        datasets,
    )

    # Bigger batches mean fewer updates per epoch (about 55,000 / batch_size),
    # but each update averages more images, so it points more accurately downhill.
    run_sweep(
        "Batch size sweep (sizes [784, 128, 64, 10], lr 0.1, 5 epochs)", "batch size",
        [(str(bs), baseline(batch_size=bs)) for bs in [16, 64, 256]],
        datasets,
    )

    # Batch 16 won above, but it also took ~1.6x longer. Give batch 64 the same
    # time budget (8 epochs is about as long as 5 epochs of batch 16) to see
    # whether batch 16 is better per second, or just ahead because it trained longer.
    run_sweep(
        "Equal time budget (sizes [784, 128, 64, 10], lr 0.1)", "batch, epochs",
        [("16, 5 epochs", baseline(batch_size=16, epochs=5)),
         ("64, 8 epochs", baseline(batch_size=64, epochs=8))],
        datasets,
    )
