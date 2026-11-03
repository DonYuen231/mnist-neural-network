"""
Part 9: evaluation.

Step 2 (this file so far): train the chosen network for many epochs and plot
train vs validation loss and accuracy, to see where overfitting starts and
pick the number of epochs. The test set is NOT used here.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

import data
import network
from train import train

# Chosen settings from Part 8 (see README). 20 epochs was picked from the
# curves this file plots: val accuracy plateaus around 98.3% from epoch ~16.
SIZES = [784, 128, 64, 10]
LEARNING_RATE = 0.1
BATCH_SIZE = 16
CURVE_EPOCHS = 20

FIGURE_DIR = "figures"

# Colors: blue for train, orange for validation, quiet gray text and grid
TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def plot_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """
    Save a figure with two charts side by side: loss and accuracy per epoch.

    Loss and accuracy get separate charts because they use different scales.
    A dashed line marks the epoch with the lowest validation loss: after it,
    the network keeps getting better on the training images but worse on new
    ones, which is overfitting.

    Parameters:
    train_losses, val_losses, train_accs, val_accs (list of float): One value
        per epoch, as returned by train().
    save_path (str): Where to write the PNG.

    Returns:
    int: The epoch (counting from 1) with the lowest validation loss.
    """
    epochs = np.arange(1, len(train_losses) + 1)
    best_epoch = int(np.argmin(val_losses)) + 1

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    charts = [
        (ax_loss, "Cross-entropy loss (lower is better)", train_losses, val_losses, 1),
        (ax_acc, "Accuracy (higher is better)",
         [a * 100 for a in train_accs], [a * 100 for a in val_accs], 100),
    ]
    for ax, title, train_vals, val_vals, _ in charts:
        ax.set_facecolor(SURFACE)
        ax.plot(epochs, train_vals, color=TRAIN_COLOR, linewidth=2, label="Train")
        ax.plot(epochs, val_vals, color=VAL_COLOR, linewidth=2, label="Validation")
        ax.axvline(best_epoch, color=MUTED, linewidth=1, linestyle="--")

        # Label each line at its right end, so color is never the only cue
        for vals, name, color in [(train_vals, "Train", TRAIN_COLOR),
                                  (val_vals, "Validation", VAL_COLOR)]:
            ax.annotate(name, (epochs[-1], vals[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", color=TEXT, fontsize=9)
            ax.plot(epochs[-1], vals[-1], "o", color=color, markersize=5)

        ax.set_title(title, color=TEXT, fontsize=11, loc="left")
        ax.set_xlabel("Epoch", color=MUTED)
        ax.set_xticks(epochs[::2] if len(epochs) > 10 else epochs)
        ax.set_xlim(1, epochs[-1] + 2.5)  # room for the end labels
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED, labelsize=9)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color(GRID)

    ax_acc.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax_loss.annotate(f"lowest val loss\n(epoch {best_epoch})", (best_epoch, 1),
                     xycoords=("data", "axes fraction"), xytext=(5, -4),
                     textcoords="offset points", va="top", color=MUTED, fontsize=8.5)
    ax_loss.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="center right")

    fig.suptitle(f"Training curves: {SIZES}, lr {LEARNING_RATE}, batch {BATCH_SIZE}",
                 color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return best_epoch


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = data.load_mnist()
    X_train, X_test = data.preprocess_data(X_train, X_test)
    X_train, Y_train, X_val, Y_val = data.split_data(X_train, Y_train)

    nn = network.NeuralNetwork(SIZES, seed=0)
    train_losses, val_losses, train_accs, val_accs = train(
        nn, X_train, Y_train, X_val, Y_val, epochs=CURVE_EPOCHS,
        batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, seed=0,
    )

    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, "training_curves.png")
    best_epoch = plot_curves(train_losses, val_losses, train_accs, val_accs, path)
    print(f"\nLowest val loss {min(val_losses):.4f} at epoch {best_epoch} "
          f"(val acc {val_accs[best_epoch - 1]:.2%}). Saved {path}")
