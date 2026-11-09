"""
Part 9: final training and evaluation of the chosen network.

1. Train with the settings chosen in Part 8 and plot train vs validation
   loss and accuracy per epoch (used to pick the number of epochs).
2. Score the trained network on the test set. This is the only place the
   test set is used, and it is used once: no setting was chosen from it.
3. Build the confusion matrix and plot a sample of the misclassified digits.

All figures are saved to figures/.
"""
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

import activations
import data
import network
from losses import cross_entropy_loss
from train import accuracy, train

# Chosen settings from Part 8 (see README). 20 epochs was picked from the
# curves this file plots: val accuracy plateaus around 98.3% from epoch ~16.
SIZES = [784, 128, 64, 10]
LEARNING_RATE = 0.1
BATCH_SIZE = 16
CURVE_EPOCHS = 20

FIGURE_DIR = "figures"
PARAMS_PATH = "model_params.npz"

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


def confusion_matrix(predictions, labels, num_classes=10):
    """
    Count how often each true digit was predicted as each digit.

    Rows are the true digit and columns are the predicted digit, so
    matrix[4, 9] is how many real 4s the network called a 9. The diagonal
    holds the correct predictions; everything off it is a mistake. Each row
    sums to how many images of that digit there are, and the whole matrix
    sums to the number of images.

    Parameters:
    predictions (numpy.ndarray): Predicted digits, shape (N,).
    labels (numpy.ndarray): True digits, shape (N,).
    num_classes (int): Number of possible digits (10 for MNIST).

    Returns:
    numpy.ndarray: Integer counts, shape (num_classes, num_classes).
    """
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(predictions)):
        matrix[labels[i], predictions[i]] += 1
    return matrix


def plot_confusion_matrix(matrix, save_path):
    """
    Save the confusion matrix as a grid of colored cells with counts.

    Only the mistakes (off the diagonal) are colored, darker blue for more
    mistakes, so the common confusions stand out. The diagonal holds the
    correct predictions, which are about 100 times larger; coloring them on
    the same scale would wash out every mistake, so they are shown as gray
    cells with their count instead. Cells with zero mistakes are left blank.

    Parameters:
    matrix (numpy.ndarray): Counts from confusion_matrix(), shape (10, 10).
    save_path (str): Where to write the PNG.

    Returns:
    None. Writes the image file.
    """
    n = len(matrix)
    mistakes = matrix.astype(float)
    np.fill_diagonal(mistakes, np.nan)       # nan cells are not colored
    mistakes[mistakes == 0] = np.nan         # leave zero-mistake cells blank
    worst = np.nanmax(mistakes)

    # One-hue blue scale: light for few mistakes, dark for many
    blues = LinearSegmentedColormap.from_list(
        "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

    fig, ax = plt.subplots(figsize=(7.2, 6.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.imshow(mistakes, cmap=blues, vmin=1, vmax=worst)
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, color="#ecebe7"))

    for true in range(n):
        for pred in range(n):
            count = matrix[true, pred]
            if count == 0:
                continue
            dark_cell = true != pred and count > worst * 0.55
            ax.text(pred, true, str(count), ha="center", va="center", fontsize=8.5,
                    color="white" if dark_cell else (MUTED if true == pred else TEXT))

    # Thin surface-colored lines between cells
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.tick_params(colors=MUTED, length=0, labelsize=10)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Predicted digit", color=MUTED)
    ax.set_ylabel("True digit", color=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)

    wrong = int(matrix.sum() - np.trace(matrix))
    ax.set_title(f"Confusion matrix on the test set: {wrong} mistakes out of "
                 f"{int(matrix.sum()):,}\nBlue = mistakes (darker = more), "
                 f"gray diagonal = correct", color=TEXT, fontsize=11, loc="left", pad=36)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_misclassified(X, labels, predictions, confidences, indices, save_path):
    """
    Save a grid of misclassified images, each titled with what went wrong.

    Each title shows the true digit, the predicted digit, and how confident
    the network was in its (wrong) prediction. Images are drawn dark-on-light,
    like ink on paper.

    Parameters:
    X (numpy.ndarray): Flattened images, shape (N, 784).
    labels (numpy.ndarray): True digits, shape (N,).
    predictions (numpy.ndarray): Predicted digits, shape (N,).
    confidences (numpy.ndarray): Softmax probability of each prediction, shape (N,).
    indices (numpy.ndarray): Which images to show (e.g. the first 20 mistakes).
    save_path (str): Where to write the PNG.

    Returns:
    None. Writes the image file.
    """
    cols = 5
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.2),
                             facecolor=SURFACE)
    for ax in axes.flat:
        ax.axis("off")  # also hides unused cells if indices doesn't fill the grid
    for ax, i in zip(axes.flat, indices):
        ax.imshow(X[i].reshape(28, 28), cmap="gray_r")
        ax.set_title(f"true {labels[i]}, pred {predictions[i]}\n"
                     f"{confidences[i]:.0%} sure", color=TEXT, fontsize=9)
    fig.suptitle(f"Misclassified test digits (first {len(indices)} of the mistakes)",
                 color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


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

    # The one and only test-set evaluation. Every setting was chosen
    # on validation data, so this is an honest estimate on unseen digits.
    test_loss = cross_entropy_loss(nn.forward(X_test), Y_test)
    test_acc = accuracy(nn, X_test, Y_test)
    n_wrong = round((1 - test_acc) * len(Y_test))

    print("\nFinal results (after epoch 20)")
    print(f"{'':>12} | {'loss':>6} | {'accuracy':>8}")
    print(f"{'train':>12} | {train_losses[-1]:>6.4f} | {train_accs[-1]:>8.2%}")
    print(f"{'validation':>12} | {val_losses[-1]:>6.4f} | {val_accs[-1]:>8.2%}")
    print(f"{'test':>12} | {test_loss:>6.4f} | {test_acc:>8.2%}")
    print(f"\nTest: {n_wrong} of {len(Y_test)} images wrong")

    # Save the trained parameters so export_weights.py can use them without
    # retraining. Names follow the layers: W1/b1 is 784->128, W3/b3 is 64->10.
    params = {}
    for i, layer in enumerate(nn.layers, start=1):
        params[f"W{i}"] = layer.weights
        params[f"b{i}"] = layer.biases
    np.savez(PARAMS_PATH, **params)
    print(f"Saved trained parameters to {PARAMS_PATH}")

    probs = activations.softmax(nn.forward(X_test))
    predictions = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)  # how sure the network was of its answer
    matrix = confusion_matrix(predictions, Y_test)
    print("\nConfusion matrix (rows = true digit, columns = predicted digit)")
    print(matrix)

    # Every test image is counted exactly once, so the cells must add up to the
    # number of test images. If not, the loop skipped or double-counted something.
    assert matrix.sum() == len(Y_test), \
        f"matrix counts {matrix.sum()} images, expected {len(Y_test)}"

    # The off-diagonal cells are the mistakes, so they must match the number
    # of wrong images that accuracy() found.
    assert matrix.sum() - np.trace(matrix) == n_wrong, \
        f"matrix has {matrix.sum() - np.trace(matrix)} wrong, expected {n_wrong}"

    # Accuracy per digit: the diagonal over each row's total
    per_digit = np.diag(matrix) / matrix.sum(axis=1)
    print("\nAccuracy per digit: " +
          ", ".join(f"{d}: {acc:.1%}" for d, acc in enumerate(per_digit)))

    # The most common mistakes: the largest off-diagonal cells
    mistakes = matrix.copy()
    np.fill_diagonal(mistakes, 0)
    top = np.argsort(mistakes, axis=None)[::-1][:5]
    print("Most common mistakes: " + ", ".join(
        f"{t} -> {p} ({mistakes[t, p]})" for t, p in zip(*np.unravel_index(top, mistakes.shape))))

    path = os.path.join(FIGURE_DIR, "confusion_matrix.png")
    plot_confusion_matrix(matrix, path)
    print(f"\nSaved {path}")

    # Show a sample of the mistakes, to judge whether they are messy handwriting
    # (expected) or clear digits the network should have gotten right (a problem)
    wrong = np.where(predictions != Y_test)[0]
    path = os.path.join(FIGURE_DIR, "misclassified.png")
    plot_misclassified(X_test, Y_test, predictions, confidences, wrong[:20], path)
    print(f"Saved {path}")
