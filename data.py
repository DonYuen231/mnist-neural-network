"""
MNIST data pipeline: download, flatten and scale to [0, 1], split off a
validation set, and plot sample digits.

torchvision is used only to download the raw files; everything after that is
plain NumPy.
"""
import matplotlib.pyplot as plt
import numpy as np
import torchvision


def load_mnist():
    """
    Download MNIST (if needed) and return it as plain NumPy arrays.

    torchvision is used only to fetch the raw files into ./data. The result is
    converted to NumPy right away, so nothing past this function uses torch.

    Returns:
    tuple: (X_train, Y_train, X_test, Y_test)
        X_train (numpy.ndarray): uint8 images, shape (60000, 28, 28), values 0-255.
        Y_train (numpy.ndarray): int64 labels 0-9, shape (60000,).
        X_test (numpy.ndarray): uint8 images, shape (10000, 28, 28).
        Y_test (numpy.ndarray): int64 labels 0-9, shape (10000,).
    """
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    X_train = train_dataset.data.numpy()
    Y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy()
    Y_test = test_dataset.targets.numpy()
    return X_train, Y_train, X_test, Y_test


def verify_data(X_train, Y_train, X_test, Y_test):
    """
    Print shapes, dtypes, and pixel min/max as a quick sanity check.

    Call it before and after preprocess_data() to confirm the images went from
    (N, 28, 28) uint8 in 0-255 to (N, 784) float32 in 0-1.

    Parameters:
    X_train, Y_train, X_test, Y_test (numpy.ndarray): The dataset arrays.

    Returns:
    None. Only prints.
    """
    print(X_train.shape, X_test.shape)
    print(X_train.dtype, X_test.dtype)
    print(Y_train.shape, Y_test.shape)
    print(Y_train.dtype, Y_test.dtype)
    print(X_train.min(), X_train.max())
    print(X_test.min(), X_test.max())


def preprocess_data(X_train, X_test):
    """
    Flatten each image to a row of 784 pixels and scale pixels to [0, 1].

    Parameters:
    X_train (numpy.ndarray): uint8 images, shape (N, 28, 28).
    X_test (numpy.ndarray): uint8 images, shape (M, 28, 28).

    Returns:
    tuple: (X_train, X_test) as float32 arrays of shape (N, 784) and (M, 784).
    """
    X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0
    return X_train, X_test


def split_data(X, Y, train_ratio=0.9167):
    """
    Shuffle the samples and split them into a training set and a validation set.

    Uses a fixed seed (42), so the split is the same on every run. The default
    ratio keeps about 55,000 of the 60,000 training images for training and
    holds out about 5,000 for validation.

    Parameters:
    X (numpy.ndarray): Inputs, shape (N, ...).
    Y (numpy.ndarray): Labels, shape (N,).
    train_ratio (float): Fraction of samples that go to the training set.

    Returns:
    tuple: (X_train, Y_train, X_val, Y_val)
    """
    num_samples = X.shape[0]
    num_train = int(num_samples * train_ratio)
    rng = np.random.default_rng(42)
    indices = rng.permutation(num_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    X_train, Y_train = X[train_indices], Y[train_indices]
    X_val, Y_val = X[val_indices], Y[val_indices]
    return X_train, Y_train, X_val, Y_val


def plot_samples(X, Y, n=10, save_path='sample_digits.png'):
    """
    Save a row of the first n images, each titled with its label.

    Used to check by eye that the labels match the digits.

    Parameters:
    X (numpy.ndarray): Flattened images, shape (N, 784).
    Y (numpy.ndarray): Labels, shape (N,).
    n (int): Number of images to show.
    save_path (str): Where to save the PNG.

    Returns:
    None. Writes the image file.
    """
    fig, axes = plt.subplots(1, n, figsize=(n, 1.5))
    for i in range(n):
        axes[i].imshow(X[i].reshape(28, 28), cmap='gray')
        axes[i].set_title(str(Y[i]))
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved sample plot to {save_path}")


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_mnist()
    verify_data(X_train, Y_train, X_test, Y_test)
    X_train, X_test = preprocess_data(X_train, X_test)
    verify_data(X_train, Y_train, X_test, Y_test)
    X_train, Y_train, X_val, Y_val = split_data(X_train, Y_train)
    print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)
    plot_samples(X_train, Y_train)
