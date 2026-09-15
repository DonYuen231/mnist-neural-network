import torchvision
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset using torchvision and convert the data to numpy arrays for further processing. The training and testing datasets are downloaded and stored in the './data' directory. The shapes and data types of the training and testing datasets are printed to verify the data integrity.
def load_mnist():
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    X_train = train_dataset.data.numpy()
    Y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy()
    Y_test = test_dataset.targets.numpy()
    return X_train, Y_train, X_test, Y_test

def verify_data(X_train, Y_train, X_test, Y_test):
    print(X_train.shape, X_test.shape)
    print(X_train.dtype, X_test.dtype)
    print(Y_train.shape, Y_test.shape)
    print(Y_train.dtype, Y_test.dtype)
    print(X_train.min(), X_train.max())
    print(X_test.min(), X_test.max())



def preprocess_data(X_train, X_test):
    X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0
    return X_train, X_test
def split_data(X, Y, train_ratio=0.9167):
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