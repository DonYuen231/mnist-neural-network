import torchvision


train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True)
X_train = train_dataset.data.numpy()
Y_train = train_dataset.targets.numpy()
X_test = test_dataset.data.numpy()
Y_test = test_dataset.targets.numpy()
print(X_train.shape, X_test.shape)
print(X_train.dtype, X_test.dtype)
print(Y_train.shape, Y_test.shape)
print(Y_train.dtype, Y_test.dtype)
