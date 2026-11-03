# MNIST Neural Network from Scratch

A handwritten digit classifier built with only Python and NumPy. No machine learning frameworks and no automatic differentiation. I'm writing every part by hand so I understand how a neural network actually learns.

## What works so far

- Data pipeline: loads MNIST, flattens and normalizes images, and splits the data (`data.py`)
- Single neuron with sigmoid and ReLU, vectorized and checked by hand (`neuron.py`)
- Numerically stable softmax with self-tests
- Layers and a full forward pass using He initialization (`layers.py`, `network.py`)
- Cross-entropy loss using log-softmax, stable for extreme values (`losses.py`)
- Backpropagation through every layer, derived by hand with the chain rule (`layers.py`, `network.py`)
- Gradient checking with centered finite differences: backprop matches the numerical gradients to a relative error of about 1e-9 (`gradient_check.py`)
- Mini-batch SGD training with per-epoch shuffling, tracking loss and train/validation accuracy (`train.py`)

## Results so far

A `[784, 128, 64, 10]` ReLU network trained for 5 epochs (batch size 64, learning rate 0.1) reaches **97.1% validation accuracy**:

| Epoch | Loss | Train accuracy | Validation accuracy |
|---|---|---|---|
| 1 | 0.3404 | 92.76% | 92.30% |
| 2 | 0.1553 | 95.03% | 94.30% |
| 3 | 0.1111 | 96.69% | 96.02% |
| 4 | 0.0869 | 97.59% | 96.54% |
| 5 | 0.0698 | 98.35% | 97.14% |

The test set has not been used yet. It is held back for a single final evaluation.

## A note on torchvision

`data.py` uses `torchvision` for one thing only: downloading the raw MNIST files. The data is converted to plain NumPy arrays immediately, and PyTorch is not used anywhere else. The model, forward pass, backpropagation, loss, and training are all written in NumPy.

## Next steps

- Hyperparameter experiments (learning rate, hidden layer width, batch size)
- Final test accuracy, loss and accuracy curves, a confusion matrix, and a look at misclassified digits
- A PyTorch and CNN version to compare against

## How I use AI

I use Claude as a coding assistant on this project. I review and test its output, and I verify the math by hand.
