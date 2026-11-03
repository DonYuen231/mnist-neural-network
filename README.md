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
- Hyperparameter sweeps over learning rate and hidden layer width, each repeated across 3 seeds (`experiments.py`)

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

## Hyperparameter experiments

`experiments.py` changes one hyperparameter at a time, keeping the rest at the baseline above, and compares settings on validation accuracy. Each setting is trained with 3 different seeds (different starting weights and shuffle order). The spread across seeds shows how much a score moves by chance alone, so a difference smaller than that spread doesn't count as evidence. All runs: 5 epochs.

**Learning rate** (sizes [784, 128, 64, 10], batch 64):

| Learning rate | Mean val accuracy | Range across seeds |
|---|---|---|
| 0.001 | 85.93% | 85.19% to 87.23% |
| 0.01 | 93.91% | 93.64% to 94.22% |
| **0.1** | **97.11%** | 96.94% to 97.20% |
| 0.5 | 96.57% | 95.34% to 97.34% |
| 1.0 | 61.79% | 12.10% to 93.24% |

Too small a learning rate is just slow: 0.001 is still improving after 5 epochs. Too large is unreliable: at 1.0, one seed collapsed to 12% (no better than guessing) while another reached 93%. With a single seed, 1.0 looked fine (95.6%) and 0.5 looked tied with 0.1. Running several seeds showed that both are unstable. 0.1 is both the best and the most consistent.

**Hidden layer width** (learning rate 0.1, batch 64):

| Sizes | Parameters | Mean val accuracy | Range across seeds | Train minus val | Time per run |
|---|---|---|---|---|---|
| [784, 32, 10] | ~25k | 95.59% | 95.48% to 95.72% | 0.53% | 1.9s |
| **[784, 128, 64, 10]** | ~109k | 97.11% | 96.94% to 97.20% | 1.30% | 4.7s |
| [784, 256, 128, 10] | ~235k | 97.29% | 97.06% to 97.42% | 1.46% | 9.6s |

The largest network scores 0.18% higher on average, but its range overlaps the baseline's, and it takes twice as long to train. The gap between train and validation accuracy grows with network size, which is consistent with bigger networks starting to memorize the training set. I kept [784, 128, 64, 10].

## A note on torchvision

`data.py` uses `torchvision` for one thing only: downloading the raw MNIST files. The data is converted to plain NumPy arrays immediately, and PyTorch is not used anywhere else. The model, forward pass, backpropagation, loss, and training are all written in NumPy.

## Next steps

- More hyperparameter experiments (batch size, number of epochs) and validation loss curves to show where overfitting starts
- Final test accuracy, loss and accuracy curves, a confusion matrix, and a look at misclassified digits
- A PyTorch and CNN version to compare against

## How I use AI

I use Claude as a coding assistant on this project. I review and test its output, and I verify the math by hand.
