# MNIST Neural Network from Scratch

A handwritten digit classifier built with only Python and NumPy. No machine learning frameworks and no automatic differentiation. I'm writing every part by hand so I understand how a neural network actually learns.

## What works so far

- Data pipeline: loads MNIST, flattens and normalizes images, and splits the data (`data.py`)
- Single neuron with sigmoid and ReLU, vectorized and checked by hand (`neuron.py`)
- Numerically stable softmax with self-tests
- Layers and a full forward pass using He initialization (`layers.py`, `network.py`)
- Cross-entropy loss using log-softmax, stable for extreme values (`losses.py`)

## Next steps

- Backpropagation, written and derived by hand
- Gradient checking with finite differences
- Mini-batch training and a test accuracy result
- A PyTorch and CNN version to compare against

## How I use AI

I use Claude as a coding assistant on this project. I review and test its output, and I verify the math by hand.
