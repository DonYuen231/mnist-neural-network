# Project Requirements: MNIST Classifier (From Scratch)

## Status
Originally a CS 370 course assignment. Being built as a personal portfolio project
only, NOT being submitted for a grade. The original academic integrity restriction on
AI assistance does not apply, this checklist and all code may be built with full AI
help.

## Library constraints
- Allowed: NumPy, Matplotlib, a minimal raw-MNIST loader.
- Not used in this phase: PyTorch, TensorFlow, JAX, scikit-learn's NN classes, or any
  autograd library. (A PyTorch/CNN version is a planned follow-up project, not part of
  this checklist.)

## Checklist (derived from the original spec's tasks + rubric)

### Data pipeline
- [ ] Load raw MNIST (60,000 train / 10,000 test, 28x28 grayscale)
- [ ] Flatten images to 784-dim vectors
- [ ] Normalize pixel values to [0, 1] (or standardize)
- [ ] Split off a validation set (e.g. 5,000 images) from training data
- [ ] Plot sample digits with labels to confirm correct loading

### Neuron & activations
- [ ] Implement a single neuron (weighted sum + nonlinearity)
- [ ] Implement sigmoid + derivative, numerically stable
- [ ] Implement ReLU + derivative
- [ ] Unit test neuron output against a hand-computed example

### Network architecture
- [ ] Implement DenseLayer class (forward + backward, stores W, b, dW, db)
- [ ] Use proper weight initialization (He for ReLU, Xavier/Glorot for sigmoid), not
      zero-init
- [ ] Implement NeuralNetwork class chaining arbitrary layer sizes
- [ ] Implement softmax output layer with numerically stable computation
      (subtract max(z))
- [ ] Verify forward pass produces a valid probability distribution over 10 classes

### Loss function
- [ ] Implement one-hot encoding for labels
- [ ] Implement batch-averaged cross-entropy loss (with epsilon to avoid log(0))
- [ ] Derive the softmax + cross-entropy combined gradient on paper (simplifies to
      y_hat - y), include in report

### Backpropagation
- [ ] Implement backward() on DenseLayer
- [ ] Implement backward() on NeuralNetwork (reverse order through all layers)
- [ ] Cache forward-pass values needed for backward (X, Z per layer)
- [ ] Implement numerical gradient checking (centered finite differences)
- [ ] Confirm gradient check passes within ~1e-5 relative error

### Training loop
- [ ] Implement mini-batch construction with per-epoch shuffling
- [ ] Implement plain SGD parameter update
- [ ] Track training loss, validation loss, validation accuracy per epoch
- [ ] Train to at least 90% test accuracy (floor)
- [ ] Target 97%+ test accuracy with a tuned two-hidden-layer ReLU network
- [ ] Run at least 3 hyperparameter experiments (learning rate, hidden width, batch
      size) and note observations

### Evaluation & analysis
- [ ] Report final test accuracy (measured once, after tuning is done)
- [ ] Plot train/val loss and accuracy curves vs. epoch
- [ ] Compute and visualize a 10x10 confusion matrix
- [ ] Show at least 8 misclassified examples with true/predicted labels and brief
      commentary

### Documentation
- [ ] Write a short report covering architecture, hyperparameters, the softmax +
      cross-entropy derivation, gradient-check results, curves, final accuracy,
      confusion matrix, and a reflection on backpropagation
- [ ] Write a README with setup and reproduction instructions
- [ ] Clean, modular, documented, fully vectorized code (no per-pixel Python loops)

## Stretch goals (optional, from the original bonus section)
- [ ] Momentum or Adam optimizer from scratch, compared against plain SGD
- [ ] L2 regularization or dropout implemented by hand
- [ ] A from-scratch convolutional layer, compared against the fully-connected version
- [ ] Batch normalization from scratch
- [ ] Visualize learned first-layer features (reshape W1 rows back into 28x28 images)

## Planned follow-up project (separate from this checklist)
- [ ] PyTorch reimplementation, including a CNN, targeting ~99% accuracy
- [ ] Optional web demo (Flask/FastAPI) with a draw-a-digit interface

See PROJECT_SPEC_ORIGINAL.md for the full original assignment text this checklist was
derived from.
