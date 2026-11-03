# Project Requirements: MNIST Classifier (From Scratch)

## Library constraints
- Allowed: NumPy, Matplotlib, a minimal raw-MNIST loader.
- Not used in this phase: PyTorch, TensorFlow, JAX, scikit-learn's NN classes, or any
  autograd library. (A PyTorch/CNN version is a planned follow-up project, not part of
  this checklist.)

## Checklist

### Data pipeline
- [x] Load raw MNIST (60,000 train / 10,000 test, 28x28 grayscale)
- [x] Flatten images to 784-dim vectors
- [x] Normalize pixel values to [0, 1] (or standardize)
- [x] Split off a validation set (e.g. 5,000 images) from training data (4,998 images)
- [x] Plot sample digits with labels to confirm correct loading

### Neuron & activations
- [x] Implement a single neuron (weighted sum + nonlinearity)
- [x] Implement sigmoid + derivative, numerically stable
- [x] Implement ReLU + derivative
- [x] Unit test neuron output against a hand-computed example

### Network architecture
- [x] Implement DenseLayer class (forward + backward, stores W, b, dW, db)
- [x] Use proper weight initialization (He for ReLU, Xavier/Glorot for sigmoid), not
      zero-init (He; the network only uses ReLU hidden layers)
- [x] Implement NeuralNetwork class chaining arbitrary layer sizes
- [x] Implement softmax output layer with numerically stable computation
      (subtract max(z))
- [x] Verify forward pass produces a valid probability distribution over 10 classes

### Loss function
- [x] Implement one-hot encoding for labels (done with integer-label indexing,
      `probs[np.arange(B), y]`, which is equivalent and avoids building the matrix)
- [x] Implement batch-averaged cross-entropy loss (with epsilon to avoid log(0))
      (uses log-softmax instead of an epsilon, so it never takes log(0))
- [ ] Derive the softmax + cross-entropy combined gradient on paper (simplifies to
      y_hat - y), include in report

### Backpropagation
- [x] Implement backward() on DenseLayer
- [x] Implement backward() on NeuralNetwork (reverse order through all layers)
- [x] Cache forward-pass values needed for backward (X, Z per layer)
- [x] Implement numerical gradient checking (centered finite differences)
- [x] Confirm gradient check passes within ~1e-5 relative error (worst: ~1e-9)

### Training loop
- [x] Implement mini-batch construction with per-epoch shuffling
- [x] Implement plain SGD parameter update
- [x] Track training loss, validation loss, validation accuracy per epoch
- [x] Train to at least 90% test accuracy (floor)
- [x] Target 97%+ test accuracy with a tuned two-hidden-layer ReLU network (98.32%)
- [x] Run at least 3 hyperparameter experiments (learning rate, hidden width, batch
      size) and note observations

### Evaluation & analysis
- [x] Report final test accuracy (measured once, after tuning is done)
- [x] Plot train/val loss and accuracy curves vs. epoch
- [x] Compute and visualize a 10x10 confusion matrix
- [x] Show at least 8 misclassified examples with true/predicted labels and brief
      commentary

### Documentation
- [ ] Write a short report covering architecture, hyperparameters, the softmax +
      cross-entropy derivation, gradient-check results, curves, final accuracy,
      confusion matrix, and a reflection on backpropagation
- [x] Write a README with setup and reproduction instructions
- [x] Clean, modular, documented, fully vectorized code (no per-pixel Python loops)

## Stretch goals (optional)
- [ ] Momentum or Adam optimizer from scratch, compared against plain SGD
- [ ] L2 regularization or dropout implemented by hand
- [ ] A from-scratch convolutional layer, compared against the fully-connected version
- [ ] Batch normalization from scratch
- [ ] Visualize learned first-layer features (reshape W1 rows back into 28x28 images)

## Planned follow-up project (separate from this checklist)
- [ ] PyTorch reimplementation, including a CNN, targeting ~99% accuracy
- [ ] Optional web demo (Flask/FastAPI) with a draw-a-digit interface
