# Original Course Assignment (Reference Only)

Course: CS 370 - Deep Learning and Systems
Title: Building a Neural Network from Scratch: MNIST Digit Classification
Points: 100 (+10 bonus)

NOTE: This project is no longer being submitted for a grade. It is kept here as a
reference for the technical requirements, structure, and rubric only. The original
academic integrity clause restricting AI assistance does NOT apply, this is a personal
portfolio project now and full AI assistance (including Claude Code) is permitted at
every stage.

---

## 1. Overview
Implement a feedforward neural network entirely from first principles, no PyTorch,
TensorFlow, JAX, or any library that provides automatic differentiation or pre-built
layers. Write your own neuron, layers, forward pass, and backpropagation, and use the
network to classify handwritten digits from MNIST.

### 1.1 Learning Objectives
- Understand the mathematical model of an artificial neuron (weighted sum +
  nonlinearity) and implement it as vectorized code.
- Compose neurons into layers and layers into a network, using matrix operations
  instead of explicit loops.
- Derive and implement backpropagation using the chain rule, for hidden layers and the
  output/loss layer.
- Implement numerically stable sigmoid, softmax, and cross-entropy loss.
- Implement mini-batch stochastic gradient descent and train to convergence.
- Validate the implementation using numerical gradient checking.
- Evaluate and diagnose the trained model using accuracy curves, confusion matrices,
  and qualitative error analysis.

### 1.2 What You Are Given vs. What You Must Build
Allowed: NumPy, Matplotlib, and a data-loading utility for raw MNIST.
Not allowed: PyTorch, TensorFlow, Keras, JAX, scikit-learn's neural network classes,
autograd, or any package that computes gradients for you.

## 2. Background concepts to understand
- What a single artificial neuron computes, and why nonlinearity matters.
- Why we minimize a loss function via gradient descent, and what "gradient" means for
  a function of many variables.
- The chain rule, and why backpropagation is a systematic application of it through a
  computational graph.
- Why softmax + cross-entropy is used for multi-class classification, and why their
  combined gradient is simple.

## 3. Part 1: The Artificial Neuron
z = w^T x + b, a = phi(z)
- Implement a neuron function/class.
- Implement sigmoid and its derivative, numerically stable.
- Implement ReLU and its derivative.
- Unit test against a hand-computed example.
Deliverable: neuron.py

## 4. Part 2: Building the Network
Layer as matrix operation: Z = X @ W + b, A = phi(Z)
- Implement DenseLayer class (forward method).
- Careful weight initialization (He for ReLU, Xavier/Glorot for sigmoid), not zero.
- Implement NeuralNetwork class chaining DenseLayer objects from a list of layer
  sizes and activations.
- Implement softmax output layer, numerically stable (subtract max(z)).
- Verify forward pass produces a valid probability distribution over 10 classes.

## 5. Part 3: Loss Function
Cross-entropy loss: L = -sum(y_i * log(y_hat_i)), batch-averaged.
- Implement one-hot encoding for labels.
- Implement batch-averaged cross-entropy loss with epsilon for log(0) safety.
- Derive, on paper, the gradient of loss w.r.t. pre-softmax logits z for softmax +
  cross-entropy together; show it simplifies to y_hat - y. Include in report.

## 6. Part 4: Backpropagation
For Z = XW + b, A = phi(Z), given dA:
  dZ = dA * phi'(Z)          (elementwise)
  dW = X^T @ dZ / B
  db = sum(dZ, axis=0) / B
  dX = dZ @ W^T
Output layer dZ = (y_hat - y) / B (simplified combined softmax+cross-entropy
gradient).
- Implement backward(dOut) on DenseLayer, returning dX, storing dW/db.
- Implement backward(y_true) on NeuralNetwork (reverse order through all layers).
- Cache forward-pass values needed for backward (X, Z per layer); don't recompute.
- Implement gradient checking via centered finite differences:
  dL/dw ~= (L(w+eps) - L(w-eps)) / (2*eps), eps ~= 1e-5
  Target agreement: ~1e-5 relative error with analytical gradient.

## 7. Part 5: Training Loop
For each epoch: shuffle data, iterate mini-batches, forward pass, compute loss,
backward pass, update each layer's W and b via W -= lr * dW, b -= lr * db.
- Implement mini-batch construction with per-epoch shuffling.
- Implement plain SGD update (momentum/Adam optional bonus).
- Track training loss, validation loss, validation accuracy per epoch.
- Train at least one hidden layer to >= 90% test accuracy (floor); a tuned
  two-hidden-layer ReLU network should comfortably exceed 97%.
- Run at least 3 hyperparameter experiments (learning rate, hidden width, batch size)
  and briefly report observations.

## 8. Part 6: Data Pipeline
- Load MNIST (60,000 train / 10,000 test, 28x28 grayscale) as plain NumPy arrays via
  any standard loader; from that point your own code takes over.
- Flatten each image to a 784-dim vector.
- Normalize pixel values to [0, 1] (or standardize); explain why normalization
  matters for training stability.
- Split off a validation set (e.g. 5,000 of the 60,000 training images).
- Plot sample digits with labels to confirm correct loading.

## 9. Part 7: Evaluation & Analysis
- Report final test-set accuracy (evaluated once, after tuning is complete).
- Plot training/validation loss vs. epoch and training/validation accuracy vs. epoch.
- Compute and visualize a 10x10 confusion matrix on the test set.
- Display at least 8 misclassified test images with true/predicted labels; comment on
  patterns (e.g. commonly confused digit pairs).

## 10. Suggested Program Structure
| Component | Required interface |
|---|---|
| Dense layer | DenseLayer: __init__(self, n_in, n_out); forward(self, X); backward(self, dOut); attributes W, b, dW, db |
| Activations | class Activation: forward(self, Z); backward(self, dA) -- Sigmoid, ReLU, Softmax |
| Network | NeuralNetwork: __init__(self, layer_sizes, activations); forward(self, X); backward(self, y_true); update(self, lr) |
| Loss | cross_entropy_loss(y_pred, y_true) -> scalar; cross_entropy_grad(y_pred, y_true) -> array |
| Trainer | train(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr) -> history dict |
| Gradient check | numerical_gradient_check(model, X_sample, y_sample, epsilon=1e-5) -> max relative error |

Suggested file layout:
```
mnist_from_scratch/
  data.py
  activations.py
  layers.py
  network.py
  losses.py
  train.py
  gradient_check.py
  evaluate.py
  main.py
  report.pdf
```

## 11. Suggested Timeline (5 weeks, kept as a pacing reference)
- Week 1: Data pipeline working (loaded, normalized, split, visualized). Single
  neuron + sigmoid implemented and unit-tested.
- Week 2: Full forward pass for a multi-layer network. Cross-entropy loss implemented
  and verified.
- Week 3: Backpropagation implemented for all layers. Gradient checking passes
  (< 1e-5 relative error).
- Week 4: Full training loop working end-to-end; loss decreasing, validation accuracy
  tracked. Hyperparameter tuning.
- Week 5: Final evaluation, plots, error analysis, written report, code cleanup.

## 12. Deliverables (kept as a personal quality bar)
- All source code, well-organized, documented, runnable end-to-end via a single entry
  point (python main.py).
- A short written report (3-5 pages): architecture and hyperparameter choices, the
  softmax + cross-entropy gradient derivation, gradient-checking results,
  training/validation curves, final test accuracy, confusion matrix, error analysis,
  and a reflection on what backpropagation "really is."
- A README with instructions to reproduce results.

## 13. Original Grading Rubric (kept as a personal quality checklist, not graded)
| Component | Criteria | Points |
|---|---|---|
| Neuron & activation functions | Correct forward computation; sigmoid, ReLU, softmax stable | 10 |
| Network architecture | Configurable fully-connected layers; correct forward pass; parameter initialization | 10 |
| Loss function | Cross-entropy with softmax; numerically stable (log-sum-exp) | 10 |
| Backpropagation | Gradients derived and implemented correctly for every layer/parameter | 25 |
| Gradient checking | Numerical gradient check included and passes within tolerance | 10 |
| Training loop | Mini-batch SGD, shuffling, LR handling, epoch loop, tracking | 10 |
| Data pipeline | Correct loading, normalization, train/val/test split | 5 |
| Evaluation & plots | Test accuracy, loss/accuracy curves, confusion matrix, misclassified examples | 10 |
| Written report | Clear explanation of design choices, derivations, results | 10 |
| Code quality | Readable, modular, documented, fully vectorized (no per-pixel loops) | 10 |
| **Total** | | **110*** |

*10 bonus points available, see Part 14.

## 14. Bonus Extensions (kept as optional stretch goals)
Choose one or combine smaller ones:
- Momentum or Adam optimizer from scratch, compared against plain SGD.
- L2 weight regularization or dropout (implemented by hand, including backward pass).
- A simple convolutional layer from scratch (forward and backward), compared against
  the fully-connected network.
- Batch normalization from scratch, including backward pass.
- Visualize what the first hidden layer learned (reshape rows of W1 back into 28x28
  images).

## 15. Academic Integrity (ORIGINAL CLAUSE - NO LONGER APPLICABLE)
The original spec restricted collaboration and prohibited AI-generated code, since
this was meant to be submitted for a grade. This project is NOT being submitted. Full
AI assistance, including Claude Code, is permitted and expected throughout.
