# Project Context for Claude Code

## Who I am
I'm Don, a Computer Science BS graduate from Queens College (CUNY). This project was
originally assigned as coursework (CS 370, Deep Learning and Systems), but I am NOT
submitting it for a grade. I am building it purely for my own job application
portfolio, since I graduated without a real project portfolio to show recruiters.

## Project status: not a graded submission
IMPORTANT: This is not being turned in. The academic integrity restriction in the
original spec (no AI-generated code, no from-scratch implementations found online)
does NOT apply here. Full AI assistance, including help from Claude Code, is fine and
expected at every stage. Treat this as a personal learning + portfolio project that
happens to be based on a real course spec, not as an academic submission with
integrity constraints.

## What this project is
A handwritten digit classifier trained on the MNIST dataset, implemented from first
principles in NumPy: my own neuron/layer math, my own forward pass, and my own
backpropagation, with no autograd or pre-built layers (see "Library constraints"
below). The goal is a portfolio piece that demonstrates a real, deep understanding of
how neural networks work under the hood, not just the ability to call a framework.

A PyTorch-based version (using torchvision, autograd, and eventually a CNN) may be
built afterward as a follow-up "part 2" project, once the from-scratch version is
done, to show both fundamentals and framework fluency. See "Future extension" below.

## Library constraints (from the original spec, still followed for learning value)
- Allowed: NumPy for array/matrix operations, Matplotlib for plotting, and a minimal
  loader for the raw MNIST files.
- Not allowed for this phase: PyTorch, TensorFlow, Keras, JAX, scikit-learn's neural
  network classes, autograd, or any library that computes gradients automatically.
  These constraints exist purely to maximize what I learn about backpropagation, not
  because of any external submission rule.

## Current skill level / working style
- Comfortable with basic Python.
- Coursework theory background on neural nets, limited hands-on implementation
  experience before this project.
- New to Git and GitHub workflows, still building good habits.
- Preference: explain the "why" behind suggestions, not just the "what." I'm using
  this project to actually learn, not just to get working code.

## Learning objectives (from the original spec)
- Understand the mathematical model of an artificial neuron and implement it as
  vectorized code.
- Compose neurons into layers and layers into a network using matrix operations.
- Derive and implement backpropagation via the chain rule, for hidden layers and the
  output/loss layer.
- Implement numerically stable sigmoid, softmax, and cross-entropy loss.
- Implement mini-batch stochastic gradient descent and train to convergence.
- Validate the implementation using numerical gradient checking.
- Evaluate and diagnose the trained model with accuracy curves, confusion matrices,
  and qualitative error analysis.

## Build plan (in order, based on the original 5-week course timeline)
1. Data pipeline: load raw MNIST, normalize, split into train/val/test, visualize
   sample digits.
2. Single neuron + sigmoid/ReLU activations, implemented and unit-tested on toy data.
3. DenseLayer class and full forward pass for a multi-layer network (arbitrary layer
   sizes), ending in a softmax output layer.
4. Cross-entropy loss (numerically stable), verified against a reference calculation.
5. Backpropagation for every layer, including the simplified softmax + cross-entropy
   combined gradient (dZ = (y_hat - y) / B for the output layer).
6. Gradient checking via centered finite differences (target: < 1e-5 relative error).
7. Mini-batch SGD training loop with shuffling, loss/accuracy tracking per epoch.
8. Hyperparameter experiments (learning rate, hidden layer width, batch size).
9. Evaluation: final test accuracy, loss/accuracy curves, confusion matrix,
   misclassified example analysis.
10. Clean up code, write the README and short report, push to GitHub.

## Suggested file layout (from the original spec)
```
mnist_from_scratch/
  data.py          # loading, normalization, batching
  activations.py   # sigmoid, relu, softmax + derivatives
  layers.py        # DenseLayer
  network.py       # NeuralNetwork (forward, backward, predict)
  losses.py        # cross_entropy_loss, cross_entropy_grad
  train.py         # training loop, mini-batching, SGD update
  gradient_check.py
  evaluate.py      # accuracy, confusion matrix, plots
  main.py          # ties it all together
```

## Target results
- Floor: at least 90% test accuracy.
- Realistic target: 97%+ with a tuned two-hidden-layer ReLU network.

## Future extension (after the from-scratch version is complete)
- Rebuild the same problem in PyTorch, including a CNN (Conv2d layers), to push
  accuracy toward ~99% and demonstrate framework fluency as a "part 2" project.
- Optionally wrap the PyTorch version in a small web demo (Flask/FastAPI) with a
  draw-a-digit interface for live predictions.

## Environment
- OS: Windows
- Editor: VS Code
- Python environment: venv (see /venv in project root)
- Version control: Git, remote hosted on GitHub

## Working habits I'm trying to build (please help enforce these)
- Commit locally in small, logical chunks (e.g. "data pipeline working," "gradient
  checking passes") rather than one giant commit at the end.
- Write descriptive commit messages that explain what changed and why, not just
  "update" or "fix."
- Push to GitHub once a chunk of work is actually functional, not after every tiny
  edit.
- Get working code first, then optimize/refactor.
- Document as I go rather than leaving it all to the end.

## Known environment quirks
- Installed on Python 3.14, which is very new; if a package install ever fails with a
  "no matching distribution" error, that's likely why. A 3.11 or 3.12 install
  alongside 3.14 is the fallback.
