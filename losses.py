import numpy as np
import neuron

def cross_entropy_loss(logits, y):
    """
    Compute the mean cross-entropy loss directly from logits, using a
    numerically stable log-softmax (never takes log of 0).

    Parameters:
    logits (numpy.ndarray): Raw scores from the network, shape (B, C).
    y (numpy.ndarray): Integer class labels, shape (B,).

    Returns:
    float: Cross-entropy loss averaged over the batch.
    """
    logits_shifted = np.subtract(logits, np.max(logits, axis=1, keepdims=True))
    log_sum_exp = np.log(np.sum(np.exp(logits_shifted), axis=1, keepdims=True))
    log_probs = logits_shifted - log_sum_exp
    correct_log_probs = log_probs[np.arange(logits.shape[0]), y]
    loss = -np.mean(correct_log_probs)
    return loss

def cross_entropy_grad(logits, y):
    """
    Compute the gradient of the cross-entropy loss with respect to the logits.

    Parameters:
    logits (numpy.ndarray): Raw scores from the network, shape (B, C).
    y (numpy.ndarray): Integer class labels, shape (B,).

    Returns:
    numpy.ndarray: Gradient of the loss with respect to the logits, shape (B, C).
    """
    probs = neuron.softmax(logits)  # softmax returns a new array, so logits is untouched
    B = logits.shape[0]
    probs[np.arange(B), y] -= 1     # probs - onehot: subtract 1 only at each correct label
    grad = probs / B                # loss is a mean over the batch, so scale by 1/B
    return grad

if __name__ == "__main__":

    # Hand-computed worked example (see the step-by-step table):
    # row 0: log_probs [-0.408, -1.408, -2.408], label 0 -> -0.408
    # row 1: log_probs [-3.095, -3.095, -0.095], label 2 -> -0.095
    # loss = -mean(-0.408, -0.095) ~= 0.2513
    logits = np.array([[2.0, 1.0, 0.0],
                       [0.0, 0.0, 3.0]])
    y = np.array([0, 2])
    loss = cross_entropy_loss(logits, y)
    assert np.isclose(loss, 0.25125, atol=1e-4), f"expected ~0.2513, got {loss}"
    print(f"worked example: {loss:.4f} (expected ~0.2513) OK")

    # Untrained-network sanity value: equal logits -> each class gets 0.1,
    # so loss = -log(0.1) ~= 2.3026 regardless of the labels
    loss_uniform = cross_entropy_loss(np.zeros((4, 10)), np.array([0, 3, 7, 9]))
    assert np.isclose(loss_uniform, -np.log(0.1)), f"expected ~2.3026, got {loss_uniform}"
    print(f"uniform logits: {loss_uniform:.4f} (expected ~2.3026) OK")

    # Stability: softmax would give p_correct = exactly 0 here and log(0) = -inf.
    # Log-softmax never forms that 0: log_prob for class 1 is (0 - 1000) - log(1) = -1000
    loss_big = cross_entropy_loss(np.array([[1000.0, 0.0]]), np.array([1]))
    assert np.isfinite(loss_big), f"loss is not finite: {loss_big}"
    assert np.isclose(loss_big, 1000.0), f"expected ~1000, got {loss_big}"
    print(f"stability [[1000, 0]], label 1: {loss_big} (finite, ~1000) OK")

    # Cross-check against the naive formula on ordinary-sized logits, where
    # softmax -> log is safe. Both should compute the same thing.
    rng = np.random.default_rng(0)
    B = 8
    logits_rand = rng.standard_normal((B, 10)) * 3
    y_rand = rng.integers(0, 10, size=B)
    naive = -np.mean(np.log(neuron.softmax(logits_rand)[np.arange(B), y_rand]))
    ours = cross_entropy_loss(logits_rand, y_rand)
    assert np.isclose(ours, naive), f"log-softmax {ours} != naive {naive}"
    print(f"matches naive softmax -> log: {ours:.6f} vs {naive:.6f} OK")

    # Gradient, worked example: logits [2, 1, 0], label 0
    # probs [0.665, 0.245, 0.090] - onehot [1, 0, 0] = [-0.335, 0.245, 0.090], / B (B = 1)
    grad = cross_entropy_grad(np.array([[2.0, 1.0, 0.0]]), np.array([0]))
    assert np.allclose(grad, [[-0.33476, 0.24473, 0.09003]], atol=1e-4), f"got {grad}"
    print(f"grad worked example: {grad.round(3)} OK")

    # Shape matches logits, and every row sums to 0 (probs sum to 1, onehot sums to 1)
    grad_rand = cross_entropy_grad(logits_rand, y_rand)
    assert grad_rand.shape == logits_rand.shape, f"got shape {grad_rand.shape}"
    assert np.allclose(grad_rand.sum(axis=1), 0), f"row sums {grad_rand.sum(axis=1)}"
    print("grad shape matches logits, rows sum to 0 OK")

    # Batch scaling: the same image repeated B times gives each row 1/B of the B=1 gradient
    grad_batch = cross_entropy_grad(np.array([[2.0, 1.0, 0.0]] * 4), np.array([0] * 4))
    assert np.allclose(grad_batch, grad / 4), "batch gradient not scaled by 1/B"
    print("grad scaled by 1/B OK")

    # The function must not modify the caller's logits
    logits_copy = logits_rand.copy()
    cross_entropy_grad(logits_rand, y_rand)
    assert np.array_equal(logits_rand, logits_copy), "grad function modified its input"
    print("grad leaves logits unchanged OK")

    print("\nAll losses.py checks passed.")
