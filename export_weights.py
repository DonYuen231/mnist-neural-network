"""
Export the trained network for the portfolio website, which runs the same
network in TypeScript in the browser.

Reads model_params.npz (written by evaluate.py) and writes three files to export/:

  weights.bin    W1, b1, W2, b2, W3, b3 as little-endian float32, row-major,
                 in that order, with the shapes listed in model.json.
  model.json     Architecture, hyperparameters, normalization, and the layout
                 of weights.bin.
  fixtures.json  Values computed in Python that the site's tests compare its own
                 forward pass, backward pass, and SGD step against.

The file stores float32 values, but the training math is float64. So every
number in model.json and fixtures.json is computed from the float32 weights
(read back into float64) using float64 math. A float64 TypeScript port that
loads weights.bin should then match the fixtures to about 1e-12.

Run after evaluate.py: python export_weights.py
"""
import datetime
import json
import os
import subprocess

import numpy as np

import activations
import data
import network
from evaluate import BATCH_SIZE, LEARNING_RATE, PARAMS_PATH, SIZES
from losses import cross_entropy_loss
from train import accuracy

EXPORT_DIR = "export"
PARAM_NAMES = ["W1", "b1", "W2", "b2", "W3", "b3"]

# How many gradient values to sample per parameter (200 in total). b3 has
# only 10 entries, so all of them are included.
GRAD_SAMPLES = {"W1": 70, "b1": 30, "W2": 40, "b2": 20, "W3": 30, "b3": 10}
SAMPLE_SEED = 0

NORMALIZATION = ("x / 255.0, range [0,1]; 28x28 image flattened row-major "
                 "(row 0 left to right, then row 1, ...); 0 = background (black), "
                 "1 = ink (white)")


def load_float32_params(path):
    """
    Load the saved parameters, rounded to float32 the way weights.bin stores them.

    Biases are saved with shape (1, n); they are flattened to (n,) here, which
    is the shape model.json lists.

    Returns:
    dict: name -> float32 array, in PARAM_NAMES order.
    """
    saved = np.load(path)
    params = {}
    for name in PARAM_NAMES:
        array = saved[name].astype("<f4")
        params[name] = array.reshape(-1) if name.startswith("b") else array
    return params


def build_network(params):
    """
    Build a NeuralNetwork that holds the float32 weights as float64 arrays.

    Uses the same NeuralNetwork and DenseLayer code that training used, so the
    forward and backward passes are exactly the trained model's.
    """
    nn = network.NeuralNetwork(SIZES)
    for i, layer in enumerate(nn.layers, start=1):
        layer.weights = params[f"W{i}"].astype(np.float64)
        layer.biases = params[f"b{i}"].astype(np.float64).reshape(1, -1)
    return nn


def git_version():
    """Today's date plus the short git SHA, e.g. '2026-11-09-c8ca17c'."""
    today = datetime.date.today().isoformat()
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        return f"{today}-{sha}"
    except (OSError, subprocess.CalledProcessError):
        return today


def pixel_list(x):
    """
    One image's 784 pixels as JSON-friendly floats.

    The network sees float32 values (data.py casts before dividing), so each
    value is written as the float32 number, read as a float64.
    """
    return [float(v) for v in np.asarray(x, dtype=np.float32)]


def forward_fixtures(nn, X_test, Y_test):
    """The first 2 test images of each digit, with Python's softmax outputs."""
    fixtures = []
    for digit in range(10):
        for index in np.where(Y_test == digit)[0][:2]:
            x = X_test[index:index + 1].astype(np.float64)
            probs = activations.softmax(nn.forward(x))[0]
            fixtures.append({
                "index": int(index),
                "label": int(Y_test[index]),
                "pixels": pixel_list(X_test[index]),
                "probs": [float(p) for p in probs],
            })
    return fixtures


def sample_gradients(grads, rng):
    """
    Pick GRAD_SAMPLES[name] flat indices from each gradient array.

    Most input pixels are 0 (background), and a 0 pixel always gets a 0 weight
    gradient. So a plain random sample of W1 would be almost all zeros and
    would test very little. Instead, 3/4 of each parameter's sample comes from
    entries with a nonzero gradient and the rest from zero entries, so the site's
    tests check both cases.

    Returns:
    list of dict: {"param", "index", "value"}, sorted by param order then index.
    """
    samples = []
    for name in PARAM_NAMES:
        flat = grads[name].reshape(-1)
        count = GRAD_SAMPLES[name]
        if count >= flat.size:
            chosen = np.arange(flat.size)
        else:
            nonzero = np.flatnonzero(flat != 0)
            zero = np.flatnonzero(flat == 0)
            n_nonzero = min(len(nonzero), count - min(len(zero), count // 4))
            chosen = np.concatenate([
                rng.choice(nonzero, n_nonzero, replace=False),
                rng.choice(zero, count - n_nonzero, replace=False),
            ])
        for index in np.sort(chosen):
            samples.append({"param": name, "index": int(index), "value": float(flat[index])})
    return samples


def backward_fixture(params, X_test, Y_test):
    """
    One image's loss, a sample of its gradients, and the probabilities after
    one SGD step on that image alone (batch size 1) at the training learning rate.

    The image is the test image the network is least sure about: the one whose
    probability for the correct digit is closest to 0.5. Most test images are
    classified with 99.99%+ confidence, which makes every gradient about 1e-10
    or smaller. Gradients that small would pass a 1e-5 test even if the site's
    backward pass were wrong, and one SGD step would barely change anything.
    """
    nn = build_network(params)
    probs_all = activations.softmax(nn.forward(X_test.astype(np.float64)))
    p_true = probs_all[np.arange(len(Y_test)), Y_test]
    index = int(np.argmin(np.abs(p_true - 0.5)))
    x =X_test[index:index + 1].astype(np.float64)
    y = Y_test[index:index + 1]

    logits = nn.forward(x)
    loss = cross_entropy_loss(logits, y)
    nn.backward(logits, y)

    grads = {}
    for i, layer in enumerate(nn.layers, start=1):
        grads[f"W{i}"] = layer.grad_weights
        grads[f"b{i}"] = layer.grad_biases.reshape(-1)
    samples = sample_gradients(grads, np.random.default_rng(SAMPLE_SEED))

    # One SGD step, the same update train_one_epoch() makes
    for layer in nn.layers:
        layer.weights -= LEARNING_RATE * layer.grad_weights
        layer.biases -= LEARNING_RATE * layer.grad_biases
    logits_after = nn.forward(x)[0]
    probs_after = activations.softmax(logits_after[None, :])[0]

    # Logits are included too: after the step the probabilities saturate near
    # 0 and 1, where a wrong update can still land within 1e-5, but the logits
    # don't saturate.
    return {
        "index": index,
        "label": int(Y_test[index]),
        "pixels": pixel_list(X_test[index]),
        "logits": [float(z) for z in logits[0]],
        "probs": [float(p) for p in activations.softmax(logits)[0]],
        "loss": float(loss),
        "gradients": samples,
        "learningRate": LEARNING_RATE,
        "logitsAfterStep": [float(z) for z in logits_after],
        "probsAfterStep": [float(p) for p in probs_after],
    }


def sample_images(nn, X_test, Y_test):
    """
    One clear test image per digit for the site's sample buttons.

    "Clear" here means classified correctly with at least 99.9% confidence, and
    among those, the one whose amount of ink is closest to the median for that
    digit. That picks a typical-looking digit rather than an unusually thick or
    thin one.
    """
    probs = activations.softmax(nn.forward(X_test.astype(np.float64)))
    predictions = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    ink = X_test.sum(axis=1)
    samples = []
    for digit in range(10):
        candidates = np.where((Y_test == digit) & (predictions == digit) & (confidence >= 0.999))[0]
        median_ink = np.median(ink[Y_test == digit])
        index = candidates[np.argmin(np.abs(ink[candidates] - median_ink))]
        samples.append({"index": int(index), "label": digit, "pixels": pixel_list(X_test[index])})
    return samples


def write_export(params, test_acc, X_test, Y_test):
    """Write weights.bin, model.json, and fixtures.json to EXPORT_DIR."""
    os.makedirs(EXPORT_DIR, exist_ok=True)
    nn = build_network(params)

    with open(os.path.join(EXPORT_DIR, "weights.bin"), "wb") as f:
        for name in PARAM_NAMES:
            f.write(np.ascontiguousarray(params[name], dtype="<f4").tobytes(order="C"))

    model = {
        "version": git_version(),
        "layers": SIZES,
        "activation": "relu",
        "output": "softmax",
        "loss": "cross-entropy",
        "learningRate": LEARNING_RATE,
        "batchSize": BATCH_SIZE,
        "testAccuracy": round(float(test_acc), 6),
        "normalization": NORMALIZATION,
        "forward": "x @ W + b",
        "dtype": "float32 little-endian",
        "order": [{"name": name, "shape": list(params[name].shape)} for name in PARAM_NAMES],
    }
    with open(os.path.join(EXPORT_DIR, "model.json"), "w") as f:
        json.dump(model, f, indent=2)

    fixtures = {
        "forward": forward_fixtures(nn, X_test, Y_test),
        "backward": backward_fixture(params, X_test, Y_test),
        "samples": sample_images(nn, X_test, Y_test),
    }
    with open(os.path.join(EXPORT_DIR, "fixtures.json"), "w") as f:
        json.dump(fixtures, f)


def verify_export(X_test, Y_test):
    """
    Read the exported files back from disk and check them.

    1. weights.bin, read with np.fromfile and reshaped from model.json alone,
       reproduces the test accuracy in model.json.
    2. It reproduces every forward-fixture probability.
    3. The sampled gradients match centered finite differences (the same check
       as gradient_check.py), so the backward fixture is right, not just
       consistent with itself.
    4. One SGD step reproduces probsAfterStep.
    """
    with open(os.path.join(EXPORT_DIR, "model.json")) as f:
        model = json.load(f)
    with open(os.path.join(EXPORT_DIR, "fixtures.json")) as f:
        fixtures = json.load(f)

    flat = np.fromfile(os.path.join(EXPORT_DIR, "weights.bin"), dtype="<f4")
    params, offset = {}, 0
    for entry in model["order"]:
        size = int(np.prod(entry["shape"]))
        params[entry["name"]] = flat[offset:offset + size].reshape(entry["shape"])
        offset += size
    assert offset == flat.size, f"weights.bin has {flat.size} values, model.json lists {offset}"

    nn = build_network(params)
    acc = accuracy(nn, X_test.astype(np.float64), Y_test)
    assert round(acc, 6) == model["testAccuracy"], f"accuracy {acc} != {model['testAccuracy']}"
    print(f"weights.bin reproduces test accuracy {acc:.4f} OK")

    worst = 0.0
    for fx in fixtures["forward"]:
        x = np.array([fx["pixels"]])
        probs = activations.softmax(nn.forward(x))[0]
        worst = max(worst, np.max(np.abs(probs - fx["probs"])))
    assert worst < 1e-12, f"forward fixture probs differ by {worst}"
    print(f"forward fixtures reproduced, max diff {worst:.1e} OK")

    bw = fixtures["backward"]
    x, y = np.array([bw["pixels"]]), np.array([bw["label"]])
    assert abs(cross_entropy_loss(nn.forward(x), y) - bw["loss"]) < 1e-12, "loss differs"
    arrays = {"W1": nn.layers[0].weights, "b1": nn.layers[0].biases,
              "W2": nn.layers[1].weights, "b2": nn.layers[1].biases,
              "W3": nn.layers[2].weights, "b3": nn.layers[2].biases}
    eps, worst_rel, checked = 1e-5, 0.0, 0
    for g in bw["gradients"]:
        values = arrays[g["param"]].reshape(-1)  # a view, so edits change the network
        original = values[g["index"]]
        values[g["index"]] = original + eps
        loss_plus = cross_entropy_loss(nn.forward(x), y)
        values[g["index"]] = original - eps
        loss_minus = cross_entropy_loss(nn.forward(x), y)
        values[g["index"]] = original
        numeric = (loss_plus - loss_minus) / (2 * eps)
        rel = abs(numeric - g["value"]) / max(abs(numeric) + abs(g["value"]), 1e-12)
        # Finite differences can't resolve slopes much below 1e-9: rounding in
        # the loss is ~1e-16, divided by 2 * eps. For gradients that tiny
        # (including exact zeros) relative error is meaningless, so check the
        # absolute difference instead.
        if max(abs(numeric), abs(g["value"])) < 1e-7:
            assert abs(numeric - g["value"]) < 1e-9, \
                f"{g['param']}[{g['index']}]: backprop {g['value']}, numeric {numeric}"
            continue
        checked += 1
        worst_rel = max(worst_rel, rel)
        assert rel < 1e-5, f"{g['param']}[{g['index']}]: backprop {g['value']}, numeric {numeric}"
    print(f"{len(bw['gradients'])} sampled gradients match finite differences "
          f"(within 1e-9, or relative error < 1e-5 on the {checked} larger ones: "
          f"max {worst_rel:.1e}) OK")

    logits = nn.forward(x)
    nn.backward(logits, y)
    for layer in nn.layers:
        layer.weights -= model["learningRate"] * layer.grad_weights
        layer.biases -= model["learningRate"] * layer.grad_biases
    logits_after = nn.forward(x)
    after = activations.softmax(logits_after)[0]
    diff = max(np.max(np.abs(after - bw["probsAfterStep"])),
               np.max(np.abs(logits_after[0] - bw["logitsAfterStep"])))
    assert diff < 1e-12, f"after-step fixture differs by {diff}"
    print(f"logitsAfterStep and probsAfterStep reproduced, max diff {diff:.1e} OK")


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = data.load_mnist()
    X_train, X_test = data.preprocess_data(X_train, X_test)

    params = load_float32_params(PARAMS_PATH)
    nn = build_network(params)
    test_acc = accuracy(nn, X_test.astype(np.float64), Y_test)
    print(f"Test accuracy with float32 weights: {test_acc:.4%}")

    write_export(params, test_acc, X_test, Y_test)
    for name in ["weights.bin", "model.json", "fixtures.json"]:
        path = os.path.join(EXPORT_DIR, name)
        print(f"Wrote {path} ({os.path.getsize(path):,} bytes)")

    print("\nVerifying the exported files:")
    verify_export(X_test, Y_test)
    print("\nExport verified.")
