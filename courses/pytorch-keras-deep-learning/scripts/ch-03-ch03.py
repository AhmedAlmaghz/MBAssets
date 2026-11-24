#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — من الخلية العصبية إلى الشبكة
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 03 — The Foundations

Book slug : pytorch-keras-deep-learning
Chapter   : ch-03-ch03 (lesson 3)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-03-ch03.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — من الخلية العصبية إلى الشبكة
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 03 — The Foundations

Book slug : pytorch-keras-deep-learning
Chapter   : ch-03-ch03 (lesson 3)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch03.html
# ────────────────────────────────────────────────────────────────────────
# خلية عصبية واحدة، مكتوبة من الصفر
import numpy as np

def neuron(x, w, b, activation="relu"):
    z = np.dot(x, w) + b
    if activation == "relu":
        return max(0.0, z)
    if activation == "sigmoid":
        return 1 / (1 + np.exp(-z))
    raise ValueError("unknown activation")

x = np.array([0.8, -0.3, 0.5])
w = np.array([1.2, -0.7, 0.4])
b = 0.1

print("z      :", round(float(np.dot(x, w) + b), 4))   # الناتج: 1.05
print("relu   :", neuron(x, w, b, "relu"))              # الناتج: 1.05
print("sigmoid:", round(neuron(x, w, b, "sigmoid"), 4)) # الناتج: 0.7410

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch03.html
# ────────────────────────────────────────────────────────────────────────
# شبكة_feed-forward بطبقتين مخفيتين — من الصفر بـnumpy
import numpy as np

def relu(z): return np.maximum(0, z)

def forward(X, W1, b1, W2, b2, W3, b3):
    a1 = relu(X @ W1 + b1)     # الطبقة المخفية 1
    a2 = relu(a1 @ W2 + b2)    # الطبقة المخفية 2
    out = a2 @ W3 + b3         # طبقة المخرجات (بلا تفعيل: انحدار)
    return out, (a1, a2)

rng = np.random.default_rng(0)
n, d_in, d_h, d_out = 200, 10, 32, 1
X = rng.normal(size=(n, d_in))
y = (X[:, :1] * 2 - 0.5 + rng.normal(scale=0.05, size=(n, 1)))

W1 = rng.normal(scale=0.1, size=(d_in, d_h));  b1 = np.zeros(d_h)
W2 = rng.normal(scale=0.1, size=(d_h, d_h));   b2 = np.zeros(d_h)
W3 = rng.normal(scale=0.1, size=(d_h, d_out)); b3 = np.zeros(d_out)

pred, _ = forward(X, W1, b1, W2, b2, W3, b3)
print("mse:", round(float(np.mean((pred - y) ** 2)), 4))   # الناتج: 4.12xx (م untrained)

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch03.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

layer = nn.Linear(in_features=784, out_features=256)
x = torch.randn(32, 784)          # (batch, features)
y = layer(x)

print("weights:", tuple(layer.weight.shape))   # (256, 784)
print("bias   :", tuple(layer.bias.shape))     # (256,)
print("output :", tuple(y.shape))              # (32, 256)
print("params :", sum(p.numel() for p in layer.parameters()))
# الناتج: params: 201216

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch03.html
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
x = rng.uniform(-3, 3, (2000, 1))
y = np.sin(3 * x) + 0.5 * x

W1 = rng.normal(scale=0.5, size=(1, 32)); b1 = np.zeros((1, 32))
W2 = rng.normal(scale=0.5, size=(32, 16)); b2 = np.zeros((1, 16))
W3 = rng.normal(scale=0.5, size=(16, 1));  b3 = np.zeros((1, 1))

def forward(X):
    a1 = np.maximum(0, X @ W1 + b1)
    a2 = np.maximum(0, a1 @ W2 + b2)
    return a2 @ W3 + b3, (a1, a2)

def train(lr, epochs=500):
    losses = []
    for _ in range(epochs):
        out, (a1, a2) = forward(x)
        loss = np.mean((out - y) ** 2)
        losses.append(loss)
        d = 2 * (out - y) / len(x)            # مشتقّة الخسارة
        dW3 = a2.T @ d; db3 = d.sum(0)
        da2 = d @ W3.T
        da1 = da2 * (a2 > 0)
        dW2 = a1.T @ da2; db2 = da2.sum(0)
        da1 = da1 @ W2.T
        dW1 = x.T @ da1; db1 = da1.sum(0)
        for W, dW, b, db in ((W1, dW1, b1, db1), (W2, dW2, b2, db2), (W3, dW3, b3, db3)):
            W -= lr * dW; b -= lr * db
    return losses

for lr in (0.01, 0.5, 0.0001):
    losses = train(lr)
    print(f"lr={lr:<8} final loss={losses[-1]:.5f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-03-ch03.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# خلية عصبية واحدة، مكتوبة من الصفر
import numpy as np

def neuron(x, w, b, activation="relu"):
    z = np.dot(x, w) + b
    if activation == "relu":
        return max(0.0, z)
    if activation == "sigmoid":
        return 1 / (1 + np.exp(-z))
    raise ValueError("unknown activation")

x = np.array([0.8, -0.3, 0.5])
w = np.array([1.2, -0.7, 0.4])
b = 0.1

print("z      :", round(float(np.dot(x, w) + b), 4))   # الناتج: 1.05
print("relu   :", neuron(x, w, b, "relu"))              # الناتج: 1.05
print("sigmoid:", round(neuron(x, w, b, "sigmoid"), 4)) # الناتج: 0.7410

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-03-ch03.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# شبكة_feed-forward بطبقتين مخفيتين — من الصفر بـnumpy
import numpy as np

def relu(z): return np.maximum(0, z)

def forward(X, W1, b1, W2, b2, W3, b3):
    a1 = relu(X @ W1 + b1)     # الطبقة المخفية 1
    a2 = relu(a1 @ W2 + b2)    # الطبقة المخفية 2
    out = a2 @ W3 + b3         # طبقة المخرجات (بلا تفعيل: انحدار)
    return out, (a1, a2)

rng = np.random.default_rng(0)
n, d_in, d_h, d_out = 200, 10, 32, 1
X = rng.normal(size=(n, d_in))
y = (X[:, :1] * 2 - 0.5 + rng.normal(scale=0.05, size=(n, 1)))

W1 = rng.normal(scale=0.1, size=(d_in, d_h));  b1 = np.zeros(d_h)
W2 = rng.normal(scale=0.1, size=(d_h, d_h));   b2 = np.zeros(d_h)
W3 = rng.normal(scale=0.1, size=(d_h, d_out)); b3 = np.zeros(d_out)

pred, _ = forward(X, W1, b1, W2, b2, W3, b3)
print("mse:", round(float(np.mean((pred - y) ** 2)), 4))   # الناتج: 4.12xx (م untrained)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-03-ch03.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

layer = nn.Linear(in_features=784, out_features=256)
x = torch.randn(32, 784)          # (batch, features)
y = layer(x)

print("weights:", tuple(layer.weight.shape))   # (256, 784)
print("bias   :", tuple(layer.bias.shape))     # (256,)
print("output :", tuple(y.shape))              # (32, 256)
print("params :", sum(p.numel() for p in layer.parameters()))
# الناتج: params: 201216

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-03-ch03.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
x = rng.uniform(-3, 3, (2000, 1))
y = np.sin(3 * x) + 0.5 * x

W1 = rng.normal(scale=0.5, size=(1, 32)); b1 = np.zeros((1, 32))
W2 = rng.normal(scale=0.5, size=(32, 16)); b2 = np.zeros((1, 16))
W3 = rng.normal(scale=0.5, size=(16, 1));  b3 = np.zeros((1, 1))

def forward(X):
    a1 = np.maximum(0, X @ W1 + b1)
    a2 = np.maximum(0, a1 @ W2 + b2)
    return a2 @ W3 + b3, (a1, a2)

def train(lr, epochs=500):
    losses = []
    for _ in range(epochs):
        out, (a1, a2) = forward(x)
        loss = np.mean((out - y) ** 2)
        losses.append(loss)
        d = 2 * (out - y) / len(x)            # مشتقّة الخسارة
        dW3 = a2.T @ d; db3 = d.sum(0)
        da2 = d @ W3.T
        da1 = da2 * (a2 > 0)
        dW2 = a1.T @ da2; db2 = da2.sum(0)
        da1 = da1 @ W2.T
        dW1 = x.T @ da1; db1 = da1.sum(0)
        for W, dW, b, db in ((W1, dW1, b1, db1), (W2, dW2, b2, db2), (W3, dW3, b3, db3)):
            W -= lr * dW; b -= lr * db
    return losses

for lr in (0.01, 0.5, 0.0001):
    losses = train(lr)
    print(f"lr={lr:<8} final loss={losses[-1]:.5f}")
