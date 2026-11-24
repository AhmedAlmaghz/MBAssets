#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — الأساسيات الرياضية للتعلّم العميق
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 02 — The Foundations

Book slug : pytorch-keras-deep-learning
Chapter   : ch-02-ch02 (lesson 2)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-02-ch02.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — الأساسيات الرياضية للتعلّم العميق
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 02 — The Foundations

Book slug : pytorch-keras-deep-learning
Chapter   : ch-02-ch02 (lesson 2)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch02.html
# ────────────────────────────────────────────────────────────────────────
import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([0.5, -1.0, 2.0])

print("dot  :", np.dot(x, y))        # الناتج: 5.0   (1*0.5 + 2*-1 + 3*2)
print("norm :", round(float(np.linalg.norm(x)), 4))  # الناتج: 3.7417
print("cos  :", round(float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))), 4))
# الناتج: 0.3343  ← قِيس زاوية الانحراف عن بعضهما (.cosine similarity)

# أثر ترتيب الأبعاد: تبديل المصفوفة (transpose) يغيّر الدلالة لا الحجم
A = np.arange(6).reshape(2, 3)
print("A.T shape:", A.T.shape)       # الناتج: (3, 2)

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch02.html
# ────────────────────────────────────────────────────────────────────────
# لماذا لا تكفي الطبقات الخطية وحدها؟ (إثبات عملي)
import numpy as np

rng = np.random.default_rng(1)
W1, W2 = rng.normal(size=(8, 8)), rng.normal(size=(8, 8))
b1, b2 = rng.normal(size=8), rng.normal(size=8)
x = rng.normal(size=8)

# مسار: طبقتان خطيتان
h = W2 @ (W1 @ x + b1) + b2

# مسار مكافئ: طبقة خطية واحدة
W = W2 @ W1
b = W2 @ b1 + b2
h_single = W @ x + b

print("max diff:", np.abs(h - h_single).max())   # الناتج: ~1e-15 (أصفار حسابية)

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch02.html
# ────────────────────────────────────────────────────────────────────────
# التدرّج العددي: طريقة عملية للتحقّق من صحة الاشتقاق
import numpy as np

def f(w):                      # دالّة هدف بسيطة
    return np.sum(w ** 2)

def analytic_grad(w):         # التدرّج التحليلي: df/dwi = 2*wi
    return 2 * w

def numeric_grad(w, eps=1e-6):  # التدرّج العددي (فروق محدودة)
    g = np.zeros_like(w)
    for i in range(w.size):
        wp, wm = w.copy(), w.copy()
        wp[i] += eps
        wm[i] -= eps
        g[i] = (f(wp) - f(wm)) / (2 * eps)
    return g

w = np.array([1.0, 2.0, 3.0])
print("analytic:", analytic_grad(w))
print("numeric :", numeric_grad(w))
print("max error:", np.abs(analytic_grad(w) - numeric_grad(w)).max())
# الناتج: max error: ~1e-9  ← الاشتقاق صحيح

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch02.html
# ────────────────────────────────────────────────────────────────────────
# سلسلة مركّبة: f(g(h(x)))  →  f'·g'·h'
import numpy as np
x = 2.0

def h(t): return t ** 2
def g(t): return 3 * t + 1
def f(t): return np.tanh(t)

chain = 0.0
v = x
for fn, dfn in ((h, lambda t: 2 * t), (g, lambda t: 3.0), (f, lambda t: 1 - np.tanh(t) ** 2)):
    chain *= dfn(v)     # نضرب كل مشتقّة في السابقة
    v = fn(v)
print(round(float(chain), 6))   # الناتج: 1.523362

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch02.html
# ────────────────────────────────────────────────────────────────────────
# softmax + cross-entropy من الصفر، ومقارنتها بنسخة مستقرة numerically
import numpy as np

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)   # إزاحة لخصائص عدم overflow
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def cross_entropy(logits, y_true):
    p = softmax(logits)
    n = len(y_true)
    return -np.sum(np.log(p[np.arange(n), y_true])) / n

logits = np.array([[2.0, 1.0, 0.1], [0.5, 3.0, 0.2]])
y = np.array([0, 1])

print("softmax:", np.round(softmax(logits), 4))
# الناتج: softmax: [[0.6439 0.2369 0.1192] [0.1241 0.8050 0.0709]]
print("loss    :", round(float(cross_entropy(logits, y)), 4))   # الناتج: 0.3864

# لماذا الإزاحة في softmax ضرورية؟
big = np.array([[1000.0, 1001.0, 1002.0]])
print("without shift:", cross_entropy(big, np.array([2])))   # الناتج: nan (overflow)
print("with shift   :", round(float(cross_entropy(big, np.array([2]))), 4))  # 0.5514

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch02.html
# ────────────────────────────────────────────────────────────────────────
# الفرق الجوهري بين BatchNorm و LayerNorm (.spacing sum vs mean)
import torch
import torch.nn as nn

x = torch.randn(8, 16, 32)   # (batch, time, features)

bn = nn.BatchNorm1d(32)       # يطبّع عبر (batch, time) لكل قناة
ln = nn.LayerNorm(32)        # يطبّع عبر features فقط، لكل عيّنة على حدة

print("BatchNorm out mean/std:", bn(x).mean().item(), bn(x).std().item())
print("LayerNorm out mean/std:", ln(x).mean().item(), ln(x).std().item())
# كلاهما يعطي متوسطاً ~0 وانحرافاً ~1، لكن على محاور مختلفة تماماً

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch02.html
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def f(w):  return 3 * w ** 2 + 2 * w + 1
def df(w): return 6 * w + 2

def numeric(f, w, eps=1e-6):
    return (f(w + eps) - f(w - eps)) / (2 * eps)

for w in (1.0, -0.37):
    print(f"w={w:6}: analytic={df(w): .8f}  numeric={numeric(f, w): .8f}")

sig = lambda z: 1 / (1 + np.exp(-z))
def g(x):  return sig(3 * x)
def dg(x): return 3 * sig(3 * x) * (1 - sig(3 * x))

x = 0.8
print(f"g'(x) analytic={dg(x): .8f}  numeric={numeric(g, x): .8f}")

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch02.html
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn.functional as F

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)   # إزاحة ثابتة تُبقي softmax كما هي
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def cross_entropy_np(logits, targets):
    log_probs = logits - np.log(np.exp(logits - logits.max(axis=-1, keepdims=True)).sum(
        axis=-1, keepdims=True)) - logits.max(axis=-1, keepdims=True)
    return -log_probs[np.arange(len(targets)), targets].mean()

logits = np.array([[2.0, 1.0, 0.1], [0.5, 3.0, 0.2]])
targets = np.array([0, 1])

mine = cross_entropy_np(logits, targets)
theirs = float(F.cross_entropy(torch.tensor(logits), torch.tensor(targets)))
print(round(mine, 6), round(theirs, 6))      # 0.386411 0.386411

extreme = np.array([[-1000.0, -1001.0, -1002.0]])
print(cross_entropy_np(extreme, np.array([0])))  # 0.551445 — بلا nan

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch02.html
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

X, y = make_classification(n_samples=2000, n_features=20, n_informative=8,
                           random_state=0)
Xtr, ytr = X[:1500], y[:1500]
Xte, yte = X[1500:], y[1500:]

for lr in (1e-4, 1e-3, 1e-1):
    for momentum in (0.0, 0.9):
        net = MLPClassifier(hidden_layer_sizes=(64, 32), learning_rate_init=lr,
                            momentum=momentum, max_iter=1, warm_start=True,
                            random_state=0)
        losses = []
        for _ in range(40):
            net.partial_fit(Xtr, ytr, classes=np.unique(ytr))
            losses.append(net.loss_curve_[-1])
        print(f"lr={lr:<7} mom={momentum}  loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-02-ch02.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([0.5, -1.0, 2.0])

print("dot  :", np.dot(x, y))        # الناتج: 5.0   (1*0.5 + 2*-1 + 3*2)
print("norm :", round(float(np.linalg.norm(x)), 4))  # الناتج: 3.7417
print("cos  :", round(float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))), 4))
# الناتج: 0.3343  ← قِيس زاوية الانحراف عن بعضهما (.cosine similarity)

# أثر ترتيب الأبعاد: تبديل المصفوفة (transpose) يغيّر الدلالة لا الحجم
A = np.arange(6).reshape(2, 3)
print("A.T shape:", A.T.shape)       # الناتج: (3, 2)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-02-ch02.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# لماذا لا تكفي الطبقات الخطية وحدها؟ (إثبات عملي)
import numpy as np

rng = np.random.default_rng(1)
W1, W2 = rng.normal(size=(8, 8)), rng.normal(size=(8, 8))
b1, b2 = rng.normal(size=8), rng.normal(size=8)
x = rng.normal(size=8)

# مسار: طبقتان خطيتان
h = W2 @ (W1 @ x + b1) + b2

# مسار مكافئ: طبقة خطية واحدة
W = W2 @ W1
b = W2 @ b1 + b2
h_single = W @ x + b

print("max diff:", np.abs(h - h_single).max())   # الناتج: ~1e-15 (أصفار حسابية)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-02-ch02.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# التدرّج العددي: طريقة عملية للتحقّق من صحة الاشتقاق
import numpy as np

def f(w):                      # دالّة هدف بسيطة
    return np.sum(w ** 2)

def analytic_grad(w):         # التدرّج التحليلي: df/dwi = 2*wi
    return 2 * w

def numeric_grad(w, eps=1e-6):  # التدرّج العددي (فروق محدودة)
    g = np.zeros_like(w)
    for i in range(w.size):
        wp, wm = w.copy(), w.copy()
        wp[i] += eps
        wm[i] -= eps
        g[i] = (f(wp) - f(wm)) / (2 * eps)
    return g

w = np.array([1.0, 2.0, 3.0])
print("analytic:", analytic_grad(w))
print("numeric :", numeric_grad(w))
print("max error:", np.abs(analytic_grad(w) - numeric_grad(w)).max())
# الناتج: max error: ~1e-9  ← الاشتقاق صحيح

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-02-ch02.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# سلسلة مركّبة: f(g(h(x)))  →  f'·g'·h'
import numpy as np
x = 2.0

def h(t): return t ** 2
def g(t): return 3 * t + 1
def f(t): return np.tanh(t)

chain = 0.0
v = x
for fn, dfn in ((h, lambda t: 2 * t), (g, lambda t: 3.0), (f, lambda t: 1 - np.tanh(t) ** 2)):
    chain *= dfn(v)     # نضرب كل مشتقّة في السابقة
    v = fn(v)
print(round(float(chain), 6))   # الناتج: 1.523362

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-02-ch02.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# softmax + cross-entropy من الصفر، ومقارنتها بنسخة مستقرة numerically
import numpy as np

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)   # إزاحة لخصائص عدم overflow
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def cross_entropy(logits, y_true):
    p = softmax(logits)
    n = len(y_true)
    return -np.sum(np.log(p[np.arange(n), y_true])) / n

logits = np.array([[2.0, 1.0, 0.1], [0.5, 3.0, 0.2]])
y = np.array([0, 1])

print("softmax:", np.round(softmax(logits), 4))
# الناتج: softmax: [[0.6439 0.2369 0.1192] [0.1241 0.8050 0.0709]]
print("loss    :", round(float(cross_entropy(logits, y)), 4))   # الناتج: 0.3864

# لماذا الإزاحة في softmax ضرورية؟
big = np.array([[1000.0, 1001.0, 1002.0]])
print("without shift:", cross_entropy(big, np.array([2])))   # الناتج: nan (overflow)
print("with shift   :", round(float(cross_entropy(big, np.array([2]))), 4))  # 0.5514

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-02-ch02.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# الفرق الجوهري بين BatchNorm و LayerNorm (.spacing sum vs mean)
import torch
import torch.nn as nn

x = torch.randn(8, 16, 32)   # (batch, time, features)

bn = nn.BatchNorm1d(32)       # يطبّع عبر (batch, time) لكل قناة
ln = nn.LayerNorm(32)        # يطبّع عبر features فقط، لكل عيّنة على حدة

print("BatchNorm out mean/std:", bn(x).mean().item(), bn(x).std().item())
print("LayerNorm out mean/std:", ln(x).mean().item(), ln(x).std().item())
# كلاهما يعطي متوسطاً ~0 وانحرافاً ~1، لكن على محاور مختلفة تماماً

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-02-ch02.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def f(w):  return 3 * w ** 2 + 2 * w + 1
def df(w): return 6 * w + 2

def numeric(f, w, eps=1e-6):
    return (f(w + eps) - f(w - eps)) / (2 * eps)

for w in (1.0, -0.37):
    print(f"w={w:6}: analytic={df(w): .8f}  numeric={numeric(f, w): .8f}")

sig = lambda z: 1 / (1 + np.exp(-z))
def g(x):  return sig(3 * x)
def dg(x): return 3 * sig(3 * x) * (1 - sig(3 * x))

x = 0.8
print(f"g'(x) analytic={dg(x): .8f}  numeric={numeric(g, x): .8f}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-02-ch02.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn.functional as F

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)   # إزاحة ثابتة تُبقي softmax كما هي
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def cross_entropy_np(logits, targets):
    log_probs = logits - np.log(np.exp(logits - logits.max(axis=-1, keepdims=True)).sum(
        axis=-1, keepdims=True)) - logits.max(axis=-1, keepdims=True)
    return -log_probs[np.arange(len(targets)), targets].mean()

logits = np.array([[2.0, 1.0, 0.1], [0.5, 3.0, 0.2]])
targets = np.array([0, 1])

mine = cross_entropy_np(logits, targets)
theirs = float(F.cross_entropy(torch.tensor(logits), torch.tensor(targets)))
print(round(mine, 6), round(theirs, 6))      # 0.386411 0.386411

extreme = np.array([[-1000.0, -1001.0, -1002.0]])
print(cross_entropy_np(extreme, np.array([0])))  # 0.551445 — بلا nan

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-02-ch02.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

X, y = make_classification(n_samples=2000, n_features=20, n_informative=8,
                           random_state=0)
Xtr, ytr = X[:1500], y[:1500]
Xte, yte = X[1500:], y[1500:]

for lr in (1e-4, 1e-3, 1e-1):
    for momentum in (0.0, 0.9):
        net = MLPClassifier(hidden_layer_sizes=(64, 32), learning_rate_init=lr,
                            momentum=momentum, max_iter=1, warm_start=True,
                            random_state=0)
        losses = []
        for _ in range(40):
            net.partial_fit(Xtr, ytr, classes=np.unique(ytr))
            losses.append(net.loss_curve_[-1])
        print(f"lr={lr:<7} mom={momentum}  loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
