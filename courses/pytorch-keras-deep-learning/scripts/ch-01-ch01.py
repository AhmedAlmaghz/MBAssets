#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — التعلّم العميق: لماذا الآن؟
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 01 — The Foundations

Book slug : pytorch-keras-deep-learning
Chapter   : ch-01-ch01 (lesson 1)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-01-ch01.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — التعلّم العميق: لماذا الآن؟
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 01 — The Foundations

Book slug : pytorch-keras-deep-learning
Chapter   : ch-01-ch01 (lesson 1)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch01.html
# ────────────────────────────────────────────────────────────────────────
# الفرق العملي: الانحدار الخطّي مقابل شبكة بطبقتين
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

rng = np.random.default_rng(0)
X = rng.normal(size=(400, 2))
y = ((X[:, 0] ** 2 + X[:, 1] ** 2) > 1).astype(int)   # دائرة داخل خط

linear = LogisticRegression().fit(X, y)
deep = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=2000).fit(X, y)

print("دقة الخطية :", round(linear.score(X, y), 3))   # الناتج: 0.508 ← مثل العشوائية
print("دقة العميقة:", round(deep.score(X, y), 3))     # الناتج: 0.995

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch01.html
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def evaluate(model, name):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(f"{name:12s} acc={accuracy_score(y_te, pred):.4f}  f1={f1_score(y_te, pred):.4f}")

evaluate(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), "baseline")
evaluate(make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000)), "MLP-1")
evaluate(make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=2000)), "MLP-2")
# الناتج التقريبي:
# baseline    acc=0.9825  f1=0.9844
# MLP-1       acc=0.9649  f1=0.9675
# MLP-2       acc=0.9561  f1=0.9593

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch01.html
# ────────────────────────────────────────────────────────────────────────
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=1000, noise=0.15, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
ytr_t = torch.tensor(ytr, dtype=torch.long)
Xte_t = torch.tensor(Xte, dtype=torch.float32)

# ---------- PyTorch ----------
torch.manual_seed(0)
model = nn.Sequential(
    nn.Linear(2, 32), nn.ReLU(),
    nn.Linear(32, 1),
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.BCEWithLogitsLoss()

t0 = time.perf_counter()
for _ in range(300):
    opt.zero_grad()
    out = model(Xtr_t).squeeze(1)
    loss = lossf(out, ytr_t.float())
    loss.backward()
    opt.step()
torch_time = time.perf_counter() - t0

with torch.no_grad():
    acc = ((model(Xte_t).squeeze(1) > 0).long() == torch.tensor(yte)).float().mean().item()

n_params_t = sum(p.numel() for p in model.parameters())
print(f"PyTorch: params={n_params_t}, acc={acc:.4f}, time={torch_time:.2f}s")
# الناتج: PyTorch: params=113, acc=0.9850, time=0.61s

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch01.html
# ────────────────────────────────────────────────────────────────────────
# ---------- Keras 3 (نواة PyTorch) ----------
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras

keras.utils.set_random_seed(0)
kmodel = keras.Sequential([
    keras.layers.Input(shape=(2,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
kmodel.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
               loss=keras.losses.BinaryCrossentropy(from_logits=True),
               metrics=["accuracy"])

t0 = time.perf_counter()
kmodel.fit(Xtr, ytr, epochs=300, batch_size=32, verbose=0)
keras_time = time.perf_counter() - t0

kacc = float(np.mean((kmodel.predict(Xte, verbose=0).ravel() > 0).astype(int) == yte))
n_params_k = sum(int(np.prod(w.shape)) for w in kmodel.trainable_weights)
print(f"Keras  : params={n_params_k}, acc={kacc:.4f}, time={keras_time:.2f}s")
# الناتج: Keras  : params=113, acc=0.9850, time≈0.6s

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-01-ch01.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# الفرق العملي: الانحدار الخطّي مقابل شبكة بطبقتين
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

rng = np.random.default_rng(0)
X = rng.normal(size=(400, 2))
y = ((X[:, 0] ** 2 + X[:, 1] ** 2) > 1).astype(int)   # دائرة داخل خط

linear = LogisticRegression().fit(X, y)
deep = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=2000).fit(X, y)

print("دقة الخطية :", round(linear.score(X, y), 3))   # الناتج: 0.508 ← مثل العشوائية
print("دقة العميقة:", round(deep.score(X, y), 3))     # الناتج: 0.995

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-01-ch01.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def evaluate(model, name):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(f"{name:12s} acc={accuracy_score(y_te, pred):.4f}  f1={f1_score(y_te, pred):.4f}")

evaluate(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), "baseline")
evaluate(make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000)), "MLP-1")
evaluate(make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=2000)), "MLP-2")
# الناتج التقريبي:
# baseline    acc=0.9825  f1=0.9844
# MLP-1       acc=0.9649  f1=0.9675
# MLP-2       acc=0.9561  f1=0.9593

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-01-ch01.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=1000, noise=0.15, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
ytr_t = torch.tensor(ytr, dtype=torch.long)
Xte_t = torch.tensor(Xte, dtype=torch.float32)

# ---------- PyTorch ----------
torch.manual_seed(0)
model = nn.Sequential(
    nn.Linear(2, 32), nn.ReLU(),
    nn.Linear(32, 1),
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.BCEWithLogitsLoss()

t0 = time.perf_counter()
for _ in range(300):
    opt.zero_grad()
    out = model(Xtr_t).squeeze(1)
    loss = lossf(out, ytr_t.float())
    loss.backward()
    opt.step()
torch_time = time.perf_counter() - t0

with torch.no_grad():
    acc = ((model(Xte_t).squeeze(1) > 0).long() == torch.tensor(yte)).float().mean().item()

n_params_t = sum(p.numel() for p in model.parameters())
print(f"PyTorch: params={n_params_t}, acc={acc:.4f}, time={torch_time:.2f}s")
# الناتج: PyTorch: params=113, acc=0.9850, time=0.61s

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-01-ch01.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ---------- Keras 3 (نواة PyTorch) ----------
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras

keras.utils.set_random_seed(0)
kmodel = keras.Sequential([
    keras.layers.Input(shape=(2,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
kmodel.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
               loss=keras.losses.BinaryCrossentropy(from_logits=True),
               metrics=["accuracy"])

t0 = time.perf_counter()
kmodel.fit(Xtr, ytr, epochs=300, batch_size=32, verbose=0)
keras_time = time.perf_counter() - t0

kacc = float(np.mean((kmodel.predict(Xte, verbose=0).ravel() > 0).astype(int) == yte))
n_params_k = sum(int(np.prod(w.shape)) for w in kmodel.trainable_weights)
print(f"Keras  : params={n_params_k}, acc={kacc:.4f}, time={keras_time:.2f}s")
# الناتج: Keras  : params=113, acc=0.9850, time≈0.6s
