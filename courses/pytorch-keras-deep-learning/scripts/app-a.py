#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — خريطة المقابلة بين PyTorch و Keras
PyTorch and Keras: The Complete Deep Learning Guide — Appendix A — خريطة المقابلة بين PyTorch و Keras

Book slug : pytorch-keras-deep-learning
Chapter   : app-a (lesson a)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/app-a.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — خريطة المقابلة بين PyTorch و Keras
PyTorch and Keras: The Complete Deep Learning Guide — Appendix A — خريطة المقابلة بين PyTorch و Keras

Book slug : pytorch-keras-deep-learning
Chapter   : app-a (lesson a)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/appA.html
# ────────────────────────────────────────────────────────────────────────
# ---------- PyTorch ----------
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=256, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

model_pt = MLP()
opt_pt = torch.optim.AdamW(model_pt.parameters(), lr=1e-3, weight_decay=1e-2)
loss_pt = nn.CrossEntropyLoss()

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/appA.html
# ────────────────────────────────────────────────────────────────────────
# ---------- Keras 3 ----------
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras

model_ks = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10),
])

model_ks.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-2),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/appA.html
# ────────────────────────────────────────────────────────────────────────
# الحالة 1: إعادة بناء البنية في Keras ثم نسخ الأوزان
import torch
import keras

# على افتراض وجود model_pt (nn.Module)
weights = {k: v.detach().numpy() for k, v in model_pt.state_dict().items()}

# keras Dense Lovel: kernel (in, out)  vs  torch Linear.weight (out, in)
model_ks = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(256, activation="relu", name="dense_0"),
    keras.layers.Dense(10, name="dense_1"),
])

model_ks.get_layer("dense_0").set_weights([
    weights["net.0.weight"].T, weights["net.0.bias"]
])
model_ks.get_layer("dense_1").set_weights([
    weights["net.4.weight"].T, weights["net.4.bias"]
])
print("تم نقل الأوزان بنجاح")

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/appA.html
# ────────────────────────────────────────────────────────────────────────
# النسخة الأولى: PyTorch
import torch
import torch.nn as nn

torch_net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

# النسخة الثانية: Keras — البنية نفسها، أربع أسطر
import keras
from keras import layers

keras_net = keras.Sequential([
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10),
])

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/app-a.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ---------- PyTorch ----------
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=256, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

model_pt = MLP()
opt_pt = torch.optim.AdamW(model_pt.parameters(), lr=1e-3, weight_decay=1e-2)
loss_pt = nn.CrossEntropyLoss()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/app-a.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ---------- Keras 3 ----------
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras

model_ks = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10),
])

model_ks.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-2),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/app-a.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# الحالة 1: إعادة بناء البنية في Keras ثم نسخ الأوزان
import torch
import keras

# على افتراض وجود model_pt (nn.Module)
weights = {k: v.detach().numpy() for k, v in model_pt.state_dict().items()}

# keras Dense Lovel: kernel (in, out)  vs  torch Linear.weight (out, in)
model_ks = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(256, activation="relu", name="dense_0"),
    keras.layers.Dense(10, name="dense_1"),
])

model_ks.get_layer("dense_0").set_weights([
    weights["net.0.weight"].T, weights["net.0.bias"]
])
model_ks.get_layer("dense_1").set_weights([
    weights["net.4.weight"].T, weights["net.4.bias"]
])
print("تم نقل الأوزان بنجاح")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/app-a.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# النسخة الأولى: PyTorch
import torch
import torch.nn as nn

torch_net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

# النسخة الثانية: Keras — البنية نفسها، أربع أسطر
import keras
from keras import layers

keras_net = keras.Sequential([
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10),
])
