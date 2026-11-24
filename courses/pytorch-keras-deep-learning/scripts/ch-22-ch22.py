#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — حلقة التدريب و Callbacks في Keras
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 22 — compile

Book slug : pytorch-keras-deep-learning
Chapter   : ch-22-ch22 (lesson 22)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-22-ch22.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — حلقة التدريب و Callbacks في Keras
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 22 — compile

Book slug : pytorch-keras-deep-learning
Chapter   : ch-22-ch22 (lesson 22)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch22.html
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Rescaling(1 / 255.0),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(10),
])   # طبقة أخيرة بلا softmax عمداً

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

print(model.output_shape)   # (None, 10)

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch22.html
# ────────────────────────────────────────────────────────────────────────
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    batch_size=64,
    shuffle=True,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            mode="min",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
        keras.callbacks.ModelCheckpoint(
            "best.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
    ],
)
print(history.history.keys())
# الناتج: dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy', 'lr'])

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch22.html
# ────────────────────────────────────────────────────────────────────────
import keras


class StopAtLossPlateau(keras.callbacks.Callback):
    """يوقف التدريب إذا stagned التحقّق وتحسّن بطيء."""

    def __init__(self, monitor="val_loss", patience=3, min_delta=1e-3):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
        if current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True


class SaveBestAndLog(keras.callbacks.Callback):
    """يحفظ أفضل نموذج ويطبع ملخّصاً في كلّ حقبة."""

    def __init__(self, path="best.keras"):
        super().__init__()
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        val = logs.get("val_loss")
        if val is not None and val == min(logs.values()):
            self.model.save(self.path)
        print(f"epoch {epoch}: {logs}")

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch22.html
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1])
y_proba = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9])

print(classification_report(y_true, y_pred, digits=2))
print(confusion_matrix(y_true, y_pred))
print(roc_auc_score(y_true, y_proba))

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch22.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.distributed as dist
import os


def setup_ddp(rank, world_size):
    """تهيئة عملية واحدة من التوزيع."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch22.html
# ────────────────────────────────────────────────────────────────────────
import keras

callbacks = [
    keras.callbacks.TensorBoard(log_dir="logs"),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
    ),
    keras.callbacks.ModelCheckpoint(
        "best.keras", monitor="val_loss", save_best_only=True
    ),
]
history = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)

losses = history.history["val_loss"]
best = int(min(range(len(losses)), key=losses.__getitem__))
print("best epoch:", best, "val_loss:", losses[best])

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch22.html
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import precision_recall_curve

proba = model.predict(val_ds)
y_true = np.concatenate([y.numpy() for _, y in val_ds])
y_score = proba[:, 1] if proba.ndim == 2 else proba.ravel()

precision, recall, thresholds = precision_recall_curve(y_true, y_score)
ok = np.where(recall >= 0.90)[0]
idx = ok[-1]
print("threshold:", thresholds[idx],
      "precision:", precision[idx], "recall:", recall[idx])

n_pos = int(y_true.sum())
n_pred = int((y_score >= thresholds[idx]).sum())
daily_alerts = n_pred * 5.0   # خمسة تنبيهات لكلّ حالة

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-22-ch22.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Rescaling(1 / 255.0),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(10),
])   # طبقة أخيرة بلا softmax عمداً

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

print(model.output_shape)   # (None, 10)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-22-ch22.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    batch_size=64,
    shuffle=True,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            mode="min",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
        keras.callbacks.ModelCheckpoint(
            "best.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
    ],
)
print(history.history.keys())
# الناتج: dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy', 'lr'])

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-22-ch22.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import keras


class StopAtLossPlateau(keras.callbacks.Callback):
    """يوقف التدريب إذا stagned التحقّق وتحسّن بطيء."""

    def __init__(self, monitor="val_loss", patience=3, min_delta=1e-3):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
        if current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True


class SaveBestAndLog(keras.callbacks.Callback):
    """يحفظ أفضل نموذج ويطبع ملخّصاً في كلّ حقبة."""

    def __init__(self, path="best.keras"):
        super().__init__()
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        val = logs.get("val_loss")
        if val is not None and val == min(logs.values()):
            self.model.save(self.path)
        print(f"epoch {epoch}: {logs}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-22-ch22.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1])
y_proba = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9])

print(classification_report(y_true, y_pred, digits=2))
print(confusion_matrix(y_true, y_pred))
print(roc_auc_score(y_true, y_proba))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-22-ch22.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.distributed as dist
import os


def setup_ddp(rank, world_size):
    """تهيئة عملية واحدة من التوزيع."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-22-ch22.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras

callbacks = [
    keras.callbacks.TensorBoard(log_dir="logs"),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
    ),
    keras.callbacks.ModelCheckpoint(
        "best.keras", monitor="val_loss", save_best_only=True
    ),
]
history = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)

losses = history.history["val_loss"]
best = int(min(range(len(losses)), key=losses.__getitem__))
print("best epoch:", best, "val_loss:", losses[best])

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-22-ch22.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import precision_recall_curve

proba = model.predict(val_ds)
y_true = np.concatenate([y.numpy() for _, y in val_ds])
y_score = proba[:, 1] if proba.ndim == 2 else proba.ravel()

precision, recall, thresholds = precision_recall_curve(y_true, y_score)
ok = np.where(recall >= 0.90)[0]
idx = ok[-1]
print("threshold:", thresholds[idx],
      "precision:", precision[idx], "recall:", recall[idx])

n_pos = int(y_true.sum())
n_pred = int((y_score >= thresholds[idx]).sum())
daily_alerts = n_pred * 5.0   # خمسة تنبيهات لكلّ حالة
