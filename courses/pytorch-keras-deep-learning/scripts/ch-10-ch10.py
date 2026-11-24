#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — حلقة التدريب الاحترافية
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 10 — AMP

Book slug : pytorch-keras-deep-learning
Chapter   : ch-10-ch10 (lesson 10)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-10-ch10.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — حلقة التدريب الاحترافية
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 10 — AMP

Book slug : pytorch-keras-deep-learning
Chapter   : ch-10-ch10 (lesson 10)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch10.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"

model = nn.Linear(16, 3).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

x = torch.randn(32, 16, device=device)
y = torch.randint(0, 3, (32,), device=device)

optimizer.zero_grad(set_to_none=True)
with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
    logits = model(x)
    loss = criterion(logits, y)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()

print("loss:", round(loss.item(), 4))
print("grad norm:", round(float(total_norm), 4))
print("current scale:", scaler.get_scale())

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch10.html
# ────────────────────────────────────────────────────────────────────────
import os
import random

import numpy as np
import torch
import torch.nn as nn


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_metric):
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_metric": best_metric,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
    }
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"])
    return payload["epoch"] + 1, payload["best_metric"]


model = nn.Linear(8, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
scaler = torch.amp.GradScaler("cuda", enabled=False)

save_checkpoint("demo.pt", model, optimizer, scheduler, scaler, epoch=4, best_metric=0.87)
resumed_epoch, best = load_checkpoint("demo.pt", model, optimizer, scheduler, scaler)
print("resumed at epoch:", resumed_epoch)
print("best metric carried over:", best)

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch10.html
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn

total_steps = 1000
warmup_steps = 50
base_lr = 3e-4


def lr_lambda(step: int) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + np.cos(np.pi * progress))


model = nn.Linear(4, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

observed = []
for _ in range(total_steps):
    optimizer.step()
    scheduler.step()
    observed.append(optimizer.param_groups[0]["lr"])

print("lr at the end of warmup:", f"{observed[warmup_steps - 1]:.6f}")
print("lr at the end of training:", f"{observed[-1]:.6f}")

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch10.html
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

keras.mixed_precision.set_global_policy("mixed_float16")

rng = np.random.default_rng(5)
x_train = rng.normal(size=(1024, 20)).astype("float32")
y_train = (x_train[:, 0] * x_train[:, 1] > 0).astype("int64")
x_val = rng.normal(size=(256, 20)).astype("float32")
y_val = (x_val[:, 0] * x_val[:, 1] > 0).astype("int64")

schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=100,
    alpha=0.1,
    warmup_steps=10,
    warmup_target=1e-4,
)

model = keras.Sequential([
    keras.layers.Input(shape=(20,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2),
])

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=schedule, weight_decay=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=64,
    verbose=0,
)
print("final validation accuracy:", round(float(history.history["val_acc"][-1]), 4))
print("lr at the end of the schedule:", float(schedule(99)))

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch10.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, clip_value=1.0):
    model.train()
    loss_sum, correct, seen = 0.0, 0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(xb)
            loss = criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * len(yb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        seen += len(yb)
    print(f"grad norm (last batch): {float(grad_norm):.4f}")
    return loss_sum / seen, correct / seen

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch10.html
# ────────────────────────────────────────────────────────────────────────
from copy import deepcopy

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.module = deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_value, model_value in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.module.state_dict(), strict=True)


model = nn.Linear(4, 2)
ema = ModelEMA(model, decay=0.99)
for _ in range(50):
    with torch.no_grad():
        model.weight.add_(torch.randn_like(model.weight) * 0.1)
    ema.update(model)

distance = float((model.weight - ema.module.weight).norm())
print(f"distance between live and EMA weights: {distance:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch10.html
# ────────────────────────────────────────────────────────────────────────
for xb, yb in loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        loss = criterion(model(xb), yb)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch10.html
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

rng = np.random.default_rng(11)
x_train = rng.normal(size=(2048, 32)).astype("float32")
y_train = (x_train[:, 0] + 0.5 * x_train[:, 3] > 0).astype("int64")
x_val = rng.normal(size=(512, 32)).astype("float32")
y_val = (x_val[:, 0] + 0.5 * x_val[:, 3] > 0).astype("int64")

model = keras.Sequential([
    keras.layers.Input(shape=(32,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2),
])

model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=1e-3, weight_decay=1e-4, global_clipnorm=1.0, use_ema=True, ema_momentum=0.99,
    ),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    keras.callbacks.CSVLogger("logs.csv"),
]

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=20, batch_size=128,
    callbacks=callbacks,
    verbose=2,
)
print("best validation loss:", min(history.history["val_loss"]))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-10-ch10.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"

model = nn.Linear(16, 3).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

x = torch.randn(32, 16, device=device)
y = torch.randint(0, 3, (32,), device=device)

optimizer.zero_grad(set_to_none=True)
with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
    logits = model(x)
    loss = criterion(logits, y)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()

print("loss:", round(loss.item(), 4))
print("grad norm:", round(float(total_norm), 4))
print("current scale:", scaler.get_scale())

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-10-ch10.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import os
import random

import numpy as np
import torch
import torch.nn as nn


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_metric):
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_metric": best_metric,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
    }
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"])
    return payload["epoch"] + 1, payload["best_metric"]


model = nn.Linear(8, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
scaler = torch.amp.GradScaler("cuda", enabled=False)

save_checkpoint("demo.pt", model, optimizer, scheduler, scaler, epoch=4, best_metric=0.87)
resumed_epoch, best = load_checkpoint("demo.pt", model, optimizer, scheduler, scaler)
print("resumed at epoch:", resumed_epoch)
print("best metric carried over:", best)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-10-ch10.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn

total_steps = 1000
warmup_steps = 50
base_lr = 3e-4


def lr_lambda(step: int) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + np.cos(np.pi * progress))


model = nn.Linear(4, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

observed = []
for _ in range(total_steps):
    optimizer.step()
    scheduler.step()
    observed.append(optimizer.param_groups[0]["lr"])

print("lr at the end of warmup:", f"{observed[warmup_steps - 1]:.6f}")
print("lr at the end of training:", f"{observed[-1]:.6f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-10-ch10.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

keras.mixed_precision.set_global_policy("mixed_float16")

rng = np.random.default_rng(5)
x_train = rng.normal(size=(1024, 20)).astype("float32")
y_train = (x_train[:, 0] * x_train[:, 1] > 0).astype("int64")
x_val = rng.normal(size=(256, 20)).astype("float32")
y_val = (x_val[:, 0] * x_val[:, 1] > 0).astype("int64")

schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=100,
    alpha=0.1,
    warmup_steps=10,
    warmup_target=1e-4,
)

model = keras.Sequential([
    keras.layers.Input(shape=(20,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2),
])

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=schedule, weight_decay=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=64,
    verbose=0,
)
print("final validation accuracy:", round(float(history.history["val_acc"][-1]), 4))
print("lr at the end of the schedule:", float(schedule(99)))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-10-ch10.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, clip_value=1.0):
    model.train()
    loss_sum, correct, seen = 0.0, 0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(xb)
            loss = criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * len(yb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        seen += len(yb)
    print(f"grad norm (last batch): {float(grad_norm):.4f}")
    return loss_sum / seen, correct / seen

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-10-ch10.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from copy import deepcopy

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.module = deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_value, model_value in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.module.state_dict(), strict=True)


model = nn.Linear(4, 2)
ema = ModelEMA(model, decay=0.99)
for _ in range(50):
    with torch.no_grad():
        model.weight.add_(torch.randn_like(model.weight) * 0.1)
    ema.update(model)

distance = float((model.weight - ema.module.weight).norm())
print(f"distance between live and EMA weights: {distance:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-10-ch10.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
for xb, yb in loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        loss = criterion(model(xb), yb)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-10-ch10.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

rng = np.random.default_rng(11)
x_train = rng.normal(size=(2048, 32)).astype("float32")
y_train = (x_train[:, 0] + 0.5 * x_train[:, 3] > 0).astype("int64")
x_val = rng.normal(size=(512, 32)).astype("float32")
y_val = (x_val[:, 0] + 0.5 * x_val[:, 3] > 0).astype("int64")

model = keras.Sequential([
    keras.layers.Input(shape=(32,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2),
])

model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=1e-3, weight_decay=1e-4, global_clipnorm=1.0, use_ema=True, ema_momentum=0.99,
    ),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    keras.callbacks.CSVLogger("logs.csv"),
]

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=20, batch_size=128,
    callbacks=callbacks,
    verbose=2,
)
print("best validation loss:", min(history.history["val_loss"]))
