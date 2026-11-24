#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — بيئة عمل احترافية على Python 3.14
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 05 — Python

Book slug : pytorch-keras-deep-learning
Chapter   : ch-05-ch05 (lesson 5)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-05-ch05.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — بيئة عمل احترافية على Python 3.14
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 05 — Python

Book slug : pytorch-keras-deep-learning
Chapter   : ch-05-ch05 (lesson 5)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch05.html
# ────────────────────────────────────────────────────────────────────────
import torch
import platform

def describe_hardware():
  info = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
  }
  if torch.cuda.is_available():
    info["gpu"] = torch.cuda.get_device_name(0)
    info["capability"] = torch.cuda.get_device_capability(0)
    info["memory_GB"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
  info["mps"] = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
  return info

print(describe_hardware())
# الناتج التقريبي:
# {'python': '3.14.6', 'torch': '2.14.1', 'cuda_available': True,
# 'gpu': 'NVIDIA RTX 4070', 'capability': (8, 9), 'memory_GB': 12.0, 'mps': False}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch05.html
# ────────────────────────────────────────────────────────────────────────
import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = False) -> None:
  """يثبّت كل مصادر العشوائية."""
  os.environ["PYTHONHASHSEED"] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
  else:
    torch.backends.cudnn.benchmark = True

set_seed(42)
a = torch.randn(1000)
set_seed(42)
b = torch.randn(1000)
print("identical:", torch.equal(a, b))  # الناتج: identical: True

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch05.html
# ────────────────────────────────────────────────────────────────────────
# تسجيل بسيط بدون تبعيات خارجية: JSONL
import json
import time
from pathlib import Path

class RunLogger:
  def __init__(self, out_dir="artifacts/logs", name="run"):
    self.path = Path(out_dir) / f"{name}.jsonl"
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self.t0 = time.perf_counter()

  def log(self, **kwargs):
    row = {"elapsed_s": round(time.perf_counter() - self.t0, 2), **kwargs}
    with self.path.open("a", encoding="utf-8") as fh:
      fh.write(json.dumps(row, ensure_ascii=False) + "\n")

  def config(self, cfg: dict):
    self.log(event="config", **cfg)

log = RunLogger(name="exp01")
log.config(lr=3e-4, batch_size=64, model="resnet50", seed=42)
for epoch in range(3):
  log.event = "epoch"
  log(epoch=epoch, train_loss=0.42 - epoch * 0.05, val_acc=0.90 + epoch * 0.01)
print("wrote", log.path)

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch05.html
# ────────────────────────────────────────────────────────────────────────
# اختيار النواة برمجياً داخل السكربت (قبل استيراد keras)
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
print(keras.backend.backend())   # torch
print(keras.__version__)      # 3.15.1

# اختيار الجهاز
backend = keras.backend
if backend.backend() == "torch":
  import torch
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print("device:", device)

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch05.html
# ────────────────────────────────────────────────────────────────────────
# src/train.py
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import sys
sys.path.append("src")

import torch
from torch import nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from utils.seed import set_seed
from utils.logging import RunLogger

def main():
  set_seed(42)
  log = RunLogger(name="moons")

  X, y = make_moons(n_samples=2000, noise=0.2, random_state=0)
  Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
  Xtr = torch.tensor(Xtr, dtype=torch.float32)
  ytr = torch.tensor(ytr, dtype=torch.long)
  Xte = torch.tensor(Xte, dtype=torch.float32)
  yte = torch.tensor(yte, dtype=torch.long)

  model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
  opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
  lossf = nn.CrossEntropyLoss()

  for epoch in range(10):
    model.train()
    opt.zero_grad()
    loss = lossf(model(Xtr), ytr)
    loss.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
      acc = (model(Xte).argmax(1) == yte).float().mean().item()
    log(epoch=epoch, loss=round(loss.item(), 4), val_acc=round(acc, 4))
    print(f"epoch {epoch:02d} loss={loss.item():.4f} val_acc={acc:.4f}")

if __name__ == "__main__":
  main()

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch05.html
# ────────────────────────────────────────────────────────────────────────
# tests/test_shapes.py
import torch
from torch import nn

def test_output_shape():
  model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
  out = model(torch.randn(5, 2))
  assert out.shape == (5, 2)

def test_backward_runs():
  model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
  loss = model(torch.randn(5, 2)).sum()
  loss.backward()
  assert model[0].weight.grad is not None

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-05-ch05.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import torch
import platform

def describe_hardware():
  info = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
  }
  if torch.cuda.is_available():
    info["gpu"] = torch.cuda.get_device_name(0)
    info["capability"] = torch.cuda.get_device_capability(0)
    info["memory_GB"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
  info["mps"] = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
  return info

print(describe_hardware())
# الناتج التقريبي:
# {'python': '3.14.6', 'torch': '2.14.1', 'cuda_available': True,
# 'gpu': 'NVIDIA RTX 4070', 'capability': (8, 9), 'memory_GB': 12.0, 'mps': False}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-05-ch05.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = False) -> None:
  """يثبّت كل مصادر العشوائية."""
  os.environ["PYTHONHASHSEED"] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
  else:
    torch.backends.cudnn.benchmark = True

set_seed(42)
a = torch.randn(1000)
set_seed(42)
b = torch.randn(1000)
print("identical:", torch.equal(a, b))  # الناتج: identical: True

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-05-ch05.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# تسجيل بسيط بدون تبعيات خارجية: JSONL
import json
import time
from pathlib import Path

class RunLogger:
  def __init__(self, out_dir="artifacts/logs", name="run"):
    self.path = Path(out_dir) / f"{name}.jsonl"
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self.t0 = time.perf_counter()

  def log(self, **kwargs):
    row = {"elapsed_s": round(time.perf_counter() - self.t0, 2), **kwargs}
    with self.path.open("a", encoding="utf-8") as fh:
      fh.write(json.dumps(row, ensure_ascii=False) + "\n")

  def config(self, cfg: dict):
    self.log(event="config", **cfg)

log = RunLogger(name="exp01")
log.config(lr=3e-4, batch_size=64, model="resnet50", seed=42)
for epoch in range(3):
  log.event = "epoch"
  log(epoch=epoch, train_loss=0.42 - epoch * 0.05, val_acc=0.90 + epoch * 0.01)
print("wrote", log.path)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-05-ch05.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# اختيار النواة برمجياً داخل السكربت (قبل استيراد keras)
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
print(keras.backend.backend())   # torch
print(keras.__version__)      # 3.15.1

# اختيار الجهاز
backend = keras.backend
if backend.backend() == "torch":
  import torch
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print("device:", device)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-05-ch05.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# src/train.py
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import sys
sys.path.append("src")

import torch
from torch import nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from utils.seed import set_seed
from utils.logging import RunLogger

def main():
  set_seed(42)
  log = RunLogger(name="moons")

  X, y = make_moons(n_samples=2000, noise=0.2, random_state=0)
  Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
  Xtr = torch.tensor(Xtr, dtype=torch.float32)
  ytr = torch.tensor(ytr, dtype=torch.long)
  Xte = torch.tensor(Xte, dtype=torch.float32)
  yte = torch.tensor(yte, dtype=torch.long)

  model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
  opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
  lossf = nn.CrossEntropyLoss()

  for epoch in range(10):
    model.train()
    opt.zero_grad()
    loss = lossf(model(Xtr), ytr)
    loss.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
      acc = (model(Xte).argmax(1) == yte).float().mean().item()
    log(epoch=epoch, loss=round(loss.item(), 4), val_acc=round(acc, 4))
    print(f"epoch {epoch:02d} loss={loss.item():.4f} val_acc={acc:.4f}")

if __name__ == "__main__":
  main()

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-05-ch05.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# tests/test_shapes.py
import torch
from torch import nn

def test_output_shape():
  model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
  out = model(torch.randn(5, 2))
  assert out.shape == (5, 2)

def test_backward_runs():
  model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
  loss = model(torch.randn(5, 2)).sum()
  loss.backward()
  assert model[0].weight.grad is not None
