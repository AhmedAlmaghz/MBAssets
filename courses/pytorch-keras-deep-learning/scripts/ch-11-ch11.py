#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — التنظيم والتعميم
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 11 — Deep Dive into PyTorch

Book slug : pytorch-keras-deep-learning
Chapter   : ch-11-ch11 (lesson 11)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-11-ch11.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — التنظيم والتعميم
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 11 — Deep Dive into PyTorch

Book slug : pytorch-keras-deep-learning
Chapter   : ch-11-ch11 (lesson 11)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch11.html
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn

# 1) الطريقة الصحيحة: مجموعتا معاملات ( decaying للمصفوفات، بلا decay للـbias/BN)
def param_groups(model, weight_decay=0.01):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "bn" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.BatchNorm1d(256),
                      nn.Linear(256, 10))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
# أو الأدق:
opt = torch.optim.AdamW(param_groups(model, 0.01), lr=1e-3)

# 2) L2 صريح (بديل عن weight decay)
l2 = nn.MSELoss() + 1e-4 * sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch11.html
# ────────────────────────────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(784, 512), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 10),
)

# أثناء التدريب: model.train()  ⇒ dropout فعّال
# أثناء التقييم: model.eval()   ⇒ dropout معطّل
# measurements: فعّل/تعطيل تلقائياً
model.train()
out_train = model(x_train[:8])
model.eval()
with torch.no_grad():
    out_eval = model(x_train[:8])
print("different outputs:", not torch.allclose(out_train, out_eval))

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch11.html
# ────────────────────────────────────────────────────────────────────────
# Mixup: يخلط صورتين وأهدافهما — تنظيم قوي جداً بلا معاملات إضافية
def mixup(x, y, alpha=0.2):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0))
    x_mix = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return x_mix, y_a, y_b, lam

# في حلقة التدريب:
# loss = lam * lossf(out, y_a) + (1 - lam) * lossf(out, y_b)

# Label smoothing: نخفّف التسميات الصعبة
lossf = nn.CrossEntropyLoss(label_smoothing=0.1)

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch11.html
# ────────────────────────────────────────────────────────────────────────
# الفكرة: طبّع ثم أعد القياس بمعاملَين قابلَين للتعلّم
class LayerNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mu) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch11.html
# ────────────────────────────────────────────────────────────────────────
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

best_val = float("inf")
patience_counter = 0
for epoch in range(100):
    model.train()
    for xb, yb in train_loader:
        opt.zero_grad(set_to_none=True)
        loss = lossf(model(xb), yb)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_loss = sum(lossf(model(xb), yb).item() for xb, yb in val_loader) / len(val_loader)

    scheduler.step(val_loss)

    if val_loss < best_val - 1e-4:
        best_val = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best.pt")
    else:
        patience_counter += 1
        if patience_counter >= 8:          # صبر (patience) = 8 حقبات
            print(f"early stop at epoch {epoch}")
            break

model.load_state_dict(torch.load("best.pt"))   # استرجع أفضل الأوزان

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch11.html
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=1000, noise=0.25, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
Xtr = torch.tensor(Xtr, dtype=torch.float32); ytr = torch.tensor(ytr, dtype=torch.long)
Xte = torch.tensor(Xte, dtype=torch.float32); yte = torch.tensor(yte, dtype=torch.long)

def run(dropout=0.0, wd=0.0, epochs=200, patience=10):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Dropout(dropout),
                           nn.Linear(64, 64), nn.ReLU(), nn.Dropout(dropout),
                           nn.Linear(64, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    best, bad, best_state = 1e9, 0, None
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        lossf(model(Xtr), ytr).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xte), yte).item()
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
    return best, acc, ep

for name, kw in (("بلا تنظيم", {}), ("dropout", {"dropout": 0.5}),
                 ("wd", {"wd": 0.01}), ("كلاهما", {"dropout": 0.4, "wd": 0.01})):
    loss, acc, ep = run(**kw)
    print(f"{name:12s} val_loss={loss:.4f} acc={acc:.4f} stopped@{ep}")

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch11.html
# ────────────────────────────────────────────────────────────────────────
import hashlib
import json
import platform
import subprocess
from pathlib import Path


def environment_fingerprint() -> dict:
    """بصمة البيئة: تُحفظ بجانب النموذج للتحقّق من التطابق."""
    import torch
    import keras

    result = subprocess.run(
        ["pip", "freeze"], capture_output=True, text=True, check=False
    )
    packages = sorted(result.stdout.split())
    digest = hashlib.sha256("\n".join(packages).encode()).hexdigest()

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "keras": keras.__version__,
        "keras_backend": keras.backend.backend(),
        "packages_sha256": digest,
        "packages": packages,
    }


def save_fingerprint(path: str = "env.json") -> None:
    Path(path).write_text(
        json.dumps(environment_fingerprint(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    save_fingerprint()
    print(json.dumps(environment_fingerprint(), ensure_ascii=False, indent=2))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-11-ch11.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn

# 1) الطريقة الصحيحة: مجموعتا معاملات ( decaying للمصفوفات، بلا decay للـbias/BN)
def param_groups(model, weight_decay=0.01):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "bn" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.BatchNorm1d(256),
                      nn.Linear(256, 10))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
# أو الأدق:
opt = torch.optim.AdamW(param_groups(model, 0.01), lr=1e-3)

# 2) L2 صريح (بديل عن weight decay)
l2 = nn.MSELoss() + 1e-4 * sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-11-ch11.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(784, 512), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 10),
)

# أثناء التدريب: model.train()  ⇒ dropout فعّال
# أثناء التقييم: model.eval()   ⇒ dropout معطّل
# measurements: فعّل/تعطيل تلقائياً
model.train()
out_train = model(x_train[:8])
model.eval()
with torch.no_grad():
    out_eval = model(x_train[:8])
print("different outputs:", not torch.allclose(out_train, out_eval))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-11-ch11.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# Mixup: يخلط صورتين وأهدافهما — تنظيم قوي جداً بلا معاملات إضافية
def mixup(x, y, alpha=0.2):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0))
    x_mix = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return x_mix, y_a, y_b, lam

# في حلقة التدريب:
# loss = lam * lossf(out, y_a) + (1 - lam) * lossf(out, y_b)

# Label smoothing: نخفّف التسميات الصعبة
lossf = nn.CrossEntropyLoss(label_smoothing=0.1)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-11-ch11.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# الفكرة: طبّع ثم أعد القياس بمعاملَين قابلَين للتعلّم
class LayerNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mu) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-11-ch11.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

best_val = float("inf")
patience_counter = 0
for epoch in range(100):
    model.train()
    for xb, yb in train_loader:
        opt.zero_grad(set_to_none=True)
        loss = lossf(model(xb), yb)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_loss = sum(lossf(model(xb), yb).item() for xb, yb in val_loader) / len(val_loader)

    scheduler.step(val_loss)

    if val_loss < best_val - 1e-4:
        best_val = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best.pt")
    else:
        patience_counter += 1
        if patience_counter >= 8:          # صبر (patience) = 8 حقبات
            print(f"early stop at epoch {epoch}")
            break

model.load_state_dict(torch.load("best.pt"))   # استرجع أفضل الأوزان

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-11-ch11.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=1000, noise=0.25, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
Xtr = torch.tensor(Xtr, dtype=torch.float32); ytr = torch.tensor(ytr, dtype=torch.long)
Xte = torch.tensor(Xte, dtype=torch.float32); yte = torch.tensor(yte, dtype=torch.long)

def run(dropout=0.0, wd=0.0, epochs=200, patience=10):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Dropout(dropout),
                           nn.Linear(64, 64), nn.ReLU(), nn.Dropout(dropout),
                           nn.Linear(64, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    best, bad, best_state = 1e9, 0, None
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        lossf(model(Xtr), ytr).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xte), yte).item()
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
    return best, acc, ep

for name, kw in (("بلا تنظيم", {}), ("dropout", {"dropout": 0.5}),
                 ("wd", {"wd": 0.01}), ("كلاهما", {"dropout": 0.4, "wd": 0.01})):
    loss, acc, ep = run(**kw)
    print(f"{name:12s} val_loss={loss:.4f} acc={acc:.4f} stopped@{ep}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-11-ch11.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import hashlib
import json
import platform
import subprocess
from pathlib import Path


def environment_fingerprint() -> dict:
    """بصمة البيئة: تُحفظ بجانب النموذج للتحقّق من التطابق."""
    import torch
    import keras

    result = subprocess.run(
        ["pip", "freeze"], capture_output=True, text=True, check=False
    )
    packages = sorted(result.stdout.split())
    digest = hashlib.sha256("\n".join(packages).encode()).hexdigest()

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "keras": keras.__version__,
        "keras_backend": keras.backend.backend(),
        "packages_sha256": digest,
        "packages": packages,
    }


def save_fingerprint(path: str = "env.json") -> None:
    Path(path).write_text(
        json.dumps(environment_fingerprint(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    save_fingerprint()
    print(json.dumps(environment_fingerprint(), ensure_ascii=False, indent=2))
