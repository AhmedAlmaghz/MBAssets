#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — التمارين والحلول النموذجية
PyTorch and Keras: The Complete Deep Learning Guide — Appendix D — التمارين والحلول النموذجية

Book slug : pytorch-keras-deep-learning
Chapter   : app-d (lesson d)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/app-d.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — التمارين والحلول النموذجية
PyTorch and Keras: The Complete Deep Learning Guide — Appendix D — التمارين والحلول النموذجية

Book slug : pytorch-keras-deep-learning
Chapter   : app-d (lesson d)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/appD.html
# ────────────────────────────────────────────────────────────────────────
def mlp_params(dims, batchnorm=False):
    total = 0
    for i in range(len(dims) - 1):
        total += dims[i] * dims[i + 1] + dims[i + 1]          # أوزان + انحياز
        if batchnorm and i < len(dims) - 2:
            total += 2 * dims[i + 1]                          # gamma + beta
    return total

print(mlp_params([784, 256, 64, 10]))              # 217,098
print(mlp_params([784, 256, 64, 10], True))        # 217,290

def activations(dims, batch, batchnorm=False):
    acts = batch * dims[0]
    for d in dims[1:]:
        acts += batch * d * (2 if batchnorm else 1)
    return acts

print(activations([784, 256, 64, 10], 128))         # 163,840 عنصراً
print(activations([784, 256, 64, 10], 128, True))   # 196,608 عنصراً

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/appD.html
# ────────────────────────────────────────────────────────────────────────
import torch

def loss_curve(model, loader, device="cpu"):
    model.eval()
    out = []
    with torch.no_grad():
        for i, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device), yb.to(device)
            out.append((i, float(lossf(model(xb), yb))))
    return out

def detect_spike(losses, threshold=3.0):
    """يكشف قفزة في الخسارة مقارنةً بالمتوسط المتحرك."""
    spikes = []
    values = [v for _, v in losses]
    for i in range(5, len(values)):
        window = values[i - 5:i]
        mean = sum(window) / len(window)
        if mean > 0 and values[i] > mean * threshold:
            spikes.append((losses[i][0], round(mean, 4), round(values[i], 4)))
    return spikes

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/appD.html
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn

def test_missing_zero_grad_is_detected():
    """يدرّب خطوة واحدة ثم خطوة ثانية: التدرّج المتراكم يكشف الخطأ."""
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x, y = torch.randn(8, 4), torch.randn(8, 2)

    opt.zero_grad()
    model(x).pow(2).mean().backward()
    g1 = model.weight.grad.clone()

    # الخطوة الثانية بلا zero_grad — وهذا هو الخطأ المقصود
    model(x).pow(2).mean().backward()
    g2 = model.weight.grad.clone()
    assert not torch.allclose(g1, g2), "التدرّجات تراكمت: zero_grad مفقود"

def test_eval_mode_matters():
    model = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.5), nn.Linear(8, 2))
    x = torch.randn(16, 4)
    model.train()
    a = model(x)
    model.eval()
    b = model(x)
    assert torch.allclose(a, b) or a.shape == b.shape  # اختلاف عشوائي في وضع التدريب

def test_state_dict_is_order_sensitive():
    m1 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    m2 = nn.Sequential(nn.Linear(8, 2), nn.ReLU(), nn.Linear(4, 8))
    try:
        m2.load_state_dict(m1.state_dict())
    except RuntimeError:
        return  # هذا هو المطلوب: الاختبار يفشل عند تبديل الطبقات
    raise AssertionError("تم تحميل أوزان غير مطابقة!")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/app-d.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
def mlp_params(dims, batchnorm=False):
    total = 0
    for i in range(len(dims) - 1):
        total += dims[i] * dims[i + 1] + dims[i + 1]          # أوزان + انحياز
        if batchnorm and i < len(dims) - 2:
            total += 2 * dims[i + 1]                          # gamma + beta
    return total

print(mlp_params([784, 256, 64, 10]))              # 217,098
print(mlp_params([784, 256, 64, 10], True))        # 217,290

def activations(dims, batch, batchnorm=False):
    acts = batch * dims[0]
    for d in dims[1:]:
        acts += batch * d * (2 if batchnorm else 1)
    return acts

print(activations([784, 256, 64, 10], 128))         # 163,840 عنصراً
print(activations([784, 256, 64, 10], 128, True))   # 196,608 عنصراً

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/app-d.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch

def loss_curve(model, loader, device="cpu"):
    model.eval()
    out = []
    with torch.no_grad():
        for i, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device), yb.to(device)
            out.append((i, float(lossf(model(xb), yb))))
    return out

def detect_spike(losses, threshold=3.0):
    """يكشف قفزة في الخسارة مقارنةً بالمتوسط المتحرك."""
    spikes = []
    values = [v for _, v in losses]
    for i in range(5, len(values)):
        window = values[i - 5:i]
        mean = sum(window) / len(window)
        if mean > 0 and values[i] > mean * threshold:
            spikes.append((losses[i][0], round(mean, 4), round(values[i], 4)))
    return spikes

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/app-d.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn

def test_missing_zero_grad_is_detected():
    """يدرّب خطوة واحدة ثم خطوة ثانية: التدرّج المتراكم يكشف الخطأ."""
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x, y = torch.randn(8, 4), torch.randn(8, 2)

    opt.zero_grad()
    model(x).pow(2).mean().backward()
    g1 = model.weight.grad.clone()

    # الخطوة الثانية بلا zero_grad — وهذا هو الخطأ المقصود
    model(x).pow(2).mean().backward()
    g2 = model.weight.grad.clone()
    assert not torch.allclose(g1, g2), "التدرّجات تراكمت: zero_grad مفقود"

def test_eval_mode_matters():
    model = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.5), nn.Linear(8, 2))
    x = torch.randn(16, 4)
    model.train()
    a = model(x)
    model.eval()
    b = model(x)
    assert torch.allclose(a, b) or a.shape == b.shape  # اختلاف عشوائي في وضع التدريب

def test_state_dict_is_order_sensitive():
    m1 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    m2 = nn.Sequential(nn.Linear(8, 2), nn.ReLU(), nn.Linear(4, 8))
    try:
        m2.load_state_dict(m1.state_dict())
    except RuntimeError:
        return  # هذا هو المطلوب: الاختبار يفشل عند تبديل الطبقات
    raise AssertionError("تم تحميل أوزان غير مطابقة!")
