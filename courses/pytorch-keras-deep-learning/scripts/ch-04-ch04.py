#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — كيف تتعلّم الشبكة؟ الانتشار العكسي ونزول التدرّج
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 04 — The Foundations

Book slug : pytorch-keras-deep-learning
Chapter   : ch-04-ch04 (lesson 4)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-04-ch04.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — كيف تتعلّم الشبكة؟ الانتشار العكسي ونزول التدرّج
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 04 — The Foundations

Book slug : pytorch-keras-deep-learning
Chapter   : ch-04-ch04 (lesson 4)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch04.html
# ────────────────────────────────────────────────────────────────────────
# انتشار عكسي مكتوب يدوياً — للتحقّق فقط (لا تستخدمه في الإنتاج)
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-z))

def forward_backward(x, y, w1, w2, b):
    z = w1 * x[0] + w2 * x[1] + b
    a = sigmoid(z)
    loss = 0.5 * (a - y) ** 2

    da = (a - y)                 # ∂L/∂a
    dz = da * a * (1 - a)        # ∂L/∂z  (σ'(z) = a(1-a))
    dw1 = dz * x[0]              # ∂L/∂w1
    dw2 = dz * x[1]
    db  = dz
    return loss, (dw1, dw2, db)

x = np.array([0.5, -1.2]); y = 1.0
w1, w2, b = 0.3, -0.2, 0.1
loss, grads = forward_backward(x, y, w1, w2, b)

# تحقّق عددي
eps = 1e-6
num = lambda f: (f(w1 + eps, w2, b) - f(w1 - eps, w2, b)) / (2 * eps)
loss_only = lambda a, b_, c: forward_backward(x, y, a, b_, c)[0]
print("analytic dw1:", round(grads[0], 8))
print("numeric  dw1:", round(num(loss_only), 8))
# الناتج: analytic dw1: 0.01946694  numeric dw1: 0.01946694

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch04.html
# ────────────────────────────────────────────────────────────────────────
import torch

# 1) تدرّج يدوي (للتعلّم فقط)
w = torch.tensor([0.3, -0.2], requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)
x = torch.tensor([0.5, -1.2])
y = torch.tensor(1.0)

a = torch.sigmoid((w * x).sum() + b)
loss = 0.5 * (a - y) ** 2
loss.backward()

print("grad w:", w.grad)      # الناتج: tensor([0.0244336, -0.0586406])
print("grad b:", b.grad)      # الناتج: tensor(0.0488672)

# 2) التحقّق من صحّة التنفيذ بأداة PyTorch نفسها
from torch.autograd import gradcheck

def f(params):
    ww, bb = params
    out = torch.sigmoid((ww * x).sum() + bb)
    return 0.5 * (out - y) ** 2

print("gradcheck:", gradcheck(f, (w.detach().clone().requires_grad_(),
                                  b.detach().clone().requires_grad_())))
# الناتج: gradcheck: True

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch04.html
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn

model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(10):
    model.train()                      # 1) وضع التدريب (يبدّل سلوك Dropout/BatchNorm)
    for xb, yb in loader:              # 2) جلب دفعة
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)   # 3) تصفير التدرّجات
        pred = model(xb).squeeze(1)            # 4) تمرير أمامي
        loss = loss_fn(pred, yb.float())       # 5) الخسارة
        loss.backward()                        # 6) انتشار عكسي
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 7) قصّ التدرّج
        optimizer.step()                       # 8) تحديث الأوزان
    # 9) تقييم على التحقّق + 10) جدولة معدّل التعلّم
    scheduler.step()

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch04.html
# ────────────────────────────────────────────────────────────────────────
import torch

def sgd(params, grads, lr):
    for p, g in zip(params, grads):
        p.data -= lr * g

def momentum(params, grads, lr, velocity, beta=0.9):
    for i, (p, g) in enumerate(zip(params, grads)):
        velocity[i] = beta * velocity[i] + g
        p.data -= lr * velocity[i]

def adam(params, grads, lr, m, v, t, beta1=0.9, beta2=0.999, eps=1e-8):
    for i, (p, g) in enumerate(zip(params, grads)):
        m[i] = beta1 * m[i] + (1 - beta1) * g
        v[i] = beta2 * v[i] + (1 - beta2) * g ** 2
        mhat = m[i] / (1 - beta1 ** t)
        vhat = v[i] / (1 - beta2 ** t)
        p.data -= lr * mhat / (torch.sqrt(vhat) + eps)

# تشغيل المقارنة على problem بسيط: y = w*x
w = torch.tensor([5.0], requires_grad=True)
x, y = torch.tensor([1.0]), torch.tensor([3.0])
opt = torch.optim.SGD([w], lr=0.05)   # أو جرّب Adam على نفس المسألة
for _ in range(400):
    opt.zero_grad()
    loss = ((w * x - y) ** 2).mean()
    loss.backward()
    opt.step()
print("w =", w.item())   # الناتج: w = tensor(3.0000) بعد 400 خطوة

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch04.html
# ────────────────────────────────────────────────────────────────────────
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

model = torch.nn.Linear(20, 1)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

# جدول cosine بسيط
sched = CosineAnnealingLR(opt, T_max=100)
for epoch in range(100):
    # ... (دورة التدريب) ...
    sched.step()

# OneCycle: إقلاع ثم انحدار (الأداء الأعلى عادةً في CNN)
opt2 = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
sched2 = OneCycleLR(opt2, max_lr=0.5, epochs=50, steps_per_epoch=10)
print("start lr:", sched2.get_last_lr())
# الناتج: start lr: [0.0049999999...]

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch04.html
# ────────────────────────────────────────────────────────────────────────
import torch

def adam_manual(params, lr, m, v, t, beta1=0.9, beta2=0.999, eps=1e-8):
    for i, p in enumerate(params):
        g = p.grad
        m[i] = beta1 * m[i] + (1 - beta1) * g
        v[i] = beta2 * v[i] + (1 - beta2) * g ** 2
        mhat = m[i] / (1 - beta1 ** t)
        vhat = v[i] / (1 - beta2 ** t)
        p.data -= lr * mhat / (torch.sqrt(vhat) + eps)

def run(lr, beta2, manual=True, epochs=300):
    torch.manual_seed(0)
    w = torch.tensor([5.0], requires_grad=True)
    x, y = torch.tensor([1.0]), torch.tensor([3.0])
    m = [torch.zeros_like(w)]; v = [torch.zeros_like(w)]
    opt = None if manual else torch.optim.Adam([w], lr=lr, betas=(0.9, beta2))
    for t in range(1, epochs + 1):
        if manual:
            w.grad = None
            loss = ((w * x - y) ** 2).mean()
            loss.backward()
            adam_manual([w], lr, m, v, t, beta2=beta2)
        else:
            opt.zero_grad()
            loss = ((w * x - y) ** 2).mean()
            loss.backward()
            opt.step()
    return w.item()

print("manual   :", round(run(0.05, 0.999, True), 4))
print("builtin  :", round(run(0.05, 0.999, False), 4))
print("beta2=0.99:", round(run(0.05, 0.99, True), 4))

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch04.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=1000, noise=0.2, random_state=0)
Xt = torch.tensor(X, dtype=torch.float32)
yt = torch.tensor(y, dtype=torch.long)

def train(lr, clip=None, epochs=200):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 2))
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    lossf = nn.CrossEntropyLoss()
    hist = []
    for ep in range(epochs):
        opt.zero_grad()
        loss = lossf(model(Xt), yt)
        loss.backward()
        if clip:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        hist.append(loss.item())
    return hist

for lr in (0.5, 0.05, 0.005):
    h = train(lr)
    print(f"lr={lr:<7} first={h[0]:.4f} mid={h[100]:.4f} last={h[-1]:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-04-ch04.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# انتشار عكسي مكتوب يدوياً — للتحقّق فقط (لا تستخدمه في الإنتاج)
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-z))

def forward_backward(x, y, w1, w2, b):
    z = w1 * x[0] + w2 * x[1] + b
    a = sigmoid(z)
    loss = 0.5 * (a - y) ** 2

    da = (a - y)                 # ∂L/∂a
    dz = da * a * (1 - a)        # ∂L/∂z  (σ'(z) = a(1-a))
    dw1 = dz * x[0]              # ∂L/∂w1
    dw2 = dz * x[1]
    db  = dz
    return loss, (dw1, dw2, db)

x = np.array([0.5, -1.2]); y = 1.0
w1, w2, b = 0.3, -0.2, 0.1
loss, grads = forward_backward(x, y, w1, w2, b)

# تحقّق عددي
eps = 1e-6
num = lambda f: (f(w1 + eps, w2, b) - f(w1 - eps, w2, b)) / (2 * eps)
loss_only = lambda a, b_, c: forward_backward(x, y, a, b_, c)[0]
print("analytic dw1:", round(grads[0], 8))
print("numeric  dw1:", round(num(loss_only), 8))
# الناتج: analytic dw1: 0.01946694  numeric dw1: 0.01946694

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-04-ch04.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch

# 1) تدرّج يدوي (للتعلّم فقط)
w = torch.tensor([0.3, -0.2], requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)
x = torch.tensor([0.5, -1.2])
y = torch.tensor(1.0)

a = torch.sigmoid((w * x).sum() + b)
loss = 0.5 * (a - y) ** 2
loss.backward()

print("grad w:", w.grad)      # الناتج: tensor([0.0244336, -0.0586406])
print("grad b:", b.grad)      # الناتج: tensor(0.0488672)

# 2) التحقّق من صحّة التنفيذ بأداة PyTorch نفسها
from torch.autograd import gradcheck

def f(params):
    ww, bb = params
    out = torch.sigmoid((ww * x).sum() + bb)
    return 0.5 * (out - y) ** 2

print("gradcheck:", gradcheck(f, (w.detach().clone().requires_grad_(),
                                  b.detach().clone().requires_grad_())))
# الناتج: gradcheck: True

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-04-ch04.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn

model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(10):
    model.train()                      # 1) وضع التدريب (يبدّل سلوك Dropout/BatchNorm)
    for xb, yb in loader:              # 2) جلب دفعة
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)   # 3) تصفير التدرّجات
        pred = model(xb).squeeze(1)            # 4) تمرير أمامي
        loss = loss_fn(pred, yb.float())       # 5) الخسارة
        loss.backward()                        # 6) انتشار عكسي
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 7) قصّ التدرّج
        optimizer.step()                       # 8) تحديث الأوزان
    # 9) تقييم على التحقّق + 10) جدولة معدّل التعلّم
    scheduler.step()

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-04-ch04.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch

def sgd(params, grads, lr):
    for p, g in zip(params, grads):
        p.data -= lr * g

def momentum(params, grads, lr, velocity, beta=0.9):
    for i, (p, g) in enumerate(zip(params, grads)):
        velocity[i] = beta * velocity[i] + g
        p.data -= lr * velocity[i]

def adam(params, grads, lr, m, v, t, beta1=0.9, beta2=0.999, eps=1e-8):
    for i, (p, g) in enumerate(zip(params, grads)):
        m[i] = beta1 * m[i] + (1 - beta1) * g
        v[i] = beta2 * v[i] + (1 - beta2) * g ** 2
        mhat = m[i] / (1 - beta1 ** t)
        vhat = v[i] / (1 - beta2 ** t)
        p.data -= lr * mhat / (torch.sqrt(vhat) + eps)

# تشغيل المقارنة على problem بسيط: y = w*x
w = torch.tensor([5.0], requires_grad=True)
x, y = torch.tensor([1.0]), torch.tensor([3.0])
opt = torch.optim.SGD([w], lr=0.05)   # أو جرّب Adam على نفس المسألة
for _ in range(400):
    opt.zero_grad()
    loss = ((w * x - y) ** 2).mean()
    loss.backward()
    opt.step()
print("w =", w.item())   # الناتج: w = tensor(3.0000) بعد 400 خطوة

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-04-ch04.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

model = torch.nn.Linear(20, 1)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

# جدول cosine بسيط
sched = CosineAnnealingLR(opt, T_max=100)
for epoch in range(100):
    # ... (دورة التدريب) ...
    sched.step()

# OneCycle: إقلاع ثم انحدار (الأداء الأعلى عادةً في CNN)
opt2 = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
sched2 = OneCycleLR(opt2, max_lr=0.5, epochs=50, steps_per_epoch=10)
print("start lr:", sched2.get_last_lr())
# الناتج: start lr: [0.0049999999...]

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-04-ch04.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import torch

def adam_manual(params, lr, m, v, t, beta1=0.9, beta2=0.999, eps=1e-8):
    for i, p in enumerate(params):
        g = p.grad
        m[i] = beta1 * m[i] + (1 - beta1) * g
        v[i] = beta2 * v[i] + (1 - beta2) * g ** 2
        mhat = m[i] / (1 - beta1 ** t)
        vhat = v[i] / (1 - beta2 ** t)
        p.data -= lr * mhat / (torch.sqrt(vhat) + eps)

def run(lr, beta2, manual=True, epochs=300):
    torch.manual_seed(0)
    w = torch.tensor([5.0], requires_grad=True)
    x, y = torch.tensor([1.0]), torch.tensor([3.0])
    m = [torch.zeros_like(w)]; v = [torch.zeros_like(w)]
    opt = None if manual else torch.optim.Adam([w], lr=lr, betas=(0.9, beta2))
    for t in range(1, epochs + 1):
        if manual:
            w.grad = None
            loss = ((w * x - y) ** 2).mean()
            loss.backward()
            adam_manual([w], lr, m, v, t, beta2=beta2)
        else:
            opt.zero_grad()
            loss = ((w * x - y) ** 2).mean()
            loss.backward()
            opt.step()
    return w.item()

print("manual   :", round(run(0.05, 0.999, True), 4))
print("builtin  :", round(run(0.05, 0.999, False), 4))
print("beta2=0.99:", round(run(0.05, 0.99, True), 4))

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-04-ch04.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=1000, noise=0.2, random_state=0)
Xt = torch.tensor(X, dtype=torch.float32)
yt = torch.tensor(y, dtype=torch.long)

def train(lr, clip=None, epochs=200):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 2))
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    lossf = nn.CrossEntropyLoss()
    hist = []
    for ep in range(epochs):
        opt.zero_grad()
        loss = lossf(model(Xt), yt)
        loss.backward()
        if clip:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        hist.append(loss.item())
    return hist

for lr in (0.5, 0.05, 0.005):
    h = train(lr)
    print(f"lr={lr:<7} first={h[0]:.4f} mid={h[100]:.4f} last={h[-1]:.4f}")
