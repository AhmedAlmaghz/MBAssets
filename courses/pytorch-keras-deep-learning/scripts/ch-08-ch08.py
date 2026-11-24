#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — Autograd: محرّك التمايز التلقائي
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 08 — Autograd

Book slug : pytorch-keras-deep-learning
Chapter   : ch-08-ch08 (lesson 8)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-08-ch08.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — Autograd: محرّك التمايز التلقائي
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 08 — Autograd

Book slug : pytorch-keras-deep-learning
Chapter   : ch-08-ch08 (lesson 8)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch08.html
# ────────────────────────────────────────────────────────────────────────
import torch

# الاشتقاق اليدوي: f(x) = x^2  =>  f'(x) = 2x
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)       # الناتج: tensor([4., 6.])   (2 × [2, 3])

# لا يوجد سطر واحد يكتب الاشتقاق: المحرّك يعرف أنّ SquareDerivative = 2x

# من المتغيّر إلى كل شيء آخر في الرسم
a = torch.tensor(2.0, requires_grad=True)
b = a ** 3
c = b * 4
print(b.grad_fn)    # الناتج:
print(c.grad_fn)    # الناتج:
print(c.requires_grad)   # الناتج: True  (خاصية الرصد تنتقل تلقائياً)

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch08.html
# ────────────────────────────────────────────────────────────────────────
import torch

w = torch.tensor(2.0, requires_grad=True)

def square_internally(v):
    inner = v * v          # مرصودة لأن v مرصودة
    return inner

out = square_internally(w)
out.backward()
print(w.grad)        # الناتج: tensor(4.)   الرسم وصل إلى w

# الحالة الخطرة: شدّة مؤقتة تُبنى داخل الدالة
def bad_scale(v):
    tmp = v.detach() * 3   # detach يقطع الرصد
    return tmp

out2 = bad_scale(w)
print(out2.requires_grad)   # الناتج: False
try:
    out2.backward()
except RuntimeError as err:
    print("RuntimeError:", err)
    # الناتج: RuntimeError: element 0 of tensors does not require grad ...

# operations التي تقطع الرصد
z = w * 2
print(z.item())          # item() تعيد Python float وتقطع الرصد
print(z.numpy())        # الناتج: array([4.], dtype=float32)  ( cutting the graph )

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch08.html
# ────────────────────────────────────────────────────────────────────────
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)

loss = (x ** 2).sum()
loss.backward()
print(x.grad)          # الناتج: tensor([2., 4.])

loss2 = (x ** 2).sum()  # الخسارة نفسها مرة أخرى
loss2.backward()
print(x.grad)          # الناتج: tensor([4., 8.])  (تراكمت!)

# --- طريقة بديلة: لا تتراكم ولا تكتب في .grad ---
g = torch.autograd.grad(loss, x, retain_graph=True)
print(g)               # الناتج: (tensor([2., 4.]),)  تُعيد مجموعة
print(x.grad)          # الناتج: tensor([2., 4.])  لم تتغيّر

# --- المشتقّة الثانية: تحتاج retain_graph وcreate_graph ---
g1 = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)
g2 = torch.autograd.grad(g1[0].sum(), x)
print(g2)              # الناتج: (tensor([2., 2.]),)  المشتقّة الثانية لـ x^2

# --- تدرّج مع شدّة لم تُراقب:.grad = None ---
p = torch.tensor(3.0)                 # بدون requires_grad
print(p.grad)         # الناتج: None

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch08.html
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn

model = nn.Linear(4, 2)
x = torch.randn(3, 4, requires_grad=True)

with torch.no_grad():
    a = model(x)
print(a.requires_grad)     # الناتج: False

with torch.inference_mode():
    b = model(x)
print(b.requires_grad)     # الناتج: False

# الفرق الجوهري: ناتج inference_mode لا يمكن أن يُستعمل لاحقاً في رسم تدرّج
try:
    c = (b * 2).sum()
    c.backward()
except RuntimeError as err:
    print("RuntimeError:", err)
    # الناتج: RuntimeError: element 0 of tensors does not require grad ...

# detach: نسخة بلا رصد، قابلة للاستعمال في حساب لاحق
d = x.detach()
print(d.requires_grad)     # الناتج: False
e = (d * 3).sum()
print(e.requires_grad)     # الناتج: False

#.detach_ بم_SIGNES المعاكس: تقطع الرصد في مكانه
y = x * 2
print(y.requires_grad)     # الناتج: True
y.detach_()
print(y.requires_grad)     # الناتج: False

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch08.html
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn

# 1) خطّاف على شدّة: يعدّل التدرّج أو يراقبه
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

def scale_grad(grad):
    print("grad seen by hook:", grad.item())
    return grad * 10        # إرجاع قيمة جديدة يعدّل التدرّج

h = y.register_hook(scale_grad)
y.backward()
print(x.grad)                # الناتج: tensor(40.)  لأن الخطّاف ضاعف 4
h.remove()

# 2) خطّاف خلفي كامل على وحدة: يرى تدرّج الخرج وتدرّج المدخل
class Doubler(nn.Module):
    def forward(self, t):
        return t * 2

d = Doubler()
records = []
hb = d.register_full_backward_hook(
    lambda module, grad_input, grad_output: records.append((grad_input[0], grad_output[0]))
)
t = torch.tensor(3.0, requires_grad=True)
d(t).backward()
hb.remove()
print(records[0][1].item())   # الناتج: 2.0  (تدرّج الخرج)
print(records[0][0].item())   # الناتج: 2.0  (تدرّج المدخل: d(2t)/dt = 2)

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch08.html
# ────────────────────────────────────────────────────────────────────────
import keras

# train_step مخصّص في Keras 3: يتصرّف بشكل مختلف حسب النواة
class WeightedLossModel(keras.Model):
    def __init__(self, units=16):
        super().__init__()
        self.d1 = keras.layers.Dense(units, activation="relu")
        self.d2 = keras.layers.Dense(1)

    def call(self, x, training=False):
        h = self.d1(x, training=training)
        return self.d2(h)

    def train_step(self, data):
        x, y = data
        with self.compute_loss(x, y, sample_weight=None) as loss:
            y_pred = self(x, training=True)
            loss = keras.ops.mean(keras.ops.square(y_pred - y), axis=-1)
        # على نواة torch ينفّذ المحسّن step() بنفسه؛ الوسيط أعلاه يوفّر الخسارة
        return self.compute_metrics(x, y, y_pred, sample_weight=None)

model = WeightedLossModel()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

import numpy as np
rng = np.random.default_rng(0)
x = rng.normal(size=(256, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

model.fit(x, y, batch_size=32, epochs=2, verbose=0)
print("training finished")

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch08.html
# ────────────────────────────────────────────────────────────────────────
import torch

def f(x):
    return 3 * x ** 2 + 2 * x - 5

x = torch.tensor(1.0, requires_grad=True)
y = f(x)
y.backward()
print(x.grad)          # الناتج: tensor(8.)   لأن 6*1 + 2 = 8

# المشتقّة الثانية
x2 = torch.tensor(1.0, requires_grad=True)
y2 = f(x2)
g1 = torch.autograd.grad(y2, x2, create_graph=True, retain_graph=True)
g2 = torch.autograd.grad(g1[0].sum(), x2)
print(g2)              # الناتج: (tensor(6.),)

# بلا تراكم
x3 = torch.tensor(1.0, requires_grad=True)
y3 = f(x3)
g = torch.autograd.grad(y3, x3)
print(g, x3.grad)      # الناتج: (tensor(8.),) None

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch08.html
# ────────────────────────────────────────────────────────────────────────
import torch

def square_and_scale(v, k=2.0):
    return (v ** 2) * k

x = torch.tensor([2.0, 3.0], requires_grad=True)
y = square_and_scale(x)

def watch(grad):
    print("hook saw:", grad.shape, grad.tolist())
    return grad

h = y.register_hook(watch)
y.backward()
print("x.grad =", x.grad.tolist())   # الناتج: [16.0, 36.0]
h.remove()

# خطّاف يعدّل التدرّج
x2 = torch.tensor([2.0, 3.0], requires_grad=True)
y2 = square_and_scale(x2)
h2 = y2.register_hook(lambda g: g * 0.5)
y2.backward()
print("x.grad =", x2.grad.tolist())  # الناتج: [8.0, 18.0]
h2.remove()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-08-ch08.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import torch

# الاشتقاق اليدوي: f(x) = x^2  =>  f'(x) = 2x
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)       # الناتج: tensor([4., 6.])   (2 × [2, 3])

# لا يوجد سطر واحد يكتب الاشتقاق: المحرّك يعرف أنّ SquareDerivative = 2x

# من المتغيّر إلى كل شيء آخر في الرسم
a = torch.tensor(2.0, requires_grad=True)
b = a ** 3
c = b * 4
print(b.grad_fn)    # الناتج:
print(c.grad_fn)    # الناتج:
print(c.requires_grad)   # الناتج: True  (خاصية الرصد تنتقل تلقائياً)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-08-ch08.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch

w = torch.tensor(2.0, requires_grad=True)

def square_internally(v):
    inner = v * v          # مرصودة لأن v مرصودة
    return inner

out = square_internally(w)
out.backward()
print(w.grad)        # الناتج: tensor(4.)   الرسم وصل إلى w

# الحالة الخطرة: شدّة مؤقتة تُبنى داخل الدالة
def bad_scale(v):
    tmp = v.detach() * 3   # detach يقطع الرصد
    return tmp

out2 = bad_scale(w)
print(out2.requires_grad)   # الناتج: False
try:
    out2.backward()
except RuntimeError as err:
    print("RuntimeError:", err)
    # الناتج: RuntimeError: element 0 of tensors does not require grad ...

# operations التي تقطع الرصد
z = w * 2
print(z.item())          # item() تعيد Python float وتقطع الرصد
print(z.numpy())        # الناتج: array([4.], dtype=float32)  ( cutting the graph )

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-08-ch08.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)

loss = (x ** 2).sum()
loss.backward()
print(x.grad)          # الناتج: tensor([2., 4.])

loss2 = (x ** 2).sum()  # الخسارة نفسها مرة أخرى
loss2.backward()
print(x.grad)          # الناتج: tensor([4., 8.])  (تراكمت!)

# --- طريقة بديلة: لا تتراكم ولا تكتب في .grad ---
g = torch.autograd.grad(loss, x, retain_graph=True)
print(g)               # الناتج: (tensor([2., 4.]),)  تُعيد مجموعة
print(x.grad)          # الناتج: tensor([2., 4.])  لم تتغيّر

# --- المشتقّة الثانية: تحتاج retain_graph وcreate_graph ---
g1 = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)
g2 = torch.autograd.grad(g1[0].sum(), x)
print(g2)              # الناتج: (tensor([2., 2.]),)  المشتقّة الثانية لـ x^2

# --- تدرّج مع شدّة لم تُراقب:.grad = None ---
p = torch.tensor(3.0)                 # بدون requires_grad
print(p.grad)         # الناتج: None

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-08-ch08.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn

model = nn.Linear(4, 2)
x = torch.randn(3, 4, requires_grad=True)

with torch.no_grad():
    a = model(x)
print(a.requires_grad)     # الناتج: False

with torch.inference_mode():
    b = model(x)
print(b.requires_grad)     # الناتج: False

# الفرق الجوهري: ناتج inference_mode لا يمكن أن يُستعمل لاحقاً في رسم تدرّج
try:
    c = (b * 2).sum()
    c.backward()
except RuntimeError as err:
    print("RuntimeError:", err)
    # الناتج: RuntimeError: element 0 of tensors does not require grad ...

# detach: نسخة بلا رصد، قابلة للاستعمال في حساب لاحق
d = x.detach()
print(d.requires_grad)     # الناتج: False
e = (d * 3).sum()
print(e.requires_grad)     # الناتج: False

#.detach_ بم_SIGNES المعاكس: تقطع الرصد في مكانه
y = x * 2
print(y.requires_grad)     # الناتج: True
y.detach_()
print(y.requires_grad)     # الناتج: False

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-08-ch08.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn

# 1) خطّاف على شدّة: يعدّل التدرّج أو يراقبه
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

def scale_grad(grad):
    print("grad seen by hook:", grad.item())
    return grad * 10        # إرجاع قيمة جديدة يعدّل التدرّج

h = y.register_hook(scale_grad)
y.backward()
print(x.grad)                # الناتج: tensor(40.)  لأن الخطّاف ضاعف 4
h.remove()

# 2) خطّاف خلفي كامل على وحدة: يرى تدرّج الخرج وتدرّج المدخل
class Doubler(nn.Module):
    def forward(self, t):
        return t * 2

d = Doubler()
records = []
hb = d.register_full_backward_hook(
    lambda module, grad_input, grad_output: records.append((grad_input[0], grad_output[0]))
)
t = torch.tensor(3.0, requires_grad=True)
d(t).backward()
hb.remove()
print(records[0][1].item())   # الناتج: 2.0  (تدرّج الخرج)
print(records[0][0].item())   # الناتج: 2.0  (تدرّج المدخل: d(2t)/dt = 2)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-08-ch08.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras

# train_step مخصّص في Keras 3: يتصرّف بشكل مختلف حسب النواة
class WeightedLossModel(keras.Model):
    def __init__(self, units=16):
        super().__init__()
        self.d1 = keras.layers.Dense(units, activation="relu")
        self.d2 = keras.layers.Dense(1)

    def call(self, x, training=False):
        h = self.d1(x, training=training)
        return self.d2(h)

    def train_step(self, data):
        x, y = data
        with self.compute_loss(x, y, sample_weight=None) as loss:
            y_pred = self(x, training=True)
            loss = keras.ops.mean(keras.ops.square(y_pred - y), axis=-1)
        # على نواة torch ينفّذ المحسّن step() بنفسه؛ الوسيط أعلاه يوفّر الخسارة
        return self.compute_metrics(x, y, y_pred, sample_weight=None)

model = WeightedLossModel()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

import numpy as np
rng = np.random.default_rng(0)
x = rng.normal(size=(256, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

model.fit(x, y, batch_size=32, epochs=2, verbose=0)
print("training finished")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-08-ch08.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import torch

def f(x):
    return 3 * x ** 2 + 2 * x - 5

x = torch.tensor(1.0, requires_grad=True)
y = f(x)
y.backward()
print(x.grad)          # الناتج: tensor(8.)   لأن 6*1 + 2 = 8

# المشتقّة الثانية
x2 = torch.tensor(1.0, requires_grad=True)
y2 = f(x2)
g1 = torch.autograd.grad(y2, x2, create_graph=True, retain_graph=True)
g2 = torch.autograd.grad(g1[0].sum(), x2)
print(g2)              # الناتج: (tensor(6.),)

# بلا تراكم
x3 = torch.tensor(1.0, requires_grad=True)
y3 = f(x3)
g = torch.autograd.grad(y3, x3)
print(g, x3.grad)      # الناتج: (tensor(8.),) None

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-08-ch08.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import torch

def square_and_scale(v, k=2.0):
    return (v ** 2) * k

x = torch.tensor([2.0, 3.0], requires_grad=True)
y = square_and_scale(x)

def watch(grad):
    print("hook saw:", grad.shape, grad.tolist())
    return grad

h = y.register_hook(watch)
y.backward()
print("x.grad =", x.grad.tolist())   # الناتج: [16.0, 36.0]
h.remove()

# خطّاف يعدّل التدرّج
x2 = torch.tensor([2.0, 3.0], requires_grad=True)
y2 = square_and_scale(x2)
h2 = y2.register_hook(lambda g: g * 0.5)
y2.backward()
print("x.grad =", x2.grad.tolist())  # الناتج: [8.0, 18.0]
h2.remove()
