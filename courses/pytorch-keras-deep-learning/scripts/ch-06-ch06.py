#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — Tensor: حجر الأساس في PyTorch
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 06 — Tensor

Book slug : pytorch-keras-deep-learning
Chapter   : ch-06-ch06 (lesson 6)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-06-ch06.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — Tensor: حجر الأساس في PyTorch
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 06 — Tensor

Book slug : pytorch-keras-deep-learning
Chapter   : ch-06-ch06 (lesson 6)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch06.html
# ────────────────────────────────────────────────────────────────────────
import torch

# تثبيت مصدر العشوائية: نفس البذرة تعطي نفس النتائج دائماً
torch.manual_seed(42)

t = torch.arange(12, dtype=torch.float32).reshape(3, 4)

print(t.shape)                 # الناتج: torch.Size([3, 4])
print(t.dtype)                 # الناتج: torch.float32
print(t.device)                # الناتج: cpu
print(t.ndim, t.numel())       # الناتج: 2 12
print(t[1, 2])                 # الناتج: tensor(6.)

# الطرق الأخرى لصنع الشدّات
print(torch.zeros(2, 3).shape)       # الناتج: torch.Size([2, 3])
print(torch.ones(2, 3).shape)        # الناتج: torch.Size([2, 3])
print(torch.rand(2, 3).shape)        # الناتج: torch.Size([2, 3])  قيم في [0, 1)
print(torch.randn(2, 3).shape)       # الناتج: torch.Size([2, 3])  توزيع طبيعي
print(torch.linspace(0, 1, 5).shape)  # الناتج: torch.Size([5])       خمس نقاط متساوية

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch06.html
# ────────────────────────────────────────────────────────────────────────
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)                       # الناتج: cuda (أو cpu إن لم تتوفّر بطاقة)

w = torch.randn(512, 512)
w_gpu = w.to(device)                # نسخة على الجهاز الجديد
print(w.device, w_gpu.device)       # الناتج: cpu cuda

# التحويل في النوع أيضاً، ويمكن دمج الأمرين في نداء واحد
w_half = w.to(device=device, dtype=torch.float16)
print(w_half.dtype)                 # الناتج: torch.float16

# الأجهزة المتاحة فعلياً
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))   # اسم بطاقة الرسوميات
    print(torch.cuda.device_count())        # الناتج: 1

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch06.html
# ────────────────────────────────────────────────────────────────────────
import torch

a = torch.arange(6).reshape(3, 2)          # شكل (3, 2)
b = torch.tensor([[10.0], [20.0], [30.0]])  # شكل (3, 1)

print(torch.broadcast_shapes(a.shape, b.shape))  # الناتج: (3, 2)
c = a + b                                   # البثّ: (3, 2) + (3, 1) -> (3, 2)
print(c)
# الناتج:
# tensor([[10., 11.],
#         [22., 23.],
#         [34., 35.]])

# البثّ في الاتجاه الخطأ يفشل بوضوح
try:
    bad = a + torch.tensor([100.0, 200.0, 300.0])   # المتجه طوله 3 لا 2
except RuntimeError as err:
    print(err)   # الناتج: The size of tensor a (2) must match the size of tensor b (3)...

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch06.html
# ────────────────────────────────────────────────────────────────────────
import torch

x = torch.arange(12).reshape(3, 4)

v = x.t()      # تبديل المحاور: منظور (view) على الذاكرة نفسها
c = x.clone()  # نسخة مستقلة تماماً

print(x.data_ptr() == v.data_ptr())      # الناتج: True   -> لا ذاكرة جديدة
print(x.data_ptr() == c.data_ptr())      # الناتج: False  -> ذاكرة جديدة
print(x.stride(), v.stride())            # الناتج: (1, 4) (4, 1)
print(x.is_contiguous(), v.is_contiguous())   # الناتج: True False

# reshape على شدّة متّصلة يُعيد منظوراً، وعلى شدّة غير متّصلة قد ينسخ
r = x.reshape(4, 3)
print(r.data_ptr() == x.data_ptr())      # الناتج: True

# لإجبار النسخ أو تنسيق الذاكرة
vc = v.contiguous()
print(vc.is_contiguous())                # الناتج: True

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch06.html
# ────────────────────────────────────────────────────────────────────────
import torch

t = torch.arange(12).reshape(3, 4)

print(t[0])        # الناتج: tensor([0, 1, 2, 3])       فهرسة أساسية: منظور
print(t[:, 1])     # الناتج: tensor([1, 5, 9])          شريحة عمود: منظور
print(t[1, 2])     # الناتج: tensor(6)                  عنصر واحد: منظور مربوط

mask = t % 2 == 0                       # قناع منطقي: القيم الزوجية فقط
print(t[mask])        # الناتج: tensor([ 0,  2,  4,  6,  8, 10])   فهرسة متقدّمة: نسخة

idx = torch.tensor([0, 2])              # اختيار صفوف بعينها
print(t[idx])
# الناتج: tensor([[ 0,  1,  2,  3],
#                 [ 8,  9, 10, 11]])

# دمج شرطَيْن في عملية واحدة
cond = t > 5
print(torch.where(cond, t, torch.zeros_like(t)))
# الناتج: tensor([[ 0,  0,  0,  0],
#                 [ 6,  7,  8,  9],
#                 [10, 11,  0,  0]])

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch06.html
# ────────────────────────────────────────────────────────────────────────
import torch

# ضرب مصفوفتين: (2, 3) × (3, 4) -> (2, 4)
a = torch.randn(2, 3)
b = torch.randn(3, 4)
ab = torch.einsum("bi,ij->bj", a, b)
print(ab.shape)          # الناتج: torch.Size([2, 4])

# ضرب داخلي بين متجهين: ناتج عدد واحد
u = torch.randn(4)
v = torch.randn(4)
print(torch.einsum("i,i->", u, v).shape)   # الناتج: torch.Size([])

# دفعة من المصفوفات: كل عيّنة لها مصفوفة خاصة بها
batch = torch.randn(5, 3, 3)
out = torch.einsum("bij,bkj->bik", batch, batch)
print(out.shape)         # الناتج: torch.Size([5, 3, 3])

# جمع على محور مشترك مع الحفاظ على البعد (مفيد للجداول)
scores = torch.randn(2, 100)
per_class = torch.einsum("bc,c->b", scores, torch.ones(100))
print(per_class.shape)   # الناتج: torch.Size([2])

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch06.html
# ────────────────────────────────────────────────────────────────────────
import torch

torch.manual_seed(0)
B, C = 8, 5
scores = torch.rand(B, C)

best = scores.max(dim=1).values          # الناتج: torch.Size([8])
mean = scores.mean(dim=1)                # الناتج: torch.Size([8])
total = scores.sum(dim=1, keepdim=True)  # الناتج: torch.Size([8, 1])

probs = scores / total                   # البثّ: (8, 5) / (8, 1) -> (8, 5)
print(probs.sum(dim=1))                   # الناتج: tensor([1., 1., 1., 1., 1., 1., 1., 1.])

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch06.html
# ────────────────────────────────────────────────────────────────────────
import torch

x = torch.arange(24).reshape(4, 6)

y = x.t()
y[0, 0] = 99
print(x[0, 0])                      # الناتج: tensor(99)  -> تغيّر الأصل
print(shares_memory(x, y))          # الناتج: True

z = x.clone().t()
z[0, 0] = -1
print(x[0, 0])                      # الناتج: tensor(99)  -> الأصل لم يتغيّر
print(shares_memory(x, z))          # الناتج: False

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch06.html
# ────────────────────────────────────────────────────────────────────────
def shares_memory(a, b):
    """تتحقّق ممّا إذا كانت الشدّتان تشتركان في المخزن نفسه."""
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-06-ch06.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import torch

# تثبيت مصدر العشوائية: نفس البذرة تعطي نفس النتائج دائماً
torch.manual_seed(42)

t = torch.arange(12, dtype=torch.float32).reshape(3, 4)

print(t.shape)                 # الناتج: torch.Size([3, 4])
print(t.dtype)                 # الناتج: torch.float32
print(t.device)                # الناتج: cpu
print(t.ndim, t.numel())       # الناتج: 2 12
print(t[1, 2])                 # الناتج: tensor(6.)

# الطرق الأخرى لصنع الشدّات
print(torch.zeros(2, 3).shape)       # الناتج: torch.Size([2, 3])
print(torch.ones(2, 3).shape)        # الناتج: torch.Size([2, 3])
print(torch.rand(2, 3).shape)        # الناتج: torch.Size([2, 3])  قيم في [0, 1)
print(torch.randn(2, 3).shape)       # الناتج: torch.Size([2, 3])  توزيع طبيعي
print(torch.linspace(0, 1, 5).shape)  # الناتج: torch.Size([5])       خمس نقاط متساوية

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-06-ch06.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)                       # الناتج: cuda (أو cpu إن لم تتوفّر بطاقة)

w = torch.randn(512, 512)
w_gpu = w.to(device)                # نسخة على الجهاز الجديد
print(w.device, w_gpu.device)       # الناتج: cpu cuda

# التحويل في النوع أيضاً، ويمكن دمج الأمرين في نداء واحد
w_half = w.to(device=device, dtype=torch.float16)
print(w_half.dtype)                 # الناتج: torch.float16

# الأجهزة المتاحة فعلياً
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))   # اسم بطاقة الرسوميات
    print(torch.cuda.device_count())        # الناتج: 1

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-06-ch06.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch

a = torch.arange(6).reshape(3, 2)          # شكل (3, 2)
b = torch.tensor([[10.0], [20.0], [30.0]])  # شكل (3, 1)

print(torch.broadcast_shapes(a.shape, b.shape))  # الناتج: (3, 2)
c = a + b                                   # البثّ: (3, 2) + (3, 1) -> (3, 2)
print(c)
# الناتج:
# tensor([[10., 11.],
#         [22., 23.],
#         [34., 35.]])

# البثّ في الاتجاه الخطأ يفشل بوضوح
try:
    bad = a + torch.tensor([100.0, 200.0, 300.0])   # المتجه طوله 3 لا 2
except RuntimeError as err:
    print(err)   # الناتج: The size of tensor a (2) must match the size of tensor b (3)...

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-06-ch06.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch

x = torch.arange(12).reshape(3, 4)

v = x.t()      # تبديل المحاور: منظور (view) على الذاكرة نفسها
c = x.clone()  # نسخة مستقلة تماماً

print(x.data_ptr() == v.data_ptr())      # الناتج: True   -> لا ذاكرة جديدة
print(x.data_ptr() == c.data_ptr())      # الناتج: False  -> ذاكرة جديدة
print(x.stride(), v.stride())            # الناتج: (1, 4) (4, 1)
print(x.is_contiguous(), v.is_contiguous())   # الناتج: True False

# reshape على شدّة متّصلة يُعيد منظوراً، وعلى شدّة غير متّصلة قد ينسخ
r = x.reshape(4, 3)
print(r.data_ptr() == x.data_ptr())      # الناتج: True

# لإجبار النسخ أو تنسيق الذاكرة
vc = v.contiguous()
print(vc.is_contiguous())                # الناتج: True

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-06-ch06.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import torch

t = torch.arange(12).reshape(3, 4)

print(t[0])        # الناتج: tensor([0, 1, 2, 3])       فهرسة أساسية: منظور
print(t[:, 1])     # الناتج: tensor([1, 5, 9])          شريحة عمود: منظور
print(t[1, 2])     # الناتج: tensor(6)                  عنصر واحد: منظور مربوط

mask = t % 2 == 0                       # قناع منطقي: القيم الزوجية فقط
print(t[mask])        # الناتج: tensor([ 0,  2,  4,  6,  8, 10])   فهرسة متقدّمة: نسخة

idx = torch.tensor([0, 2])              # اختيار صفوف بعينها
print(t[idx])
# الناتج: tensor([[ 0,  1,  2,  3],
#                 [ 8,  9, 10, 11]])

# دمج شرطَيْن في عملية واحدة
cond = t > 5
print(torch.where(cond, t, torch.zeros_like(t)))
# الناتج: tensor([[ 0,  0,  0,  0],
#                 [ 6,  7,  8,  9],
#                 [10, 11,  0,  0]])

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-06-ch06.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import torch

# ضرب مصفوفتين: (2, 3) × (3, 4) -> (2, 4)
a = torch.randn(2, 3)
b = torch.randn(3, 4)
ab = torch.einsum("bi,ij->bj", a, b)
print(ab.shape)          # الناتج: torch.Size([2, 4])

# ضرب داخلي بين متجهين: ناتج عدد واحد
u = torch.randn(4)
v = torch.randn(4)
print(torch.einsum("i,i->", u, v).shape)   # الناتج: torch.Size([])

# دفعة من المصفوفات: كل عيّنة لها مصفوفة خاصة بها
batch = torch.randn(5, 3, 3)
out = torch.einsum("bij,bkj->bik", batch, batch)
print(out.shape)         # الناتج: torch.Size([5, 3, 3])

# جمع على محور مشترك مع الحفاظ على البعد (مفيد للجداول)
scores = torch.randn(2, 100)
per_class = torch.einsum("bc,c->b", scores, torch.ones(100))
print(per_class.shape)   # الناتج: torch.Size([2])

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-06-ch06.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import torch

torch.manual_seed(0)
B, C = 8, 5
scores = torch.rand(B, C)

best = scores.max(dim=1).values          # الناتج: torch.Size([8])
mean = scores.mean(dim=1)                # الناتج: torch.Size([8])
total = scores.sum(dim=1, keepdim=True)  # الناتج: torch.Size([8, 1])

probs = scores / total                   # البثّ: (8, 5) / (8, 1) -> (8, 5)
print(probs.sum(dim=1))                   # الناتج: tensor([1., 1., 1., 1., 1., 1., 1., 1.])

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-06-ch06.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import torch

x = torch.arange(24).reshape(4, 6)

y = x.t()
y[0, 0] = 99
print(x[0, 0])                      # الناتج: tensor(99)  -> تغيّر الأصل
print(shares_memory(x, y))          # الناتج: True

z = x.clone().t()
z[0, 0] = -1
print(x[0, 0])                      # الناتج: tensor(99)  -> الأصل لم يتغيّر
print(shares_memory(x, z))          # الناتج: False

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-06-ch06.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
def shares_memory(a, b):
    """تتحقّق ممّا إذا كانت الشدّتان تشتركان في المخزن نفسه."""
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
