#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — تصحيح الأخطاء والأخطاء الشائعة
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 29 — Engineering and Professional Practice

Book slug : pytorch-keras-deep-learning
Chapter   : ch-29-ch29 (lesson 29)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-29-ch29.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — تصحيح الأخطاء والأخطاء الشائعة
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 29 — Engineering and Professional Practice

Book slug : pytorch-keras-deep-learning
Chapter   : ch-29-ch29 (lesson 29)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch29.html
# ────────────────────────────────────────────────────────────────────────
# تشخيص البيئة في دقيقة واحدة
import sys, torch, keras

print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("mem GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
    print("driver:", torch.cuda.driver_version())
print("keras:", keras.__version__)
print("backend:", keras.backend.backend())

# أخطاء بناء شائعة: CUDA مُبنى لـ SM غير مدعوم
# الحل: استخدم إصدار torch مبنيّاً لبنية بطاقتك، أو جرّب من المصدر عند الحاجة

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch29.html
# ────────────────────────────────────────────────────────────────────────
# فحص جودة البيانات — شغّله قبل أيّ تدريب
import numpy as np

def audit_dataset(X, y, num_classes=None, name="dataset"):
    X = np.asarray(X)
    y = np.asarray(y)
    print(f"--- {name} ---")
    print("shape:", X.shape)
    print("nan rows:", int(np.isnan(X).any(axis=1).sum()) if X.ndim > 1 else int(np.isnan(X).sum()))
    print("inf values:", int(np.isinf(X).sum()))
    print("zero rows:", int((np.abs(X).sum(axis=1) == 0).sum()) if X.ndim > 1 else 0)
    if num_classes:
        counts = np.bincount(y.astype(int), minlength=num_classes)
        print("class counts:", counts)
        print("imbalance:", round(counts.max() / max(counts.min(), 1), 1), "x")
    # أسماء الفئات: الأخطاء الإملائية تخلق فئات وهمية
    if y.dtype.kind in "US":
        print("unique labels:", len(np.unique(y)))

audit_dataset(X_train, y_train, num_classes=10, name="train")
audit_dataset(X_test, y_test, num_classes=10, name="test")

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch29.html
# ────────────────────────────────────────────────────────────────────────
# اختبار دخان (smoke test) يلتقط 80% من الأخطاء البنيوية
import torch
from torch import nn

def smoke_test(model, x, y, lossf):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    # 1) هل يمرّ التمرير الأمامي ويُنتج شكلاً منطقياً؟
    out = model(x)
    assert out.shape[0] == x.shape[0], f"شكل المخرجات خاطئ: {out.shape}"
    assert not torch.isnan(out).any(), "المخرجات فيها NaN"

    # 2) هل الخسارة رقمي معقول؟
    loss = lossf(out, y)
    assert torch.isfinite(loss), "الخسارة ليست رقماً منتهياً"
    assert loss.item() >= 0, "خسارة سالبة — تحقّق من دالّة الخسارة"

    # 3) هل التدرّجات تصل؟
    opt.zero_grad()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), "لا توجد تدرّجات!"
    total = sum(float(g.abs().sum()) for g in grads if g is not None)
    assert total > 0, "كل التدرّجات صفر — هل requires_grad مفعّل؟"

    # 4) هل الأوزان تتحرّك فعلاً؟
    before = model.weight.detach().clone()
    opt.step()
    assert not torch.allclose(before, model.weight), "الأوزان لم تتحرّك"

    print("smoke test: PASS")

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch29.html
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=1000, noise=0.2, random_state=0)
Xt = torch.tensor(X, dtype=torch.float32)
yt = torch.tensor(y, dtype=torch.long)

def build():
    return nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))

def run(lr=1e-2, zero=True, no_grad=False, steps=200):
    torch.manual_seed(0)
    model = build()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    hist = []
    for _ in range(steps):
        if zero:
            opt.zero_grad()
        if no_grad:
            with torch.no_grad():
                out = model(Xt)
            loss = lossf(out, yt)
        else:
            out = model(Xt)
            loss = lossf(out, yt)
        loss.backward()
        opt.step()
        hist.append(loss.item())
    return hist

print("سليم:", round(run()[0], 4), "->", round(run()[-1], 4))
print("بلا zero:", round(run(zero=False)[0], 4), "->", round(run(zero=False)[-1], 4))
print("no_grad:", round(run(no_grad=True)[0], 4), "->", round(run(no_grad=True)[-1], 4))
print("lr=10:", round(run(lr=10.0)[0], 4), "->", round(run(lr=10.0)[-1], 4))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-29-ch29.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# تشخيص البيئة في دقيقة واحدة
import sys, torch, keras

print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("mem GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
    print("driver:", torch.cuda.driver_version())
print("keras:", keras.__version__)
print("backend:", keras.backend.backend())

# أخطاء بناء شائعة: CUDA مُبنى لـ SM غير مدعوم
# الحل: استخدم إصدار torch مبنيّاً لبنية بطاقتك، أو جرّب من المصدر عند الحاجة

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-29-ch29.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# فحص جودة البيانات — شغّله قبل أيّ تدريب
import numpy as np

def audit_dataset(X, y, num_classes=None, name="dataset"):
    X = np.asarray(X)
    y = np.asarray(y)
    print(f"--- {name} ---")
    print("shape:", X.shape)
    print("nan rows:", int(np.isnan(X).any(axis=1).sum()) if X.ndim > 1 else int(np.isnan(X).sum()))
    print("inf values:", int(np.isinf(X).sum()))
    print("zero rows:", int((np.abs(X).sum(axis=1) == 0).sum()) if X.ndim > 1 else 0)
    if num_classes:
        counts = np.bincount(y.astype(int), minlength=num_classes)
        print("class counts:", counts)
        print("imbalance:", round(counts.max() / max(counts.min(), 1), 1), "x")
    # أسماء الفئات: الأخطاء الإملائية تخلق فئات وهمية
    if y.dtype.kind in "US":
        print("unique labels:", len(np.unique(y)))

audit_dataset(X_train, y_train, num_classes=10, name="train")
audit_dataset(X_test, y_test, num_classes=10, name="test")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-29-ch29.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# اختبار دخان (smoke test) يلتقط 80% من الأخطاء البنيوية
import torch
from torch import nn

def smoke_test(model, x, y, lossf):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    # 1) هل يمرّ التمرير الأمامي ويُنتج شكلاً منطقياً؟
    out = model(x)
    assert out.shape[0] == x.shape[0], f"شكل المخرجات خاطئ: {out.shape}"
    assert not torch.isnan(out).any(), "المخرجات فيها NaN"

    # 2) هل الخسارة رقمي معقول؟
    loss = lossf(out, y)
    assert torch.isfinite(loss), "الخسارة ليست رقماً منتهياً"
    assert loss.item() >= 0, "خسارة سالبة — تحقّق من دالّة الخسارة"

    # 3) هل التدرّجات تصل؟
    opt.zero_grad()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), "لا توجد تدرّجات!"
    total = sum(float(g.abs().sum()) for g in grads if g is not None)
    assert total > 0, "كل التدرّجات صفر — هل requires_grad مفعّل؟"

    # 4) هل الأوزان تتحرّك فعلاً؟
    before = model.weight.detach().clone()
    opt.step()
    assert not torch.allclose(before, model.weight), "الأوزان لم تتحرّك"

    print("smoke test: PASS")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-29-ch29.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=1000, noise=0.2, random_state=0)
Xt = torch.tensor(X, dtype=torch.float32)
yt = torch.tensor(y, dtype=torch.long)

def build():
    return nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))

def run(lr=1e-2, zero=True, no_grad=False, steps=200):
    torch.manual_seed(0)
    model = build()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    hist = []
    for _ in range(steps):
        if zero:
            opt.zero_grad()
        if no_grad:
            with torch.no_grad():
                out = model(Xt)
            loss = lossf(out, yt)
        else:
            out = model(Xt)
            loss = lossf(out, yt)
        loss.backward()
        opt.step()
        hist.append(loss.item())
    return hist

print("سليم:", round(run()[0], 4), "->", round(run()[-1], 4))
print("بلا zero:", round(run(zero=False)[0], 4), "->", round(run(zero=False)[-1], 4))
print("no_grad:", round(run(no_grad=True)[0], 4), "->", round(run(no_grad=True)[-1], 4))
print("lr=10:", round(run(lr=10.0)[0], 4), "->", round(run(lr=10.0)[-1], 4))
