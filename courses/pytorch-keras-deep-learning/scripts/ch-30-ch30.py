#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — الأخلاقيات والتحيّز والخصوصية والأمن
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 30 — Engineering and Professional Practice

Book slug : pytorch-keras-deep-learning
Chapter   : ch-30-ch30 (lesson 30)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-30-ch30.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — الأخلاقيات والتحيّز والخصوصية والأمن
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 30 — Engineering and Professional Practice

Book slug : pytorch-keras-deep-learning
Chapter   : ch-30-ch30 (lesson 30)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch30.html
# ────────────────────────────────────────────────────────────────────────
# قياس أداء النموذج على المجموعات الفرعية: الأداة الأساسية
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def subgroup_report(y_true, y_pred, sensitive, names=None):
    """تقرير أداء مفصّل لكل مجموعة فرعية."""
    names = names or [f"group_{i}" for i in np.unique(sensitive)]
    rows = []
    for g in np.unique(sensitive):
        mask = sensitive == g
        yt, yp = y_true[mask], y_pred[mask]
        if len(np.unique(yt)) < 2:
            rows.append((names[g], len(yt), None, None, None, None))
            continue
        rows.append((
            names[g], len(yt),
            accuracy_score(yt, yp),
            precision_score(yt, yp, zero_division=0),
            recall_score(yt, yp, zero_division=0),
            f1_score(yt, yp, zero_division=0),
        ))
    return rows

# اقرأ النتيجة هكذا: هل فارق الـF1 بين المجموعات يتجاوز 10 نقاط؟
# إذا كان كذلك: المشكلة في البيانات أو في دالّة الخسارة، لا في المحسّن.

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch30.html
# ────────────────────────────────────────────────────────────────────────
# مثال مبسّط: إضافة ضوضاء إلى تدرّجات (DP-SGD مبسّطة)
import torch

def dp_sgd_step(model, loss, optimizer, clip_norm=1.0, noise_multiplier=1.1, max_batch=64):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # 1) قصّ التدرّج على مستوى كل عيّنة
    for p in model.parameters():
        if p.grad is None:
            continue
        p.grad = p.grad.clamp(-clip_norm, clip_norm)

    # 2) جمع ضجيج غاوسي يحقّق الخصوصية
    if torch.rand(1).item() < 0.01:      #_rate limiter في الإنتاج
        for p in model.parameters():
            if p.grad is not None:
                noise = torch.randn_like(p.grad) * (noise_multiplier * clip_norm / max_batch)
                p.grad += noise

    optimizer.step()

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch30.html
# ────────────────────────────────────────────────────────────────────────
# اختبار هشاشة عدائية مبسّط: FGSM
import torch

def fgsm_attack(model, x, y, eps=8/255, lossf=None):
    lossf = lossf or torch.nn.CrossEntropyLoss()
    x = x.clone().detach().requires_grad_(True)
    loss = lossf(model(x), y)
    loss.backward()
    # خطوة في اتجاه تعظيم الخسارة
    grad = x.grad.sign()
    return (x + eps * grad).detach()

# اختبار: دقّة قبل وبعد
model.eval()
acc_clean = accuracy(model(x_test), y_test)
acc_adv = accuracy(model(fgsm_attack(model, x_test, y_test)), y_test)
print(f"clean={acc_clean:.3f}  adversarial={acc_adv:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-30-ch30.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# قياس أداء النموذج على المجموعات الفرعية: الأداة الأساسية
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def subgroup_report(y_true, y_pred, sensitive, names=None):
    """تقرير أداء مفصّل لكل مجموعة فرعية."""
    names = names or [f"group_{i}" for i in np.unique(sensitive)]
    rows = []
    for g in np.unique(sensitive):
        mask = sensitive == g
        yt, yp = y_true[mask], y_pred[mask]
        if len(np.unique(yt)) < 2:
            rows.append((names[g], len(yt), None, None, None, None))
            continue
        rows.append((
            names[g], len(yt),
            accuracy_score(yt, yp),
            precision_score(yt, yp, zero_division=0),
            recall_score(yt, yp, zero_division=0),
            f1_score(yt, yp, zero_division=0),
        ))
    return rows

# اقرأ النتيجة هكذا: هل فارق الـF1 بين المجموعات يتجاوز 10 نقاط؟
# إذا كان كذلك: المشكلة في البيانات أو في دالّة الخسارة، لا في المحسّن.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-30-ch30.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# مثال مبسّط: إضافة ضوضاء إلى تدرّجات (DP-SGD مبسّطة)
import torch

def dp_sgd_step(model, loss, optimizer, clip_norm=1.0, noise_multiplier=1.1, max_batch=64):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # 1) قصّ التدرّج على مستوى كل عيّنة
    for p in model.parameters():
        if p.grad is None:
            continue
        p.grad = p.grad.clamp(-clip_norm, clip_norm)

    # 2) جمع ضجيج غاوسي يحقّق الخصوصية
    if torch.rand(1).item() < 0.01:      #_rate limiter في الإنتاج
        for p in model.parameters():
            if p.grad is not None:
                noise = torch.randn_like(p.grad) * (noise_multiplier * clip_norm / max_batch)
                p.grad += noise

    optimizer.step()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-30-ch30.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# اختبار هشاشة عدائية مبسّط: FGSM
import torch

def fgsm_attack(model, x, y, eps=8/255, lossf=None):
    lossf = lossf or torch.nn.CrossEntropyLoss()
    x = x.clone().detach().requires_grad_(True)
    loss = lossf(model(x), y)
    loss.backward()
    # خطوة في اتجاه تعظيم الخسارة
    grad = x.grad.sign()
    return (x + eps * grad).detach()

# اختبار: دقّة قبل وبعد
model.eval()
acc_clean = accuracy(model(x_test), y_test)
acc_adv = accuracy(model(fgsm_attack(model, x_test, y_test)), y_test)
print(f"clean={acc_clean:.3f}  adversarial={acc_adv:.3f}")
