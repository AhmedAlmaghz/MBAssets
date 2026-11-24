#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — Backpropagation والتحسين (Optimization)
Large Language Model Engineering — Backpropagation

Book slug : llm-engineering-69-chapters
Chapter   : ch-03-chapter-03 (lesson 3)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-03-chapter-03.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — Backpropagation والتحسين (Optimization)
Large Language Model Engineering — Backpropagation

Book slug : llm-engineering-69-chapters
Chapter   : ch-03-chapter-03 (lesson 3)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-03-chapter-03.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — Backpropagation والتحسين (Optimization)
Large Language Model Engineering — Backpropagation

Book slug : llm-engineering-69-chapters
Chapter   : ch-03-chapter-03 (lesson 3)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_03.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 03: الفصل الثالث: Backpropagation والتحسين (Optimization)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_gradient() -> None:
    parameter, rate = (8.0, 0.15)
    trace = []
    for _ in range(30):
        loss = (parameter - 2.0) ** 2
        analytic = 2.0 * (parameter - 2.0)
        epsilon = 1e-05
        numeric = ((parameter + epsilon - 2) ** 2 - (parameter - epsilon - 2) ** 2) / (2 * epsilon)
        trace.append(loss)
        parameter -= rate * analytic
    print('final parameter:', round(parameter, 5))
    print('initial/final loss:', round(trace[0], 4), round(trace[-1], 8))
    print('finite-difference gradient agrees:', abs(analytic - numeric) < 1e-05)

if __name__ == "__main__":
    print("Chapter 03 lab: Gradient descent and finite differences")
    run_gradient()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-01-foundations/chapter-03.md
# ────────────────────────────────────────────────────────────────────────
"""
backprop_manual.py
تطبيق يدوي كامل لـ Forward + Backward Pass لشبكة MLP بطبقتين،
ثم التحقق من صحته مقابل torch.autograd.
Python 3.13+ | NumPy فقط للجزء اليدوي، ثم PyTorch للمقارنة.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

rng = np.random.default_rng(seed=42)


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def softmax(z: np.ndarray) -> np.ndarray:
    z_shifted = z - z.max(axis=1, keepdims=True)  # استقرار عددي
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def cross_entropy(y_pred: np.ndarray, y_true_idx: np.ndarray) -> float:
    n = y_pred.shape[0]
    log_likelihood = -np.log(y_pred[np.arange(n), y_true_idx] + 1e-12)
    return float(log_likelihood.mean())


def manual_forward_backward(
    x: np.ndarray,
    y_idx: np.ndarray,
    w1: np.ndarray, b1: np.ndarray,
    w2: np.ndarray, b2: np.ndarray,
) -> dict[str, np.ndarray]:
    n = x.shape[0]
    num_classes = w2.shape[1]

    # ---------- Forward Pass ----------
    z1 = x @ w1 + b1
    a1 = relu(z1)
    z2 = a1 @ w2 + b2
    y_hat = softmax(z2)
    loss = cross_entropy(y_hat, y_idx)

    # ---------- Backward Pass (يدويًا، بقاعدة السلسلة) ----------
    y_onehot = np.zeros_like(y_hat)
    y_onehot[np.arange(n), y_idx] = 1.0

    d_z2 = (y_hat - y_onehot) / n          # الصيغة الأنيقة Softmax+CE
    d_w2 = a1.T @ d_z2
    d_b2 = d_z2.sum(axis=0)
    d_a1 = d_z2 @ w2.T
    d_z1 = d_a1 * (z1 > 0).astype(np.float64)   # اشتقاق ReLU
    d_w1 = x.T @ d_z1
    d_b1 = d_z1.sum(axis=0)

    return {
        "loss": np.array(loss),
        "d_w1": d_w1, "d_b1": d_b1,
        "d_w2": d_w2, "d_b2": d_b2,
    }


def torch_forward_backward(
    x: np.ndarray, y_idx: np.ndarray,
    w1: np.ndarray, b1: np.ndarray,
    w2: np.ndarray, b2: np.ndarray,
) -> dict[str, torch.Tensor]:
    x_t = torch.tensor(x, dtype=torch.float64)
    y_t = torch.tensor(y_idx, dtype=torch.long)
    w1_t = torch.tensor(w1, dtype=torch.float64, requires_grad=True)
    b1_t = torch.tensor(b1, dtype=torch.float64, requires_grad=True)
    w2_t = torch.tensor(w2, dtype=torch.float64, requires_grad=True)
    b2_t = torch.tensor(b2, dtype=torch.float64, requires_grad=True)

    z1 = x_t @ w1_t + b1_t
    a1 = torch.relu(z1)
    z2 = a1 @ w2_t + b2_t
    loss = F.cross_entropy(z2, y_t)   # يطبّق log-softmax + NLL داخليًا
    loss.backward()

    return {
        "loss": loss.detach(),
        "d_w1": w1_t.grad, "d_b1": b1_t.grad,
        "d_w2": w2_t.grad, "d_b2": b2_t.grad,
    }


if __name__ == "__main__":
    n_samples, in_dim, hidden_dim, out_dim = 8, 4, 5, 3
    x = rng.normal(size=(n_samples, in_dim))
    y_idx = rng.integers(0, out_dim, size=n_samples)
    w1 = rng.normal(scale=0.1, size=(in_dim, hidden_dim))
    b1 = np.zeros(hidden_dim)
    w2 = rng.normal(scale=0.1, size=(hidden_dim, out_dim))
    b2 = np.zeros(out_dim)

    manual = manual_forward_backward(x, y_idx, w1, b1, w2, b2)
    auto = torch_forward_backward(x, y_idx, w1, b1, w2, b2)

    print(f"Loss (يدوي)   : {manual['loss']:.6f}")
    print(f"Loss (PyTorch): {auto['loss'].item():.6f}")

    for name in ["d_w1", "d_b1", "d_w2", "d_b2"]:
        diff = np.abs(manual[name] - auto[name].numpy()).max()
        print(f"أقصى فرق في {name}: {diff:.2e}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-01-foundations/chapter-03.md
# ────────────────────────────────────────────────────────────────────────
**لاحظ هذه التفاصيل الاحترافية في الكود أعلاه، فكل واحدة منها "أفضل ممارسة" حقيقية:**
- `optimizer.zero_grad(set_to_none=True)` أسرع من `zero_grad()` العادية (يُحرّر الذاكرة بدل تصفيرها).
- ترتيب العمليات: `backward()` ← `clip_grad_norm_()` ← `optimizer.step()` ← `scheduler.step()` — هذا الترتيب **إلزامي** ولا يجوز تبديله.
- `scheduler.step()` يُستدعى **بعد** كل `optimizer.step()` (بمستوى الـ Step وليس الـ Epoch) عند استخدام Warmup/Cosine.
- طباعة `grad_norm` في كل Epoch هي أداة تشخيصية بسيطة لكنها قوية جدًا لرصد مشاكل التدريب مبكرًا.

---

## 5. 💡 نصائح وحيل (Tips & Tricks)

- **قاعدة ذهبية لمعدل التعلم:** عند مضاعفة حجم الدفعة (Batch Size)، يمكن تقريبًا مضاعفة معدل التعلم أيضًا (Linear Scaling Rule من ورقة Goyal et al. 2017)، لكن يجب إضافة Warmup أطول لتعويض عدم الاستقرار الأولي.
- **ابدأ دائمًا بـ AdamW** كخيار افتراضي آمن لتدريب الشبكات العميقة والمحوّلات، إلا إذا كان لديك سبب محدد (كـ SGD+Momentum في بعض نماذج الرؤية الكلاسيكية التي تحقق تعميمًا أفضل أحيانًا).
- **راقب نسبة `grad_norm / max_norm`:** إذا كانت القيمة تتجاوز الحد الأقصى في أغلب الخطوات (أي أن القص يحدث دائمًا)، هذا مؤشر خطر — قد يعني معدل تعلم مرتفع جدًا أو مشكلة في تهيئة الأوزان (Initialization).
- **Gradient Accumulation:** إذا كانت الذاكرة لا تكفي لدفعة كبيرة، اجمع التدرجات عبر عدة دفعات صغيرة قبل استدعاء `optimizer.step()` — هذا يُحاكي تأثير دفعة أكبر دون الحاجة لذاكرة أكبر (أساسي في تدريب LLM).
- **قيمة β2 الافتراضية 0.999 قد لا تناسب كل الحالات:** في تدريب الـ LLM الكبيرة أحيانًا يُستخدم `β2=0.95` بدل `0.999` (كما في بعض إعدادات LLaMA) لاستجابة أسرع للتغيرات، خاصة مع معدلات تعلم مرتفعة نسبيًا.
- **استخدم `torch.compile()`** (المتاح من PyTorch 2.0+) لتسريع حلقة التدريب تلقائيًا دون تغيير منطق الكود، عبر دمج العمليات (Kernel Fusion) خلف الكواليس.

---

## 6. ⚠️ أخطاء شائعة ومصائد (Common Pitfalls)

### الخطأ 1: نسيان `optimizer.zero_grad()`

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-03-chapter-03.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 03: الفصل الثالث: Backpropagation والتحسين (Optimization)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_gradient() -> None:
    parameter, rate = (8.0, 0.15)
    trace = []
    for _ in range(30):
        loss = (parameter - 2.0) ** 2
        analytic = 2.0 * (parameter - 2.0)
        epsilon = 1e-05
        numeric = ((parameter + epsilon - 2) ** 2 - (parameter - epsilon - 2) ** 2) / (2 * epsilon)
        trace.append(loss)
        parameter -= rate * analytic
    print('final parameter:', round(parameter, 5))
    print('initial/final loss:', round(trace[0], 4), round(trace[-1], 8))
    print('finite-difference gradient agrees:', abs(analytic - numeric) < 1e-05)

if __name__ == "__main__":
    print("Chapter 03 lab: Gradient descent and finite differences")
    run_gradient()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-03-chapter-03.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
backprop_manual.py
تطبيق يدوي كامل لـ Forward + Backward Pass لشبكة MLP بطبقتين،
ثم التحقق من صحته مقابل torch.autograd.
Python 3.13+ | NumPy فقط للجزء اليدوي، ثم PyTorch للمقارنة.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

rng = np.random.default_rng(seed=42)


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def softmax(z: np.ndarray) -> np.ndarray:
    z_shifted = z - z.max(axis=1, keepdims=True)  # استقرار عددي
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def cross_entropy(y_pred: np.ndarray, y_true_idx: np.ndarray) -> float:
    n = y_pred.shape[0]
    log_likelihood = -np.log(y_pred[np.arange(n), y_true_idx] + 1e-12)
    return float(log_likelihood.mean())


def manual_forward_backward(
    x: np.ndarray,
    y_idx: np.ndarray,
    w1: np.ndarray, b1: np.ndarray,
    w2: np.ndarray, b2: np.ndarray,
) -> dict[str, np.ndarray]:
    n = x.shape[0]
    num_classes = w2.shape[1]

    # ---------- Forward Pass ----------
    z1 = x @ w1 + b1
    a1 = relu(z1)
    z2 = a1 @ w2 + b2
    y_hat = softmax(z2)
    loss = cross_entropy(y_hat, y_idx)

    # ---------- Backward Pass (يدويًا، بقاعدة السلسلة) ----------
    y_onehot = np.zeros_like(y_hat)
    y_onehot[np.arange(n), y_idx] = 1.0

    d_z2 = (y_hat - y_onehot) / n          # الصيغة الأنيقة Softmax+CE
    d_w2 = a1.T @ d_z2
    d_b2 = d_z2.sum(axis=0)
    d_a1 = d_z2 @ w2.T
    d_z1 = d_a1 * (z1 > 0).astype(np.float64)   # اشتقاق ReLU
    d_w1 = x.T @ d_z1
    d_b1 = d_z1.sum(axis=0)

    return {
        "loss": np.array(loss),
        "d_w1": d_w1, "d_b1": d_b1,
        "d_w2": d_w2, "d_b2": d_b2,
    }


def torch_forward_backward(
    x: np.ndarray, y_idx: np.ndarray,
    w1: np.ndarray, b1: np.ndarray,
    w2: np.ndarray, b2: np.ndarray,
) -> dict[str, torch.Tensor]:
    x_t = torch.tensor(x, dtype=torch.float64)
    y_t = torch.tensor(y_idx, dtype=torch.long)
    w1_t = torch.tensor(w1, dtype=torch.float64, requires_grad=True)
    b1_t = torch.tensor(b1, dtype=torch.float64, requires_grad=True)
    w2_t = torch.tensor(w2, dtype=torch.float64, requires_grad=True)
    b2_t = torch.tensor(b2, dtype=torch.float64, requires_grad=True)

    z1 = x_t @ w1_t + b1_t
    a1 = torch.relu(z1)
    z2 = a1 @ w2_t + b2_t
    loss = F.cross_entropy(z2, y_t)   # يطبّق log-softmax + NLL داخليًا
    loss.backward()

    return {
        "loss": loss.detach(),
        "d_w1": w1_t.grad, "d_b1": b1_t.grad,
        "d_w2": w2_t.grad, "d_b2": b2_t.grad,
    }


if __name__ == "__main__":
    n_samples, in_dim, hidden_dim, out_dim = 8, 4, 5, 3
    x = rng.normal(size=(n_samples, in_dim))
    y_idx = rng.integers(0, out_dim, size=n_samples)
    w1 = rng.normal(scale=0.1, size=(in_dim, hidden_dim))
    b1 = np.zeros(hidden_dim)
    w2 = rng.normal(scale=0.1, size=(hidden_dim, out_dim))
    b2 = np.zeros(out_dim)

    manual = manual_forward_backward(x, y_idx, w1, b1, w2, b2)
    auto = torch_forward_backward(x, y_idx, w1, b1, w2, b2)

    print(f"Loss (يدوي)   : {manual['loss']:.6f}")
    print(f"Loss (PyTorch): {auto['loss'].item():.6f}")

    for name in ["d_w1", "d_b1", "d_w2", "d_b2"]:
        diff = np.abs(manual[name] - auto[name].numpy()).max()
        print(f"أقصى فرق في {name}: {diff:.2e}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-03-chapter-03.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
**لاحظ هذه التفاصيل الاحترافية في الكود أعلاه، فكل واحدة منها "أفضل ممارسة" حقيقية:**
- `optimizer.zero_grad(set_to_none=True)` أسرع من `zero_grad()` العادية (يُحرّر الذاكرة بدل تصفيرها).
- ترتيب العمليات: `backward()` ← `clip_grad_norm_()` ← `optimizer.step()` ← `scheduler.step()` — هذا الترتيب **إلزامي** ولا يجوز تبديله.
- `scheduler.step()` يُستدعى **بعد** كل `optimizer.step()` (بمستوى الـ Step وليس الـ Epoch) عند استخدام Warmup/Cosine.
- طباعة `grad_norm` في كل Epoch هي أداة تشخيصية بسيطة لكنها قوية جدًا لرصد مشاكل التدريب مبكرًا.

---

## 5. 💡 نصائح وحيل (Tips & Tricks)

- **قاعدة ذهبية لمعدل التعلم:** عند مضاعفة حجم الدفعة (Batch Size)، يمكن تقريبًا مضاعفة معدل التعلم أيضًا (Linear Scaling Rule من ورقة Goyal et al. 2017)، لكن يجب إضافة Warmup أطول لتعويض عدم الاستقرار الأولي.
- **ابدأ دائمًا بـ AdamW** كخيار افتراضي آمن لتدريب الشبكات العميقة والمحوّلات، إلا إذا كان لديك سبب محدد (كـ SGD+Momentum في بعض نماذج الرؤية الكلاسيكية التي تحقق تعميمًا أفضل أحيانًا).
- **راقب نسبة `grad_norm / max_norm`:** إذا كانت القيمة تتجاوز الحد الأقصى في أغلب الخطوات (أي أن القص يحدث دائمًا)، هذا مؤشر خطر — قد يعني معدل تعلم مرتفع جدًا أو مشكلة في تهيئة الأوزان (Initialization).
- **Gradient Accumulation:** إذا كانت الذاكرة لا تكفي لدفعة كبيرة، اجمع التدرجات عبر عدة دفعات صغيرة قبل استدعاء `optimizer.step()` — هذا يُحاكي تأثير دفعة أكبر دون الحاجة لذاكرة أكبر (أساسي في تدريب LLM).
- **قيمة β2 الافتراضية 0.999 قد لا تناسب كل الحالات:** في تدريب الـ LLM الكبيرة أحيانًا يُستخدم `β2=0.95` بدل `0.999` (كما في بعض إعدادات LLaMA) لاستجابة أسرع للتغيرات، خاصة مع معدلات تعلم مرتفعة نسبيًا.
- **استخدم `torch.compile()`** (المتاح من PyTorch 2.0+) لتسريع حلقة التدريب تلقائيًا دون تغيير منطق الكود، عبر دمج العمليات (Kernel Fusion) خلف الكواليس.

---

## 6. ⚠️ أخطاء شائعة ومصائد (Common Pitfalls)

### الخطأ 1: نسيان `optimizer.zero_grad()`

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-03-chapter-03.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — Backpropagation والتحسين (Optimization)
Large Language Model Engineering — Backpropagation

Book slug : llm-engineering-69-chapters
Chapter   : ch-03-chapter-03 (lesson 3)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_03.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 03: الفصل الثالث: Backpropagation والتحسين (Optimization)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_gradient() -> None:
    parameter, rate = (8.0, 0.15)
    trace = []
    for _ in range(30):
        loss = (parameter - 2.0) ** 2
        analytic = 2.0 * (parameter - 2.0)
        epsilon = 1e-05
        numeric = ((parameter + epsilon - 2) ** 2 - (parameter - epsilon - 2) ** 2) / (2 * epsilon)
        trace.append(loss)
        parameter -= rate * analytic
    print('final parameter:', round(parameter, 5))
    print('initial/final loss:', round(trace[0], 4), round(trace[-1], 8))
    print('finite-difference gradient agrees:', abs(analytic - numeric) < 1e-05)

if __name__ == "__main__":
    print("Chapter 03 lab: Gradient descent and finite differences")
    run_gradient()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-01-foundations/chapter-03.md
# ────────────────────────────────────────────────────────────────────────
"""
backprop_manual.py
تطبيق يدوي كامل لـ Forward + Backward Pass لشبكة MLP بطبقتين،
ثم التحقق من صحته مقابل torch.autograd.
Python 3.13+ | NumPy فقط للجزء اليدوي، ثم PyTorch للمقارنة.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

rng = np.random.default_rng(seed=42)


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def softmax(z: np.ndarray) -> np.ndarray:
    z_shifted = z - z.max(axis=1, keepdims=True)  # استقرار عددي
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def cross_entropy(y_pred: np.ndarray, y_true_idx: np.ndarray) -> float:
    n = y_pred.shape[0]
    log_likelihood = -np.log(y_pred[np.arange(n), y_true_idx] + 1e-12)
    return float(log_likelihood.mean())


def manual_forward_backward(
    x: np.ndarray,
    y_idx: np.ndarray,
    w1: np.ndarray, b1: np.ndarray,
    w2: np.ndarray, b2: np.ndarray,
) -> dict[str, np.ndarray]:
    n = x.shape[0]
    num_classes = w2.shape[1]

    # ---------- Forward Pass ----------
    z1 = x @ w1 + b1
    a1 = relu(z1)
    z2 = a1 @ w2 + b2
    y_hat = softmax(z2)
    loss = cross_entropy(y_hat, y_idx)

    # ---------- Backward Pass (يدويًا، بقاعدة السلسلة) ----------
    y_onehot = np.zeros_like(y_hat)
    y_onehot[np.arange(n), y_idx] = 1.0

    d_z2 = (y_hat - y_onehot) / n          # الصيغة الأنيقة Softmax+CE
    d_w2 = a1.T @ d_z2
    d_b2 = d_z2.sum(axis=0)
    d_a1 = d_z2 @ w2.T
    d_z1 = d_a1 * (z1 > 0).astype(np.float64)   # اشتقاق ReLU
    d_w1 = x.T @ d_z1
    d_b1 = d_z1.sum(axis=0)

    return {
        "loss": np.array(loss),
        "d_w1": d_w1, "d_b1": d_b1,
        "d_w2": d_w2, "d_b2": d_b2,
    }


def torch_forward_backward(
    x: np.ndarray, y_idx: np.ndarray,
    w1: np.ndarray, b1: np.ndarray,
    w2: np.ndarray, b2: np.ndarray,
) -> dict[str, torch.Tensor]:
    x_t = torch.tensor(x, dtype=torch.float64)
    y_t = torch.tensor(y_idx, dtype=torch.long)
    w1_t = torch.tensor(w1, dtype=torch.float64, requires_grad=True)
    b1_t = torch.tensor(b1, dtype=torch.float64, requires_grad=True)
    w2_t = torch.tensor(w2, dtype=torch.float64, requires_grad=True)
    b2_t = torch.tensor(b2, dtype=torch.float64, requires_grad=True)

    z1 = x_t @ w1_t + b1_t
    a1 = torch.relu(z1)
    z2 = a1 @ w2_t + b2_t
    loss = F.cross_entropy(z2, y_t)   # يطبّق log-softmax + NLL داخليًا
    loss.backward()

    return {
        "loss": loss.detach(),
        "d_w1": w1_t.grad, "d_b1": b1_t.grad,
        "d_w2": w2_t.grad, "d_b2": b2_t.grad,
    }


if __name__ == "__main__":
    n_samples, in_dim, hidden_dim, out_dim = 8, 4, 5, 3
    x = rng.normal(size=(n_samples, in_dim))
    y_idx = rng.integers(0, out_dim, size=n_samples)
    w1 = rng.normal(scale=0.1, size=(in_dim, hidden_dim))
    b1 = np.zeros(hidden_dim)
    w2 = rng.normal(scale=0.1, size=(hidden_dim, out_dim))
    b2 = np.zeros(out_dim)

    manual = manual_forward_backward(x, y_idx, w1, b1, w2, b2)
    auto = torch_forward_backward(x, y_idx, w1, b1, w2, b2)

    print(f"Loss (يدوي)   : {manual['loss']:.6f}")
    print(f"Loss (PyTorch): {auto['loss'].item():.6f}")

    for name in ["d_w1", "d_b1", "d_w2", "d_b2"]:
        diff = np.abs(manual[name] - auto[name].numpy()).max()
        print(f"أقصى فرق في {name}: {diff:.2e}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-01-foundations/chapter-03.md
# ────────────────────────────────────────────────────────────────────────
**لاحظ هذه التفاصيل الاحترافية في الكود أعلاه، فكل واحدة منها "أفضل ممارسة" حقيقية:**
- `optimizer.zero_grad(set_to_none=True)` أسرع من `zero_grad()` العادية (يُحرّر الذاكرة بدل تصفيرها).
- ترتيب العمليات: `backward()` ← `clip_grad_norm_()` ← `optimizer.step()` ← `scheduler.step()` — هذا الترتيب **إلزامي** ولا يجوز تبديله.
- `scheduler.step()` يُستدعى **بعد** كل `optimizer.step()` (بمستوى الـ Step وليس الـ Epoch) عند استخدام Warmup/Cosine.
- طباعة `grad_norm` في كل Epoch هي أداة تشخيصية بسيطة لكنها قوية جدًا لرصد مشاكل التدريب مبكرًا.

---

## 5. 💡 نصائح وحيل (Tips & Tricks)

- **قاعدة ذهبية لمعدل التعلم:** عند مضاعفة حجم الدفعة (Batch Size)، يمكن تقريبًا مضاعفة معدل التعلم أيضًا (Linear Scaling Rule من ورقة Goyal et al. 2017)، لكن يجب إضافة Warmup أطول لتعويض عدم الاستقرار الأولي.
- **ابدأ دائمًا بـ AdamW** كخيار افتراضي آمن لتدريب الشبكات العميقة والمحوّلات، إلا إذا كان لديك سبب محدد (كـ SGD+Momentum في بعض نماذج الرؤية الكلاسيكية التي تحقق تعميمًا أفضل أحيانًا).
- **راقب نسبة `grad_norm / max_norm`:** إذا كانت القيمة تتجاوز الحد الأقصى في أغلب الخطوات (أي أن القص يحدث دائمًا)، هذا مؤشر خطر — قد يعني معدل تعلم مرتفع جدًا أو مشكلة في تهيئة الأوزان (Initialization).
- **Gradient Accumulation:** إذا كانت الذاكرة لا تكفي لدفعة كبيرة، اجمع التدرجات عبر عدة دفعات صغيرة قبل استدعاء `optimizer.step()` — هذا يُحاكي تأثير دفعة أكبر دون الحاجة لذاكرة أكبر (أساسي في تدريب LLM).
- **قيمة β2 الافتراضية 0.999 قد لا تناسب كل الحالات:** في تدريب الـ LLM الكبيرة أحيانًا يُستخدم `β2=0.95` بدل `0.999` (كما في بعض إعدادات LLaMA) لاستجابة أسرع للتغيرات، خاصة مع معدلات تعلم مرتفعة نسبيًا.
- **استخدم `torch.compile()`** (المتاح من PyTorch 2.0+) لتسريع حلقة التدريب تلقائيًا دون تغيير منطق الكود، عبر دمج العمليات (Kernel Fusion) خلف الكواليس.

---

## 6. ⚠️ أخطاء شائعة ومصائد (Common Pitfalls)

### الخطأ 1: نسيان `optimizer.zero_grad()`

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-03-chapter-03.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 03: الفصل الثالث: Backpropagation والتحسين (Optimization)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_gradient() -> None:
    parameter, rate = (8.0, 0.15)
    trace = []
    for _ in range(30):
        loss = (parameter - 2.0) ** 2
        analytic = 2.0 * (parameter - 2.0)
        epsilon = 1e-05
        numeric = ((parameter + epsilon - 2) ** 2 - (parameter - epsilon - 2) ** 2) / (2 * epsilon)
        trace.append(loss)
        parameter -= rate * analytic
    print('final parameter:', round(parameter, 5))
    print('initial/final loss:', round(trace[0], 4), round(trace[-1], 8))
    print('finite-difference gradient agrees:', abs(analytic - numeric) < 1e-05)

if __name__ == "__main__":
    print("Chapter 03 lab: Gradient descent and finite differences")
    run_gradient()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-03-chapter-03.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
backprop_manual.py
تطبيق يدوي كامل لـ Forward + Backward Pass لشبكة MLP بطبقتين،
ثم التحقق من صحته مقابل torch.autograd.
Python 3.13+ | NumPy فقط للجزء اليدوي، ثم PyTorch للمقارنة.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

rng = np.random.default_rng(seed=42)


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def softmax(z: np.ndarray) -> np.ndarray:
    z_shifted = z - z.max(axis=1, keepdims=True)  # استقرار عددي
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def cross_entropy(y_pred: np.ndarray, y_true_idx: np.ndarray) -> float:
    n = y_pred.shape[0]
    log_likelihood = -np.log(y_pred[np.arange(n), y_true_idx] + 1e-12)
    return float(log_likelihood.mean())


def manual_forward_backward(
    x: np.ndarray,
    y_idx: np.ndarray,
    w1: np.ndarray, b1: np.ndarray,
    w2: np.ndarray, b2: np.ndarray,
) -> dict[str, np.ndarray]:
    n = x.shape[0]
    num_classes = w2.shape[1]

    # ---------- Forward Pass ----------
    z1 = x @ w1 + b1
    a1 = relu(z1)
    z2 = a1 @ w2 + b2
    y_hat = softmax(z2)
    loss = cross_entropy(y_hat, y_idx)

    # ---------- Backward Pass (يدويًا، بقاعدة السلسلة) ----------
    y_onehot = np.zeros_like(y_hat)
    y_onehot[np.arange(n), y_idx] = 1.0

    d_z2 = (y_hat - y_onehot) / n          # الصيغة الأنيقة Softmax+CE
    d_w2 = a1.T @ d_z2
    d_b2 = d_z2.sum(axis=0)
    d_a1 = d_z2 @ w2.T
    d_z1 = d_a1 * (z1 > 0).astype(np.float64)   # اشتقاق ReLU
    d_w1 = x.T @ d_z1
    d_b1 = d_z1.sum(axis=0)

    return {
        "loss": np.array(loss),
        "d_w1": d_w1, "d_b1": d_b1,
        "d_w2": d_w2, "d_b2": d_b2,
    }


def torch_forward_backward(
    x: np.ndarray, y_idx: np.ndarray,
    w1: np.ndarray, b1: np.ndarray,
    w2: np.ndarray, b2: np.ndarray,
) -> dict[str, torch.Tensor]:
    x_t = torch.tensor(x, dtype=torch.float64)
    y_t = torch.tensor(y_idx, dtype=torch.long)
    w1_t = torch.tensor(w1, dtype=torch.float64, requires_grad=True)
    b1_t = torch.tensor(b1, dtype=torch.float64, requires_grad=True)
    w2_t = torch.tensor(w2, dtype=torch.float64, requires_grad=True)
    b2_t = torch.tensor(b2, dtype=torch.float64, requires_grad=True)

    z1 = x_t @ w1_t + b1_t
    a1 = torch.relu(z1)
    z2 = a1 @ w2_t + b2_t
    loss = F.cross_entropy(z2, y_t)   # يطبّق log-softmax + NLL داخليًا
    loss.backward()

    return {
        "loss": loss.detach(),
        "d_w1": w1_t.grad, "d_b1": b1_t.grad,
        "d_w2": w2_t.grad, "d_b2": b2_t.grad,
    }


if __name__ == "__main__":
    n_samples, in_dim, hidden_dim, out_dim = 8, 4, 5, 3
    x = rng.normal(size=(n_samples, in_dim))
    y_idx = rng.integers(0, out_dim, size=n_samples)
    w1 = rng.normal(scale=0.1, size=(in_dim, hidden_dim))
    b1 = np.zeros(hidden_dim)
    w2 = rng.normal(scale=0.1, size=(hidden_dim, out_dim))
    b2 = np.zeros(out_dim)

    manual = manual_forward_backward(x, y_idx, w1, b1, w2, b2)
    auto = torch_forward_backward(x, y_idx, w1, b1, w2, b2)

    print(f"Loss (يدوي)   : {manual['loss']:.6f}")
    print(f"Loss (PyTorch): {auto['loss'].item():.6f}")

    for name in ["d_w1", "d_b1", "d_w2", "d_b2"]:
        diff = np.abs(manual[name] - auto[name].numpy()).max()
        print(f"أقصى فرق في {name}: {diff:.2e}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-03-chapter-03.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
**لاحظ هذه التفاصيل الاحترافية في الكود أعلاه، فكل واحدة منها "أفضل ممارسة" حقيقية:**
- `optimizer.zero_grad(set_to_none=True)` أسرع من `zero_grad()` العادية (يُحرّر الذاكرة بدل تصفيرها).
- ترتيب العمليات: `backward()` ← `clip_grad_norm_()` ← `optimizer.step()` ← `scheduler.step()` — هذا الترتيب **إلزامي** ولا يجوز تبديله.
- `scheduler.step()` يُستدعى **بعد** كل `optimizer.step()` (بمستوى الـ Step وليس الـ Epoch) عند استخدام Warmup/Cosine.
- طباعة `grad_norm` في كل Epoch هي أداة تشخيصية بسيطة لكنها قوية جدًا لرصد مشاكل التدريب مبكرًا.

---

## 5. 💡 نصائح وحيل (Tips & Tricks)

- **قاعدة ذهبية لمعدل التعلم:** عند مضاعفة حجم الدفعة (Batch Size)، يمكن تقريبًا مضاعفة معدل التعلم أيضًا (Linear Scaling Rule من ورقة Goyal et al. 2017)، لكن يجب إضافة Warmup أطول لتعويض عدم الاستقرار الأولي.
- **ابدأ دائمًا بـ AdamW** كخيار افتراضي آمن لتدريب الشبكات العميقة والمحوّلات، إلا إذا كان لديك سبب محدد (كـ SGD+Momentum في بعض نماذج الرؤية الكلاسيكية التي تحقق تعميمًا أفضل أحيانًا).
- **راقب نسبة `grad_norm / max_norm`:** إذا كانت القيمة تتجاوز الحد الأقصى في أغلب الخطوات (أي أن القص يحدث دائمًا)، هذا مؤشر خطر — قد يعني معدل تعلم مرتفع جدًا أو مشكلة في تهيئة الأوزان (Initialization).
- **Gradient Accumulation:** إذا كانت الذاكرة لا تكفي لدفعة كبيرة، اجمع التدرجات عبر عدة دفعات صغيرة قبل استدعاء `optimizer.step()` — هذا يُحاكي تأثير دفعة أكبر دون الحاجة لذاكرة أكبر (أساسي في تدريب LLM).
- **قيمة β2 الافتراضية 0.999 قد لا تناسب كل الحالات:** في تدريب الـ LLM الكبيرة أحيانًا يُستخدم `β2=0.95` بدل `0.999` (كما في بعض إعدادات LLaMA) لاستجابة أسرع للتغيرات، خاصة مع معدلات تعلم مرتفعة نسبيًا.
- **استخدم `torch.compile()`** (المتاح من PyTorch 2.0+) لتسريع حلقة التدريب تلقائيًا دون تغيير منطق الكود، عبر دمج العمليات (Kernel Fusion) خلف الكواليس.

---

## 6. ⚠️ أخطاء شائعة ومصائد (Common Pitfalls)

### الخطأ 1: نسيان `optimizer.zero_grad()`
