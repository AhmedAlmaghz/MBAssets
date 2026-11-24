#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — مقدّمات رياضية سريعة — اللغة السرّية التي يتحدّثها كل نموذج ذكاء اصطناعي
Large Language Model Engineering — Chapter 01 — Foundations

Book slug : llm-engineering-69-chapters
Chapter   : ch-01-chapter-01 (lesson 1)
Code files: 24 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-01-chapter-01.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — مقدّمات رياضية سريعة — اللغة السرّية التي يتحدّثها كل نموذج ذكاء اصطناعي
Large Language Model Engineering — Chapter 01 — Foundations

Book slug : llm-engineering-69-chapters
Chapter   : ch-01-chapter-01 (lesson 1)
Code files: 23 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-01-chapter-01.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — مقدّمات رياضية سريعة — اللغة السرّية التي يتحدّثها كل نموذج ذكاء اصطناعي
Large Language Model Engineering — Chapter 01 — Foundations

Book slug : llm-engineering-69-chapters
Chapter   : ch-01-chapter-01 (lesson 1)
Code files: 22 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_01.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 01: الفصل الأول: مقدّمات رياضية سريعة — اللغة السرّية التي يتحدّثها كل نموذج ذكاء اصطناعي."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponents = [math.exp(value - maximum) for value in values]
    denominator = sum(exponents)
    return [value / denominator for value in exponents]

def cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum((x * x for x in left)) * sum((y * y for y in right)))
    return sum((x * y for x, y in zip(left, right))) / denominator if denominator else 0.0

def run_math() -> None:
    vector_a, vector_b = ([1.0, 2.0, 0.0], [2.0, 1.0, 2.0])
    probabilities = softmax([1.2, 0.3, -0.5])
    entropy = -sum((p * math.log(p) for p in probabilities))
    print('dot product:', sum((a * b for a, b in zip(vector_a, vector_b))))
    print('cosine similarity:', round(cosine(vector_a, vector_b), 4))
    print('softmax:', [round(p, 4) for p in probabilities])
    print('entropy (nats):', round(entropy, 4))

if __name__ == "__main__":
    print("Chapter 01 lab: Linear algebra and probability")
    run_math()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# تخيّل فضاءً مبسّطًا بُعدين فقط لأغراض التوضيح البصري
king   = np.array([0.90, 0.85])   # "ملك": قيمة عالية في [السلطة, الذكورة]
queen  = np.array([0.88, 0.15])   # "ملكة": سلطة عالية، لكن أنثى
man    = np.array([0.10, 0.90])   # "رجل": سلطة منخفضة، ذكورة عالية
woman  = np.array([0.08, 0.12])   # "امرأة": سلطة منخفضة، أنثى

# العملية السحرية الشهيرة: king - man + woman ≈ queen ؟
result = king - man + woman
print("الناتج:", result)          # تقريبًا: [0.88, 0.07]
print("queen فعليًا:", queen)      # [0.88, 0.15]  ← قريب جدًا!

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# مثال محسوس: دفعة من 3 أمثلة (Batch)، كل مثال له 4 خصائص (Features)
X = np.array([
    [1.0, 2.0, 0.5, 0.1],   # المثال الأول
    [0.3, 0.8, 1.2, 0.9],   # المثال الثاني
    [2.1, 0.4, 0.3, 1.5],   # المثال الثالث
])  # shape: (3, 4) → (batch_size, input_features)

# طبقة تُحوّل كل مثال من 4 خصائص إلى تمثيل بـ 2 خصيصة جديدة
W = np.array([
    [0.5, -0.2],
    [0.1,  0.4],
    [-0.3, 0.6],
    [0.8, -0.1],
])  # shape: (4, 2) → (input_features, output_features)

b = np.array([0.1, -0.1])  # shape: (2,)

Y = X @ W + b  # @ هو رمز الضرب المصفوفي في بايثون الحديث
print("شكل المخرجات:", Y.shape)   # (3, 2)
print(Y)

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """تُطبّع الضرب النقطي بأطوال المتجهين لعزل تأثير الزاوية فقط"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


vec_king  = np.array([0.90, 0.85])
vec_queen = np.array([0.88, 0.15])
vec_apple = np.array([-0.70, 0.05])   # كلمة "تفاحة" - غير ذات صلة

print(f"تشابه (ملك, ملكة) = {cosine_similarity(vec_king, vec_queen):.3f}")
print(f"تشابه (ملك, تفاحة) = {cosine_similarity(vec_king, vec_apple):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
logits = np.array([2.0, 1.0, 0.1, -1.0, 3.5])  # ["سجادة", "كرسي", "طاولة", "سماء", "أريكة"]

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
def softmax_naive(z: np.ndarray) -> np.ndarray:
    """التطبيق الساذج المباشر - احذر، سنرى مشكلته بعد قليل!"""
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


probs = softmax_naive(logits)
print("الاحتمالات:", np.round(probs, 3))
print("المجموع:", probs.sum())  # يجب أن يكون 1.0 بالضبط

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
large_logits = np.array([1000.0, 1001.0, 999.0])
probs = softmax_naive(large_logits)
print(probs)

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
def softmax_stable(z: np.ndarray) -> np.ndarray:
    """
    التطبيق الصحيح والآمن عدديًا - استخدم هذا دائمًا في الإنتاج
    الإثبات الرياضي: softmax(z) = softmax(z - c) لأي ثابت c
    """
    z_shifted = z - np.max(z)   # الآن أكبر قيمة = 0، والباقي سالب أو صفر
    exp_z = np.exp(z_shifted)   # exp(0)=1 كحد أقصى، لا فيضان أبدًا!
    return exp_z / np.sum(exp_z)


large_logits = np.array([1000.0, 1001.0, 999.0])
print(softmax_stable(large_logits))
# [0.2447 0.6652 0.0900]  ← نتيجة صحيحة تمامًا، بلا أي NaN!

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
true_label = np.array([0, 0, 0, 0, 1])  # 1 في موضع "أريكة" فقط، صفر في الباقي
predicted  = np.array([0.196, 0.072, 0.029, 0.010, 0.692])  # من Softmax السابقة

# ────────────────────────────────────────────────────────────────────────
# [010] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """
    eps صغيرة جدًا تُضاف لتجنّب log(0) الذي ينتج -inf
    (مصيدة شائعة أخرى يجب الانتباه لها!)
    """
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    return -np.sum(y_true * np.log(y_pred_clipped))


loss = cross_entropy_loss(true_label, predicted)
print(f"قيمة الخسارة: {loss:.4f}")   # 0.3677

# ────────────────────────────────────────────────────────────────────────
# [011] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

confidence = np.linspace(0.01, 1.0, 100)
loss_values = -np.log(confidence)

plt.plot(confidence, loss_values)
plt.xlabel("ثقة النموذج بالإجابة الصحيحة")
plt.ylabel("قيمة Cross-Entropy Loss")
plt.title("كلما قلّت ثقة النموذج بالإجابة الصحيحة، تصاعدت العقوبة بشكل حاد")
plt.grid(True)
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [012] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """يقيس 'كم من المعلومات نفقدها' عند تقريب التوزيع P بالتوزيع Q"""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * np.log(p / q))


p_true  = np.array([0.1, 0.2, 0.7])   # التوزيع الحقيقي/المرجعي
q_model = np.array([0.15, 0.25, 0.6])  # تنبؤات النموذج

print(f"KL Divergence: {kl_divergence(p_true, q_model):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [013] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
import torch

# نُعرّف وزنًا بسيطًا، ونطلب من PyTorch تتبع مشتقته تلقائيًا
w = torch.tensor(3.0, requires_grad=True)

# دالة خسارة تجريبية بسيطة: L(w) = (w - 5)^2
loss = (w - 5) ** 2

loss.backward()  # PyTorch يحسب dL/dw تلقائيًا عبر Autograd!

print(f"القيمة الحالية لـ w: {w.item()}")
print(f"المشتقة dL/dw عند هذه النقطة: {w.grad.item()}")
# dL/dw = 2(w-5) = 2(3-5) = -4

# ────────────────────────────────────────────────────────────────────────
# [014] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
learning_rate = 0.1
with torch.no_grad():  # نُعطّل تتبع المشتقات أثناء التحديث نفسه
    w -= learning_rate * w.grad

print(f"الوزن بعد خطوة تحديث واحدة: {w.item()}")  # 3.0 - 0.1*(-4) = 3.4

# ────────────────────────────────────────────────────────────────────────
# [015] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
# دالة خسارة تعتمد على وزنين معًا
w1 = torch.tensor(2.0, requires_grad=True)
w2 = torch.tensor(-1.0, requires_grad=True)

loss = (w1 - 4) ** 2 + (w2 + 2) ** 2   # الحد الأدنى عند w1=4, w2=-2

loss.backward()

print(f"∂L/∂w1 = {w1.grad.item()}")   # 2(2-4) = -4
print(f"∂L/∂w2 = {w2.grad.item()}")   # 2(-1+2) = 2

# ────────────────────────────────────────────────────────────────────────
# [016] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
# مثال توضيحي: سلسلة من 3 دوال متتالية (محاكاة مبسّطة لـ 3 طبقات)
x = torch.tensor(2.0, requires_grad=True)

h1 = x ** 2          # الطبقة الأولى: h1 = x²
h2 = torch.sin(h1)   # الطبقة الثانية: h2 = sin(h1)
L  = h2 ** 3          # الطبقة الثالثة (دالة الخسارة): L = h2³

L.backward()

print(f"dL/dx (محسوبة تلقائيًا عبر Chain Rule): {x.grad.item():.4f}")

# التحقق اليدوي بتطبيق Chain Rule خطوة بخطوة:
# dL/dh2 = 3·h2²
# dh2/dh1 = cos(h1)
# dh1/dx = 2x
# dL/dx = 3·h2² · cos(h1) · 2x
import math
h1_val = 2.0 ** 2
h2_val = math.sin(h1_val)
manual_grad = 3 * (h2_val ** 2) * math.cos(h1_val) * 2 * 2.0
print(f"dL/dx (محسوبة يدويًا للتحقق): {manual_grad:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [017] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

z = np.array([2.0, 0.5, -1.0])
true_class_index = 0

z_shifted = z - np.max(z)
exp_z = np.exp(z_shifted)
probs = exp_z / np.sum(exp_z)

print("Softmax Probabilities:", np.round(probs, 3))

loss = -np.log(probs[true_class_index])
print(f"Cross-Entropy Loss: {loss:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [018] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

logits = torch.tensor([[2.0, 0.5, -1.0]])   # شكل: (batch_size=1, num_classes=3)
true_label = torch.tensor([0])               # الفئة الصحيحة: index 0

# ملاحظة احترافية مهمة جدًا:
# F.cross_entropy في PyTorch تُطبّق Softmax و Cross-Entropy معًا داخليًا
# بطريقة مُحسَّنة عدديًا (log-sum-exp trick) - لا تُطبّق Softmax يدويًا قبلها!
loss = F.cross_entropy(logits, true_label)
print(f"PyTorch Loss: {loss.item():.4f}")

# للتحقق من الاحتمالات نفسها، نستخدم softmax بشكل منفصل فقط للعرض
probs = F.softmax(logits, dim=-1)
print(f"PyTorch Probabilities: {probs}")

# ────────────────────────────────────────────────────────────────────────
# [019] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: ينهار مع أي Logits كبيرة نسبيًا
def softmax_wrong(z):
    return np.exp(z) / np.sum(np.exp(z))

# ✅ صحيح: آمن دائمًا بغض النظر عن حجم القيم
def softmax_correct(z):
    z_shifted = z - np.max(z)
    return np.exp(z_shifted) / np.sum(np.exp(z_shifted))

# ────────────────────────────────────────────────────────────────────────
# [020] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: إذا كانت إحدى الاحتمالات 0.0 بالضبط (نادر لكن ممكن)
loss = -np.log(0.0)   # ينتج -inf ثم يُدمّر كل التدريب لاحقًا

# ✅ صحيح: قيمة صغيرة جدًا تمنع الانهيار دون تأثير يُذكر على الدقة
loss = -np.log(np.clip(prob, 1e-12, 1.0))

# ────────────────────────────────────────────────────────────────────────
# [021] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع جدًا للمبتدئين: افتراض أن الضرب يعمل "تلقائيًا" بغض النظر عن الشكل
X = torch.randn(32, 10)   # batch_size=32, features=10
W = torch.randn(10, 5)
W_wrong = torch.randn(5, 10)   # معكوسة بالخطأ!

Y = X @ W         # ✅ يعمل: (32,10)@(10,5) = (32,5)
Y2 = X @ W_wrong  # ❌ RuntimeError: shapes mismatch

# ────────────────────────────────────────────────────────────────────────
# [022] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
def manual_forward_and_loss(
    logits: np.ndarray,       # shape: (batch_size, num_classes)
    true_labels: np.ndarray,  # shape: (batch_size,) - أرقام الفئات الصحيحة
) -> tuple[np.ndarray, float]:
    """
    يجب أن تُرجع:
      1. مصفوفة الاحتمالات بعد Softmax (آمنة عدديًا!)
      2. متوسط Cross-Entropy Loss عبر كل أمثلة الـ Batch

    المتطلبات:
    - طبّق خدعة الاستقرار العددي (طرح max لكل صف على حدة، ليس للمصفوفة كاملة!)
    - تعامل مع batch كامل (أكثر من مثال واحد) وليس مثالًا واحدًا فقط
    - أضف eps صغيرة لتجنب log(0)
    """
    # اكتب الحل هنا
    ...


# اختبر حلك على هذه البيانات:
test_logits = np.array([
    [2.0, 0.5, -1.0],
    [0.1, 3.0, 0.2],
    [-1.0, -0.5, 2.5],
])
test_labels = np.array([0, 1, 2])

probs, loss = manual_forward_and_loss(test_logits, test_labels)
print("الاحتمالات:\n", probs)
print("متوسط الخسارة:", loss)

# تحقق من صحة حلك بمقارنته بـ PyTorch:
import torch
import torch.nn.functional as F
torch_loss = F.cross_entropy(
    torch.tensor(test_logits, dtype=torch.float32),
    torch.tensor(test_labels, dtype=torch.long)
)
print("خسارة PyTorch للمقارنة:", torch_loss.item())
assert abs(loss - torch_loss.item()) < 1e-5, "يوجد خطأ في حلك!"
print("✅ الحل صحيح تمامًا!")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 01: الفصل الأول: مقدّمات رياضية سريعة — اللغة السرّية التي يتحدّثها كل نموذج ذكاء اصطناعي."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponents = [math.exp(value - maximum) for value in values]
    denominator = sum(exponents)
    return [value / denominator for value in exponents]

def cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum((x * x for x in left)) * sum((y * y for y in right)))
    return sum((x * y for x, y in zip(left, right))) / denominator if denominator else 0.0

def run_math() -> None:
    vector_a, vector_b = ([1.0, 2.0, 0.0], [2.0, 1.0, 2.0])
    probabilities = softmax([1.2, 0.3, -0.5])
    entropy = -sum((p * math.log(p) for p in probabilities))
    print('dot product:', sum((a * b for a, b in zip(vector_a, vector_b))))
    print('cosine similarity:', round(cosine(vector_a, vector_b), 4))
    print('softmax:', [round(p, 4) for p in probabilities])
    print('entropy (nats):', round(entropy, 4))

if __name__ == "__main__":
    print("Chapter 01 lab: Linear algebra and probability")
    run_math()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# تخيّل فضاءً مبسّطًا بُعدين فقط لأغراض التوضيح البصري
king   = np.array([0.90, 0.85])   # "ملك": قيمة عالية في [السلطة, الذكورة]
queen  = np.array([0.88, 0.15])   # "ملكة": سلطة عالية، لكن أنثى
man    = np.array([0.10, 0.90])   # "رجل": سلطة منخفضة، ذكورة عالية
woman  = np.array([0.08, 0.12])   # "امرأة": سلطة منخفضة، أنثى

# العملية السحرية الشهيرة: king - man + woman ≈ queen ؟
result = king - man + woman
print("الناتج:", result)          # تقريبًا: [0.88, 0.07]
print("queen فعليًا:", queen)      # [0.88, 0.15]  ← قريب جدًا!

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# مثال محسوس: دفعة من 3 أمثلة (Batch)، كل مثال له 4 خصائص (Features)
X = np.array([
    [1.0, 2.0, 0.5, 0.1],   # المثال الأول
    [0.3, 0.8, 1.2, 0.9],   # المثال الثاني
    [2.1, 0.4, 0.3, 1.5],   # المثال الثالث
])  # shape: (3, 4) → (batch_size, input_features)

# طبقة تُحوّل كل مثال من 4 خصائص إلى تمثيل بـ 2 خصيصة جديدة
W = np.array([
    [0.5, -0.2],
    [0.1,  0.4],
    [-0.3, 0.6],
    [0.8, -0.1],
])  # shape: (4, 2) → (input_features, output_features)

b = np.array([0.1, -0.1])  # shape: (2,)

Y = X @ W + b  # @ هو رمز الضرب المصفوفي في بايثون الحديث
print("شكل المخرجات:", Y.shape)   # (3, 2)
print(Y)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """تُطبّع الضرب النقطي بأطوال المتجهين لعزل تأثير الزاوية فقط"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


vec_king  = np.array([0.90, 0.85])
vec_queen = np.array([0.88, 0.15])
vec_apple = np.array([-0.70, 0.05])   # كلمة "تفاحة" - غير ذات صلة

print(f"تشابه (ملك, ملكة) = {cosine_similarity(vec_king, vec_queen):.3f}")
print(f"تشابه (ملك, تفاحة) = {cosine_similarity(vec_king, vec_apple):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
logits = np.array([2.0, 1.0, 0.1, -1.0, 3.5])  # ["سجادة", "كرسي", "طاولة", "سماء", "أريكة"]

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def softmax_naive(z: np.ndarray) -> np.ndarray:
    """التطبيق الساذج المباشر - احذر، سنرى مشكلته بعد قليل!"""
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


probs = softmax_naive(logits)
print("الاحتمالات:", np.round(probs, 3))
print("المجموع:", probs.sum())  # يجب أن يكون 1.0 بالضبط

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
large_logits = np.array([1000.0, 1001.0, 999.0])
probs = softmax_naive(large_logits)
print(probs)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def softmax_stable(z: np.ndarray) -> np.ndarray:
    """
    التطبيق الصحيح والآمن عدديًا - استخدم هذا دائمًا في الإنتاج
    الإثبات الرياضي: softmax(z) = softmax(z - c) لأي ثابت c
    """
    z_shifted = z - np.max(z)   # الآن أكبر قيمة = 0، والباقي سالب أو صفر
    exp_z = np.exp(z_shifted)   # exp(0)=1 كحد أقصى، لا فيضان أبدًا!
    return exp_z / np.sum(exp_z)


large_logits = np.array([1000.0, 1001.0, 999.0])
print(softmax_stable(large_logits))
# [0.2447 0.6652 0.0900]  ← نتيجة صحيحة تمامًا، بلا أي NaN!

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
true_label = np.array([0, 0, 0, 0, 1])  # 1 في موضع "أريكة" فقط، صفر في الباقي
predicted  = np.array([0.196, 0.072, 0.029, 0.010, 0.692])  # من Softmax السابقة

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """
    eps صغيرة جدًا تُضاف لتجنّب log(0) الذي ينتج -inf
    (مصيدة شائعة أخرى يجب الانتباه لها!)
    """
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    return -np.sum(y_true * np.log(y_pred_clipped))


loss = cross_entropy_loss(true_label, predicted)
print(f"قيمة الخسارة: {loss:.4f}")   # 0.3677

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

confidence = np.linspace(0.01, 1.0, 100)
loss_values = -np.log(confidence)

plt.plot(confidence, loss_values)
plt.xlabel("ثقة النموذج بالإجابة الصحيحة")
plt.ylabel("قيمة Cross-Entropy Loss")
plt.title("كلما قلّت ثقة النموذج بالإجابة الصحيحة، تصاعدت العقوبة بشكل حاد")
plt.grid(True)
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """يقيس 'كم من المعلومات نفقدها' عند تقريب التوزيع P بالتوزيع Q"""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * np.log(p / q))


p_true  = np.array([0.1, 0.2, 0.7])   # التوزيع الحقيقي/المرجعي
q_model = np.array([0.15, 0.25, 0.6])  # تنبؤات النموذج

print(f"KL Divergence: {kl_divergence(p_true, q_model):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import torch

# نُعرّف وزنًا بسيطًا، ونطلب من PyTorch تتبع مشتقته تلقائيًا
w = torch.tensor(3.0, requires_grad=True)

# دالة خسارة تجريبية بسيطة: L(w) = (w - 5)^2
loss = (w - 5) ** 2

loss.backward()  # PyTorch يحسب dL/dw تلقائيًا عبر Autograd!

print(f"القيمة الحالية لـ w: {w.item()}")
print(f"المشتقة dL/dw عند هذه النقطة: {w.grad.item()}")
# dL/dw = 2(w-5) = 2(3-5) = -4

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
learning_rate = 0.1
with torch.no_grad():  # نُعطّل تتبع المشتقات أثناء التحديث نفسه
    w -= learning_rate * w.grad

print(f"الوزن بعد خطوة تحديث واحدة: {w.item()}")  # 3.0 - 0.1*(-4) = 3.4

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
# دالة خسارة تعتمد على وزنين معًا
w1 = torch.tensor(2.0, requires_grad=True)
w2 = torch.tensor(-1.0, requires_grad=True)

loss = (w1 - 4) ** 2 + (w2 + 2) ** 2   # الحد الأدنى عند w1=4, w2=-2

loss.backward()

print(f"∂L/∂w1 = {w1.grad.item()}")   # 2(2-4) = -4
print(f"∂L/∂w2 = {w2.grad.item()}")   # 2(-1+2) = 2

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
# مثال توضيحي: سلسلة من 3 دوال متتالية (محاكاة مبسّطة لـ 3 طبقات)
x = torch.tensor(2.0, requires_grad=True)

h1 = x ** 2          # الطبقة الأولى: h1 = x²
h2 = torch.sin(h1)   # الطبقة الثانية: h2 = sin(h1)
L  = h2 ** 3          # الطبقة الثالثة (دالة الخسارة): L = h2³

L.backward()

print(f"dL/dx (محسوبة تلقائيًا عبر Chain Rule): {x.grad.item():.4f}")

# التحقق اليدوي بتطبيق Chain Rule خطوة بخطوة:
# dL/dh2 = 3·h2²
# dh2/dh1 = cos(h1)
# dh1/dx = 2x
# dL/dx = 3·h2² · cos(h1) · 2x
import math
h1_val = 2.0 ** 2
h2_val = math.sin(h1_val)
manual_grad = 3 * (h2_val ** 2) * math.cos(h1_val) * 2 * 2.0
print(f"dL/dx (محسوبة يدويًا للتحقق): {manual_grad:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
import numpy as np

z = np.array([2.0, 0.5, -1.0])
true_class_index = 0

z_shifted = z - np.max(z)
exp_z = np.exp(z_shifted)
probs = exp_z / np.sum(exp_z)

print("Softmax Probabilities:", np.round(probs, 3))

loss = -np.log(probs[true_class_index])
print(f"Cross-Entropy Loss: {loss:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

logits = torch.tensor([[2.0, 0.5, -1.0]])   # شكل: (batch_size=1, num_classes=3)
true_label = torch.tensor([0])               # الفئة الصحيحة: index 0

# ملاحظة احترافية مهمة جدًا:
# F.cross_entropy في PyTorch تُطبّق Softmax و Cross-Entropy معًا داخليًا
# بطريقة مُحسَّنة عدديًا (log-sum-exp trick) - لا تُطبّق Softmax يدويًا قبلها!
loss = F.cross_entropy(logits, true_label)
print(f"PyTorch Loss: {loss.item():.4f}")

# للتحقق من الاحتمالات نفسها، نستخدم softmax بشكل منفصل فقط للعرض
probs = F.softmax(logits, dim=-1)
print(f"PyTorch Probabilities: {probs}")

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: ينهار مع أي Logits كبيرة نسبيًا
def softmax_wrong(z):
    return np.exp(z) / np.sum(np.exp(z))

# ✅ صحيح: آمن دائمًا بغض النظر عن حجم القيم
def softmax_correct(z):
    z_shifted = z - np.max(z)
    return np.exp(z_shifted) / np.sum(np.exp(z_shifted))

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: إذا كانت إحدى الاحتمالات 0.0 بالضبط (نادر لكن ممكن)
loss = -np.log(0.0)   # ينتج -inf ثم يُدمّر كل التدريب لاحقًا

# ✅ صحيح: قيمة صغيرة جدًا تمنع الانهيار دون تأثير يُذكر على الدقة
loss = -np.log(np.clip(prob, 1e-12, 1.0))

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع جدًا للمبتدئين: افتراض أن الضرب يعمل "تلقائيًا" بغض النظر عن الشكل
X = torch.randn(32, 10)   # batch_size=32, features=10
W = torch.randn(10, 5)
W_wrong = torch.randn(5, 10)   # معكوسة بالخطأ!

Y = X @ W         # ✅ يعمل: (32,10)@(10,5) = (32,5)
Y2 = X @ W_wrong  # ❌ RuntimeError: shapes mismatch

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
def manual_forward_and_loss(
    logits: np.ndarray,       # shape: (batch_size, num_classes)
    true_labels: np.ndarray,  # shape: (batch_size,) - أرقام الفئات الصحيحة
) -> tuple[np.ndarray, float]:
    """
    يجب أن تُرجع:
      1. مصفوفة الاحتمالات بعد Softmax (آمنة عدديًا!)
      2. متوسط Cross-Entropy Loss عبر كل أمثلة الـ Batch

    المتطلبات:
    - طبّق خدعة الاستقرار العددي (طرح max لكل صف على حدة، ليس للمصفوفة كاملة!)
    - تعامل مع batch كامل (أكثر من مثال واحد) وليس مثالًا واحدًا فقط
    - أضف eps صغيرة لتجنب log(0)
    """
    # اكتب الحل هنا
    ...


# اختبر حلك على هذه البيانات:
test_logits = np.array([
    [2.0, 0.5, -1.0],
    [0.1, 3.0, 0.2],
    [-1.0, -0.5, 2.5],
])
test_labels = np.array([0, 1, 2])

probs, loss = manual_forward_and_loss(test_logits, test_labels)
print("الاحتمالات:\n", probs)
print("متوسط الخسارة:", loss)

# تحقق من صحة حلك بمقارنته بـ PyTorch:
import torch
import torch.nn.functional as F
torch_loss = F.cross_entropy(
    torch.tensor(test_logits, dtype=torch.float32),
    torch.tensor(test_labels, dtype=torch.long)
)
print("خسارة PyTorch للمقارنة:", torch_loss.item())
assert abs(loss - torch_loss.item()) < 1e-5, "يوجد خطأ في حلك!"
print("✅ الحل صحيح تمامًا!")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — مقدّمات رياضية سريعة — اللغة السرّية التي يتحدّثها كل نموذج ذكاء اصطناعي
Large Language Model Engineering — Chapter 01 — Foundations

Book slug : llm-engineering-69-chapters
Chapter   : ch-01-chapter-01 (lesson 1)
Code files: 22 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_01.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 01: الفصل الأول: مقدّمات رياضية سريعة — اللغة السرّية التي يتحدّثها كل نموذج ذكاء اصطناعي."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponents = [math.exp(value - maximum) for value in values]
    denominator = sum(exponents)
    return [value / denominator for value in exponents]

def cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum((x * x for x in left)) * sum((y * y for y in right)))
    return sum((x * y for x, y in zip(left, right))) / denominator if denominator else 0.0

def run_math() -> None:
    vector_a, vector_b = ([1.0, 2.0, 0.0], [2.0, 1.0, 2.0])
    probabilities = softmax([1.2, 0.3, -0.5])
    entropy = -sum((p * math.log(p) for p in probabilities))
    print('dot product:', sum((a * b for a, b in zip(vector_a, vector_b))))
    print('cosine similarity:', round(cosine(vector_a, vector_b), 4))
    print('softmax:', [round(p, 4) for p in probabilities])
    print('entropy (nats):', round(entropy, 4))

if __name__ == "__main__":
    print("Chapter 01 lab: Linear algebra and probability")
    run_math()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# تخيّل فضاءً مبسّطًا بُعدين فقط لأغراض التوضيح البصري
king   = np.array([0.90, 0.85])   # "ملك": قيمة عالية في [السلطة, الذكورة]
queen  = np.array([0.88, 0.15])   # "ملكة": سلطة عالية، لكن أنثى
man    = np.array([0.10, 0.90])   # "رجل": سلطة منخفضة، ذكورة عالية
woman  = np.array([0.08, 0.12])   # "امرأة": سلطة منخفضة، أنثى

# العملية السحرية الشهيرة: king - man + woman ≈ queen ؟
result = king - man + woman
print("الناتج:", result)          # تقريبًا: [0.88, 0.07]
print("queen فعليًا:", queen)      # [0.88, 0.15]  ← قريب جدًا!

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# مثال محسوس: دفعة من 3 أمثلة (Batch)، كل مثال له 4 خصائص (Features)
X = np.array([
    [1.0, 2.0, 0.5, 0.1],   # المثال الأول
    [0.3, 0.8, 1.2, 0.9],   # المثال الثاني
    [2.1, 0.4, 0.3, 1.5],   # المثال الثالث
])  # shape: (3, 4) → (batch_size, input_features)

# طبقة تُحوّل كل مثال من 4 خصائص إلى تمثيل بـ 2 خصيصة جديدة
W = np.array([
    [0.5, -0.2],
    [0.1,  0.4],
    [-0.3, 0.6],
    [0.8, -0.1],
])  # shape: (4, 2) → (input_features, output_features)

b = np.array([0.1, -0.1])  # shape: (2,)

Y = X @ W + b  # @ هو رمز الضرب المصفوفي في بايثون الحديث
print("شكل المخرجات:", Y.shape)   # (3, 2)
print(Y)

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """تُطبّع الضرب النقطي بأطوال المتجهين لعزل تأثير الزاوية فقط"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


vec_king  = np.array([0.90, 0.85])
vec_queen = np.array([0.88, 0.15])
vec_apple = np.array([-0.70, 0.05])   # كلمة "تفاحة" - غير ذات صلة

print(f"تشابه (ملك, ملكة) = {cosine_similarity(vec_king, vec_queen):.3f}")
print(f"تشابه (ملك, تفاحة) = {cosine_similarity(vec_king, vec_apple):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
logits = np.array([2.0, 1.0, 0.1, -1.0, 3.5])  # ["سجادة", "كرسي", "طاولة", "سماء", "أريكة"]

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
def softmax_naive(z: np.ndarray) -> np.ndarray:
    """التطبيق الساذج المباشر - احذر، سنرى مشكلته بعد قليل!"""
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


probs = softmax_naive(logits)
print("الاحتمالات:", np.round(probs, 3))
print("المجموع:", probs.sum())  # يجب أن يكون 1.0 بالضبط

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
large_logits = np.array([1000.0, 1001.0, 999.0])
probs = softmax_naive(large_logits)
print(probs)

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
def softmax_stable(z: np.ndarray) -> np.ndarray:
    """
    التطبيق الصحيح والآمن عدديًا - استخدم هذا دائمًا في الإنتاج
    الإثبات الرياضي: softmax(z) = softmax(z - c) لأي ثابت c
    """
    z_shifted = z - np.max(z)   # الآن أكبر قيمة = 0، والباقي سالب أو صفر
    exp_z = np.exp(z_shifted)   # exp(0)=1 كحد أقصى، لا فيضان أبدًا!
    return exp_z / np.sum(exp_z)


large_logits = np.array([1000.0, 1001.0, 999.0])
print(softmax_stable(large_logits))
# [0.2447 0.6652 0.0900]  ← نتيجة صحيحة تمامًا، بلا أي NaN!

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
true_label = np.array([0, 0, 0, 0, 1])  # 1 في موضع "أريكة" فقط، صفر في الباقي
predicted  = np.array([0.196, 0.072, 0.029, 0.010, 0.692])  # من Softmax السابقة

# ────────────────────────────────────────────────────────────────────────
# [010] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """
    eps صغيرة جدًا تُضاف لتجنّب log(0) الذي ينتج -inf
    (مصيدة شائعة أخرى يجب الانتباه لها!)
    """
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    return -np.sum(y_true * np.log(y_pred_clipped))


loss = cross_entropy_loss(true_label, predicted)
print(f"قيمة الخسارة: {loss:.4f}")   # 0.3677

# ────────────────────────────────────────────────────────────────────────
# [011] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

confidence = np.linspace(0.01, 1.0, 100)
loss_values = -np.log(confidence)

plt.plot(confidence, loss_values)
plt.xlabel("ثقة النموذج بالإجابة الصحيحة")
plt.ylabel("قيمة Cross-Entropy Loss")
plt.title("كلما قلّت ثقة النموذج بالإجابة الصحيحة، تصاعدت العقوبة بشكل حاد")
plt.grid(True)
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [012] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """يقيس 'كم من المعلومات نفقدها' عند تقريب التوزيع P بالتوزيع Q"""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * np.log(p / q))


p_true  = np.array([0.1, 0.2, 0.7])   # التوزيع الحقيقي/المرجعي
q_model = np.array([0.15, 0.25, 0.6])  # تنبؤات النموذج

print(f"KL Divergence: {kl_divergence(p_true, q_model):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [013] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
import torch

# نُعرّف وزنًا بسيطًا، ونطلب من PyTorch تتبع مشتقته تلقائيًا
w = torch.tensor(3.0, requires_grad=True)

# دالة خسارة تجريبية بسيطة: L(w) = (w - 5)^2
loss = (w - 5) ** 2

loss.backward()  # PyTorch يحسب dL/dw تلقائيًا عبر Autograd!

print(f"القيمة الحالية لـ w: {w.item()}")
print(f"المشتقة dL/dw عند هذه النقطة: {w.grad.item()}")
# dL/dw = 2(w-5) = 2(3-5) = -4

# ────────────────────────────────────────────────────────────────────────
# [014] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
learning_rate = 0.1
with torch.no_grad():  # نُعطّل تتبع المشتقات أثناء التحديث نفسه
    w -= learning_rate * w.grad

print(f"الوزن بعد خطوة تحديث واحدة: {w.item()}")  # 3.0 - 0.1*(-4) = 3.4

# ────────────────────────────────────────────────────────────────────────
# [015] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
# دالة خسارة تعتمد على وزنين معًا
w1 = torch.tensor(2.0, requires_grad=True)
w2 = torch.tensor(-1.0, requires_grad=True)

loss = (w1 - 4) ** 2 + (w2 + 2) ** 2   # الحد الأدنى عند w1=4, w2=-2

loss.backward()

print(f"∂L/∂w1 = {w1.grad.item()}")   # 2(2-4) = -4
print(f"∂L/∂w2 = {w2.grad.item()}")   # 2(-1+2) = 2

# ────────────────────────────────────────────────────────────────────────
# [016] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
# مثال توضيحي: سلسلة من 3 دوال متتالية (محاكاة مبسّطة لـ 3 طبقات)
x = torch.tensor(2.0, requires_grad=True)

h1 = x ** 2          # الطبقة الأولى: h1 = x²
h2 = torch.sin(h1)   # الطبقة الثانية: h2 = sin(h1)
L  = h2 ** 3          # الطبقة الثالثة (دالة الخسارة): L = h2³

L.backward()

print(f"dL/dx (محسوبة تلقائيًا عبر Chain Rule): {x.grad.item():.4f}")

# التحقق اليدوي بتطبيق Chain Rule خطوة بخطوة:
# dL/dh2 = 3·h2²
# dh2/dh1 = cos(h1)
# dh1/dx = 2x
# dL/dx = 3·h2² · cos(h1) · 2x
import math
h1_val = 2.0 ** 2
h2_val = math.sin(h1_val)
manual_grad = 3 * (h2_val ** 2) * math.cos(h1_val) * 2 * 2.0
print(f"dL/dx (محسوبة يدويًا للتحقق): {manual_grad:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [017] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

z = np.array([2.0, 0.5, -1.0])
true_class_index = 0

z_shifted = z - np.max(z)
exp_z = np.exp(z_shifted)
probs = exp_z / np.sum(exp_z)

print("Softmax Probabilities:", np.round(probs, 3))

loss = -np.log(probs[true_class_index])
print(f"Cross-Entropy Loss: {loss:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [018] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

logits = torch.tensor([[2.0, 0.5, -1.0]])   # شكل: (batch_size=1, num_classes=3)
true_label = torch.tensor([0])               # الفئة الصحيحة: index 0

# ملاحظة احترافية مهمة جدًا:
# F.cross_entropy في PyTorch تُطبّق Softmax و Cross-Entropy معًا داخليًا
# بطريقة مُحسَّنة عدديًا (log-sum-exp trick) - لا تُطبّق Softmax يدويًا قبلها!
loss = F.cross_entropy(logits, true_label)
print(f"PyTorch Loss: {loss.item():.4f}")

# للتحقق من الاحتمالات نفسها، نستخدم softmax بشكل منفصل فقط للعرض
probs = F.softmax(logits, dim=-1)
print(f"PyTorch Probabilities: {probs}")

# ────────────────────────────────────────────────────────────────────────
# [019] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: ينهار مع أي Logits كبيرة نسبيًا
def softmax_wrong(z):
    return np.exp(z) / np.sum(np.exp(z))

# ✅ صحيح: آمن دائمًا بغض النظر عن حجم القيم
def softmax_correct(z):
    z_shifted = z - np.max(z)
    return np.exp(z_shifted) / np.sum(np.exp(z_shifted))

# ────────────────────────────────────────────────────────────────────────
# [020] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: إذا كانت إحدى الاحتمالات 0.0 بالضبط (نادر لكن ممكن)
loss = -np.log(0.0)   # ينتج -inf ثم يُدمّر كل التدريب لاحقًا

# ✅ صحيح: قيمة صغيرة جدًا تمنع الانهيار دون تأثير يُذكر على الدقة
loss = -np.log(np.clip(prob, 1e-12, 1.0))

# ────────────────────────────────────────────────────────────────────────
# [021] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع جدًا للمبتدئين: افتراض أن الضرب يعمل "تلقائيًا" بغض النظر عن الشكل
X = torch.randn(32, 10)   # batch_size=32, features=10
W = torch.randn(10, 5)
W_wrong = torch.randn(5, 10)   # معكوسة بالخطأ!

Y = X @ W         # ✅ يعمل: (32,10)@(10,5) = (32,5)
Y2 = X @ W_wrong  # ❌ RuntimeError: shapes mismatch

# ────────────────────────────────────────────────────────────────────────
# [022] book/chapters/part-01-foundations/chapter-01.md
# ────────────────────────────────────────────────────────────────────────
def manual_forward_and_loss(
    logits: np.ndarray,       # shape: (batch_size, num_classes)
    true_labels: np.ndarray,  # shape: (batch_size,) - أرقام الفئات الصحيحة
) -> tuple[np.ndarray, float]:
    """
    يجب أن تُرجع:
      1. مصفوفة الاحتمالات بعد Softmax (آمنة عدديًا!)
      2. متوسط Cross-Entropy Loss عبر كل أمثلة الـ Batch

    المتطلبات:
    - طبّق خدعة الاستقرار العددي (طرح max لكل صف على حدة، ليس للمصفوفة كاملة!)
    - تعامل مع batch كامل (أكثر من مثال واحد) وليس مثالًا واحدًا فقط
    - أضف eps صغيرة لتجنب log(0)
    """
    # اكتب الحل هنا
    ...


# اختبر حلك على هذه البيانات:
test_logits = np.array([
    [2.0, 0.5, -1.0],
    [0.1, 3.0, 0.2],
    [-1.0, -0.5, 2.5],
])
test_labels = np.array([0, 1, 2])

probs, loss = manual_forward_and_loss(test_logits, test_labels)
print("الاحتمالات:\n", probs)
print("متوسط الخسارة:", loss)

# تحقق من صحة حلك بمقارنته بـ PyTorch:
import torch
import torch.nn.functional as F
torch_loss = F.cross_entropy(
    torch.tensor(test_logits, dtype=torch.float32),
    torch.tensor(test_labels, dtype=torch.long)
)
print("خسارة PyTorch للمقارنة:", torch_loss.item())
assert abs(loss - torch_loss.item()) < 1e-5, "يوجد خطأ في حلك!"
print("✅ الحل صحيح تمامًا!")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 01: الفصل الأول: مقدّمات رياضية سريعة — اللغة السرّية التي يتحدّثها كل نموذج ذكاء اصطناعي."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponents = [math.exp(value - maximum) for value in values]
    denominator = sum(exponents)
    return [value / denominator for value in exponents]

def cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum((x * x for x in left)) * sum((y * y for y in right)))
    return sum((x * y for x, y in zip(left, right))) / denominator if denominator else 0.0

def run_math() -> None:
    vector_a, vector_b = ([1.0, 2.0, 0.0], [2.0, 1.0, 2.0])
    probabilities = softmax([1.2, 0.3, -0.5])
    entropy = -sum((p * math.log(p) for p in probabilities))
    print('dot product:', sum((a * b for a, b in zip(vector_a, vector_b))))
    print('cosine similarity:', round(cosine(vector_a, vector_b), 4))
    print('softmax:', [round(p, 4) for p in probabilities])
    print('entropy (nats):', round(entropy, 4))

if __name__ == "__main__":
    print("Chapter 01 lab: Linear algebra and probability")
    run_math()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# تخيّل فضاءً مبسّطًا بُعدين فقط لأغراض التوضيح البصري
king   = np.array([0.90, 0.85])   # "ملك": قيمة عالية في [السلطة, الذكورة]
queen  = np.array([0.88, 0.15])   # "ملكة": سلطة عالية، لكن أنثى
man    = np.array([0.10, 0.90])   # "رجل": سلطة منخفضة، ذكورة عالية
woman  = np.array([0.08, 0.12])   # "امرأة": سلطة منخفضة، أنثى

# العملية السحرية الشهيرة: king - man + woman ≈ queen ؟
result = king - man + woman
print("الناتج:", result)          # تقريبًا: [0.88, 0.07]
print("queen فعليًا:", queen)      # [0.88, 0.15]  ← قريب جدًا!

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# مثال محسوس: دفعة من 3 أمثلة (Batch)، كل مثال له 4 خصائص (Features)
X = np.array([
    [1.0, 2.0, 0.5, 0.1],   # المثال الأول
    [0.3, 0.8, 1.2, 0.9],   # المثال الثاني
    [2.1, 0.4, 0.3, 1.5],   # المثال الثالث
])  # shape: (3, 4) → (batch_size, input_features)

# طبقة تُحوّل كل مثال من 4 خصائص إلى تمثيل بـ 2 خصيصة جديدة
W = np.array([
    [0.5, -0.2],
    [0.1,  0.4],
    [-0.3, 0.6],
    [0.8, -0.1],
])  # shape: (4, 2) → (input_features, output_features)

b = np.array([0.1, -0.1])  # shape: (2,)

Y = X @ W + b  # @ هو رمز الضرب المصفوفي في بايثون الحديث
print("شكل المخرجات:", Y.shape)   # (3, 2)
print(Y)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """تُطبّع الضرب النقطي بأطوال المتجهين لعزل تأثير الزاوية فقط"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


vec_king  = np.array([0.90, 0.85])
vec_queen = np.array([0.88, 0.15])
vec_apple = np.array([-0.70, 0.05])   # كلمة "تفاحة" - غير ذات صلة

print(f"تشابه (ملك, ملكة) = {cosine_similarity(vec_king, vec_queen):.3f}")
print(f"تشابه (ملك, تفاحة) = {cosine_similarity(vec_king, vec_apple):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
logits = np.array([2.0, 1.0, 0.1, -1.0, 3.5])  # ["سجادة", "كرسي", "طاولة", "سماء", "أريكة"]

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def softmax_naive(z: np.ndarray) -> np.ndarray:
    """التطبيق الساذج المباشر - احذر، سنرى مشكلته بعد قليل!"""
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


probs = softmax_naive(logits)
print("الاحتمالات:", np.round(probs, 3))
print("المجموع:", probs.sum())  # يجب أن يكون 1.0 بالضبط

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
large_logits = np.array([1000.0, 1001.0, 999.0])
probs = softmax_naive(large_logits)
print(probs)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
def softmax_stable(z: np.ndarray) -> np.ndarray:
    """
    التطبيق الصحيح والآمن عدديًا - استخدم هذا دائمًا في الإنتاج
    الإثبات الرياضي: softmax(z) = softmax(z - c) لأي ثابت c
    """
    z_shifted = z - np.max(z)   # الآن أكبر قيمة = 0، والباقي سالب أو صفر
    exp_z = np.exp(z_shifted)   # exp(0)=1 كحد أقصى، لا فيضان أبدًا!
    return exp_z / np.sum(exp_z)


large_logits = np.array([1000.0, 1001.0, 999.0])
print(softmax_stable(large_logits))
# [0.2447 0.6652 0.0900]  ← نتيجة صحيحة تمامًا، بلا أي NaN!

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
true_label = np.array([0, 0, 0, 0, 1])  # 1 في موضع "أريكة" فقط، صفر في الباقي
predicted  = np.array([0.196, 0.072, 0.029, 0.010, 0.692])  # من Softmax السابقة

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """
    eps صغيرة جدًا تُضاف لتجنّب log(0) الذي ينتج -inf
    (مصيدة شائعة أخرى يجب الانتباه لها!)
    """
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    return -np.sum(y_true * np.log(y_pred_clipped))


loss = cross_entropy_loss(true_label, predicted)
print(f"قيمة الخسارة: {loss:.4f}")   # 0.3677

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

confidence = np.linspace(0.01, 1.0, 100)
loss_values = -np.log(confidence)

plt.plot(confidence, loss_values)
plt.xlabel("ثقة النموذج بالإجابة الصحيحة")
plt.ylabel("قيمة Cross-Entropy Loss")
plt.title("كلما قلّت ثقة النموذج بالإجابة الصحيحة، تصاعدت العقوبة بشكل حاد")
plt.grid(True)
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """يقيس 'كم من المعلومات نفقدها' عند تقريب التوزيع P بالتوزيع Q"""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * np.log(p / q))


p_true  = np.array([0.1, 0.2, 0.7])   # التوزيع الحقيقي/المرجعي
q_model = np.array([0.15, 0.25, 0.6])  # تنبؤات النموذج

print(f"KL Divergence: {kl_divergence(p_true, q_model):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
import torch

# نُعرّف وزنًا بسيطًا، ونطلب من PyTorch تتبع مشتقته تلقائيًا
w = torch.tensor(3.0, requires_grad=True)

# دالة خسارة تجريبية بسيطة: L(w) = (w - 5)^2
loss = (w - 5) ** 2

loss.backward()  # PyTorch يحسب dL/dw تلقائيًا عبر Autograd!

print(f"القيمة الحالية لـ w: {w.item()}")
print(f"المشتقة dL/dw عند هذه النقطة: {w.grad.item()}")
# dL/dw = 2(w-5) = 2(3-5) = -4

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
learning_rate = 0.1
with torch.no_grad():  # نُعطّل تتبع المشتقات أثناء التحديث نفسه
    w -= learning_rate * w.grad

print(f"الوزن بعد خطوة تحديث واحدة: {w.item()}")  # 3.0 - 0.1*(-4) = 3.4

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
# دالة خسارة تعتمد على وزنين معًا
w1 = torch.tensor(2.0, requires_grad=True)
w2 = torch.tensor(-1.0, requires_grad=True)

loss = (w1 - 4) ** 2 + (w2 + 2) ** 2   # الحد الأدنى عند w1=4, w2=-2

loss.backward()

print(f"∂L/∂w1 = {w1.grad.item()}")   # 2(2-4) = -4
print(f"∂L/∂w2 = {w2.grad.item()}")   # 2(-1+2) = 2

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
# مثال توضيحي: سلسلة من 3 دوال متتالية (محاكاة مبسّطة لـ 3 طبقات)
x = torch.tensor(2.0, requires_grad=True)

h1 = x ** 2          # الطبقة الأولى: h1 = x²
h2 = torch.sin(h1)   # الطبقة الثانية: h2 = sin(h1)
L  = h2 ** 3          # الطبقة الثالثة (دالة الخسارة): L = h2³

L.backward()

print(f"dL/dx (محسوبة تلقائيًا عبر Chain Rule): {x.grad.item():.4f}")

# التحقق اليدوي بتطبيق Chain Rule خطوة بخطوة:
# dL/dh2 = 3·h2²
# dh2/dh1 = cos(h1)
# dh1/dx = 2x
# dL/dx = 3·h2² · cos(h1) · 2x
import math
h1_val = 2.0 ** 2
h2_val = math.sin(h1_val)
manual_grad = 3 * (h2_val ** 2) * math.cos(h1_val) * 2 * 2.0
print(f"dL/dx (محسوبة يدويًا للتحقق): {manual_grad:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
import numpy as np

z = np.array([2.0, 0.5, -1.0])
true_class_index = 0

z_shifted = z - np.max(z)
exp_z = np.exp(z_shifted)
probs = exp_z / np.sum(exp_z)

print("Softmax Probabilities:", np.round(probs, 3))

loss = -np.log(probs[true_class_index])
print(f"Cross-Entropy Loss: {loss:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

logits = torch.tensor([[2.0, 0.5, -1.0]])   # شكل: (batch_size=1, num_classes=3)
true_label = torch.tensor([0])               # الفئة الصحيحة: index 0

# ملاحظة احترافية مهمة جدًا:
# F.cross_entropy في PyTorch تُطبّق Softmax و Cross-Entropy معًا داخليًا
# بطريقة مُحسَّنة عدديًا (log-sum-exp trick) - لا تُطبّق Softmax يدويًا قبلها!
loss = F.cross_entropy(logits, true_label)
print(f"PyTorch Loss: {loss.item():.4f}")

# للتحقق من الاحتمالات نفسها، نستخدم softmax بشكل منفصل فقط للعرض
probs = F.softmax(logits, dim=-1)
print(f"PyTorch Probabilities: {probs}")

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: ينهار مع أي Logits كبيرة نسبيًا
def softmax_wrong(z):
    return np.exp(z) / np.sum(np.exp(z))

# ✅ صحيح: آمن دائمًا بغض النظر عن حجم القيم
def softmax_correct(z):
    z_shifted = z - np.max(z)
    return np.exp(z_shifted) / np.sum(np.exp(z_shifted))

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: إذا كانت إحدى الاحتمالات 0.0 بالضبط (نادر لكن ممكن)
loss = -np.log(0.0)   # ينتج -inf ثم يُدمّر كل التدريب لاحقًا

# ✅ صحيح: قيمة صغيرة جدًا تمنع الانهيار دون تأثير يُذكر على الدقة
loss = -np.log(np.clip(prob, 1e-12, 1.0))

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع جدًا للمبتدئين: افتراض أن الضرب يعمل "تلقائيًا" بغض النظر عن الشكل
X = torch.randn(32, 10)   # batch_size=32, features=10
W = torch.randn(10, 5)
W_wrong = torch.randn(5, 10)   # معكوسة بالخطأ!

Y = X @ W         # ✅ يعمل: (32,10)@(10,5) = (32,5)
Y2 = X @ W_wrong  # ❌ RuntimeError: shapes mismatch

# ────────────────────────────────────────────────────────────────────────
# [024] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-01-chapter-01.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
def manual_forward_and_loss(
    logits: np.ndarray,       # shape: (batch_size, num_classes)
    true_labels: np.ndarray,  # shape: (batch_size,) - أرقام الفئات الصحيحة
) -> tuple[np.ndarray, float]:
    """
    يجب أن تُرجع:
      1. مصفوفة الاحتمالات بعد Softmax (آمنة عدديًا!)
      2. متوسط Cross-Entropy Loss عبر كل أمثلة الـ Batch

    المتطلبات:
    - طبّق خدعة الاستقرار العددي (طرح max لكل صف على حدة، ليس للمصفوفة كاملة!)
    - تعامل مع batch كامل (أكثر من مثال واحد) وليس مثالًا واحدًا فقط
    - أضف eps صغيرة لتجنب log(0)
    """
    # اكتب الحل هنا
    ...


# اختبر حلك على هذه البيانات:
test_logits = np.array([
    [2.0, 0.5, -1.0],
    [0.1, 3.0, 0.2],
    [-1.0, -0.5, 2.5],
])
test_labels = np.array([0, 1, 2])

probs, loss = manual_forward_and_loss(test_logits, test_labels)
print("الاحتمالات:\n", probs)
print("متوسط الخسارة:", loss)

# تحقق من صحة حلك بمقارنته بـ PyTorch:
import torch
import torch.nn.functional as F
torch_loss = F.cross_entropy(
    torch.tensor(test_logits, dtype=torch.float32),
    torch.tensor(test_labels, dtype=torch.long)
)
print("خسارة PyTorch للمقارنة:", torch_loss.item())
assert abs(loss - torch_loss.item()) < 1e-5, "يوجد خطأ في حلك!"
print("✅ الحل صحيح تمامًا!")
