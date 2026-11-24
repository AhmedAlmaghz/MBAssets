#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الشبكة العصبية الأولى من الصفر — من نقطة واحدة إلى شبكة تتعلّم
Large Language Model Engineering — Chapter 02 — Foundations

Book slug : llm-engineering-69-chapters
Chapter   : ch-02-chapter-02 (lesson 2)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-02-chapter-02.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الشبكة العصبية الأولى من الصفر — من نقطة واحدة إلى شبكة تتعلّم
Large Language Model Engineering — Chapter 02 — Foundations

Book slug : llm-engineering-69-chapters
Chapter   : ch-02-chapter-02 (lesson 2)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-02-chapter-02.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الشبكة العصبية الأولى من الصفر — من نقطة واحدة إلى شبكة تتعلّم
Large Language Model Engineering — Chapter 02 — Foundations

Book slug : llm-engineering-69-chapters
Chapter   : ch-02-chapter-02 (lesson 2)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_02.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 02: الفصل الثاني: الشبكة العصبية الأولى من الصفر — من نقطة واحدة إلى شبكة تتعلّم."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_perceptron() -> None:
    data = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
    weights, bias = ([0.0, 0.0], 0.0)
    for epoch in range(12):
        errors = 0
        for inputs, target in data:
            prediction = int(sum((w * x for w, x in zip(weights, inputs))) + bias >= 0.5)
            error = target - prediction
            errors += error != 0
            weights = [w + 0.2 * error * x for w, x in zip(weights, inputs)]
            bias += 0.2 * error
        if errors == 0:
            break
    print('AND predictions:', [int(sum((w * x for w, x in zip(weights, x))) + bias >= 0.5) for x, _ in data])
    print('weights / bias:', [round(w, 3) for w in weights], round(bias, 3))

if __name__ == "__main__":
    print("Chapter 02 lab: A trainable binary neuron")
    run_perceptron()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-01-foundations/chapter-02.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def perceptron_predict(x: np.ndarray, w: np.ndarray, b: float) -> int:
    """Perceptron بسيط جدًا - دالة عتبة خام"""
    z = np.dot(x, w) + b
    return 1 if z > 0 else 0


# دعنا نختبره على مسألة AND المنطقية (سهلة الحل بخط واحد)
w = np.array([1.0, 1.0])
b = -1.5  # نحتاج كلا المدخلين = 1 ليتجاوز المجموع العتبة

test_cases = [(0, 0), (0, 1), (1, 0), (1, 1)]
for x1, x2 in test_cases:
    result = perceptron_predict(np.array([x1, x2]), w, b)
    print(f"AND({x1}, {x2}) = {result}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-01-foundations/chapter-02.md
# ────────────────────────────────────────────────────────────────────────
# جدول الحقيقة لـ XOR:
# (0,0) → 0    (0,1) → 1    (1,0) → 1    (1,1) → 0

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-01-foundations/chapter-02.md
# ────────────────────────────────────────────────────────────────────────
def perceptron_xor_attempt():
    """محاولة محكوم عليها بالفشل - جرّب أي أوزان تريدها!"""
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_true = np.array([0, 1, 1, 0])

    # جرّب أي قيم أوزان - لن تجد تركيبة تحل المشكلة بالكامل أبدًا
    w = np.array([1.0, 1.0])
    b = -0.5

    for x, y in zip(X, y_true):
        pred = perceptron_predict(x, w, b)
        status = "✅" if pred == y else "❌"
        print(f"XOR({x[0]}, {x[1]}) = {y}, التنبؤ = {pred} {status}")

perceptron_xor_attempt()

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-01-foundations/chapter-02.md
# ────────────────────────────────────────────────────────────────────────
"""
activation_functions.py
التطبيق الكامل لأهم 4 دوال تفعيل مع مشتقاتها
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z: np.ndarray) -> np.ndarray:
    """تُخرج قيمة بين (0, 1) - مفيدة للاحتمالات الثنائية"""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z: np.ndarray) -> np.ndarray:
    """تُخرج قيمة بين (-1, 1) - متمركزة حول الصفر، أفضل من Sigmoid غالبًا"""
    return np.tanh(z)


def tanh_derivative(z: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(z) ** 2


def relu(z: np.ndarray) -> np.ndarray:
    """تُخرج z نفسها إن كانت موجبة، وإلا صفر - بسيطة وسريعة جدًا"""
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(np.float32)


def gelu(z: np.ndarray) -> np.ndarray:
    """
    Gaussian Error Linear Unit - التقريب الشائع المستخدم في BERT/GPT
    GELU(z) ≈ 0.5 * z * (1 + tanh(√(2/π) * (z + 0.044715*z³)))
    """
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))


# رسم الدوال الأربع ومشتقاتها جنبًا إلى جنب
z = np.linspace(-5, 5, 500)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(z, sigmoid(z), label="Sigmoid", color="tab:blue")
axes[0, 0].plot(z, sigmoid_derivative(z), label="المشتقة", linestyle="--", color="tab:blue")
axes[0, 0].set_title("Sigmoid")
axes[0, 0].legend(); axes[0, 0].grid(True)

axes[0, 1].plot(z, tanh(z), label="Tanh", color="tab:orange")
axes[0, 1].plot(z, tanh_derivative(z), label="المشتقة", linestyle="--", color="tab:orange")
axes[0, 1].set_title("Tanh")
axes[0, 1].legend(); axes[0, 1].grid(True)

axes[1, 0].plot(z, relu(z), label="ReLU", color="tab:green")
axes[1, 0].plot(z, relu_derivative(z), label="المشتقة", linestyle="--", color="tab:green")
axes[1, 0].set_title("ReLU")
axes[1, 0].legend(); axes[1, 0].grid(True)

axes[1, 1].plot(z, gelu(z), label="GELU", color="tab:red")
axes[1, 1].set_title("GELU")
axes[1, 1].legend(); axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-01-foundations/chapter-02.md
# ────────────────────────────────────────────────────────────────────────
"""
mlp_from_scratch.py
شبكة عصبية كاملة (2 → 4 → 1) من الصفر بـ NumPy فقط
تحل مسألة XOR التي عجز عنها Perceptron البسيط
يتطلب: Python 3.13+, NumPy 2.x
"""
from __future__ import annotations

import numpy as np


class MLPFromScratch:
    """
    شبكة عصبية بطبقة خفية واحدة، مبنية يدويًا بالكامل
    البنية: Input(2) → Hidden(4, sigmoid) → Output(1, sigmoid)
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)  # الأسلوب الحديث الموصى به في NumPy 2.x

        # تهيئة الأوزان بقيم صغيرة عشوائية (Xavier-like initialization)
        self.W1 = rng.normal(0, np.sqrt(1 / input_dim), size=(input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = rng.normal(0, np.sqrt(1 / hidden_dim), size=(hidden_dim, output_dim))
        self.b2 = np.zeros((1, output_dim))

        # متغيرات نخزّنها أثناء Forward لاستخدامها لاحقًا في Backward
        self._cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        التمرير الأمامي: يحسب تنبؤ الشبكة خطوة بخطوة
        X shape: (batch_size, input_dim)
        """
        z1 = X @ self.W1 + self.b1          # (batch, hidden_dim)
        a1 = self._sigmoid(z1)               # تفعيل الطبقة الخفية

        z2 = a1 @ self.W2 + self.b2          # (batch, output_dim)
        a2 = self._sigmoid(z2)               # التنبؤ النهائي

        # نخزّن كل القيم الوسيطة - سنحتاجها بالضبط في Backward Pass
        self._cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return a2

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Binary Cross-Entropy Loss - مع eps لتجنب log(0)"""
        eps = 1e-12
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(
            y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        return float(loss)

    def backward(self, y_true: np.ndarray, learning_rate: float = 0.1) -> None:
        """
        التمرير الخلفي: تطبيق Chain Rule يدويًا لحساب كل التدرجات
        ثم تحديث الأوزان مباشرة (دمج Gradient Descent هنا للتبسيط)
        """
        m = y_true.shape[0]  # عدد الأمثلة في الدفعة (Batch Size)

        X, a1, a2 = self._cache["X"], self._cache["a1"], self._cache["a2"]

        # ----- الخطوة 1: تدرّج طبقة الإخراج -----
        # dL/dz2 = (a2 - y_true) — مشتقة مبسّطة ناتجة عن دمج Sigmoid + BCE معًا
        dz2 = a2 - y_true                              # (batch, output_dim)
        dW2 = (a1.T @ dz2) / m                          # (hidden_dim, output_dim)
        db2 = np.sum(dz2, axis=0, keepdims=True) / m    # (1, output_dim)

        # ----- الخطوة 2: نشر التدرّج عكسيًا للطبقة الخفية (Chain Rule!) -----
        da1 = dz2 @ self.W2.T                           # (batch, hidden_dim)
        dz1 = da1 * (a1 * (1 - a1))                      # ضرب بمشتقة Sigmoid للطبقة الخفية
        dW1 = (X.T @ dz1) / m                            # (input_dim, hidden_dim)
        db1 = np.sum(dz1, axis=0, keepdims=True) / m     # (1, hidden_dim)

        # ----- الخطوة 3: تحديث الأوزان (Gradient Descent) -----
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(
        self, X: np.ndarray, y: np.ndarray, epochs: int = 5000, learning_rate: float = 0.5
    ) -> list[float]:
        """حلقة التدريب الكاملة"""
        loss_history: list[float] = []

        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y_pred, y)
            self.backward(y, learning_rate)
            loss_history.append(loss)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

        return loss_history

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-01-foundations/chapter-02.md
# ────────────────────────────────────────────────────────────────────────
# بيانات XOR الكاملة
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

model = MLPFromScratch(input_dim=2, hidden_dim=4, output_dim=1, seed=42)
history = model.train(X, y, epochs=5000, learning_rate=0.5)

print("\n--- النتائج النهائية ---")
final_predictions = model.forward(X)
for i, (x, y_true, y_pred) in enumerate(zip(X, y, final_predictions)):
    print(f"XOR({int(x[0])}, {int(x[1])}) = {int(y_true[0])}, "
          f"التنبؤ = {y_pred[0]:.4f} → {'✅' if round(y_pred[0]) == y_true[0] else '❌'}")

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-01-foundations/chapter-02.md
# ────────────────────────────────────────────────────────────────────────
"""
mlp_pytorch.py
نفس الشبكة بالضبط، لكن بـ PyTorch 2.x
قارن حجم الكود مع النسخة اليدوية أعلاه!
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPTorch(nn.Module):
    """نفس البنية تمامًا: Input(2) → Hidden(4, sigmoid) → Output(1, sigmoid)"""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 4, output_dim: int = 1) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation1 = nn.Sigmoid()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.activation2 = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation1(self.layer1(x))
        x = self.activation2(self.layer2(x))
        return x


# البيانات
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

# إعداد النموذج والتدريب
torch.manual_seed(42)
model = MLPTorch()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(5000):
    optimizer.zero_grad()           # 🔑 لا تنسَ هذا أبدًا! (سنراها بالتفصيل في الفصل 3)
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()                  # Backpropagation تلقائيًا بالكامل - سطر واحد فقط!
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f}")

print("\n--- النتائج النهائية ---")
with torch.no_grad():
    predictions = model(X)
    for x, y_true, y_pred in zip(X, y, predictions):
        print(f"XOR({int(x[0])}, {int(x[1])}) = {int(y_true[0])}, "
              f"التنبؤ = {y_pred[0]:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-01-foundations/chapter-02.md
# ────────────────────────────────────────────────────────────────────────
# أوزان مُبسَّطة يدويًا (طبقة خفية بعصبونين فقط لتبسيط التتبع)
W1 = np.array([[2.0, -2.0], [2.0, -2.0]])   # (input_dim=2, hidden_dim=2)
b1 = np.array([[-1.0, 3.0]])
W2 = np.array([[2.0], [2.0]])                # (hidden_dim=2, output_dim=1)
b2 = np.array([[-3.0]])

x = np.array([[1.0, 0.0]])  # المدخل: XOR(1, 0) يجب أن يُعطي 1

# ===== Forward Pass يدويًا =====
z1 = x @ W1 + b1
print(f"z1 = {z1}")   # [[1.0, -1.0]] = (1*2+0*2-1, 1*(-2)+0*(-2)+3)

a1 = 1 / (1 + np.exp(-z1))
print(f"a1 (بعد Sigmoid) = {a1}")   # [[0.731, 0.269]]

z2 = a1 @ W2 + b2
print(f"z2 = {z2}")   # 0.731*2 + 0.269*2 - 3 = -1.0

a2 = 1 / (1 + np.exp(-z2))
print(f"a2 (التنبؤ النهائي) = {a2}")   # ≈ 0.269 (ليس مثاليًا بهذه الأوزان العشوائية البسيطة)

# ────────────────────────────────────────────────────────────────────────
# [010] book/chapters/part-01-foundations/chapter-02.md
# ────────────────────────────────────────────────────────────────────────
   model = nn.Sequential(
       nn.Linear(2, 4), nn.Sigmoid(),
       nn.Linear(4, 1), nn.Sigmoid()
   )
   ```

---

## ⚠️ 2.5 أخطاء شائعة ومصائد (احذر منها!)

### ❌ الخطأ #1: نسيان دالة التفعيل غير الخطية تمامًا

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-02-chapter-02.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 02: الفصل الثاني: الشبكة العصبية الأولى من الصفر — من نقطة واحدة إلى شبكة تتعلّم."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_perceptron() -> None:
    data = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
    weights, bias = ([0.0, 0.0], 0.0)
    for epoch in range(12):
        errors = 0
        for inputs, target in data:
            prediction = int(sum((w * x for w, x in zip(weights, inputs))) + bias >= 0.5)
            error = target - prediction
            errors += error != 0
            weights = [w + 0.2 * error * x for w, x in zip(weights, inputs)]
            bias += 0.2 * error
        if errors == 0:
            break
    print('AND predictions:', [int(sum((w * x for w, x in zip(weights, x))) + bias >= 0.5) for x, _ in data])
    print('weights / bias:', [round(w, 3) for w in weights], round(bias, 3))

if __name__ == "__main__":
    print("Chapter 02 lab: A trainable binary neuron")
    run_perceptron()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-02-chapter-02.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def perceptron_predict(x: np.ndarray, w: np.ndarray, b: float) -> int:
    """Perceptron بسيط جدًا - دالة عتبة خام"""
    z = np.dot(x, w) + b
    return 1 if z > 0 else 0


# دعنا نختبره على مسألة AND المنطقية (سهلة الحل بخط واحد)
w = np.array([1.0, 1.0])
b = -1.5  # نحتاج كلا المدخلين = 1 ليتجاوز المجموع العتبة

test_cases = [(0, 0), (0, 1), (1, 0), (1, 1)]
for x1, x2 in test_cases:
    result = perceptron_predict(np.array([x1, x2]), w, b)
    print(f"AND({x1}, {x2}) = {result}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-02-chapter-02.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# جدول الحقيقة لـ XOR:
# (0,0) → 0    (0,1) → 1    (1,0) → 1    (1,1) → 0

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-02-chapter-02.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def perceptron_xor_attempt():
    """محاولة محكوم عليها بالفشل - جرّب أي أوزان تريدها!"""
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_true = np.array([0, 1, 1, 0])

    # جرّب أي قيم أوزان - لن تجد تركيبة تحل المشكلة بالكامل أبدًا
    w = np.array([1.0, 1.0])
    b = -0.5

    for x, y in zip(X, y_true):
        pred = perceptron_predict(x, w, b)
        status = "✅" if pred == y else "❌"
        print(f"XOR({x[0]}, {x[1]}) = {y}, التنبؤ = {pred} {status}")

perceptron_xor_attempt()

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-02-chapter-02.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
"""
activation_functions.py
التطبيق الكامل لأهم 4 دوال تفعيل مع مشتقاتها
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z: np.ndarray) -> np.ndarray:
    """تُخرج قيمة بين (0, 1) - مفيدة للاحتمالات الثنائية"""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z: np.ndarray) -> np.ndarray:
    """تُخرج قيمة بين (-1, 1) - متمركزة حول الصفر، أفضل من Sigmoid غالبًا"""
    return np.tanh(z)


def tanh_derivative(z: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(z) ** 2


def relu(z: np.ndarray) -> np.ndarray:
    """تُخرج z نفسها إن كانت موجبة، وإلا صفر - بسيطة وسريعة جدًا"""
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(np.float32)


def gelu(z: np.ndarray) -> np.ndarray:
    """
    Gaussian Error Linear Unit - التقريب الشائع المستخدم في BERT/GPT
    GELU(z) ≈ 0.5 * z * (1 + tanh(√(2/π) * (z + 0.044715*z³)))
    """
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))


# رسم الدوال الأربع ومشتقاتها جنبًا إلى جنب
z = np.linspace(-5, 5, 500)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(z, sigmoid(z), label="Sigmoid", color="tab:blue")
axes[0, 0].plot(z, sigmoid_derivative(z), label="المشتقة", linestyle="--", color="tab:blue")
axes[0, 0].set_title("Sigmoid")
axes[0, 0].legend(); axes[0, 0].grid(True)

axes[0, 1].plot(z, tanh(z), label="Tanh", color="tab:orange")
axes[0, 1].plot(z, tanh_derivative(z), label="المشتقة", linestyle="--", color="tab:orange")
axes[0, 1].set_title("Tanh")
axes[0, 1].legend(); axes[0, 1].grid(True)

axes[1, 0].plot(z, relu(z), label="ReLU", color="tab:green")
axes[1, 0].plot(z, relu_derivative(z), label="المشتقة", linestyle="--", color="tab:green")
axes[1, 0].set_title("ReLU")
axes[1, 0].legend(); axes[1, 0].grid(True)

axes[1, 1].plot(z, gelu(z), label="GELU", color="tab:red")
axes[1, 1].set_title("GELU")
axes[1, 1].legend(); axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-02-chapter-02.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
"""
mlp_from_scratch.py
شبكة عصبية كاملة (2 → 4 → 1) من الصفر بـ NumPy فقط
تحل مسألة XOR التي عجز عنها Perceptron البسيط
يتطلب: Python 3.13+, NumPy 2.x
"""
from __future__ import annotations

import numpy as np


class MLPFromScratch:
    """
    شبكة عصبية بطبقة خفية واحدة، مبنية يدويًا بالكامل
    البنية: Input(2) → Hidden(4, sigmoid) → Output(1, sigmoid)
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)  # الأسلوب الحديث الموصى به في NumPy 2.x

        # تهيئة الأوزان بقيم صغيرة عشوائية (Xavier-like initialization)
        self.W1 = rng.normal(0, np.sqrt(1 / input_dim), size=(input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = rng.normal(0, np.sqrt(1 / hidden_dim), size=(hidden_dim, output_dim))
        self.b2 = np.zeros((1, output_dim))

        # متغيرات نخزّنها أثناء Forward لاستخدامها لاحقًا في Backward
        self._cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        التمرير الأمامي: يحسب تنبؤ الشبكة خطوة بخطوة
        X shape: (batch_size, input_dim)
        """
        z1 = X @ self.W1 + self.b1          # (batch, hidden_dim)
        a1 = self._sigmoid(z1)               # تفعيل الطبقة الخفية

        z2 = a1 @ self.W2 + self.b2          # (batch, output_dim)
        a2 = self._sigmoid(z2)               # التنبؤ النهائي

        # نخزّن كل القيم الوسيطة - سنحتاجها بالضبط في Backward Pass
        self._cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return a2

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Binary Cross-Entropy Loss - مع eps لتجنب log(0)"""
        eps = 1e-12
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(
            y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        return float(loss)

    def backward(self, y_true: np.ndarray, learning_rate: float = 0.1) -> None:
        """
        التمرير الخلفي: تطبيق Chain Rule يدويًا لحساب كل التدرجات
        ثم تحديث الأوزان مباشرة (دمج Gradient Descent هنا للتبسيط)
        """
        m = y_true.shape[0]  # عدد الأمثلة في الدفعة (Batch Size)

        X, a1, a2 = self._cache["X"], self._cache["a1"], self._cache["a2"]

        # ----- الخطوة 1: تدرّج طبقة الإخراج -----
        # dL/dz2 = (a2 - y_true) — مشتقة مبسّطة ناتجة عن دمج Sigmoid + BCE معًا
        dz2 = a2 - y_true                              # (batch, output_dim)
        dW2 = (a1.T @ dz2) / m                          # (hidden_dim, output_dim)
        db2 = np.sum(dz2, axis=0, keepdims=True) / m    # (1, output_dim)

        # ----- الخطوة 2: نشر التدرّج عكسيًا للطبقة الخفية (Chain Rule!) -----
        da1 = dz2 @ self.W2.T                           # (batch, hidden_dim)
        dz1 = da1 * (a1 * (1 - a1))                      # ضرب بمشتقة Sigmoid للطبقة الخفية
        dW1 = (X.T @ dz1) / m                            # (input_dim, hidden_dim)
        db1 = np.sum(dz1, axis=0, keepdims=True) / m     # (1, hidden_dim)

        # ----- الخطوة 3: تحديث الأوزان (Gradient Descent) -----
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(
        self, X: np.ndarray, y: np.ndarray, epochs: int = 5000, learning_rate: float = 0.5
    ) -> list[float]:
        """حلقة التدريب الكاملة"""
        loss_history: list[float] = []

        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y_pred, y)
            self.backward(y, learning_rate)
            loss_history.append(loss)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

        return loss_history

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-02-chapter-02.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# بيانات XOR الكاملة
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

model = MLPFromScratch(input_dim=2, hidden_dim=4, output_dim=1, seed=42)
history = model.train(X, y, epochs=5000, learning_rate=0.5)

print("\n--- النتائج النهائية ---")
final_predictions = model.forward(X)
for i, (x, y_true, y_pred) in enumerate(zip(X, y, final_predictions)):
    print(f"XOR({int(x[0])}, {int(x[1])}) = {int(y_true[0])}, "
          f"التنبؤ = {y_pred[0]:.4f} → {'✅' if round(y_pred[0]) == y_true[0] else '❌'}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-02-chapter-02.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
"""
mlp_pytorch.py
نفس الشبكة بالضبط، لكن بـ PyTorch 2.x
قارن حجم الكود مع النسخة اليدوية أعلاه!
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPTorch(nn.Module):
    """نفس البنية تمامًا: Input(2) → Hidden(4, sigmoid) → Output(1, sigmoid)"""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 4, output_dim: int = 1) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation1 = nn.Sigmoid()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.activation2 = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation1(self.layer1(x))
        x = self.activation2(self.layer2(x))
        return x


# البيانات
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

# إعداد النموذج والتدريب
torch.manual_seed(42)
model = MLPTorch()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(5000):
    optimizer.zero_grad()           # 🔑 لا تنسَ هذا أبدًا! (سنراها بالتفصيل في الفصل 3)
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()                  # Backpropagation تلقائيًا بالكامل - سطر واحد فقط!
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f}")

print("\n--- النتائج النهائية ---")
with torch.no_grad():
    predictions = model(X)
    for x, y_true, y_pred in zip(X, y, predictions):
        print(f"XOR({int(x[0])}, {int(x[1])}) = {int(y_true[0])}, "
              f"التنبؤ = {y_pred[0]:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-02-chapter-02.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# أوزان مُبسَّطة يدويًا (طبقة خفية بعصبونين فقط لتبسيط التتبع)
W1 = np.array([[2.0, -2.0], [2.0, -2.0]])   # (input_dim=2, hidden_dim=2)
b1 = np.array([[-1.0, 3.0]])
W2 = np.array([[2.0], [2.0]])                # (hidden_dim=2, output_dim=1)
b2 = np.array([[-3.0]])

x = np.array([[1.0, 0.0]])  # المدخل: XOR(1, 0) يجب أن يُعطي 1

# ===== Forward Pass يدويًا =====
z1 = x @ W1 + b1
print(f"z1 = {z1}")   # [[1.0, -1.0]] = (1*2+0*2-1, 1*(-2)+0*(-2)+3)

a1 = 1 / (1 + np.exp(-z1))
print(f"a1 (بعد Sigmoid) = {a1}")   # [[0.731, 0.269]]

z2 = a1 @ W2 + b2
print(f"z2 = {z2}")   # 0.731*2 + 0.269*2 - 3 = -1.0

a2 = 1 / (1 + np.exp(-z2))
print(f"a2 (التنبؤ النهائي) = {a2}")   # ≈ 0.269 (ليس مثاليًا بهذه الأوزان العشوائية البسيطة)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-02-chapter-02.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
   model = nn.Sequential(
       nn.Linear(2, 4), nn.Sigmoid(),
       nn.Linear(4, 1), nn.Sigmoid()
   )
   ```

---

## ⚠️ 2.5 أخطاء شائعة ومصائد (احذر منها!)

### ❌ الخطأ #1: نسيان دالة التفعيل غير الخطية تمامًا

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-02-chapter-02.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الشبكة العصبية الأولى من الصفر — من نقطة واحدة إلى شبكة تتعلّم
Large Language Model Engineering — Chapter 02 — Foundations

Book slug : llm-engineering-69-chapters
Chapter   : ch-02-chapter-02 (lesson 2)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_02.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 02: الفصل الثاني: الشبكة العصبية الأولى من الصفر — من نقطة واحدة إلى شبكة تتعلّم."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_perceptron() -> None:
    data = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
    weights, bias = ([0.0, 0.0], 0.0)
    for epoch in range(12):
        errors = 0
        for inputs, target in data:
            prediction = int(sum((w * x for w, x in zip(weights, inputs))) + bias >= 0.5)
            error = target - prediction
            errors += error != 0
            weights = [w + 0.2 * error * x for w, x in zip(weights, inputs)]
            bias += 0.2 * error
        if errors == 0:
            break
    print('AND predictions:', [int(sum((w * x for w, x in zip(weights, x))) + bias >= 0.5) for x, _ in data])
    print('weights / bias:', [round(w, 3) for w in weights], round(bias, 3))

if __name__ == "__main__":
    print("Chapter 02 lab: A trainable binary neuron")
    run_perceptron()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-01-foundations/chapter-02.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def perceptron_predict(x: np.ndarray, w: np.ndarray, b: float) -> int:
    """Perceptron بسيط جدًا - دالة عتبة خام"""
    z = np.dot(x, w) + b
    return 1 if z > 0 else 0


# دعنا نختبره على مسألة AND المنطقية (سهلة الحل بخط واحد)
w = np.array([1.0, 1.0])
b = -1.5  # نحتاج كلا المدخلين = 1 ليتجاوز المجموع العتبة

test_cases = [(0, 0), (0, 1), (1, 0), (1, 1)]
for x1, x2 in test_cases:
    result = perceptron_predict(np.array([x1, x2]), w, b)
    print(f"AND({x1}, {x2}) = {result}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-01-foundations/chapter-02.md
# ────────────────────────────────────────────────────────────────────────
# جدول الحقيقة لـ XOR:
# (0,0) → 0    (0,1) → 1    (1,0) → 1    (1,1) → 0

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-01-foundations/chapter-02.md
# ────────────────────────────────────────────────────────────────────────
def perceptron_xor_attempt():
    """محاولة محكوم عليها بالفشل - جرّب أي أوزان تريدها!"""
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_true = np.array([0, 1, 1, 0])

    # جرّب أي قيم أوزان - لن تجد تركيبة تحل المشكلة بالكامل أبدًا
    w = np.array([1.0, 1.0])
    b = -0.5

    for x, y in zip(X, y_true):
        pred = perceptron_predict(x, w, b)
        status = "✅" if pred == y else "❌"
        print(f"XOR({x[0]}, {x[1]}) = {y}, التنبؤ = {pred} {status}")

perceptron_xor_attempt()

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-01-foundations/chapter-02.md
# ────────────────────────────────────────────────────────────────────────
"""
activation_functions.py
التطبيق الكامل لأهم 4 دوال تفعيل مع مشتقاتها
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z: np.ndarray) -> np.ndarray:
    """تُخرج قيمة بين (0, 1) - مفيدة للاحتمالات الثنائية"""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z: np.ndarray) -> np.ndarray:
    """تُخرج قيمة بين (-1, 1) - متمركزة حول الصفر، أفضل من Sigmoid غالبًا"""
    return np.tanh(z)


def tanh_derivative(z: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(z) ** 2


def relu(z: np.ndarray) -> np.ndarray:
    """تُخرج z نفسها إن كانت موجبة، وإلا صفر - بسيطة وسريعة جدًا"""
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(np.float32)


def gelu(z: np.ndarray) -> np.ndarray:
    """
    Gaussian Error Linear Unit - التقريب الشائع المستخدم في BERT/GPT
    GELU(z) ≈ 0.5 * z * (1 + tanh(√(2/π) * (z + 0.044715*z³)))
    """
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))


# رسم الدوال الأربع ومشتقاتها جنبًا إلى جنب
z = np.linspace(-5, 5, 500)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(z, sigmoid(z), label="Sigmoid", color="tab:blue")
axes[0, 0].plot(z, sigmoid_derivative(z), label="المشتقة", linestyle="--", color="tab:blue")
axes[0, 0].set_title("Sigmoid")
axes[0, 0].legend(); axes[0, 0].grid(True)

axes[0, 1].plot(z, tanh(z), label="Tanh", color="tab:orange")
axes[0, 1].plot(z, tanh_derivative(z), label="المشتقة", linestyle="--", color="tab:orange")
axes[0, 1].set_title("Tanh")
axes[0, 1].legend(); axes[0, 1].grid(True)

axes[1, 0].plot(z, relu(z), label="ReLU", color="tab:green")
axes[1, 0].plot(z, relu_derivative(z), label="المشتقة", linestyle="--", color="tab:green")
axes[1, 0].set_title("ReLU")
axes[1, 0].legend(); axes[1, 0].grid(True)

axes[1, 1].plot(z, gelu(z), label="GELU", color="tab:red")
axes[1, 1].set_title("GELU")
axes[1, 1].legend(); axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-01-foundations/chapter-02.md
# ────────────────────────────────────────────────────────────────────────
"""
mlp_from_scratch.py
شبكة عصبية كاملة (2 → 4 → 1) من الصفر بـ NumPy فقط
تحل مسألة XOR التي عجز عنها Perceptron البسيط
يتطلب: Python 3.13+, NumPy 2.x
"""
from __future__ import annotations

import numpy as np


class MLPFromScratch:
    """
    شبكة عصبية بطبقة خفية واحدة، مبنية يدويًا بالكامل
    البنية: Input(2) → Hidden(4, sigmoid) → Output(1, sigmoid)
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)  # الأسلوب الحديث الموصى به في NumPy 2.x

        # تهيئة الأوزان بقيم صغيرة عشوائية (Xavier-like initialization)
        self.W1 = rng.normal(0, np.sqrt(1 / input_dim), size=(input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = rng.normal(0, np.sqrt(1 / hidden_dim), size=(hidden_dim, output_dim))
        self.b2 = np.zeros((1, output_dim))

        # متغيرات نخزّنها أثناء Forward لاستخدامها لاحقًا في Backward
        self._cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        التمرير الأمامي: يحسب تنبؤ الشبكة خطوة بخطوة
        X shape: (batch_size, input_dim)
        """
        z1 = X @ self.W1 + self.b1          # (batch, hidden_dim)
        a1 = self._sigmoid(z1)               # تفعيل الطبقة الخفية

        z2 = a1 @ self.W2 + self.b2          # (batch, output_dim)
        a2 = self._sigmoid(z2)               # التنبؤ النهائي

        # نخزّن كل القيم الوسيطة - سنحتاجها بالضبط في Backward Pass
        self._cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return a2

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Binary Cross-Entropy Loss - مع eps لتجنب log(0)"""
        eps = 1e-12
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(
            y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        return float(loss)

    def backward(self, y_true: np.ndarray, learning_rate: float = 0.1) -> None:
        """
        التمرير الخلفي: تطبيق Chain Rule يدويًا لحساب كل التدرجات
        ثم تحديث الأوزان مباشرة (دمج Gradient Descent هنا للتبسيط)
        """
        m = y_true.shape[0]  # عدد الأمثلة في الدفعة (Batch Size)

        X, a1, a2 = self._cache["X"], self._cache["a1"], self._cache["a2"]

        # ----- الخطوة 1: تدرّج طبقة الإخراج -----
        # dL/dz2 = (a2 - y_true) — مشتقة مبسّطة ناتجة عن دمج Sigmoid + BCE معًا
        dz2 = a2 - y_true                              # (batch, output_dim)
        dW2 = (a1.T @ dz2) / m                          # (hidden_dim, output_dim)
        db2 = np.sum(dz2, axis=0, keepdims=True) / m    # (1, output_dim)

        # ----- الخطوة 2: نشر التدرّج عكسيًا للطبقة الخفية (Chain Rule!) -----
        da1 = dz2 @ self.W2.T                           # (batch, hidden_dim)
        dz1 = da1 * (a1 * (1 - a1))                      # ضرب بمشتقة Sigmoid للطبقة الخفية
        dW1 = (X.T @ dz1) / m                            # (input_dim, hidden_dim)
        db1 = np.sum(dz1, axis=0, keepdims=True) / m     # (1, hidden_dim)

        # ----- الخطوة 3: تحديث الأوزان (Gradient Descent) -----
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(
        self, X: np.ndarray, y: np.ndarray, epochs: int = 5000, learning_rate: float = 0.5
    ) -> list[float]:
        """حلقة التدريب الكاملة"""
        loss_history: list[float] = []

        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y_pred, y)
            self.backward(y, learning_rate)
            loss_history.append(loss)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

        return loss_history

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-01-foundations/chapter-02.md
# ────────────────────────────────────────────────────────────────────────
# بيانات XOR الكاملة
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

model = MLPFromScratch(input_dim=2, hidden_dim=4, output_dim=1, seed=42)
history = model.train(X, y, epochs=5000, learning_rate=0.5)

print("\n--- النتائج النهائية ---")
final_predictions = model.forward(X)
for i, (x, y_true, y_pred) in enumerate(zip(X, y, final_predictions)):
    print(f"XOR({int(x[0])}, {int(x[1])}) = {int(y_true[0])}, "
          f"التنبؤ = {y_pred[0]:.4f} → {'✅' if round(y_pred[0]) == y_true[0] else '❌'}")

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-01-foundations/chapter-02.md
# ────────────────────────────────────────────────────────────────────────
"""
mlp_pytorch.py
نفس الشبكة بالضبط، لكن بـ PyTorch 2.x
قارن حجم الكود مع النسخة اليدوية أعلاه!
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPTorch(nn.Module):
    """نفس البنية تمامًا: Input(2) → Hidden(4, sigmoid) → Output(1, sigmoid)"""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 4, output_dim: int = 1) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation1 = nn.Sigmoid()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.activation2 = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation1(self.layer1(x))
        x = self.activation2(self.layer2(x))
        return x


# البيانات
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

# إعداد النموذج والتدريب
torch.manual_seed(42)
model = MLPTorch()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(5000):
    optimizer.zero_grad()           # 🔑 لا تنسَ هذا أبدًا! (سنراها بالتفصيل في الفصل 3)
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()                  # Backpropagation تلقائيًا بالكامل - سطر واحد فقط!
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f}")

print("\n--- النتائج النهائية ---")
with torch.no_grad():
    predictions = model(X)
    for x, y_true, y_pred in zip(X, y, predictions):
        print(f"XOR({int(x[0])}, {int(x[1])}) = {int(y_true[0])}, "
              f"التنبؤ = {y_pred[0]:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-01-foundations/chapter-02.md
# ────────────────────────────────────────────────────────────────────────
# أوزان مُبسَّطة يدويًا (طبقة خفية بعصبونين فقط لتبسيط التتبع)
W1 = np.array([[2.0, -2.0], [2.0, -2.0]])   # (input_dim=2, hidden_dim=2)
b1 = np.array([[-1.0, 3.0]])
W2 = np.array([[2.0], [2.0]])                # (hidden_dim=2, output_dim=1)
b2 = np.array([[-3.0]])

x = np.array([[1.0, 0.0]])  # المدخل: XOR(1, 0) يجب أن يُعطي 1

# ===== Forward Pass يدويًا =====
z1 = x @ W1 + b1
print(f"z1 = {z1}")   # [[1.0, -1.0]] = (1*2+0*2-1, 1*(-2)+0*(-2)+3)

a1 = 1 / (1 + np.exp(-z1))
print(f"a1 (بعد Sigmoid) = {a1}")   # [[0.731, 0.269]]

z2 = a1 @ W2 + b2
print(f"z2 = {z2}")   # 0.731*2 + 0.269*2 - 3 = -1.0

a2 = 1 / (1 + np.exp(-z2))
print(f"a2 (التنبؤ النهائي) = {a2}")   # ≈ 0.269 (ليس مثاليًا بهذه الأوزان العشوائية البسيطة)

# ────────────────────────────────────────────────────────────────────────
# [010] book/chapters/part-01-foundations/chapter-02.md
# ────────────────────────────────────────────────────────────────────────
   model = nn.Sequential(
       nn.Linear(2, 4), nn.Sigmoid(),
       nn.Linear(4, 1), nn.Sigmoid()
   )
   ```

---

## ⚠️ 2.5 أخطاء شائعة ومصائد (احذر منها!)

### ❌ الخطأ #1: نسيان دالة التفعيل غير الخطية تمامًا

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-02-chapter-02.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 02: الفصل الثاني: الشبكة العصبية الأولى من الصفر — من نقطة واحدة إلى شبكة تتعلّم."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_perceptron() -> None:
    data = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
    weights, bias = ([0.0, 0.0], 0.0)
    for epoch in range(12):
        errors = 0
        for inputs, target in data:
            prediction = int(sum((w * x for w, x in zip(weights, inputs))) + bias >= 0.5)
            error = target - prediction
            errors += error != 0
            weights = [w + 0.2 * error * x for w, x in zip(weights, inputs)]
            bias += 0.2 * error
        if errors == 0:
            break
    print('AND predictions:', [int(sum((w * x for w, x in zip(weights, x))) + bias >= 0.5) for x, _ in data])
    print('weights / bias:', [round(w, 3) for w in weights], round(bias, 3))

if __name__ == "__main__":
    print("Chapter 02 lab: A trainable binary neuron")
    run_perceptron()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-02-chapter-02.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def perceptron_predict(x: np.ndarray, w: np.ndarray, b: float) -> int:
    """Perceptron بسيط جدًا - دالة عتبة خام"""
    z = np.dot(x, w) + b
    return 1 if z > 0 else 0


# دعنا نختبره على مسألة AND المنطقية (سهلة الحل بخط واحد)
w = np.array([1.0, 1.0])
b = -1.5  # نحتاج كلا المدخلين = 1 ليتجاوز المجموع العتبة

test_cases = [(0, 0), (0, 1), (1, 0), (1, 1)]
for x1, x2 in test_cases:
    result = perceptron_predict(np.array([x1, x2]), w, b)
    print(f"AND({x1}, {x2}) = {result}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-02-chapter-02.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# جدول الحقيقة لـ XOR:
# (0,0) → 0    (0,1) → 1    (1,0) → 1    (1,1) → 0

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-02-chapter-02.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def perceptron_xor_attempt():
    """محاولة محكوم عليها بالفشل - جرّب أي أوزان تريدها!"""
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_true = np.array([0, 1, 1, 0])

    # جرّب أي قيم أوزان - لن تجد تركيبة تحل المشكلة بالكامل أبدًا
    w = np.array([1.0, 1.0])
    b = -0.5

    for x, y in zip(X, y_true):
        pred = perceptron_predict(x, w, b)
        status = "✅" if pred == y else "❌"
        print(f"XOR({x[0]}, {x[1]}) = {y}, التنبؤ = {pred} {status}")

perceptron_xor_attempt()

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-02-chapter-02.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
"""
activation_functions.py
التطبيق الكامل لأهم 4 دوال تفعيل مع مشتقاتها
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z: np.ndarray) -> np.ndarray:
    """تُخرج قيمة بين (0, 1) - مفيدة للاحتمالات الثنائية"""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z: np.ndarray) -> np.ndarray:
    """تُخرج قيمة بين (-1, 1) - متمركزة حول الصفر، أفضل من Sigmoid غالبًا"""
    return np.tanh(z)


def tanh_derivative(z: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(z) ** 2


def relu(z: np.ndarray) -> np.ndarray:
    """تُخرج z نفسها إن كانت موجبة، وإلا صفر - بسيطة وسريعة جدًا"""
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(np.float32)


def gelu(z: np.ndarray) -> np.ndarray:
    """
    Gaussian Error Linear Unit - التقريب الشائع المستخدم في BERT/GPT
    GELU(z) ≈ 0.5 * z * (1 + tanh(√(2/π) * (z + 0.044715*z³)))
    """
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))


# رسم الدوال الأربع ومشتقاتها جنبًا إلى جنب
z = np.linspace(-5, 5, 500)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(z, sigmoid(z), label="Sigmoid", color="tab:blue")
axes[0, 0].plot(z, sigmoid_derivative(z), label="المشتقة", linestyle="--", color="tab:blue")
axes[0, 0].set_title("Sigmoid")
axes[0, 0].legend(); axes[0, 0].grid(True)

axes[0, 1].plot(z, tanh(z), label="Tanh", color="tab:orange")
axes[0, 1].plot(z, tanh_derivative(z), label="المشتقة", linestyle="--", color="tab:orange")
axes[0, 1].set_title("Tanh")
axes[0, 1].legend(); axes[0, 1].grid(True)

axes[1, 0].plot(z, relu(z), label="ReLU", color="tab:green")
axes[1, 0].plot(z, relu_derivative(z), label="المشتقة", linestyle="--", color="tab:green")
axes[1, 0].set_title("ReLU")
axes[1, 0].legend(); axes[1, 0].grid(True)

axes[1, 1].plot(z, gelu(z), label="GELU", color="tab:red")
axes[1, 1].set_title("GELU")
axes[1, 1].legend(); axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-02-chapter-02.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
"""
mlp_from_scratch.py
شبكة عصبية كاملة (2 → 4 → 1) من الصفر بـ NumPy فقط
تحل مسألة XOR التي عجز عنها Perceptron البسيط
يتطلب: Python 3.13+, NumPy 2.x
"""
from __future__ import annotations

import numpy as np


class MLPFromScratch:
    """
    شبكة عصبية بطبقة خفية واحدة، مبنية يدويًا بالكامل
    البنية: Input(2) → Hidden(4, sigmoid) → Output(1, sigmoid)
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)  # الأسلوب الحديث الموصى به في NumPy 2.x

        # تهيئة الأوزان بقيم صغيرة عشوائية (Xavier-like initialization)
        self.W1 = rng.normal(0, np.sqrt(1 / input_dim), size=(input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = rng.normal(0, np.sqrt(1 / hidden_dim), size=(hidden_dim, output_dim))
        self.b2 = np.zeros((1, output_dim))

        # متغيرات نخزّنها أثناء Forward لاستخدامها لاحقًا في Backward
        self._cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        التمرير الأمامي: يحسب تنبؤ الشبكة خطوة بخطوة
        X shape: (batch_size, input_dim)
        """
        z1 = X @ self.W1 + self.b1          # (batch, hidden_dim)
        a1 = self._sigmoid(z1)               # تفعيل الطبقة الخفية

        z2 = a1 @ self.W2 + self.b2          # (batch, output_dim)
        a2 = self._sigmoid(z2)               # التنبؤ النهائي

        # نخزّن كل القيم الوسيطة - سنحتاجها بالضبط في Backward Pass
        self._cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return a2

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Binary Cross-Entropy Loss - مع eps لتجنب log(0)"""
        eps = 1e-12
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(
            y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        return float(loss)

    def backward(self, y_true: np.ndarray, learning_rate: float = 0.1) -> None:
        """
        التمرير الخلفي: تطبيق Chain Rule يدويًا لحساب كل التدرجات
        ثم تحديث الأوزان مباشرة (دمج Gradient Descent هنا للتبسيط)
        """
        m = y_true.shape[0]  # عدد الأمثلة في الدفعة (Batch Size)

        X, a1, a2 = self._cache["X"], self._cache["a1"], self._cache["a2"]

        # ----- الخطوة 1: تدرّج طبقة الإخراج -----
        # dL/dz2 = (a2 - y_true) — مشتقة مبسّطة ناتجة عن دمج Sigmoid + BCE معًا
        dz2 = a2 - y_true                              # (batch, output_dim)
        dW2 = (a1.T @ dz2) / m                          # (hidden_dim, output_dim)
        db2 = np.sum(dz2, axis=0, keepdims=True) / m    # (1, output_dim)

        # ----- الخطوة 2: نشر التدرّج عكسيًا للطبقة الخفية (Chain Rule!) -----
        da1 = dz2 @ self.W2.T                           # (batch, hidden_dim)
        dz1 = da1 * (a1 * (1 - a1))                      # ضرب بمشتقة Sigmoid للطبقة الخفية
        dW1 = (X.T @ dz1) / m                            # (input_dim, hidden_dim)
        db1 = np.sum(dz1, axis=0, keepdims=True) / m     # (1, hidden_dim)

        # ----- الخطوة 3: تحديث الأوزان (Gradient Descent) -----
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(
        self, X: np.ndarray, y: np.ndarray, epochs: int = 5000, learning_rate: float = 0.5
    ) -> list[float]:
        """حلقة التدريب الكاملة"""
        loss_history: list[float] = []

        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y_pred, y)
            self.backward(y, learning_rate)
            loss_history.append(loss)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

        return loss_history

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-02-chapter-02.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# بيانات XOR الكاملة
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

model = MLPFromScratch(input_dim=2, hidden_dim=4, output_dim=1, seed=42)
history = model.train(X, y, epochs=5000, learning_rate=0.5)

print("\n--- النتائج النهائية ---")
final_predictions = model.forward(X)
for i, (x, y_true, y_pred) in enumerate(zip(X, y, final_predictions)):
    print(f"XOR({int(x[0])}, {int(x[1])}) = {int(y_true[0])}, "
          f"التنبؤ = {y_pred[0]:.4f} → {'✅' if round(y_pred[0]) == y_true[0] else '❌'}")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-02-chapter-02.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
"""
mlp_pytorch.py
نفس الشبكة بالضبط، لكن بـ PyTorch 2.x
قارن حجم الكود مع النسخة اليدوية أعلاه!
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPTorch(nn.Module):
    """نفس البنية تمامًا: Input(2) → Hidden(4, sigmoid) → Output(1, sigmoid)"""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 4, output_dim: int = 1) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation1 = nn.Sigmoid()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.activation2 = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation1(self.layer1(x))
        x = self.activation2(self.layer2(x))
        return x


# البيانات
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

# إعداد النموذج والتدريب
torch.manual_seed(42)
model = MLPTorch()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(5000):
    optimizer.zero_grad()           # 🔑 لا تنسَ هذا أبدًا! (سنراها بالتفصيل في الفصل 3)
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()                  # Backpropagation تلقائيًا بالكامل - سطر واحد فقط!
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f}")

print("\n--- النتائج النهائية ---")
with torch.no_grad():
    predictions = model(X)
    for x, y_true, y_pred in zip(X, y, predictions):
        print(f"XOR({int(x[0])}, {int(x[1])}) = {int(y_true[0])}, "
              f"التنبؤ = {y_pred[0]:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-02-chapter-02.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# أوزان مُبسَّطة يدويًا (طبقة خفية بعصبونين فقط لتبسيط التتبع)
W1 = np.array([[2.0, -2.0], [2.0, -2.0]])   # (input_dim=2, hidden_dim=2)
b1 = np.array([[-1.0, 3.0]])
W2 = np.array([[2.0], [2.0]])                # (hidden_dim=2, output_dim=1)
b2 = np.array([[-3.0]])

x = np.array([[1.0, 0.0]])  # المدخل: XOR(1, 0) يجب أن يُعطي 1

# ===== Forward Pass يدويًا =====
z1 = x @ W1 + b1
print(f"z1 = {z1}")   # [[1.0, -1.0]] = (1*2+0*2-1, 1*(-2)+0*(-2)+3)

a1 = 1 / (1 + np.exp(-z1))
print(f"a1 (بعد Sigmoid) = {a1}")   # [[0.731, 0.269]]

z2 = a1 @ W2 + b2
print(f"z2 = {z2}")   # 0.731*2 + 0.269*2 - 3 = -1.0

a2 = 1 / (1 + np.exp(-z2))
print(f"a2 (التنبؤ النهائي) = {a2}")   # ≈ 0.269 (ليس مثاليًا بهذه الأوزان العشوائية البسيطة)

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-02-chapter-02.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
   model = nn.Sequential(
       nn.Linear(2, 4), nn.Sigmoid(),
       nn.Linear(4, 1), nn.Sigmoid()
   )
   ```

---

## ⚠️ 2.5 أخطاء شائعة ومصائد (احذر منها!)

### ❌ الخطأ #1: نسيان دالة التفعيل غير الخطية تمامًا
