#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الشبكات العصبية من الصفر: MLP والانتشار العكسي
Mastering Transformers and Large Language Models — Mlp Backpropagation

Book slug : mastering-llms-transformers
Chapter   : ch-03-ch03-mlp-backpropagation (lesson 3)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-03-ch03-mlp-backpropagation.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الشبكات العصبية من الصفر: MLP والانتشار العكسي
Mastering Transformers and Large Language Models — Mlp Backpropagation

Book slug : mastering-llms-transformers
Chapter   : ch-03-ch03-mlp-backpropagation (lesson 3)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-03-ch03-mlp-backpropagation.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الشبكات العصبية من الصفر: MLP والانتشار العكسي
Mastering Transformers and Large Language Models — Mlp Backpropagation

Book slug : mastering-llms-transformers
Chapter   : ch-03-ch03-mlp-backpropagation (lesson 3)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-01-foundations/ch03-mlp-backpropagation.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# === إعداد البيئة ===
# يفحص هذا القسم الحزم المطلوبة ويحاول تثبيت الناقص منها تلقائيًا.
# شغّل هذه الخلية أولًا، ثم بقية خلايا الدفتر.
import importlib.util
import subprocess
import sys

print("Python:", sys.version.split()[0])

# (اسم وحدة الاستيراد، اسم حزمة pip)
REQUIRED = [
    ("numpy", "numpy"),
    ("torch", "torch")
]
AUTO_INSTALL = {"PIL", "bs4", "datasets", "faiss", "fasttext", "librosa", "networkx", "opentelemetry", "peft", "prometheus_client", "pydantic", "rank_bm25", "requests", "sentence_transformers", "sentencepiece", "soundfile", "tokenizers", "trafilatura", "transformers", "wandb", "yaml"}


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


if not REQUIRED:
    print("لا توجد مكتبات خارجية مطلوبة؛ يكفي Python القياسي.")
else:
    missing = [(module, package) for module, package in REQUIRED if not has_module(module)]
    auto_specs = []
    manual_specs = []
    for module, package in missing:
        specs = package.split()
        if module in AUTO_INSTALL:
            auto_specs.extend(specs)
        else:
            manual_specs.extend(specs)
    auto_specs = list(dict.fromkeys(auto_specs))
    manual_specs = list(dict.fromkeys(manual_specs))

    if auto_specs:
        print("⏳ تثبيت تلقائي:", " ".join(auto_specs))
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *auto_specs])
            print("✅ تم التثبيت.")
        except Exception as exc:
            print("⚠️ تعذّر التثبيت التلقائي:", exc)
            print("شغّل يدويًا: %pip install", " ".join(auto_specs))
    if manual_specs:
        print("⚠️ حزم تحتاج تثبيتًا يدويًا:", " ".join(manual_specs))
        print("شغّل: %pip install", " ".join(manual_specs))
    if not auto_specs and not manual_specs:
        print("✅ كل الحزم المطلوبة متاحة.")

    print("\nالحالة النهائية:")
    for module, package in REQUIRED:
        print(("✅" if has_module(module) else "⚠️"), package)

# ────────────────────────────────────────────────────────────────────────
# [002] content/notebooks/part-01-foundations/ch03-mlp-backpropagation.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# إثبات عملي: تركيب طبقتين خطيتين = طبقة خطية واحدة (بدون تفعيل)
W1 = np.random.randn(4, 3)
W2 = np.random.randn(2, 4)
x = np.random.randn(3)

# التمرير عبر طبقتين منفصلتين
output_two_layers = W2 @ (W1 @ x)

# مكافئ رياضيًا لطبقة واحدة بمصفوفة مُركَّبة W_combined = W2 @ W1
W_combined = W2 @ W1
output_single_layer = W_combined @ x

print(f"متطابقان؟ {np.allclose(output_two_layers, output_single_layer)}")
# متطابقان؟ True  <- هذا بالضبط سبب حاجتنا الحتمية لدوال التفعيل غير الخطية

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-01-foundations/ch03-mlp-backpropagation.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np

np.random.seed(42)

class MLPFromScratch:
    """
    شبكة عصبية ذات طبقة مخفية واحدة، مُطبَّقة بالكامل بـ NumPy
    بدون أي مكتبة تعلّم عميق، لحل مسألة XOR الكلاسيكية
    (غير قابلة للفصل خطيًا — تحتاج لاخطية حقيقية لحلها).
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 learning_rate: float = 0.1):
        # تهيئة Xavier/Glorot: تمنع انفجار أو تلاشي التنشيطات الأولية
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        self.lr = learning_rate

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # clip لمنع overflow

    @staticmethod
    def sigmoid_derivative(a):
        # مشتقة sigmoid بدلالة الناتج a=sigmoid(z) نفسه (حيلة حسابية موفرة)
        return a * (1 - a)

    def forward(self, X: np.ndarray) -> dict:
        """
        التمرير الأمامي (Forward Pass): نحسب التنشيطات طبقة تلو أخرى،
        ونُخزِّن كل القيم الوسيطة لأننا سنحتاجها لاحقًا في Backpropagation
        (هذا هو "Cache" الحسابي القياسي في كل أطر العمل الحديثة).
        """
        z1 = X @ self.W1 + self.b1
        a1 = self.sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.sigmoid(z2)  # الناتج النهائي (احتمال، لمسألة ثنائية)

        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return cache

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Binary Cross-Entropy Loss، تطبيق مباشر لمفاهيم الفصل 2."""
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return loss

    def backward(self, cache: dict, y_true: np.ndarray):
        """
        التمرير الخلفي (Backpropagation): نُطبِّق قاعدة السلسلة
        (Chain Rule) يدويًا، طبقة بطبقة من الإخراج إلى الإدخال،
        لحساب تدرج الخسارة بالنسبة لكل وزن ومتحيّز (bias) في الشبكة.
        """
        m = y_true.shape[0]  # حجم الدفعة (batch size)
        X, a1, a2 = cache["X"], cache["a1"], cache["a2"]

        # الخطوة 1: تدرج الخسارة بالنسبة لمخرجات الطبقة الأخيرة (z2)
        # للـ BCE Loss مع sigmoid، هذا الاشتقاق يبسّط جميلاً إلى:
        dz2 = (a2 - y_true) / m              # (batch, output_dim)

        # الخطوة 2: تدرجات الطبقة الثانية (W2, b2) عبر قاعدة السلسلة
        dW2 = a1.T @ dz2                      # ∂L/∂W2 = a1ᵀ · dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # الخطوة 3: انتشار التدرج للخلف نحو الطبقة المخفية
        da1 = dz2 @ self.W2.T                 # ∂L/∂a1 = dz2 · W2ᵀ
        dz1 = da1 * self.sigmoid_derivative(a1)  # تطبيق قاعدة السلسلة عبر التفعيل

        # الخطوة 4: تدرجات الطبقة الأولى (W1, b1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def update_parameters(self, grads: dict):
        """تحديث الأوزان عبر Gradient Descent البسيط (سنُحسّن هذا في الفصل 4)."""
        self.W1 -= self.lr * grads["dW1"]
        self.b1 -= self.lr * grads["db1"]
        self.W2 -= self.lr * grads["dW2"]
        self.b2 -= self.lr * grads["db2"]

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 5000,
              verbose_every: int = 500):
        history = []
        for epoch in range(epochs):
            cache = self.forward(X)
            loss = self.compute_loss(y, cache["a2"])
            grads = self.backward(cache, y)
            self.update_parameters(grads)

            history.append(loss)
            if epoch % verbose_every == 0:
                predictions = (cache["a2"] > 0.5).astype(int)
                accuracy = np.mean(predictions == y)
                print(f"Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.2%}")
        return history

# === تطبيق على مسألة XOR ===
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])  # XOR: غير قابل للفصل خطيًا!

mlp = MLPFromScratch(input_dim=2, hidden_dim=4, output_dim=1, learning_rate=0.5)
history = mlp.train(X_xor, y_xor, epochs=3000, verbose_every=500)

final_predictions = mlp.forward(X_xor)["a2"]
print(f"\nالتنبؤات النهائية:\n{final_predictions.round(3)}")
print(f"القيم الحقيقية:\n{y_xor}")

# Epoch 0: loss=0.7102, accuracy=50.00%
# Epoch 500: loss=0.1234, accuracy=100.00%
# ...
# Epoch 2500: loss=0.0089, accuracy=100.00%
# التنبؤات النهائية: [[0.012], [0.986], [0.985], [0.015]]

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-01-foundations/ch03-mlp-backpropagation.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class MLPTorch(nn.Module):
    """
    نفس الشبكة بالضبط، لكن باستخدام PyTorch autograd.
    لاحظ: لا نكتب backward() يدويًا إطلاقًا — PyTorch يبني
    "الرسم البياني الحسابي" (computational graph) تلقائيًا أثناء
    forward()، ويحسب كل التدرجات عبره عند استدعاء loss.backward().
    """
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Sigmoid()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.output_activation(self.layer2(x))
        return x

X_xor_t = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y_xor_t = torch.tensor([[0.], [1.], [1.], [0.]])

model = MLPTorch()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(3000):
    optimizer.zero_grad()           # تصفير التدرجات المتراكمة من الخطوة السابقة
    predictions = model(X_xor_t)    # Forward pass (يبني الرسم البياني تلقائيًا)
    loss = criterion(predictions, y_xor_t)
    loss.backward()                 # Backward pass التلقائي الكامل!
    optimizer.step()                # تحديث الأوزان

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-01-foundations/ch03-mlp-backpropagation.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-03-ch03-mlp-backpropagation.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# === إعداد البيئة ===
# يفحص هذا القسم الحزم المطلوبة ويحاول تثبيت الناقص منها تلقائيًا.
# شغّل هذه الخلية أولًا، ثم بقية خلايا الدفتر.
import importlib.util
import subprocess
import sys

print("Python:", sys.version.split()[0])

# (اسم وحدة الاستيراد، اسم حزمة pip)
REQUIRED = [
    ("numpy", "numpy"),
    ("torch", "torch")
]
AUTO_INSTALL = {"PIL", "bs4", "datasets", "faiss", "fasttext", "librosa", "networkx", "opentelemetry", "peft", "prometheus_client", "pydantic", "rank_bm25", "requests", "sentence_transformers", "sentencepiece", "soundfile", "tokenizers", "trafilatura", "transformers", "wandb", "yaml"}


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


if not REQUIRED:
    print("لا توجد مكتبات خارجية مطلوبة؛ يكفي Python القياسي.")
else:
    missing = [(module, package) for module, package in REQUIRED if not has_module(module)]
    auto_specs = []
    manual_specs = []
    for module, package in missing:
        specs = package.split()
        if module in AUTO_INSTALL:
            auto_specs.extend(specs)
        else:
            manual_specs.extend(specs)
    auto_specs = list(dict.fromkeys(auto_specs))
    manual_specs = list(dict.fromkeys(manual_specs))

    if auto_specs:
        print("⏳ تثبيت تلقائي:", " ".join(auto_specs))
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *auto_specs])
            print("✅ تم التثبيت.")
        except Exception as exc:
            print("⚠️ تعذّر التثبيت التلقائي:", exc)
            print("شغّل يدويًا: %pip install", " ".join(auto_specs))
    if manual_specs:
        print("⚠️ حزم تحتاج تثبيتًا يدويًا:", " ".join(manual_specs))
        print("شغّل: %pip install", " ".join(manual_specs))
    if not auto_specs and not manual_specs:
        print("✅ كل الحزم المطلوبة متاحة.")

    print("\nالحالة النهائية:")
    for module, package in REQUIRED:
        print(("✅" if has_module(module) else "⚠️"), package)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-03-ch03-mlp-backpropagation.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# إثبات عملي: تركيب طبقتين خطيتين = طبقة خطية واحدة (بدون تفعيل)
W1 = np.random.randn(4, 3)
W2 = np.random.randn(2, 4)
x = np.random.randn(3)

# التمرير عبر طبقتين منفصلتين
output_two_layers = W2 @ (W1 @ x)

# مكافئ رياضيًا لطبقة واحدة بمصفوفة مُركَّبة W_combined = W2 @ W1
W_combined = W2 @ W1
output_single_layer = W_combined @ x

print(f"متطابقان؟ {np.allclose(output_two_layers, output_single_layer)}")
# متطابقان؟ True  <- هذا بالضبط سبب حاجتنا الحتمية لدوال التفعيل غير الخطية

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-03-ch03-mlp-backpropagation.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np

np.random.seed(42)

class MLPFromScratch:
    """
    شبكة عصبية ذات طبقة مخفية واحدة، مُطبَّقة بالكامل بـ NumPy
    بدون أي مكتبة تعلّم عميق، لحل مسألة XOR الكلاسيكية
    (غير قابلة للفصل خطيًا — تحتاج لاخطية حقيقية لحلها).
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 learning_rate: float = 0.1):
        # تهيئة Xavier/Glorot: تمنع انفجار أو تلاشي التنشيطات الأولية
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        self.lr = learning_rate

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # clip لمنع overflow

    @staticmethod
    def sigmoid_derivative(a):
        # مشتقة sigmoid بدلالة الناتج a=sigmoid(z) نفسه (حيلة حسابية موفرة)
        return a * (1 - a)

    def forward(self, X: np.ndarray) -> dict:
        """
        التمرير الأمامي (Forward Pass): نحسب التنشيطات طبقة تلو أخرى،
        ونُخزِّن كل القيم الوسيطة لأننا سنحتاجها لاحقًا في Backpropagation
        (هذا هو "Cache" الحسابي القياسي في كل أطر العمل الحديثة).
        """
        z1 = X @ self.W1 + self.b1
        a1 = self.sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.sigmoid(z2)  # الناتج النهائي (احتمال، لمسألة ثنائية)

        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return cache

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Binary Cross-Entropy Loss، تطبيق مباشر لمفاهيم الفصل 2."""
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return loss

    def backward(self, cache: dict, y_true: np.ndarray):
        """
        التمرير الخلفي (Backpropagation): نُطبِّق قاعدة السلسلة
        (Chain Rule) يدويًا، طبقة بطبقة من الإخراج إلى الإدخال،
        لحساب تدرج الخسارة بالنسبة لكل وزن ومتحيّز (bias) في الشبكة.
        """
        m = y_true.shape[0]  # حجم الدفعة (batch size)
        X, a1, a2 = cache["X"], cache["a1"], cache["a2"]

        # الخطوة 1: تدرج الخسارة بالنسبة لمخرجات الطبقة الأخيرة (z2)
        # للـ BCE Loss مع sigmoid، هذا الاشتقاق يبسّط جميلاً إلى:
        dz2 = (a2 - y_true) / m              # (batch, output_dim)

        # الخطوة 2: تدرجات الطبقة الثانية (W2, b2) عبر قاعدة السلسلة
        dW2 = a1.T @ dz2                      # ∂L/∂W2 = a1ᵀ · dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # الخطوة 3: انتشار التدرج للخلف نحو الطبقة المخفية
        da1 = dz2 @ self.W2.T                 # ∂L/∂a1 = dz2 · W2ᵀ
        dz1 = da1 * self.sigmoid_derivative(a1)  # تطبيق قاعدة السلسلة عبر التفعيل

        # الخطوة 4: تدرجات الطبقة الأولى (W1, b1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def update_parameters(self, grads: dict):
        """تحديث الأوزان عبر Gradient Descent البسيط (سنُحسّن هذا في الفصل 4)."""
        self.W1 -= self.lr * grads["dW1"]
        self.b1 -= self.lr * grads["db1"]
        self.W2 -= self.lr * grads["dW2"]
        self.b2 -= self.lr * grads["db2"]

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 5000,
              verbose_every: int = 500):
        history = []
        for epoch in range(epochs):
            cache = self.forward(X)
            loss = self.compute_loss(y, cache["a2"])
            grads = self.backward(cache, y)
            self.update_parameters(grads)

            history.append(loss)
            if epoch % verbose_every == 0:
                predictions = (cache["a2"] > 0.5).astype(int)
                accuracy = np.mean(predictions == y)
                print(f"Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.2%}")
        return history

# === تطبيق على مسألة XOR ===
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])  # XOR: غير قابل للفصل خطيًا!

mlp = MLPFromScratch(input_dim=2, hidden_dim=4, output_dim=1, learning_rate=0.5)
history = mlp.train(X_xor, y_xor, epochs=3000, verbose_every=500)

final_predictions = mlp.forward(X_xor)["a2"]
print(f"\nالتنبؤات النهائية:\n{final_predictions.round(3)}")
print(f"القيم الحقيقية:\n{y_xor}")

# Epoch 0: loss=0.7102, accuracy=50.00%
# Epoch 500: loss=0.1234, accuracy=100.00%
# ...
# Epoch 2500: loss=0.0089, accuracy=100.00%
# التنبؤات النهائية: [[0.012], [0.986], [0.985], [0.015]]

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-03-ch03-mlp-backpropagation.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class MLPTorch(nn.Module):
    """
    نفس الشبكة بالضبط، لكن باستخدام PyTorch autograd.
    لاحظ: لا نكتب backward() يدويًا إطلاقًا — PyTorch يبني
    "الرسم البياني الحسابي" (computational graph) تلقائيًا أثناء
    forward()، ويحسب كل التدرجات عبره عند استدعاء loss.backward().
    """
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Sigmoid()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.output_activation(self.layer2(x))
        return x

X_xor_t = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y_xor_t = torch.tensor([[0.], [1.], [1.], [0.]])

model = MLPTorch()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(3000):
    optimizer.zero_grad()           # تصفير التدرجات المتراكمة من الخطوة السابقة
    predictions = model(X_xor_t)    # Forward pass (يبني الرسم البياني تلقائيًا)
    loss = criterion(predictions, y_xor_t)
    loss.backward()                 # Backward pass التلقائي الكامل!
    optimizer.step()                # تحديث الأوزان

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-03-ch03-mlp-backpropagation.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-03-ch03-mlp-backpropagation.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الشبكات العصبية من الصفر: MLP والانتشار العكسي
Mastering Transformers and Large Language Models — Mlp Backpropagation

Book slug : mastering-llms-transformers
Chapter   : ch-03-ch03-mlp-backpropagation (lesson 3)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-01-foundations/ch03-mlp-backpropagation.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# === إعداد البيئة ===
# يفحص هذا القسم الحزم المطلوبة ويحاول تثبيت الناقص منها تلقائيًا.
# شغّل هذه الخلية أولًا، ثم بقية خلايا الدفتر.
import importlib.util
import subprocess
import sys

print("Python:", sys.version.split()[0])

# (اسم وحدة الاستيراد، اسم حزمة pip)
REQUIRED = [
    ("numpy", "numpy"),
    ("torch", "torch")
]
AUTO_INSTALL = {"PIL", "bs4", "datasets", "faiss", "fasttext", "librosa", "networkx", "opentelemetry", "peft", "prometheus_client", "pydantic", "rank_bm25", "requests", "sentence_transformers", "sentencepiece", "soundfile", "tokenizers", "trafilatura", "transformers", "wandb", "yaml"}


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


if not REQUIRED:
    print("لا توجد مكتبات خارجية مطلوبة؛ يكفي Python القياسي.")
else:
    missing = [(module, package) for module, package in REQUIRED if not has_module(module)]
    auto_specs = []
    manual_specs = []
    for module, package in missing:
        specs = package.split()
        if module in AUTO_INSTALL:
            auto_specs.extend(specs)
        else:
            manual_specs.extend(specs)
    auto_specs = list(dict.fromkeys(auto_specs))
    manual_specs = list(dict.fromkeys(manual_specs))

    if auto_specs:
        print("⏳ تثبيت تلقائي:", " ".join(auto_specs))
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *auto_specs])
            print("✅ تم التثبيت.")
        except Exception as exc:
            print("⚠️ تعذّر التثبيت التلقائي:", exc)
            print("شغّل يدويًا: %pip install", " ".join(auto_specs))
    if manual_specs:
        print("⚠️ حزم تحتاج تثبيتًا يدويًا:", " ".join(manual_specs))
        print("شغّل: %pip install", " ".join(manual_specs))
    if not auto_specs and not manual_specs:
        print("✅ كل الحزم المطلوبة متاحة.")

    print("\nالحالة النهائية:")
    for module, package in REQUIRED:
        print(("✅" if has_module(module) else "⚠️"), package)

# ────────────────────────────────────────────────────────────────────────
# [002] content/notebooks/part-01-foundations/ch03-mlp-backpropagation.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# إثبات عملي: تركيب طبقتين خطيتين = طبقة خطية واحدة (بدون تفعيل)
W1 = np.random.randn(4, 3)
W2 = np.random.randn(2, 4)
x = np.random.randn(3)

# التمرير عبر طبقتين منفصلتين
output_two_layers = W2 @ (W1 @ x)

# مكافئ رياضيًا لطبقة واحدة بمصفوفة مُركَّبة W_combined = W2 @ W1
W_combined = W2 @ W1
output_single_layer = W_combined @ x

print(f"متطابقان؟ {np.allclose(output_two_layers, output_single_layer)}")
# متطابقان؟ True  <- هذا بالضبط سبب حاجتنا الحتمية لدوال التفعيل غير الخطية

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-01-foundations/ch03-mlp-backpropagation.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np

np.random.seed(42)

class MLPFromScratch:
    """
    شبكة عصبية ذات طبقة مخفية واحدة، مُطبَّقة بالكامل بـ NumPy
    بدون أي مكتبة تعلّم عميق، لحل مسألة XOR الكلاسيكية
    (غير قابلة للفصل خطيًا — تحتاج لاخطية حقيقية لحلها).
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 learning_rate: float = 0.1):
        # تهيئة Xavier/Glorot: تمنع انفجار أو تلاشي التنشيطات الأولية
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        self.lr = learning_rate

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # clip لمنع overflow

    @staticmethod
    def sigmoid_derivative(a):
        # مشتقة sigmoid بدلالة الناتج a=sigmoid(z) نفسه (حيلة حسابية موفرة)
        return a * (1 - a)

    def forward(self, X: np.ndarray) -> dict:
        """
        التمرير الأمامي (Forward Pass): نحسب التنشيطات طبقة تلو أخرى،
        ونُخزِّن كل القيم الوسيطة لأننا سنحتاجها لاحقًا في Backpropagation
        (هذا هو "Cache" الحسابي القياسي في كل أطر العمل الحديثة).
        """
        z1 = X @ self.W1 + self.b1
        a1 = self.sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.sigmoid(z2)  # الناتج النهائي (احتمال، لمسألة ثنائية)

        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return cache

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Binary Cross-Entropy Loss، تطبيق مباشر لمفاهيم الفصل 2."""
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return loss

    def backward(self, cache: dict, y_true: np.ndarray):
        """
        التمرير الخلفي (Backpropagation): نُطبِّق قاعدة السلسلة
        (Chain Rule) يدويًا، طبقة بطبقة من الإخراج إلى الإدخال،
        لحساب تدرج الخسارة بالنسبة لكل وزن ومتحيّز (bias) في الشبكة.
        """
        m = y_true.shape[0]  # حجم الدفعة (batch size)
        X, a1, a2 = cache["X"], cache["a1"], cache["a2"]

        # الخطوة 1: تدرج الخسارة بالنسبة لمخرجات الطبقة الأخيرة (z2)
        # للـ BCE Loss مع sigmoid، هذا الاشتقاق يبسّط جميلاً إلى:
        dz2 = (a2 - y_true) / m              # (batch, output_dim)

        # الخطوة 2: تدرجات الطبقة الثانية (W2, b2) عبر قاعدة السلسلة
        dW2 = a1.T @ dz2                      # ∂L/∂W2 = a1ᵀ · dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # الخطوة 3: انتشار التدرج للخلف نحو الطبقة المخفية
        da1 = dz2 @ self.W2.T                 # ∂L/∂a1 = dz2 · W2ᵀ
        dz1 = da1 * self.sigmoid_derivative(a1)  # تطبيق قاعدة السلسلة عبر التفعيل

        # الخطوة 4: تدرجات الطبقة الأولى (W1, b1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def update_parameters(self, grads: dict):
        """تحديث الأوزان عبر Gradient Descent البسيط (سنُحسّن هذا في الفصل 4)."""
        self.W1 -= self.lr * grads["dW1"]
        self.b1 -= self.lr * grads["db1"]
        self.W2 -= self.lr * grads["dW2"]
        self.b2 -= self.lr * grads["db2"]

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 5000,
              verbose_every: int = 500):
        history = []
        for epoch in range(epochs):
            cache = self.forward(X)
            loss = self.compute_loss(y, cache["a2"])
            grads = self.backward(cache, y)
            self.update_parameters(grads)

            history.append(loss)
            if epoch % verbose_every == 0:
                predictions = (cache["a2"] > 0.5).astype(int)
                accuracy = np.mean(predictions == y)
                print(f"Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.2%}")
        return history

# === تطبيق على مسألة XOR ===
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])  # XOR: غير قابل للفصل خطيًا!

mlp = MLPFromScratch(input_dim=2, hidden_dim=4, output_dim=1, learning_rate=0.5)
history = mlp.train(X_xor, y_xor, epochs=3000, verbose_every=500)

final_predictions = mlp.forward(X_xor)["a2"]
print(f"\nالتنبؤات النهائية:\n{final_predictions.round(3)}")
print(f"القيم الحقيقية:\n{y_xor}")

# Epoch 0: loss=0.7102, accuracy=50.00%
# Epoch 500: loss=0.1234, accuracy=100.00%
# ...
# Epoch 2500: loss=0.0089, accuracy=100.00%
# التنبؤات النهائية: [[0.012], [0.986], [0.985], [0.015]]

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-01-foundations/ch03-mlp-backpropagation.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class MLPTorch(nn.Module):
    """
    نفس الشبكة بالضبط، لكن باستخدام PyTorch autograd.
    لاحظ: لا نكتب backward() يدويًا إطلاقًا — PyTorch يبني
    "الرسم البياني الحسابي" (computational graph) تلقائيًا أثناء
    forward()، ويحسب كل التدرجات عبره عند استدعاء loss.backward().
    """
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Sigmoid()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.output_activation(self.layer2(x))
        return x

X_xor_t = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y_xor_t = torch.tensor([[0.], [1.], [1.], [0.]])

model = MLPTorch()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(3000):
    optimizer.zero_grad()           # تصفير التدرجات المتراكمة من الخطوة السابقة
    predictions = model(X_xor_t)    # Forward pass (يبني الرسم البياني تلقائيًا)
    loss = criterion(predictions, y_xor_t)
    loss.backward()                 # Backward pass التلقائي الكامل!
    optimizer.step()                # تحديث الأوزان

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-01-foundations/ch03-mlp-backpropagation.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-03-ch03-mlp-backpropagation.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# === إعداد البيئة ===
# يفحص هذا القسم الحزم المطلوبة ويحاول تثبيت الناقص منها تلقائيًا.
# شغّل هذه الخلية أولًا، ثم بقية خلايا الدفتر.
import importlib.util
import subprocess
import sys

print("Python:", sys.version.split()[0])

# (اسم وحدة الاستيراد، اسم حزمة pip)
REQUIRED = [
    ("numpy", "numpy"),
    ("torch", "torch")
]
AUTO_INSTALL = {"PIL", "bs4", "datasets", "faiss", "fasttext", "librosa", "networkx", "opentelemetry", "peft", "prometheus_client", "pydantic", "rank_bm25", "requests", "sentence_transformers", "sentencepiece", "soundfile", "tokenizers", "trafilatura", "transformers", "wandb", "yaml"}


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


if not REQUIRED:
    print("لا توجد مكتبات خارجية مطلوبة؛ يكفي Python القياسي.")
else:
    missing = [(module, package) for module, package in REQUIRED if not has_module(module)]
    auto_specs = []
    manual_specs = []
    for module, package in missing:
        specs = package.split()
        if module in AUTO_INSTALL:
            auto_specs.extend(specs)
        else:
            manual_specs.extend(specs)
    auto_specs = list(dict.fromkeys(auto_specs))
    manual_specs = list(dict.fromkeys(manual_specs))

    if auto_specs:
        print("⏳ تثبيت تلقائي:", " ".join(auto_specs))
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *auto_specs])
            print("✅ تم التثبيت.")
        except Exception as exc:
            print("⚠️ تعذّر التثبيت التلقائي:", exc)
            print("شغّل يدويًا: %pip install", " ".join(auto_specs))
    if manual_specs:
        print("⚠️ حزم تحتاج تثبيتًا يدويًا:", " ".join(manual_specs))
        print("شغّل: %pip install", " ".join(manual_specs))
    if not auto_specs and not manual_specs:
        print("✅ كل الحزم المطلوبة متاحة.")

    print("\nالحالة النهائية:")
    for module, package in REQUIRED:
        print(("✅" if has_module(module) else "⚠️"), package)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-03-ch03-mlp-backpropagation.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# إثبات عملي: تركيب طبقتين خطيتين = طبقة خطية واحدة (بدون تفعيل)
W1 = np.random.randn(4, 3)
W2 = np.random.randn(2, 4)
x = np.random.randn(3)

# التمرير عبر طبقتين منفصلتين
output_two_layers = W2 @ (W1 @ x)

# مكافئ رياضيًا لطبقة واحدة بمصفوفة مُركَّبة W_combined = W2 @ W1
W_combined = W2 @ W1
output_single_layer = W_combined @ x

print(f"متطابقان؟ {np.allclose(output_two_layers, output_single_layer)}")
# متطابقان؟ True  <- هذا بالضبط سبب حاجتنا الحتمية لدوال التفعيل غير الخطية

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-03-ch03-mlp-backpropagation.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np

np.random.seed(42)

class MLPFromScratch:
    """
    شبكة عصبية ذات طبقة مخفية واحدة، مُطبَّقة بالكامل بـ NumPy
    بدون أي مكتبة تعلّم عميق، لحل مسألة XOR الكلاسيكية
    (غير قابلة للفصل خطيًا — تحتاج لاخطية حقيقية لحلها).
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 learning_rate: float = 0.1):
        # تهيئة Xavier/Glorot: تمنع انفجار أو تلاشي التنشيطات الأولية
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        self.lr = learning_rate

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # clip لمنع overflow

    @staticmethod
    def sigmoid_derivative(a):
        # مشتقة sigmoid بدلالة الناتج a=sigmoid(z) نفسه (حيلة حسابية موفرة)
        return a * (1 - a)

    def forward(self, X: np.ndarray) -> dict:
        """
        التمرير الأمامي (Forward Pass): نحسب التنشيطات طبقة تلو أخرى،
        ونُخزِّن كل القيم الوسيطة لأننا سنحتاجها لاحقًا في Backpropagation
        (هذا هو "Cache" الحسابي القياسي في كل أطر العمل الحديثة).
        """
        z1 = X @ self.W1 + self.b1
        a1 = self.sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.sigmoid(z2)  # الناتج النهائي (احتمال، لمسألة ثنائية)

        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return cache

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Binary Cross-Entropy Loss، تطبيق مباشر لمفاهيم الفصل 2."""
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return loss

    def backward(self, cache: dict, y_true: np.ndarray):
        """
        التمرير الخلفي (Backpropagation): نُطبِّق قاعدة السلسلة
        (Chain Rule) يدويًا، طبقة بطبقة من الإخراج إلى الإدخال،
        لحساب تدرج الخسارة بالنسبة لكل وزن ومتحيّز (bias) في الشبكة.
        """
        m = y_true.shape[0]  # حجم الدفعة (batch size)
        X, a1, a2 = cache["X"], cache["a1"], cache["a2"]

        # الخطوة 1: تدرج الخسارة بالنسبة لمخرجات الطبقة الأخيرة (z2)
        # للـ BCE Loss مع sigmoid، هذا الاشتقاق يبسّط جميلاً إلى:
        dz2 = (a2 - y_true) / m              # (batch, output_dim)

        # الخطوة 2: تدرجات الطبقة الثانية (W2, b2) عبر قاعدة السلسلة
        dW2 = a1.T @ dz2                      # ∂L/∂W2 = a1ᵀ · dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # الخطوة 3: انتشار التدرج للخلف نحو الطبقة المخفية
        da1 = dz2 @ self.W2.T                 # ∂L/∂a1 = dz2 · W2ᵀ
        dz1 = da1 * self.sigmoid_derivative(a1)  # تطبيق قاعدة السلسلة عبر التفعيل

        # الخطوة 4: تدرجات الطبقة الأولى (W1, b1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def update_parameters(self, grads: dict):
        """تحديث الأوزان عبر Gradient Descent البسيط (سنُحسّن هذا في الفصل 4)."""
        self.W1 -= self.lr * grads["dW1"]
        self.b1 -= self.lr * grads["db1"]
        self.W2 -= self.lr * grads["dW2"]
        self.b2 -= self.lr * grads["db2"]

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 5000,
              verbose_every: int = 500):
        history = []
        for epoch in range(epochs):
            cache = self.forward(X)
            loss = self.compute_loss(y, cache["a2"])
            grads = self.backward(cache, y)
            self.update_parameters(grads)

            history.append(loss)
            if epoch % verbose_every == 0:
                predictions = (cache["a2"] > 0.5).astype(int)
                accuracy = np.mean(predictions == y)
                print(f"Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.2%}")
        return history

# === تطبيق على مسألة XOR ===
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])  # XOR: غير قابل للفصل خطيًا!

mlp = MLPFromScratch(input_dim=2, hidden_dim=4, output_dim=1, learning_rate=0.5)
history = mlp.train(X_xor, y_xor, epochs=3000, verbose_every=500)

final_predictions = mlp.forward(X_xor)["a2"]
print(f"\nالتنبؤات النهائية:\n{final_predictions.round(3)}")
print(f"القيم الحقيقية:\n{y_xor}")

# Epoch 0: loss=0.7102, accuracy=50.00%
# Epoch 500: loss=0.1234, accuracy=100.00%
# ...
# Epoch 2500: loss=0.0089, accuracy=100.00%
# التنبؤات النهائية: [[0.012], [0.986], [0.985], [0.015]]

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-03-ch03-mlp-backpropagation.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class MLPTorch(nn.Module):
    """
    نفس الشبكة بالضبط، لكن باستخدام PyTorch autograd.
    لاحظ: لا نكتب backward() يدويًا إطلاقًا — PyTorch يبني
    "الرسم البياني الحسابي" (computational graph) تلقائيًا أثناء
    forward()، ويحسب كل التدرجات عبره عند استدعاء loss.backward().
    """
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Sigmoid()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.output_activation(self.layer2(x))
        return x

X_xor_t = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y_xor_t = torch.tensor([[0.], [1.], [1.], [0.]])

model = MLPTorch()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(3000):
    optimizer.zero_grad()           # تصفير التدرجات المتراكمة من الخطوة السابقة
    predictions = model(X_xor_t)    # Forward pass (يبني الرسم البياني تلقائيًا)
    loss = criterion(predictions, y_xor_t)
    loss.backward()                 # Backward pass التلقائي الكامل!
    optimizer.step()                # تحديث الأوزان

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-03-ch03-mlp-backpropagation.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
