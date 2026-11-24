#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التحسين والتدريب: من SGD إلى AdamW
Mastering Transformers and Large Language Models — Optimization

Book slug : mastering-llms-transformers
Chapter   : ch-04-ch04-optimization (lesson 4)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-04-ch04-optimization.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التحسين والتدريب: من SGD إلى AdamW
Mastering Transformers and Large Language Models — Optimization

Book slug : mastering-llms-transformers
Chapter   : ch-04-ch04-optimization (lesson 4)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-04-ch04-optimization.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التحسين والتدريب: من SGD إلى AdamW
Mastering Transformers and Large Language Models — Optimization

Book slug : mastering-llms-transformers
Chapter   : ch-04-ch04-optimization (lesson 4)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-01-foundations/ch04-optimization.ipynb#cell-1
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
    ("matplotlib", "matplotlib"),
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
# [002] content/notebooks/part-01-foundations/ch04-optimization.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np

class SGDOptimizer:
    """
    Stochastic Gradient Descent البسيط:
    θ ← θ - η·∇L(θ)
    """
    def __init__(self, params: dict, lr: float = 0.01):
        self.params = params
        self.lr = lr

    def step(self, grads: dict):
        for key in self.params:
            self.params[key] -= self.lr * grads[key]

class SGDMomentumOptimizer:
    """
    SGD مع Momentum: نُراكم "سرعة" (velocity) مُرجَّحة لتسريع التقارب
    في الاتجاهات ذات التدرج المستمر، وتخميد التذبذب في الاتجاهات
    المتضاربة (مثل الوادي الضيق الطويل الشهير في أدبيات التحسين).

        v ← β·v + (1-β)·∇L(θ)      [β عادة 0.9]
        θ ← θ - η·v
    """
    def __init__(self, params: dict, lr: float = 0.01, beta: float = 0.9):
        self.params = params
        self.lr = lr
        self.beta = beta
        self.velocity = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads: dict):
        for key in self.params:
            self.velocity[key] = (self.beta * self.velocity[key] +
                                   (1 - self.beta) * grads[key])
            self.params[key] -= self.lr * self.velocity[key]

class AdamOptimizer:
    """
    تطبيق Adam من الصفر (Kingma & Ba, 2014) — المُحسِّن المرجعي
    الذي بُنيت عليه كل التدريبات الحديثة تقريبًا (عبر AdamW).

    يجمع بين فكرتين:
    1. Momentum (الزخم): متوسط مُرجَّح أُسّي للتدرجات (m)
    2. Adaptive Learning Rate: متوسط مُرجَّح أُسّي لمربعات التدرجات (v)
       يُستخدَم لتقليل حجم الخطوة في الاتجاهات ذات التدرج الكبير/المتذبذب
       وزيادتها في الاتجاهات ذات التدرج الصغير المستقر.
    """
    def __init__(self, params: dict, lr: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {k: np.zeros_like(v) for k, v in params.items()}  # العزم الأول (المتوسط)
        self.v = {k: np.zeros_like(v) for k, v in params.items()}  # العزم الثاني (التباين)
        self.t = 0  # عدّاد الخطوات، ضروري لتصحيح الانحياز

    def step(self, grads: dict):
        self.t += 1
        for key in self.params:
            g = grads[key]

            # تحديث العزوم المُرجَّحة أُسّيًا
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)

            # تصحيح الانحياز (Bias Correction): ضروري لأن m,v تبدآن من الصفر
            # وتكونان متحيّزة نحو الصفر في الخطوات الأولى من التدريب
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # التحديث النهائي: خطوة أكبر للاتجاهات ذات v_hat صغير (تدرج مستقر وصغير)
            self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class AdamWOptimizer:
    """
    AdamW (Loshchilov & Hutter, 2017): الفرق الجوهري عن Adam هو فصل
    Weight Decay عن تحديث التدرج التكيّفي. في Adam التقليدي، إضافة
    L2 regularization للتدرج قبل حساب m,v تجعل تأثير الانحلال يتداخل
    بشكل غير صحيح مع التحجيم التكيّفي. AdamW يُطبِّق الانحلال مباشرة
    على الأوزان بشكل منفصل تمامًا:

        θ ← θ - η·(m̂/√v̂+ε) - η·λ·θ      [λ = weight decay]

    هذا هو المُحسِّن المعياري لتدريب كل LLM حديث تقريبًا.
    """
    def __init__(self, params: dict, lr: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8, weight_decay: float = 0.01):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict):
        self.t += 1
        for key in self.params:
            g = grads[key]
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            adaptive_update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            decay_update = self.lr * self.weight_decay * self.params[key]

            self.params[key] -= (adaptive_update + decay_update)

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-01-foundations/ch04-optimization.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def rosenbrock_loss(x, y):
    """دالة اختبار كلاسيكية للمُحسِّنات (وادٍ ضيق منحني شهير جدًا)."""
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_gradient(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x ** 2)
    dy = 200 * (y - x ** 2)
    return np.array([dx, dy])

def compare_optimizers_on_rosenbrock(n_steps: int = 1000):
    start_point = np.array([-1.5, 2.0])
    results = {}

    for name, OptClass, lr in [
        ("SGD", SGDOptimizer, 0.001),
        ("SGD+Momentum", SGDMomentumOptimizer, 0.001),
        ("Adam", AdamOptimizer, 0.01),
    ]:
        params = {"xy": start_point.copy()}
        optimizer = OptClass(params, lr=lr)
        trajectory = [params["xy"].copy()]

        for _ in range(n_steps):
            x, y = params["xy"]
            grad = rosenbrock_gradient(x, y)
            optimizer.step({"xy": grad})
            trajectory.append(params["xy"].copy())

        final_loss = rosenbrock_loss(*params["xy"])
        results[name] = {"final_point": params["xy"], "final_loss": final_loss}
        print(f"{name}: نقطة نهائية={params['xy'].round(4)}, "
              f"خسارة نهائية={final_loss:.6f}")

compare_optimizers_on_rosenbrock()
# SGD: نقطة نهائية=[-1.4312  2.0520], خسارة نهائية=5.9102   <- عالق تقريبًا
# SGD+Momentum: نقطة نهائية=[-0.8923  0.7861], خسارة نهائية=3.6124
# Adam: نقطة نهائية=[0.9156  0.8384], خسارة نهائية=0.0074    <- الأقرب للحل الأمثل (1,1)

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-01-foundations/ch04-optimization.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt

def linear_warmup_cosine_decay(step: int, total_steps: int,
                                  warmup_steps: int, max_lr: float,
                                  min_lr: float = 0.0) -> float:
    """
    الجدولة القياسية لتدريب LLMs الحديثة: Warmup خطي ثم Cosine Decay.

    لماذا Warmup؟ في بداية التدريب، تقديرات m,v في Adam غير دقيقة
    (قليلة العينات)، والأوزان عشوائية تمامًا، فاستخدام LR كامل فورًا
    قد يُسبب تحديثات كبيرة جدًا ومُزعزعة للاستقرار (خصوصًا في المحوّلات
    الكبيرة). نبدأ بـ LR صغير جدًا ونزيده تدريجيًا.

    لماذا Cosine Decay (وليس خطي)؟ يُحافظ على LR مرتفع نسبيًا لفترة
    أطول في منتصف التدريب (حيث لا يزال التعلّم فعّالًا)، وينخفض بسلاسة
    أكبر قرب النهاية مقارنة بالانخفاض الخطي الحاد.
    """
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(progress, 1.0)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine_decay

def visualize_lr_schedule(total_steps=10000, warmup_steps=500, max_lr=3e-4):
    steps = np.arange(total_steps)
    lrs = [linear_warmup_cosine_decay(s, total_steps, warmup_steps, max_lr)
           for s in steps]

    plt.figure(figsize=(10, 4))
    plt.plot(steps, lrs)
    plt.axvline(x=warmup_steps, color='r', linestyle='--', label='نهاية Warmup')
    plt.xlabel("خطوة التدريب")
    plt.ylabel("معدل التعلّم")
    plt.title("جدولة Linear Warmup + Cosine Decay")
    plt.legend()
    plt.savefig("lr_schedule.png")
    print(f"LR عند step=0: {lrs[0]:.6f}")
    print(f"LR عند step=warmup_steps: {lrs[warmup_steps]:.6f}")
    print(f"LR عند step=total_steps/2: {lrs[total_steps//2]:.6f}")
    print(f"LR عند step=total_steps-1: {lrs[-1]:.6f}")

visualize_lr_schedule()
# LR عند step=0: 0.000000
# LR عند step=warmup_steps: 0.000300
# LR عند step=total_steps/2: 0.000150
# LR عند step=total_steps-1: 0.000000

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-01-foundations/ch04-optimization.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class RegularizedMLP(nn.Module):
    """
    شبكة تطبّق أهم تقنيات Regularization الشائعة:
    Dropout + Weight Decay (عبر المُحسِّن) + Gradient Clipping (في التدريب)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)  # يُطفئ عشوائيًا نسبة من الخلايا أثناء التدريب فقط
        x = self.layer2(x)
        return x

def train_with_regularization(model, train_loader, epochs=10,
                                 max_grad_norm: float = 1.0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()  # يُفعِّل Dropout
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient Clipping: يمنع "انفجار التدرجات" (exploding gradients)
            # بتحديد أقصى قيمة لـ L2-norm الكلية لكل التدرجات مجتمعة
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

        model.eval()  # يُعطِّل Dropout أثناء التقييم (مهم جدًا!)

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-01-foundations/ch04-optimization.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def manual_adam_steps():
    theta = 1.0       # قيمة ابتدائية للمعامل
    m, v = 0.0, 0.0   # العزمان الأول والثاني، يبدآن من الصفر
    beta1, beta2, lr, eps = 0.9, 0.999, 0.1, 1e-8
    gradients = [2.0, 1.8]  # تدرجات افتراضية لخطوتين متتاليتين

    for t, g in enumerate(gradients, start=1):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)

        m_hat_no_correction = m   # بدون تصحيح انحياز (للتوضيح فقط)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        update = lr * m_hat / (np.sqrt(v_hat) + eps)
        theta -= update

        print(f"الخطوة {t}: m={m:.4f}, m_hat (مُصحَّح)={m_hat:.4f} "
              f"(بدون تصحيح كان سيكون {m_hat_no_correction:.4f})")
        print(f"          v={v:.4f}, v_hat (مُصحَّح)={v_hat:.4f}")
        print(f"          تحديث θ: -{update:.4f} → θ الجديدة = {theta:.4f}\n")

manual_adam_steps()
# الخطوة 1: m=0.2000, m_hat (مُصحَّح)=2.0000 (بدون تصحيح كان سيكون 0.2000)
#           v=0.0040, v_hat (مُصحَّح)=4.0000
#           تحديث θ: -0.1000 → θ الجديدة = 0.9000
#
# الخطوة 2: m=0.3620, m_hat (مُصحَّح)=1.9053 (بدون تصحيح كان سيكون 0.3620)
#           v=0.0072, v_hat (مُصحَّح)=3.6076
#           تحديث θ: -0.1003 → θ الجديدة = 0.7997

# ────────────────────────────────────────────────────────────────────────
# [007] content/notebooks/part-01-foundations/ch04-optimization.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-04-ch04-optimization.ipynb#cell-1
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
    ("matplotlib", "matplotlib"),
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-04-ch04-optimization.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np

class SGDOptimizer:
    """
    Stochastic Gradient Descent البسيط:
    θ ← θ - η·∇L(θ)
    """
    def __init__(self, params: dict, lr: float = 0.01):
        self.params = params
        self.lr = lr

    def step(self, grads: dict):
        for key in self.params:
            self.params[key] -= self.lr * grads[key]

class SGDMomentumOptimizer:
    """
    SGD مع Momentum: نُراكم "سرعة" (velocity) مُرجَّحة لتسريع التقارب
    في الاتجاهات ذات التدرج المستمر، وتخميد التذبذب في الاتجاهات
    المتضاربة (مثل الوادي الضيق الطويل الشهير في أدبيات التحسين).

        v ← β·v + (1-β)·∇L(θ)      [β عادة 0.9]
        θ ← θ - η·v
    """
    def __init__(self, params: dict, lr: float = 0.01, beta: float = 0.9):
        self.params = params
        self.lr = lr
        self.beta = beta
        self.velocity = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads: dict):
        for key in self.params:
            self.velocity[key] = (self.beta * self.velocity[key] +
                                   (1 - self.beta) * grads[key])
            self.params[key] -= self.lr * self.velocity[key]

class AdamOptimizer:
    """
    تطبيق Adam من الصفر (Kingma & Ba, 2014) — المُحسِّن المرجعي
    الذي بُنيت عليه كل التدريبات الحديثة تقريبًا (عبر AdamW).

    يجمع بين فكرتين:
    1. Momentum (الزخم): متوسط مُرجَّح أُسّي للتدرجات (m)
    2. Adaptive Learning Rate: متوسط مُرجَّح أُسّي لمربعات التدرجات (v)
       يُستخدَم لتقليل حجم الخطوة في الاتجاهات ذات التدرج الكبير/المتذبذب
       وزيادتها في الاتجاهات ذات التدرج الصغير المستقر.
    """
    def __init__(self, params: dict, lr: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {k: np.zeros_like(v) for k, v in params.items()}  # العزم الأول (المتوسط)
        self.v = {k: np.zeros_like(v) for k, v in params.items()}  # العزم الثاني (التباين)
        self.t = 0  # عدّاد الخطوات، ضروري لتصحيح الانحياز

    def step(self, grads: dict):
        self.t += 1
        for key in self.params:
            g = grads[key]

            # تحديث العزوم المُرجَّحة أُسّيًا
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)

            # تصحيح الانحياز (Bias Correction): ضروري لأن m,v تبدآن من الصفر
            # وتكونان متحيّزة نحو الصفر في الخطوات الأولى من التدريب
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # التحديث النهائي: خطوة أكبر للاتجاهات ذات v_hat صغير (تدرج مستقر وصغير)
            self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class AdamWOptimizer:
    """
    AdamW (Loshchilov & Hutter, 2017): الفرق الجوهري عن Adam هو فصل
    Weight Decay عن تحديث التدرج التكيّفي. في Adam التقليدي، إضافة
    L2 regularization للتدرج قبل حساب m,v تجعل تأثير الانحلال يتداخل
    بشكل غير صحيح مع التحجيم التكيّفي. AdamW يُطبِّق الانحلال مباشرة
    على الأوزان بشكل منفصل تمامًا:

        θ ← θ - η·(m̂/√v̂+ε) - η·λ·θ      [λ = weight decay]

    هذا هو المُحسِّن المعياري لتدريب كل LLM حديث تقريبًا.
    """
    def __init__(self, params: dict, lr: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8, weight_decay: float = 0.01):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict):
        self.t += 1
        for key in self.params:
            g = grads[key]
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            adaptive_update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            decay_update = self.lr * self.weight_decay * self.params[key]

            self.params[key] -= (adaptive_update + decay_update)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-04-ch04-optimization.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def rosenbrock_loss(x, y):
    """دالة اختبار كلاسيكية للمُحسِّنات (وادٍ ضيق منحني شهير جدًا)."""
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_gradient(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x ** 2)
    dy = 200 * (y - x ** 2)
    return np.array([dx, dy])

def compare_optimizers_on_rosenbrock(n_steps: int = 1000):
    start_point = np.array([-1.5, 2.0])
    results = {}

    for name, OptClass, lr in [
        ("SGD", SGDOptimizer, 0.001),
        ("SGD+Momentum", SGDMomentumOptimizer, 0.001),
        ("Adam", AdamOptimizer, 0.01),
    ]:
        params = {"xy": start_point.copy()}
        optimizer = OptClass(params, lr=lr)
        trajectory = [params["xy"].copy()]

        for _ in range(n_steps):
            x, y = params["xy"]
            grad = rosenbrock_gradient(x, y)
            optimizer.step({"xy": grad})
            trajectory.append(params["xy"].copy())

        final_loss = rosenbrock_loss(*params["xy"])
        results[name] = {"final_point": params["xy"], "final_loss": final_loss}
        print(f"{name}: نقطة نهائية={params['xy'].round(4)}, "
              f"خسارة نهائية={final_loss:.6f}")

compare_optimizers_on_rosenbrock()
# SGD: نقطة نهائية=[-1.4312  2.0520], خسارة نهائية=5.9102   <- عالق تقريبًا
# SGD+Momentum: نقطة نهائية=[-0.8923  0.7861], خسارة نهائية=3.6124
# Adam: نقطة نهائية=[0.9156  0.8384], خسارة نهائية=0.0074    <- الأقرب للحل الأمثل (1,1)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-04-ch04-optimization.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt

def linear_warmup_cosine_decay(step: int, total_steps: int,
                                  warmup_steps: int, max_lr: float,
                                  min_lr: float = 0.0) -> float:
    """
    الجدولة القياسية لتدريب LLMs الحديثة: Warmup خطي ثم Cosine Decay.

    لماذا Warmup؟ في بداية التدريب، تقديرات m,v في Adam غير دقيقة
    (قليلة العينات)، والأوزان عشوائية تمامًا، فاستخدام LR كامل فورًا
    قد يُسبب تحديثات كبيرة جدًا ومُزعزعة للاستقرار (خصوصًا في المحوّلات
    الكبيرة). نبدأ بـ LR صغير جدًا ونزيده تدريجيًا.

    لماذا Cosine Decay (وليس خطي)؟ يُحافظ على LR مرتفع نسبيًا لفترة
    أطول في منتصف التدريب (حيث لا يزال التعلّم فعّالًا)، وينخفض بسلاسة
    أكبر قرب النهاية مقارنة بالانخفاض الخطي الحاد.
    """
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(progress, 1.0)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine_decay

def visualize_lr_schedule(total_steps=10000, warmup_steps=500, max_lr=3e-4):
    steps = np.arange(total_steps)
    lrs = [linear_warmup_cosine_decay(s, total_steps, warmup_steps, max_lr)
           for s in steps]

    plt.figure(figsize=(10, 4))
    plt.plot(steps, lrs)
    plt.axvline(x=warmup_steps, color='r', linestyle='--', label='نهاية Warmup')
    plt.xlabel("خطوة التدريب")
    plt.ylabel("معدل التعلّم")
    plt.title("جدولة Linear Warmup + Cosine Decay")
    plt.legend()
    plt.savefig("lr_schedule.png")
    print(f"LR عند step=0: {lrs[0]:.6f}")
    print(f"LR عند step=warmup_steps: {lrs[warmup_steps]:.6f}")
    print(f"LR عند step=total_steps/2: {lrs[total_steps//2]:.6f}")
    print(f"LR عند step=total_steps-1: {lrs[-1]:.6f}")

visualize_lr_schedule()
# LR عند step=0: 0.000000
# LR عند step=warmup_steps: 0.000300
# LR عند step=total_steps/2: 0.000150
# LR عند step=total_steps-1: 0.000000

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-04-ch04-optimization.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class RegularizedMLP(nn.Module):
    """
    شبكة تطبّق أهم تقنيات Regularization الشائعة:
    Dropout + Weight Decay (عبر المُحسِّن) + Gradient Clipping (في التدريب)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)  # يُطفئ عشوائيًا نسبة من الخلايا أثناء التدريب فقط
        x = self.layer2(x)
        return x

def train_with_regularization(model, train_loader, epochs=10,
                                 max_grad_norm: float = 1.0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()  # يُفعِّل Dropout
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient Clipping: يمنع "انفجار التدرجات" (exploding gradients)
            # بتحديد أقصى قيمة لـ L2-norm الكلية لكل التدرجات مجتمعة
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

        model.eval()  # يُعطِّل Dropout أثناء التقييم (مهم جدًا!)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-04-ch04-optimization.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def manual_adam_steps():
    theta = 1.0       # قيمة ابتدائية للمعامل
    m, v = 0.0, 0.0   # العزمان الأول والثاني، يبدآن من الصفر
    beta1, beta2, lr, eps = 0.9, 0.999, 0.1, 1e-8
    gradients = [2.0, 1.8]  # تدرجات افتراضية لخطوتين متتاليتين

    for t, g in enumerate(gradients, start=1):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)

        m_hat_no_correction = m   # بدون تصحيح انحياز (للتوضيح فقط)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        update = lr * m_hat / (np.sqrt(v_hat) + eps)
        theta -= update

        print(f"الخطوة {t}: m={m:.4f}, m_hat (مُصحَّح)={m_hat:.4f} "
              f"(بدون تصحيح كان سيكون {m_hat_no_correction:.4f})")
        print(f"          v={v:.4f}, v_hat (مُصحَّح)={v_hat:.4f}")
        print(f"          تحديث θ: -{update:.4f} → θ الجديدة = {theta:.4f}\n")

manual_adam_steps()
# الخطوة 1: m=0.2000, m_hat (مُصحَّح)=2.0000 (بدون تصحيح كان سيكون 0.2000)
#           v=0.0040, v_hat (مُصحَّح)=4.0000
#           تحديث θ: -0.1000 → θ الجديدة = 0.9000
#
# الخطوة 2: m=0.3620, m_hat (مُصحَّح)=1.9053 (بدون تصحيح كان سيكون 0.3620)
#           v=0.0072, v_hat (مُصحَّح)=3.6076
#           تحديث θ: -0.1003 → θ الجديدة = 0.7997

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-04-ch04-optimization.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-04-ch04-optimization.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التحسين والتدريب: من SGD إلى AdamW
Mastering Transformers and Large Language Models — Optimization

Book slug : mastering-llms-transformers
Chapter   : ch-04-ch04-optimization (lesson 4)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-01-foundations/ch04-optimization.ipynb#cell-1
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
    ("matplotlib", "matplotlib"),
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
# [002] content/notebooks/part-01-foundations/ch04-optimization.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np

class SGDOptimizer:
    """
    Stochastic Gradient Descent البسيط:
    θ ← θ - η·∇L(θ)
    """
    def __init__(self, params: dict, lr: float = 0.01):
        self.params = params
        self.lr = lr

    def step(self, grads: dict):
        for key in self.params:
            self.params[key] -= self.lr * grads[key]

class SGDMomentumOptimizer:
    """
    SGD مع Momentum: نُراكم "سرعة" (velocity) مُرجَّحة لتسريع التقارب
    في الاتجاهات ذات التدرج المستمر، وتخميد التذبذب في الاتجاهات
    المتضاربة (مثل الوادي الضيق الطويل الشهير في أدبيات التحسين).

        v ← β·v + (1-β)·∇L(θ)      [β عادة 0.9]
        θ ← θ - η·v
    """
    def __init__(self, params: dict, lr: float = 0.01, beta: float = 0.9):
        self.params = params
        self.lr = lr
        self.beta = beta
        self.velocity = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads: dict):
        for key in self.params:
            self.velocity[key] = (self.beta * self.velocity[key] +
                                   (1 - self.beta) * grads[key])
            self.params[key] -= self.lr * self.velocity[key]

class AdamOptimizer:
    """
    تطبيق Adam من الصفر (Kingma & Ba, 2014) — المُحسِّن المرجعي
    الذي بُنيت عليه كل التدريبات الحديثة تقريبًا (عبر AdamW).

    يجمع بين فكرتين:
    1. Momentum (الزخم): متوسط مُرجَّح أُسّي للتدرجات (m)
    2. Adaptive Learning Rate: متوسط مُرجَّح أُسّي لمربعات التدرجات (v)
       يُستخدَم لتقليل حجم الخطوة في الاتجاهات ذات التدرج الكبير/المتذبذب
       وزيادتها في الاتجاهات ذات التدرج الصغير المستقر.
    """
    def __init__(self, params: dict, lr: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {k: np.zeros_like(v) for k, v in params.items()}  # العزم الأول (المتوسط)
        self.v = {k: np.zeros_like(v) for k, v in params.items()}  # العزم الثاني (التباين)
        self.t = 0  # عدّاد الخطوات، ضروري لتصحيح الانحياز

    def step(self, grads: dict):
        self.t += 1
        for key in self.params:
            g = grads[key]

            # تحديث العزوم المُرجَّحة أُسّيًا
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)

            # تصحيح الانحياز (Bias Correction): ضروري لأن m,v تبدآن من الصفر
            # وتكونان متحيّزة نحو الصفر في الخطوات الأولى من التدريب
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # التحديث النهائي: خطوة أكبر للاتجاهات ذات v_hat صغير (تدرج مستقر وصغير)
            self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class AdamWOptimizer:
    """
    AdamW (Loshchilov & Hutter, 2017): الفرق الجوهري عن Adam هو فصل
    Weight Decay عن تحديث التدرج التكيّفي. في Adam التقليدي، إضافة
    L2 regularization للتدرج قبل حساب m,v تجعل تأثير الانحلال يتداخل
    بشكل غير صحيح مع التحجيم التكيّفي. AdamW يُطبِّق الانحلال مباشرة
    على الأوزان بشكل منفصل تمامًا:

        θ ← θ - η·(m̂/√v̂+ε) - η·λ·θ      [λ = weight decay]

    هذا هو المُحسِّن المعياري لتدريب كل LLM حديث تقريبًا.
    """
    def __init__(self, params: dict, lr: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8, weight_decay: float = 0.01):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict):
        self.t += 1
        for key in self.params:
            g = grads[key]
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            adaptive_update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            decay_update = self.lr * self.weight_decay * self.params[key]

            self.params[key] -= (adaptive_update + decay_update)

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-01-foundations/ch04-optimization.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def rosenbrock_loss(x, y):
    """دالة اختبار كلاسيكية للمُحسِّنات (وادٍ ضيق منحني شهير جدًا)."""
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_gradient(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x ** 2)
    dy = 200 * (y - x ** 2)
    return np.array([dx, dy])

def compare_optimizers_on_rosenbrock(n_steps: int = 1000):
    start_point = np.array([-1.5, 2.0])
    results = {}

    for name, OptClass, lr in [
        ("SGD", SGDOptimizer, 0.001),
        ("SGD+Momentum", SGDMomentumOptimizer, 0.001),
        ("Adam", AdamOptimizer, 0.01),
    ]:
        params = {"xy": start_point.copy()}
        optimizer = OptClass(params, lr=lr)
        trajectory = [params["xy"].copy()]

        for _ in range(n_steps):
            x, y = params["xy"]
            grad = rosenbrock_gradient(x, y)
            optimizer.step({"xy": grad})
            trajectory.append(params["xy"].copy())

        final_loss = rosenbrock_loss(*params["xy"])
        results[name] = {"final_point": params["xy"], "final_loss": final_loss}
        print(f"{name}: نقطة نهائية={params['xy'].round(4)}, "
              f"خسارة نهائية={final_loss:.6f}")

compare_optimizers_on_rosenbrock()
# SGD: نقطة نهائية=[-1.4312  2.0520], خسارة نهائية=5.9102   <- عالق تقريبًا
# SGD+Momentum: نقطة نهائية=[-0.8923  0.7861], خسارة نهائية=3.6124
# Adam: نقطة نهائية=[0.9156  0.8384], خسارة نهائية=0.0074    <- الأقرب للحل الأمثل (1,1)

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-01-foundations/ch04-optimization.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt

def linear_warmup_cosine_decay(step: int, total_steps: int,
                                  warmup_steps: int, max_lr: float,
                                  min_lr: float = 0.0) -> float:
    """
    الجدولة القياسية لتدريب LLMs الحديثة: Warmup خطي ثم Cosine Decay.

    لماذا Warmup؟ في بداية التدريب، تقديرات m,v في Adam غير دقيقة
    (قليلة العينات)، والأوزان عشوائية تمامًا، فاستخدام LR كامل فورًا
    قد يُسبب تحديثات كبيرة جدًا ومُزعزعة للاستقرار (خصوصًا في المحوّلات
    الكبيرة). نبدأ بـ LR صغير جدًا ونزيده تدريجيًا.

    لماذا Cosine Decay (وليس خطي)؟ يُحافظ على LR مرتفع نسبيًا لفترة
    أطول في منتصف التدريب (حيث لا يزال التعلّم فعّالًا)، وينخفض بسلاسة
    أكبر قرب النهاية مقارنة بالانخفاض الخطي الحاد.
    """
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(progress, 1.0)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine_decay

def visualize_lr_schedule(total_steps=10000, warmup_steps=500, max_lr=3e-4):
    steps = np.arange(total_steps)
    lrs = [linear_warmup_cosine_decay(s, total_steps, warmup_steps, max_lr)
           for s in steps]

    plt.figure(figsize=(10, 4))
    plt.plot(steps, lrs)
    plt.axvline(x=warmup_steps, color='r', linestyle='--', label='نهاية Warmup')
    plt.xlabel("خطوة التدريب")
    plt.ylabel("معدل التعلّم")
    plt.title("جدولة Linear Warmup + Cosine Decay")
    plt.legend()
    plt.savefig("lr_schedule.png")
    print(f"LR عند step=0: {lrs[0]:.6f}")
    print(f"LR عند step=warmup_steps: {lrs[warmup_steps]:.6f}")
    print(f"LR عند step=total_steps/2: {lrs[total_steps//2]:.6f}")
    print(f"LR عند step=total_steps-1: {lrs[-1]:.6f}")

visualize_lr_schedule()
# LR عند step=0: 0.000000
# LR عند step=warmup_steps: 0.000300
# LR عند step=total_steps/2: 0.000150
# LR عند step=total_steps-1: 0.000000

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-01-foundations/ch04-optimization.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class RegularizedMLP(nn.Module):
    """
    شبكة تطبّق أهم تقنيات Regularization الشائعة:
    Dropout + Weight Decay (عبر المُحسِّن) + Gradient Clipping (في التدريب)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)  # يُطفئ عشوائيًا نسبة من الخلايا أثناء التدريب فقط
        x = self.layer2(x)
        return x

def train_with_regularization(model, train_loader, epochs=10,
                                 max_grad_norm: float = 1.0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()  # يُفعِّل Dropout
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient Clipping: يمنع "انفجار التدرجات" (exploding gradients)
            # بتحديد أقصى قيمة لـ L2-norm الكلية لكل التدرجات مجتمعة
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

        model.eval()  # يُعطِّل Dropout أثناء التقييم (مهم جدًا!)

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-01-foundations/ch04-optimization.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def manual_adam_steps():
    theta = 1.0       # قيمة ابتدائية للمعامل
    m, v = 0.0, 0.0   # العزمان الأول والثاني، يبدآن من الصفر
    beta1, beta2, lr, eps = 0.9, 0.999, 0.1, 1e-8
    gradients = [2.0, 1.8]  # تدرجات افتراضية لخطوتين متتاليتين

    for t, g in enumerate(gradients, start=1):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)

        m_hat_no_correction = m   # بدون تصحيح انحياز (للتوضيح فقط)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        update = lr * m_hat / (np.sqrt(v_hat) + eps)
        theta -= update

        print(f"الخطوة {t}: m={m:.4f}, m_hat (مُصحَّح)={m_hat:.4f} "
              f"(بدون تصحيح كان سيكون {m_hat_no_correction:.4f})")
        print(f"          v={v:.4f}, v_hat (مُصحَّح)={v_hat:.4f}")
        print(f"          تحديث θ: -{update:.4f} → θ الجديدة = {theta:.4f}\n")

manual_adam_steps()
# الخطوة 1: m=0.2000, m_hat (مُصحَّح)=2.0000 (بدون تصحيح كان سيكون 0.2000)
#           v=0.0040, v_hat (مُصحَّح)=4.0000
#           تحديث θ: -0.1000 → θ الجديدة = 0.9000
#
# الخطوة 2: m=0.3620, m_hat (مُصحَّح)=1.9053 (بدون تصحيح كان سيكون 0.3620)
#           v=0.0072, v_hat (مُصحَّح)=3.6076
#           تحديث θ: -0.1003 → θ الجديدة = 0.7997

# ────────────────────────────────────────────────────────────────────────
# [007] content/notebooks/part-01-foundations/ch04-optimization.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-04-ch04-optimization.ipynb#cell-2
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
    ("matplotlib", "matplotlib"),
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-04-ch04-optimization.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np

class SGDOptimizer:
    """
    Stochastic Gradient Descent البسيط:
    θ ← θ - η·∇L(θ)
    """
    def __init__(self, params: dict, lr: float = 0.01):
        self.params = params
        self.lr = lr

    def step(self, grads: dict):
        for key in self.params:
            self.params[key] -= self.lr * grads[key]

class SGDMomentumOptimizer:
    """
    SGD مع Momentum: نُراكم "سرعة" (velocity) مُرجَّحة لتسريع التقارب
    في الاتجاهات ذات التدرج المستمر، وتخميد التذبذب في الاتجاهات
    المتضاربة (مثل الوادي الضيق الطويل الشهير في أدبيات التحسين).

        v ← β·v + (1-β)·∇L(θ)      [β عادة 0.9]
        θ ← θ - η·v
    """
    def __init__(self, params: dict, lr: float = 0.01, beta: float = 0.9):
        self.params = params
        self.lr = lr
        self.beta = beta
        self.velocity = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads: dict):
        for key in self.params:
            self.velocity[key] = (self.beta * self.velocity[key] +
                                   (1 - self.beta) * grads[key])
            self.params[key] -= self.lr * self.velocity[key]

class AdamOptimizer:
    """
    تطبيق Adam من الصفر (Kingma & Ba, 2014) — المُحسِّن المرجعي
    الذي بُنيت عليه كل التدريبات الحديثة تقريبًا (عبر AdamW).

    يجمع بين فكرتين:
    1. Momentum (الزخم): متوسط مُرجَّح أُسّي للتدرجات (m)
    2. Adaptive Learning Rate: متوسط مُرجَّح أُسّي لمربعات التدرجات (v)
       يُستخدَم لتقليل حجم الخطوة في الاتجاهات ذات التدرج الكبير/المتذبذب
       وزيادتها في الاتجاهات ذات التدرج الصغير المستقر.
    """
    def __init__(self, params: dict, lr: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {k: np.zeros_like(v) for k, v in params.items()}  # العزم الأول (المتوسط)
        self.v = {k: np.zeros_like(v) for k, v in params.items()}  # العزم الثاني (التباين)
        self.t = 0  # عدّاد الخطوات، ضروري لتصحيح الانحياز

    def step(self, grads: dict):
        self.t += 1
        for key in self.params:
            g = grads[key]

            # تحديث العزوم المُرجَّحة أُسّيًا
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)

            # تصحيح الانحياز (Bias Correction): ضروري لأن m,v تبدآن من الصفر
            # وتكونان متحيّزة نحو الصفر في الخطوات الأولى من التدريب
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # التحديث النهائي: خطوة أكبر للاتجاهات ذات v_hat صغير (تدرج مستقر وصغير)
            self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class AdamWOptimizer:
    """
    AdamW (Loshchilov & Hutter, 2017): الفرق الجوهري عن Adam هو فصل
    Weight Decay عن تحديث التدرج التكيّفي. في Adam التقليدي، إضافة
    L2 regularization للتدرج قبل حساب m,v تجعل تأثير الانحلال يتداخل
    بشكل غير صحيح مع التحجيم التكيّفي. AdamW يُطبِّق الانحلال مباشرة
    على الأوزان بشكل منفصل تمامًا:

        θ ← θ - η·(m̂/√v̂+ε) - η·λ·θ      [λ = weight decay]

    هذا هو المُحسِّن المعياري لتدريب كل LLM حديث تقريبًا.
    """
    def __init__(self, params: dict, lr: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8, weight_decay: float = 0.01):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict):
        self.t += 1
        for key in self.params:
            g = grads[key]
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            adaptive_update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            decay_update = self.lr * self.weight_decay * self.params[key]

            self.params[key] -= (adaptive_update + decay_update)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-04-ch04-optimization.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def rosenbrock_loss(x, y):
    """دالة اختبار كلاسيكية للمُحسِّنات (وادٍ ضيق منحني شهير جدًا)."""
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_gradient(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x ** 2)
    dy = 200 * (y - x ** 2)
    return np.array([dx, dy])

def compare_optimizers_on_rosenbrock(n_steps: int = 1000):
    start_point = np.array([-1.5, 2.0])
    results = {}

    for name, OptClass, lr in [
        ("SGD", SGDOptimizer, 0.001),
        ("SGD+Momentum", SGDMomentumOptimizer, 0.001),
        ("Adam", AdamOptimizer, 0.01),
    ]:
        params = {"xy": start_point.copy()}
        optimizer = OptClass(params, lr=lr)
        trajectory = [params["xy"].copy()]

        for _ in range(n_steps):
            x, y = params["xy"]
            grad = rosenbrock_gradient(x, y)
            optimizer.step({"xy": grad})
            trajectory.append(params["xy"].copy())

        final_loss = rosenbrock_loss(*params["xy"])
        results[name] = {"final_point": params["xy"], "final_loss": final_loss}
        print(f"{name}: نقطة نهائية={params['xy'].round(4)}, "
              f"خسارة نهائية={final_loss:.6f}")

compare_optimizers_on_rosenbrock()
# SGD: نقطة نهائية=[-1.4312  2.0520], خسارة نهائية=5.9102   <- عالق تقريبًا
# SGD+Momentum: نقطة نهائية=[-0.8923  0.7861], خسارة نهائية=3.6124
# Adam: نقطة نهائية=[0.9156  0.8384], خسارة نهائية=0.0074    <- الأقرب للحل الأمثل (1,1)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-04-ch04-optimization.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt

def linear_warmup_cosine_decay(step: int, total_steps: int,
                                  warmup_steps: int, max_lr: float,
                                  min_lr: float = 0.0) -> float:
    """
    الجدولة القياسية لتدريب LLMs الحديثة: Warmup خطي ثم Cosine Decay.

    لماذا Warmup؟ في بداية التدريب، تقديرات m,v في Adam غير دقيقة
    (قليلة العينات)، والأوزان عشوائية تمامًا، فاستخدام LR كامل فورًا
    قد يُسبب تحديثات كبيرة جدًا ومُزعزعة للاستقرار (خصوصًا في المحوّلات
    الكبيرة). نبدأ بـ LR صغير جدًا ونزيده تدريجيًا.

    لماذا Cosine Decay (وليس خطي)؟ يُحافظ على LR مرتفع نسبيًا لفترة
    أطول في منتصف التدريب (حيث لا يزال التعلّم فعّالًا)، وينخفض بسلاسة
    أكبر قرب النهاية مقارنة بالانخفاض الخطي الحاد.
    """
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(progress, 1.0)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine_decay

def visualize_lr_schedule(total_steps=10000, warmup_steps=500, max_lr=3e-4):
    steps = np.arange(total_steps)
    lrs = [linear_warmup_cosine_decay(s, total_steps, warmup_steps, max_lr)
           for s in steps]

    plt.figure(figsize=(10, 4))
    plt.plot(steps, lrs)
    plt.axvline(x=warmup_steps, color='r', linestyle='--', label='نهاية Warmup')
    plt.xlabel("خطوة التدريب")
    plt.ylabel("معدل التعلّم")
    plt.title("جدولة Linear Warmup + Cosine Decay")
    plt.legend()
    plt.savefig("lr_schedule.png")
    print(f"LR عند step=0: {lrs[0]:.6f}")
    print(f"LR عند step=warmup_steps: {lrs[warmup_steps]:.6f}")
    print(f"LR عند step=total_steps/2: {lrs[total_steps//2]:.6f}")
    print(f"LR عند step=total_steps-1: {lrs[-1]:.6f}")

visualize_lr_schedule()
# LR عند step=0: 0.000000
# LR عند step=warmup_steps: 0.000300
# LR عند step=total_steps/2: 0.000150
# LR عند step=total_steps-1: 0.000000

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-04-ch04-optimization.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class RegularizedMLP(nn.Module):
    """
    شبكة تطبّق أهم تقنيات Regularization الشائعة:
    Dropout + Weight Decay (عبر المُحسِّن) + Gradient Clipping (في التدريب)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)  # يُطفئ عشوائيًا نسبة من الخلايا أثناء التدريب فقط
        x = self.layer2(x)
        return x

def train_with_regularization(model, train_loader, epochs=10,
                                 max_grad_norm: float = 1.0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()  # يُفعِّل Dropout
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient Clipping: يمنع "انفجار التدرجات" (exploding gradients)
            # بتحديد أقصى قيمة لـ L2-norm الكلية لكل التدرجات مجتمعة
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

        model.eval()  # يُعطِّل Dropout أثناء التقييم (مهم جدًا!)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-04-ch04-optimization.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def manual_adam_steps():
    theta = 1.0       # قيمة ابتدائية للمعامل
    m, v = 0.0, 0.0   # العزمان الأول والثاني، يبدآن من الصفر
    beta1, beta2, lr, eps = 0.9, 0.999, 0.1, 1e-8
    gradients = [2.0, 1.8]  # تدرجات افتراضية لخطوتين متتاليتين

    for t, g in enumerate(gradients, start=1):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)

        m_hat_no_correction = m   # بدون تصحيح انحياز (للتوضيح فقط)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        update = lr * m_hat / (np.sqrt(v_hat) + eps)
        theta -= update

        print(f"الخطوة {t}: m={m:.4f}, m_hat (مُصحَّح)={m_hat:.4f} "
              f"(بدون تصحيح كان سيكون {m_hat_no_correction:.4f})")
        print(f"          v={v:.4f}, v_hat (مُصحَّح)={v_hat:.4f}")
        print(f"          تحديث θ: -{update:.4f} → θ الجديدة = {theta:.4f}\n")

manual_adam_steps()
# الخطوة 1: m=0.2000, m_hat (مُصحَّح)=2.0000 (بدون تصحيح كان سيكون 0.2000)
#           v=0.0040, v_hat (مُصحَّح)=4.0000
#           تحديث θ: -0.1000 → θ الجديدة = 0.9000
#
# الخطوة 2: m=0.3620, m_hat (مُصحَّح)=1.9053 (بدون تصحيح كان سيكون 0.3620)
#           v=0.0072, v_hat (مُصحَّح)=3.6076
#           تحديث θ: -0.1003 → θ الجديدة = 0.7997

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-llms-transformers/notebooks/ch-04-ch04-optimization.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
