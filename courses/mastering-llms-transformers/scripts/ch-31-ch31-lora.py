#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — LoRA: التكيّف منخفض الرتبة بعمق
Mastering Transformers and Large Language Models — Lora

Book slug : mastering-llms-transformers
Chapter   : ch-31-ch31-lora (lesson 31)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-31-ch31-lora.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — LoRA: التكيّف منخفض الرتبة بعمق
Mastering Transformers and Large Language Models — Lora

Book slug : mastering-llms-transformers
Chapter   : ch-31-ch31-lora (lesson 31)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-31-ch31-lora.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — LoRA: التكيّف منخفض الرتبة بعمق
Mastering Transformers and Large Language Models — Lora

Book slug : mastering-llms-transformers
Chapter   : ch-31-ch31-lora (lesson 31)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-07-peft-alignment/ch31-lora.ipynb#cell-1
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
# [002] content/notebooks/part-07-peft-alignment/ch31-lora.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    """
    تطبيق LoRA من الصفر طبقًا لورقة Hu et al. (2021).
    يُستخدم كطبقة "إضافية" بجانب أي nn.Linear مجمّدة.
    """
    def __init__(self, in_features: int, out_features: int,
                 rank: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # معامل التحجيم α/r

        # A: تهيئة Kaiming uniform (كما في الورقة الأصلية)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # B: تهيئة بالصفر (حرج جدًا لضمان ΔW=0 في البداية)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        x = self.dropout(x)
        # ΔW·x = B·A·x ، لاحظ ترتيب العمليات لتوفير الحساب: (x·Aᵀ)·Bᵀ
        result = x @ self.lora_A.T  # (..., rank)
        result = result @ self.lora_B.T  # (..., out_features)
        return result * self.scaling

class LinearWithLoRA(nn.Module):
    """
    يُغلّف nn.Linear أصلية (مجمّدة) بطبقة LoRA مُضافة.
    هذا هو النمط الفعلي المُستخدَم في مكتبة PEFT من Hugging Face.
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 8,
                 alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base_linear = base_linear
        # تجميد الطبقة الأصلية بالكامل — هذا هو جوهر LoRA
        for param in self.base_linear.parameters():
            param.requires_grad = False

        self.lora = LoRALayer(
            in_features=base_linear.in_features,
            out_features=base_linear.out_features,
            rank=rank, alpha=alpha, dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)       # W₀·x  (بدون تدرجات)
        lora_out = self.lora(x)               # (α/r)·B·A·x  (بتدرجات)
        return base_out + lora_out

    def merge_weights(self) -> nn.Linear:
        """
        دمج أوزان LoRA في الطبقة الأصلية لإنتاج طبقة خطية واحدة
        بدون أي حمل حسابي إضافي أثناء الاستدلال (inference overhead = صفر).
        هذه هي الميزة الحاسمة لـ LoRA مقارنة بـ Adapter Layers.
        """
        merged = nn.Linear(self.base_linear.in_features,
                            self.base_linear.out_features,
                            bias=self.base_linear.bias is not None)
        with torch.no_grad():
            delta_w = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
            merged.weight.copy_(self.base_linear.weight + delta_w)
            if self.base_linear.bias is not None:
                merged.bias.copy_(self.base_linear.bias)
        return merged

def apply_lora_to_model(model: nn.Module, target_modules: list,
                         rank: int = 8, alpha: int = 16,
                         dropout: float = 0.05) -> nn.Module:
    """
    تطبيق LoRA تلقائيًا على كل الطبقات التي تطابق أسماء target_modules
    (نمط شائع: ["q_proj", "k_proj", "v_proj", "o_proj"] لنماذج Llama).
    """
    for name, module in model.named_modules():
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child_name,
                        LinearWithLoRA(module, rank=rank, alpha=alpha, dropout=dropout))

    # تجميد كل ما تبقى من النموذج (embeddings, norms, إلخ) باستثناء LoRA
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA مُطبّق: {trainable:,} / {total:,} معامل قابل للتدريب "
          f"({100*trainable/total:.4f}%)")
    return model

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-07-peft-alignment/ch31-lora.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def recommend_lora_config(task_complexity: str, model_size_billions: float):
    """
    توصيات عملية لاختيار r وα بناءً على تعقيد المهمة وحجم النموذج،
    مُستخلصة من الممارسات الشائعة في المجتمع (Hugging Face, Unsloth).
    """
    configs = {
        "simple_classification": {"r": 4, "alpha": 8},
        "domain_adaptation": {"r": 16, "alpha": 32},
        "complex_instruction_following": {"r": 32, "alpha": 64},
        "full_capability_transfer": {"r": 64, "alpha": 128},
    }
    config = configs.get(task_complexity, {"r": 16, "alpha": 32})

    # قاعدة عملية شائعة: alpha = 2×r كبداية، ثم ضبط تجريبي
    print(f"التوصية لمهمة '{task_complexity}' على نموذج {model_size_billions}B:")
    print(f"  rank (r) = {config['r']}")
    print(f"  alpha (α) = {config['alpha']}")
    print(f"  نسبة α/r = {config['alpha']/config['r']}")
    return config

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-07-peft-alignment/ch31-lora.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)

# إنشاء الطبقة الأصلية (تمثيل مبسّط لـ q_proj في محوّل)
base_layer = nn.Linear(4096, 4096, bias=False)
lora_layer = LinearWithLoRA(base_layer, rank=8, alpha=16)

# تدريب تجريبي قصير: تعليم LoRA لتقريب تحويل مستهدف معيّن
target_transform = nn.Linear(4096, 4096, bias=False)
optimizer = torch.optim.AdamW(
    [p for p in lora_layer.parameters() if p.requires_grad], lr=1e-3
)

x_sample = torch.randn(32, 4096)
for step in range(200):
    optimizer.zero_grad()
    pred = lora_layer(x_sample)
    target = target_transform(x_sample)
    loss = nn.functional.mse_loss(pred, target)
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(f"Step {step}: loss = {loss.item():.6f}")

# التحقق من تطابق الدمج
merged_layer = lora_layer.merge_weights()
with torch.no_grad():
    out_unmerged = lora_layer(x_sample)
    out_merged = merged_layer(x_sample)
    max_diff = (out_unmerged - out_merged).abs().max().item()
    print(f"\nأقصى فرق بين النموذج قبل/بعد الدمج: {max_diff:.2e}")
    # يجب أن يكون قريبًا جدًا من الصفر (خطأ عددي فقط)

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-07-peft-alignment/ch31-lora.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-31-ch31-lora.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-31-ch31-lora.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    """
    تطبيق LoRA من الصفر طبقًا لورقة Hu et al. (2021).
    يُستخدم كطبقة "إضافية" بجانب أي nn.Linear مجمّدة.
    """
    def __init__(self, in_features: int, out_features: int,
                 rank: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # معامل التحجيم α/r

        # A: تهيئة Kaiming uniform (كما في الورقة الأصلية)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # B: تهيئة بالصفر (حرج جدًا لضمان ΔW=0 في البداية)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        x = self.dropout(x)
        # ΔW·x = B·A·x ، لاحظ ترتيب العمليات لتوفير الحساب: (x·Aᵀ)·Bᵀ
        result = x @ self.lora_A.T  # (..., rank)
        result = result @ self.lora_B.T  # (..., out_features)
        return result * self.scaling

class LinearWithLoRA(nn.Module):
    """
    يُغلّف nn.Linear أصلية (مجمّدة) بطبقة LoRA مُضافة.
    هذا هو النمط الفعلي المُستخدَم في مكتبة PEFT من Hugging Face.
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 8,
                 alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base_linear = base_linear
        # تجميد الطبقة الأصلية بالكامل — هذا هو جوهر LoRA
        for param in self.base_linear.parameters():
            param.requires_grad = False

        self.lora = LoRALayer(
            in_features=base_linear.in_features,
            out_features=base_linear.out_features,
            rank=rank, alpha=alpha, dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)       # W₀·x  (بدون تدرجات)
        lora_out = self.lora(x)               # (α/r)·B·A·x  (بتدرجات)
        return base_out + lora_out

    def merge_weights(self) -> nn.Linear:
        """
        دمج أوزان LoRA في الطبقة الأصلية لإنتاج طبقة خطية واحدة
        بدون أي حمل حسابي إضافي أثناء الاستدلال (inference overhead = صفر).
        هذه هي الميزة الحاسمة لـ LoRA مقارنة بـ Adapter Layers.
        """
        merged = nn.Linear(self.base_linear.in_features,
                            self.base_linear.out_features,
                            bias=self.base_linear.bias is not None)
        with torch.no_grad():
            delta_w = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
            merged.weight.copy_(self.base_linear.weight + delta_w)
            if self.base_linear.bias is not None:
                merged.bias.copy_(self.base_linear.bias)
        return merged

def apply_lora_to_model(model: nn.Module, target_modules: list,
                         rank: int = 8, alpha: int = 16,
                         dropout: float = 0.05) -> nn.Module:
    """
    تطبيق LoRA تلقائيًا على كل الطبقات التي تطابق أسماء target_modules
    (نمط شائع: ["q_proj", "k_proj", "v_proj", "o_proj"] لنماذج Llama).
    """
    for name, module in model.named_modules():
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child_name,
                        LinearWithLoRA(module, rank=rank, alpha=alpha, dropout=dropout))

    # تجميد كل ما تبقى من النموذج (embeddings, norms, إلخ) باستثناء LoRA
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA مُطبّق: {trainable:,} / {total:,} معامل قابل للتدريب "
          f"({100*trainable/total:.4f}%)")
    return model

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-31-ch31-lora.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def recommend_lora_config(task_complexity: str, model_size_billions: float):
    """
    توصيات عملية لاختيار r وα بناءً على تعقيد المهمة وحجم النموذج،
    مُستخلصة من الممارسات الشائعة في المجتمع (Hugging Face, Unsloth).
    """
    configs = {
        "simple_classification": {"r": 4, "alpha": 8},
        "domain_adaptation": {"r": 16, "alpha": 32},
        "complex_instruction_following": {"r": 32, "alpha": 64},
        "full_capability_transfer": {"r": 64, "alpha": 128},
    }
    config = configs.get(task_complexity, {"r": 16, "alpha": 32})

    # قاعدة عملية شائعة: alpha = 2×r كبداية، ثم ضبط تجريبي
    print(f"التوصية لمهمة '{task_complexity}' على نموذج {model_size_billions}B:")
    print(f"  rank (r) = {config['r']}")
    print(f"  alpha (α) = {config['alpha']}")
    print(f"  نسبة α/r = {config['alpha']/config['r']}")
    return config

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-31-ch31-lora.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)

# إنشاء الطبقة الأصلية (تمثيل مبسّط لـ q_proj في محوّل)
base_layer = nn.Linear(4096, 4096, bias=False)
lora_layer = LinearWithLoRA(base_layer, rank=8, alpha=16)

# تدريب تجريبي قصير: تعليم LoRA لتقريب تحويل مستهدف معيّن
target_transform = nn.Linear(4096, 4096, bias=False)
optimizer = torch.optim.AdamW(
    [p for p in lora_layer.parameters() if p.requires_grad], lr=1e-3
)

x_sample = torch.randn(32, 4096)
for step in range(200):
    optimizer.zero_grad()
    pred = lora_layer(x_sample)
    target = target_transform(x_sample)
    loss = nn.functional.mse_loss(pred, target)
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(f"Step {step}: loss = {loss.item():.6f}")

# التحقق من تطابق الدمج
merged_layer = lora_layer.merge_weights()
with torch.no_grad():
    out_unmerged = lora_layer(x_sample)
    out_merged = merged_layer(x_sample)
    max_diff = (out_unmerged - out_merged).abs().max().item()
    print(f"\nأقصى فرق بين النموذج قبل/بعد الدمج: {max_diff:.2e}")
    # يجب أن يكون قريبًا جدًا من الصفر (خطأ عددي فقط)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-31-ch31-lora.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-31-ch31-lora.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — LoRA: التكيّف منخفض الرتبة بعمق
Mastering Transformers and Large Language Models — Lora

Book slug : mastering-llms-transformers
Chapter   : ch-31-ch31-lora (lesson 31)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-07-peft-alignment/ch31-lora.ipynb#cell-1
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
# [002] content/notebooks/part-07-peft-alignment/ch31-lora.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    """
    تطبيق LoRA من الصفر طبقًا لورقة Hu et al. (2021).
    يُستخدم كطبقة "إضافية" بجانب أي nn.Linear مجمّدة.
    """
    def __init__(self, in_features: int, out_features: int,
                 rank: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # معامل التحجيم α/r

        # A: تهيئة Kaiming uniform (كما في الورقة الأصلية)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # B: تهيئة بالصفر (حرج جدًا لضمان ΔW=0 في البداية)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        x = self.dropout(x)
        # ΔW·x = B·A·x ، لاحظ ترتيب العمليات لتوفير الحساب: (x·Aᵀ)·Bᵀ
        result = x @ self.lora_A.T  # (..., rank)
        result = result @ self.lora_B.T  # (..., out_features)
        return result * self.scaling

class LinearWithLoRA(nn.Module):
    """
    يُغلّف nn.Linear أصلية (مجمّدة) بطبقة LoRA مُضافة.
    هذا هو النمط الفعلي المُستخدَم في مكتبة PEFT من Hugging Face.
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 8,
                 alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base_linear = base_linear
        # تجميد الطبقة الأصلية بالكامل — هذا هو جوهر LoRA
        for param in self.base_linear.parameters():
            param.requires_grad = False

        self.lora = LoRALayer(
            in_features=base_linear.in_features,
            out_features=base_linear.out_features,
            rank=rank, alpha=alpha, dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)       # W₀·x  (بدون تدرجات)
        lora_out = self.lora(x)               # (α/r)·B·A·x  (بتدرجات)
        return base_out + lora_out

    def merge_weights(self) -> nn.Linear:
        """
        دمج أوزان LoRA في الطبقة الأصلية لإنتاج طبقة خطية واحدة
        بدون أي حمل حسابي إضافي أثناء الاستدلال (inference overhead = صفر).
        هذه هي الميزة الحاسمة لـ LoRA مقارنة بـ Adapter Layers.
        """
        merged = nn.Linear(self.base_linear.in_features,
                            self.base_linear.out_features,
                            bias=self.base_linear.bias is not None)
        with torch.no_grad():
            delta_w = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
            merged.weight.copy_(self.base_linear.weight + delta_w)
            if self.base_linear.bias is not None:
                merged.bias.copy_(self.base_linear.bias)
        return merged

def apply_lora_to_model(model: nn.Module, target_modules: list,
                         rank: int = 8, alpha: int = 16,
                         dropout: float = 0.05) -> nn.Module:
    """
    تطبيق LoRA تلقائيًا على كل الطبقات التي تطابق أسماء target_modules
    (نمط شائع: ["q_proj", "k_proj", "v_proj", "o_proj"] لنماذج Llama).
    """
    for name, module in model.named_modules():
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child_name,
                        LinearWithLoRA(module, rank=rank, alpha=alpha, dropout=dropout))

    # تجميد كل ما تبقى من النموذج (embeddings, norms, إلخ) باستثناء LoRA
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA مُطبّق: {trainable:,} / {total:,} معامل قابل للتدريب "
          f"({100*trainable/total:.4f}%)")
    return model

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-07-peft-alignment/ch31-lora.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def recommend_lora_config(task_complexity: str, model_size_billions: float):
    """
    توصيات عملية لاختيار r وα بناءً على تعقيد المهمة وحجم النموذج،
    مُستخلصة من الممارسات الشائعة في المجتمع (Hugging Face, Unsloth).
    """
    configs = {
        "simple_classification": {"r": 4, "alpha": 8},
        "domain_adaptation": {"r": 16, "alpha": 32},
        "complex_instruction_following": {"r": 32, "alpha": 64},
        "full_capability_transfer": {"r": 64, "alpha": 128},
    }
    config = configs.get(task_complexity, {"r": 16, "alpha": 32})

    # قاعدة عملية شائعة: alpha = 2×r كبداية، ثم ضبط تجريبي
    print(f"التوصية لمهمة '{task_complexity}' على نموذج {model_size_billions}B:")
    print(f"  rank (r) = {config['r']}")
    print(f"  alpha (α) = {config['alpha']}")
    print(f"  نسبة α/r = {config['alpha']/config['r']}")
    return config

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-07-peft-alignment/ch31-lora.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)

# إنشاء الطبقة الأصلية (تمثيل مبسّط لـ q_proj في محوّل)
base_layer = nn.Linear(4096, 4096, bias=False)
lora_layer = LinearWithLoRA(base_layer, rank=8, alpha=16)

# تدريب تجريبي قصير: تعليم LoRA لتقريب تحويل مستهدف معيّن
target_transform = nn.Linear(4096, 4096, bias=False)
optimizer = torch.optim.AdamW(
    [p for p in lora_layer.parameters() if p.requires_grad], lr=1e-3
)

x_sample = torch.randn(32, 4096)
for step in range(200):
    optimizer.zero_grad()
    pred = lora_layer(x_sample)
    target = target_transform(x_sample)
    loss = nn.functional.mse_loss(pred, target)
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(f"Step {step}: loss = {loss.item():.6f}")

# التحقق من تطابق الدمج
merged_layer = lora_layer.merge_weights()
with torch.no_grad():
    out_unmerged = lora_layer(x_sample)
    out_merged = merged_layer(x_sample)
    max_diff = (out_unmerged - out_merged).abs().max().item()
    print(f"\nأقصى فرق بين النموذج قبل/بعد الدمج: {max_diff:.2e}")
    # يجب أن يكون قريبًا جدًا من الصفر (خطأ عددي فقط)

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-07-peft-alignment/ch31-lora.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-31-ch31-lora.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-31-ch31-lora.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    """
    تطبيق LoRA من الصفر طبقًا لورقة Hu et al. (2021).
    يُستخدم كطبقة "إضافية" بجانب أي nn.Linear مجمّدة.
    """
    def __init__(self, in_features: int, out_features: int,
                 rank: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # معامل التحجيم α/r

        # A: تهيئة Kaiming uniform (كما في الورقة الأصلية)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # B: تهيئة بالصفر (حرج جدًا لضمان ΔW=0 في البداية)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        x = self.dropout(x)
        # ΔW·x = B·A·x ، لاحظ ترتيب العمليات لتوفير الحساب: (x·Aᵀ)·Bᵀ
        result = x @ self.lora_A.T  # (..., rank)
        result = result @ self.lora_B.T  # (..., out_features)
        return result * self.scaling

class LinearWithLoRA(nn.Module):
    """
    يُغلّف nn.Linear أصلية (مجمّدة) بطبقة LoRA مُضافة.
    هذا هو النمط الفعلي المُستخدَم في مكتبة PEFT من Hugging Face.
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 8,
                 alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base_linear = base_linear
        # تجميد الطبقة الأصلية بالكامل — هذا هو جوهر LoRA
        for param in self.base_linear.parameters():
            param.requires_grad = False

        self.lora = LoRALayer(
            in_features=base_linear.in_features,
            out_features=base_linear.out_features,
            rank=rank, alpha=alpha, dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)       # W₀·x  (بدون تدرجات)
        lora_out = self.lora(x)               # (α/r)·B·A·x  (بتدرجات)
        return base_out + lora_out

    def merge_weights(self) -> nn.Linear:
        """
        دمج أوزان LoRA في الطبقة الأصلية لإنتاج طبقة خطية واحدة
        بدون أي حمل حسابي إضافي أثناء الاستدلال (inference overhead = صفر).
        هذه هي الميزة الحاسمة لـ LoRA مقارنة بـ Adapter Layers.
        """
        merged = nn.Linear(self.base_linear.in_features,
                            self.base_linear.out_features,
                            bias=self.base_linear.bias is not None)
        with torch.no_grad():
            delta_w = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
            merged.weight.copy_(self.base_linear.weight + delta_w)
            if self.base_linear.bias is not None:
                merged.bias.copy_(self.base_linear.bias)
        return merged

def apply_lora_to_model(model: nn.Module, target_modules: list,
                         rank: int = 8, alpha: int = 16,
                         dropout: float = 0.05) -> nn.Module:
    """
    تطبيق LoRA تلقائيًا على كل الطبقات التي تطابق أسماء target_modules
    (نمط شائع: ["q_proj", "k_proj", "v_proj", "o_proj"] لنماذج Llama).
    """
    for name, module in model.named_modules():
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child_name,
                        LinearWithLoRA(module, rank=rank, alpha=alpha, dropout=dropout))

    # تجميد كل ما تبقى من النموذج (embeddings, norms, إلخ) باستثناء LoRA
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA مُطبّق: {trainable:,} / {total:,} معامل قابل للتدريب "
          f"({100*trainable/total:.4f}%)")
    return model

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-31-ch31-lora.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def recommend_lora_config(task_complexity: str, model_size_billions: float):
    """
    توصيات عملية لاختيار r وα بناءً على تعقيد المهمة وحجم النموذج،
    مُستخلصة من الممارسات الشائعة في المجتمع (Hugging Face, Unsloth).
    """
    configs = {
        "simple_classification": {"r": 4, "alpha": 8},
        "domain_adaptation": {"r": 16, "alpha": 32},
        "complex_instruction_following": {"r": 32, "alpha": 64},
        "full_capability_transfer": {"r": 64, "alpha": 128},
    }
    config = configs.get(task_complexity, {"r": 16, "alpha": 32})

    # قاعدة عملية شائعة: alpha = 2×r كبداية، ثم ضبط تجريبي
    print(f"التوصية لمهمة '{task_complexity}' على نموذج {model_size_billions}B:")
    print(f"  rank (r) = {config['r']}")
    print(f"  alpha (α) = {config['alpha']}")
    print(f"  نسبة α/r = {config['alpha']/config['r']}")
    return config

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-31-ch31-lora.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)

# إنشاء الطبقة الأصلية (تمثيل مبسّط لـ q_proj في محوّل)
base_layer = nn.Linear(4096, 4096, bias=False)
lora_layer = LinearWithLoRA(base_layer, rank=8, alpha=16)

# تدريب تجريبي قصير: تعليم LoRA لتقريب تحويل مستهدف معيّن
target_transform = nn.Linear(4096, 4096, bias=False)
optimizer = torch.optim.AdamW(
    [p for p in lora_layer.parameters() if p.requires_grad], lr=1e-3
)

x_sample = torch.randn(32, 4096)
for step in range(200):
    optimizer.zero_grad()
    pred = lora_layer(x_sample)
    target = target_transform(x_sample)
    loss = nn.functional.mse_loss(pred, target)
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(f"Step {step}: loss = {loss.item():.6f}")

# التحقق من تطابق الدمج
merged_layer = lora_layer.merge_weights()
with torch.no_grad():
    out_unmerged = lora_layer(x_sample)
    out_merged = merged_layer(x_sample)
    max_diff = (out_unmerged - out_merged).abs().max().item()
    print(f"\nأقصى فرق بين النموذج قبل/بعد الدمج: {max_diff:.2e}")
    # يجب أن يكون قريبًا جدًا من الصفر (خطأ عددي فقط)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-31-ch31-lora.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
