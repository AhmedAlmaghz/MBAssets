#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — إعداد البيئة والبنية التحتية للتطوير
Mastering Transformers and Large Language Models — Environment

Book slug : mastering-llms-transformers
Chapter   : ch-01-ch01-environment (lesson 1)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-01-ch01-environment.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — إعداد البيئة والبنية التحتية للتطوير
Mastering Transformers and Large Language Models — Environment

Book slug : mastering-llms-transformers
Chapter   : ch-01-ch01-environment (lesson 1)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-01-ch01-environment.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — إعداد البيئة والبنية التحتية للتطوير
Mastering Transformers and Large Language Models — Environment

Book slug : mastering-llms-transformers
Chapter   : ch-01-ch01-environment (lesson 1)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-01-foundations/ch01-environment.ipynb#cell-1
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
# [002] content/notebooks/part-01-foundations/ch01-environment.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ملف: verify_setup.py
# التحقق من صحة الإعداد قبل المتابعة — خطوة لا يجب تخطّيها أبدًا

import sys
import torch
import numpy as np

def verify_environment():
    """
    التحقق الشامل من بيئة العمل. شغّل هذا الملف أولًا قبل أي فصل آخر
    في الكتاب للتأكد من عدم وجود مشاكل خفية قد تُفسد نتائج التجارب لاحقًا.
    """
    print("=" * 50)
    print("🔍 التحقق من بيئة العمل")
    print("=" * 50)

    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA متاح: {cuda_available}")
    if cuda_available:
        print(f"  اسم الـ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  ذاكرة GPU الإجمالية: "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  ⚠️ لا يوجد GPU — التدريب سيعمل على CPU (بطيء للنماذج الكبيرة)")

    # اختبار عملية حسابية بسيطة للتأكد من سلامة التثبيت
    device = "cuda" if cuda_available else "cpu"
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = x @ y
    print(f"\n✅ اختبار ضرب المصفوفات نجح على: {device}")
    print(f"   شكل الناتج: {z.shape}, القيمة المتوسطة: {z.mean().item():.4f}")

    # ضبط البذرة العشوائية لضمان قابلية إعادة الإنتاج عبر كل تجارب الكتاب
    set_global_seed(42)
    print("\n✅ البيئة جاهزة بالكامل للانطلاق في رحلة الكتاب!")

def set_global_seed(seed: int = 42):
    """
    ضبط البذرة العشوائية لكل مكتبة نستخدمها.
    هذه الدالة ستُستدعى في بداية كل تجربة عبر الكتاب لضمان
    قابلية إعادة الإنتاج (reproducibility) — مهارة أساسية
    في أي عمل بحثي أو هندسي جاد.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    verify_environment()

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-01-foundations/ch01-environment.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import time
import torch

def compare_precision_performance(size: int = 4096):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for dtype, name in [(torch.float32, "FP32"), (torch.float64, "FP64")]:
        x = torch.randn(size, size, dtype=dtype, device=device)
        y = torch.randn(size, size, dtype=dtype, device=device)

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            z = x @ y
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start

        print(f"{name}: الزمن لـ 10 عمليات ضرب = {elapsed:.4f} ثانية، "
              f"حجم الذاكرة لكل مصفوفة = {x.element_size() * x.nelement() / 1e6:.1f} MB")

compare_precision_performance()
# FP32: الزمن لـ 10 عمليات ضرب = 0.0412 ثانية، حجم الذاكرة لكل مصفوفة = 67.1 MB
# FP64: الزمن لـ 10 عمليات ضرب = 0.1836 ثانية، حجم الذاكرة لكل مصفوفة = 134.2 MB

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-01-foundations/ch01-environment.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-01-ch01-environment.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-01-ch01-environment.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ملف: verify_setup.py
# التحقق من صحة الإعداد قبل المتابعة — خطوة لا يجب تخطّيها أبدًا

import sys
import torch
import numpy as np

def verify_environment():
    """
    التحقق الشامل من بيئة العمل. شغّل هذا الملف أولًا قبل أي فصل آخر
    في الكتاب للتأكد من عدم وجود مشاكل خفية قد تُفسد نتائج التجارب لاحقًا.
    """
    print("=" * 50)
    print("🔍 التحقق من بيئة العمل")
    print("=" * 50)

    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA متاح: {cuda_available}")
    if cuda_available:
        print(f"  اسم الـ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  ذاكرة GPU الإجمالية: "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  ⚠️ لا يوجد GPU — التدريب سيعمل على CPU (بطيء للنماذج الكبيرة)")

    # اختبار عملية حسابية بسيطة للتأكد من سلامة التثبيت
    device = "cuda" if cuda_available else "cpu"
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = x @ y
    print(f"\n✅ اختبار ضرب المصفوفات نجح على: {device}")
    print(f"   شكل الناتج: {z.shape}, القيمة المتوسطة: {z.mean().item():.4f}")

    # ضبط البذرة العشوائية لضمان قابلية إعادة الإنتاج عبر كل تجارب الكتاب
    set_global_seed(42)
    print("\n✅ البيئة جاهزة بالكامل للانطلاق في رحلة الكتاب!")

def set_global_seed(seed: int = 42):
    """
    ضبط البذرة العشوائية لكل مكتبة نستخدمها.
    هذه الدالة ستُستدعى في بداية كل تجربة عبر الكتاب لضمان
    قابلية إعادة الإنتاج (reproducibility) — مهارة أساسية
    في أي عمل بحثي أو هندسي جاد.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    verify_environment()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-01-ch01-environment.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import time
import torch

def compare_precision_performance(size: int = 4096):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for dtype, name in [(torch.float32, "FP32"), (torch.float64, "FP64")]:
        x = torch.randn(size, size, dtype=dtype, device=device)
        y = torch.randn(size, size, dtype=dtype, device=device)

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            z = x @ y
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start

        print(f"{name}: الزمن لـ 10 عمليات ضرب = {elapsed:.4f} ثانية، "
              f"حجم الذاكرة لكل مصفوفة = {x.element_size() * x.nelement() / 1e6:.1f} MB")

compare_precision_performance()
# FP32: الزمن لـ 10 عمليات ضرب = 0.0412 ثانية، حجم الذاكرة لكل مصفوفة = 67.1 MB
# FP64: الزمن لـ 10 عمليات ضرب = 0.1836 ثانية، حجم الذاكرة لكل مصفوفة = 134.2 MB

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-01-ch01-environment.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-01-ch01-environment.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — إعداد البيئة والبنية التحتية للتطوير
Mastering Transformers and Large Language Models — Environment

Book slug : mastering-llms-transformers
Chapter   : ch-01-ch01-environment (lesson 1)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-01-foundations/ch01-environment.ipynb#cell-1
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
# [002] content/notebooks/part-01-foundations/ch01-environment.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ملف: verify_setup.py
# التحقق من صحة الإعداد قبل المتابعة — خطوة لا يجب تخطّيها أبدًا

import sys
import torch
import numpy as np

def verify_environment():
    """
    التحقق الشامل من بيئة العمل. شغّل هذا الملف أولًا قبل أي فصل آخر
    في الكتاب للتأكد من عدم وجود مشاكل خفية قد تُفسد نتائج التجارب لاحقًا.
    """
    print("=" * 50)
    print("🔍 التحقق من بيئة العمل")
    print("=" * 50)

    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA متاح: {cuda_available}")
    if cuda_available:
        print(f"  اسم الـ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  ذاكرة GPU الإجمالية: "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  ⚠️ لا يوجد GPU — التدريب سيعمل على CPU (بطيء للنماذج الكبيرة)")

    # اختبار عملية حسابية بسيطة للتأكد من سلامة التثبيت
    device = "cuda" if cuda_available else "cpu"
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = x @ y
    print(f"\n✅ اختبار ضرب المصفوفات نجح على: {device}")
    print(f"   شكل الناتج: {z.shape}, القيمة المتوسطة: {z.mean().item():.4f}")

    # ضبط البذرة العشوائية لضمان قابلية إعادة الإنتاج عبر كل تجارب الكتاب
    set_global_seed(42)
    print("\n✅ البيئة جاهزة بالكامل للانطلاق في رحلة الكتاب!")

def set_global_seed(seed: int = 42):
    """
    ضبط البذرة العشوائية لكل مكتبة نستخدمها.
    هذه الدالة ستُستدعى في بداية كل تجربة عبر الكتاب لضمان
    قابلية إعادة الإنتاج (reproducibility) — مهارة أساسية
    في أي عمل بحثي أو هندسي جاد.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    verify_environment()

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-01-foundations/ch01-environment.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import time
import torch

def compare_precision_performance(size: int = 4096):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for dtype, name in [(torch.float32, "FP32"), (torch.float64, "FP64")]:
        x = torch.randn(size, size, dtype=dtype, device=device)
        y = torch.randn(size, size, dtype=dtype, device=device)

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            z = x @ y
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start

        print(f"{name}: الزمن لـ 10 عمليات ضرب = {elapsed:.4f} ثانية، "
              f"حجم الذاكرة لكل مصفوفة = {x.element_size() * x.nelement() / 1e6:.1f} MB")

compare_precision_performance()
# FP32: الزمن لـ 10 عمليات ضرب = 0.0412 ثانية، حجم الذاكرة لكل مصفوفة = 67.1 MB
# FP64: الزمن لـ 10 عمليات ضرب = 0.1836 ثانية، حجم الذاكرة لكل مصفوفة = 134.2 MB

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-01-foundations/ch01-environment.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-01-ch01-environment.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-01-ch01-environment.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ملف: verify_setup.py
# التحقق من صحة الإعداد قبل المتابعة — خطوة لا يجب تخطّيها أبدًا

import sys
import torch
import numpy as np

def verify_environment():
    """
    التحقق الشامل من بيئة العمل. شغّل هذا الملف أولًا قبل أي فصل آخر
    في الكتاب للتأكد من عدم وجود مشاكل خفية قد تُفسد نتائج التجارب لاحقًا.
    """
    print("=" * 50)
    print("🔍 التحقق من بيئة العمل")
    print("=" * 50)

    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA متاح: {cuda_available}")
    if cuda_available:
        print(f"  اسم الـ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  ذاكرة GPU الإجمالية: "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  ⚠️ لا يوجد GPU — التدريب سيعمل على CPU (بطيء للنماذج الكبيرة)")

    # اختبار عملية حسابية بسيطة للتأكد من سلامة التثبيت
    device = "cuda" if cuda_available else "cpu"
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = x @ y
    print(f"\n✅ اختبار ضرب المصفوفات نجح على: {device}")
    print(f"   شكل الناتج: {z.shape}, القيمة المتوسطة: {z.mean().item():.4f}")

    # ضبط البذرة العشوائية لضمان قابلية إعادة الإنتاج عبر كل تجارب الكتاب
    set_global_seed(42)
    print("\n✅ البيئة جاهزة بالكامل للانطلاق في رحلة الكتاب!")

def set_global_seed(seed: int = 42):
    """
    ضبط البذرة العشوائية لكل مكتبة نستخدمها.
    هذه الدالة ستُستدعى في بداية كل تجربة عبر الكتاب لضمان
    قابلية إعادة الإنتاج (reproducibility) — مهارة أساسية
    في أي عمل بحثي أو هندسي جاد.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    verify_environment()

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-01-ch01-environment.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import time
import torch

def compare_precision_performance(size: int = 4096):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for dtype, name in [(torch.float32, "FP32"), (torch.float64, "FP64")]:
        x = torch.randn(size, size, dtype=dtype, device=device)
        y = torch.randn(size, size, dtype=dtype, device=device)

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            z = x @ y
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start

        print(f"{name}: الزمن لـ 10 عمليات ضرب = {elapsed:.4f} ثانية، "
              f"حجم الذاكرة لكل مصفوفة = {x.element_size() * x.nelement() / 1e6:.1f} MB")

compare_precision_performance()
# FP32: الزمن لـ 10 عمليات ضرب = 0.0412 ثانية، حجم الذاكرة لكل مصفوفة = 67.1 MB
# FP64: الزمن لـ 10 عمليات ضرب = 0.1836 ثانية، حجم الذاكرة لكل مصفوفة = 134.2 MB

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-01-ch01-environment.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
