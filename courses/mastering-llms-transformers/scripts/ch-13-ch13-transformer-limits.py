#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — حدود المحوّل الكلاسيكي
Mastering Transformers and Large Language Models — Transformer Limits

Book slug : mastering-llms-transformers
Chapter   : ch-13-ch13-transformer-limits (lesson 13)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-13-ch13-transformer-limits.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — حدود المحوّل الكلاسيكي
Mastering Transformers and Large Language Models — Transformer Limits

Book slug : mastering-llms-transformers
Chapter   : ch-13-ch13-transformer-limits (lesson 13)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-13-ch13-transformer-limits.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — حدود المحوّل الكلاسيكي
Mastering Transformers and Large Language Models — Transformer Limits

Book slug : mastering-llms-transformers
Chapter   : ch-13-ch13-transformer-limits (lesson 13)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-04-modern-architectures/ch13-transformer-limits.ipynb#cell-1
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
# [002] content/notebooks/part-04-modern-architectures/ch13-transformer-limits.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import time
import matplotlib.pyplot as plt

def measure_attention_complexity(seq_lengths: list, d_model: int = 512,
                                    num_heads: int = 8, device='cuda'):
    """
    قياس عملي للوقت والذاكرة المستهلكة لـ Self-Attention
    عند أطوال تسلسل مختلفة، لإظهار النمو التربيعي بشكل ملموس.
    """
    results = {"seq_len": [], "time_ms": [], "memory_mb": []}

    for seq_len in seq_lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        x = torch.randn(1, seq_len, d_model, device=device)
        mha = MultiHeadAttention(d_model, num_heads).to(device)  # من الفصل 10

        # قياس الوقت
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            output, _ = mha(x, x, x)
        torch.cuda.synchronize()
        elapsed_ms = (time.time() - start) * 1000

        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        results["seq_len"].append(seq_len)
        results["time_ms"].append(elapsed_ms)
        results["memory_mb"].append(peak_memory_mb)

        print(f"seq_len={seq_len:>6} | time={elapsed_ms:>8.2f}ms | "
              f"peak_memory={peak_memory_mb:>8.2f}MB")

    return results

def plot_quadratic_growth(results: dict):
    """تصوير النمو التربيعي بصريًا، مع خط مرجعي نظري O(n^2)."""
    seq_lengths = results["seq_len"]
    memory = results["memory_mb"]

    # خط مرجعي نظري (مُطبَّع على أول نقطة)
    theoretical = [memory[0] * (n / seq_lengths[0]) ** 2 for n in seq_lengths]

    plt.figure(figsize=(8, 5))
    plt.plot(seq_lengths, memory, 'o-', label='الذاكرة المُقاسة فعليًا')
    plt.plot(seq_lengths, theoretical, '--', label='النمو النظري O(n²)')
    plt.xlabel('طول التسلسل (n)')
    plt.ylabel('الذاكرة المستهلكة (MB)')
    plt.title('النمو التربيعي لذاكرة Self-Attention')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('attention_quadratic_growth.png', dpi=150)

# مثال تشغيل متوقع (نتائج تقريبية توضيحية):
# seq_len=   512 | time=    2.31ms | peak_memory=   45.20MB
# seq_len=  1024 | time=    6.84ms | peak_memory=  156.40MB   (~3.5x للذاكرة عند 2x للطول)
# seq_len=  2048 | time=   24.15ms | peak_memory=  589.60MB   (~3.8x للذاكرة عند 2x للطول)
# seq_len=  4096 | time=   91.82ms | peak_memory= 2284.10MB   (~3.9x للذاكرة عند 2x للطول)

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-04-modern-architectures/ch13-transformer-limits.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-13-ch13-transformer-limits.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-13-ch13-transformer-limits.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import time
import matplotlib.pyplot as plt

def measure_attention_complexity(seq_lengths: list, d_model: int = 512,
                                    num_heads: int = 8, device='cuda'):
    """
    قياس عملي للوقت والذاكرة المستهلكة لـ Self-Attention
    عند أطوال تسلسل مختلفة، لإظهار النمو التربيعي بشكل ملموس.
    """
    results = {"seq_len": [], "time_ms": [], "memory_mb": []}

    for seq_len in seq_lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        x = torch.randn(1, seq_len, d_model, device=device)
        mha = MultiHeadAttention(d_model, num_heads).to(device)  # من الفصل 10

        # قياس الوقت
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            output, _ = mha(x, x, x)
        torch.cuda.synchronize()
        elapsed_ms = (time.time() - start) * 1000

        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        results["seq_len"].append(seq_len)
        results["time_ms"].append(elapsed_ms)
        results["memory_mb"].append(peak_memory_mb)

        print(f"seq_len={seq_len:>6} | time={elapsed_ms:>8.2f}ms | "
              f"peak_memory={peak_memory_mb:>8.2f}MB")

    return results

def plot_quadratic_growth(results: dict):
    """تصوير النمو التربيعي بصريًا، مع خط مرجعي نظري O(n^2)."""
    seq_lengths = results["seq_len"]
    memory = results["memory_mb"]

    # خط مرجعي نظري (مُطبَّع على أول نقطة)
    theoretical = [memory[0] * (n / seq_lengths[0]) ** 2 for n in seq_lengths]

    plt.figure(figsize=(8, 5))
    plt.plot(seq_lengths, memory, 'o-', label='الذاكرة المُقاسة فعليًا')
    plt.plot(seq_lengths, theoretical, '--', label='النمو النظري O(n²)')
    plt.xlabel('طول التسلسل (n)')
    plt.ylabel('الذاكرة المستهلكة (MB)')
    plt.title('النمو التربيعي لذاكرة Self-Attention')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('attention_quadratic_growth.png', dpi=150)

# مثال تشغيل متوقع (نتائج تقريبية توضيحية):
# seq_len=   512 | time=    2.31ms | peak_memory=   45.20MB
# seq_len=  1024 | time=    6.84ms | peak_memory=  156.40MB   (~3.5x للذاكرة عند 2x للطول)
# seq_len=  2048 | time=   24.15ms | peak_memory=  589.60MB   (~3.8x للذاكرة عند 2x للطول)
# seq_len=  4096 | time=   91.82ms | peak_memory= 2284.10MB   (~3.9x للذاكرة عند 2x للطول)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-13-ch13-transformer-limits.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-13-ch13-transformer-limits.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — حدود المحوّل الكلاسيكي
Mastering Transformers and Large Language Models — Transformer Limits

Book slug : mastering-llms-transformers
Chapter   : ch-13-ch13-transformer-limits (lesson 13)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-04-modern-architectures/ch13-transformer-limits.ipynb#cell-1
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
# [002] content/notebooks/part-04-modern-architectures/ch13-transformer-limits.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import time
import matplotlib.pyplot as plt

def measure_attention_complexity(seq_lengths: list, d_model: int = 512,
                                    num_heads: int = 8, device='cuda'):
    """
    قياس عملي للوقت والذاكرة المستهلكة لـ Self-Attention
    عند أطوال تسلسل مختلفة، لإظهار النمو التربيعي بشكل ملموس.
    """
    results = {"seq_len": [], "time_ms": [], "memory_mb": []}

    for seq_len in seq_lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        x = torch.randn(1, seq_len, d_model, device=device)
        mha = MultiHeadAttention(d_model, num_heads).to(device)  # من الفصل 10

        # قياس الوقت
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            output, _ = mha(x, x, x)
        torch.cuda.synchronize()
        elapsed_ms = (time.time() - start) * 1000

        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        results["seq_len"].append(seq_len)
        results["time_ms"].append(elapsed_ms)
        results["memory_mb"].append(peak_memory_mb)

        print(f"seq_len={seq_len:>6} | time={elapsed_ms:>8.2f}ms | "
              f"peak_memory={peak_memory_mb:>8.2f}MB")

    return results

def plot_quadratic_growth(results: dict):
    """تصوير النمو التربيعي بصريًا، مع خط مرجعي نظري O(n^2)."""
    seq_lengths = results["seq_len"]
    memory = results["memory_mb"]

    # خط مرجعي نظري (مُطبَّع على أول نقطة)
    theoretical = [memory[0] * (n / seq_lengths[0]) ** 2 for n in seq_lengths]

    plt.figure(figsize=(8, 5))
    plt.plot(seq_lengths, memory, 'o-', label='الذاكرة المُقاسة فعليًا')
    plt.plot(seq_lengths, theoretical, '--', label='النمو النظري O(n²)')
    plt.xlabel('طول التسلسل (n)')
    plt.ylabel('الذاكرة المستهلكة (MB)')
    plt.title('النمو التربيعي لذاكرة Self-Attention')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('attention_quadratic_growth.png', dpi=150)

# مثال تشغيل متوقع (نتائج تقريبية توضيحية):
# seq_len=   512 | time=    2.31ms | peak_memory=   45.20MB
# seq_len=  1024 | time=    6.84ms | peak_memory=  156.40MB   (~3.5x للذاكرة عند 2x للطول)
# seq_len=  2048 | time=   24.15ms | peak_memory=  589.60MB   (~3.8x للذاكرة عند 2x للطول)
# seq_len=  4096 | time=   91.82ms | peak_memory= 2284.10MB   (~3.9x للذاكرة عند 2x للطول)

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-04-modern-architectures/ch13-transformer-limits.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-13-ch13-transformer-limits.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-13-ch13-transformer-limits.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import time
import matplotlib.pyplot as plt

def measure_attention_complexity(seq_lengths: list, d_model: int = 512,
                                    num_heads: int = 8, device='cuda'):
    """
    قياس عملي للوقت والذاكرة المستهلكة لـ Self-Attention
    عند أطوال تسلسل مختلفة، لإظهار النمو التربيعي بشكل ملموس.
    """
    results = {"seq_len": [], "time_ms": [], "memory_mb": []}

    for seq_len in seq_lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        x = torch.randn(1, seq_len, d_model, device=device)
        mha = MultiHeadAttention(d_model, num_heads).to(device)  # من الفصل 10

        # قياس الوقت
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            output, _ = mha(x, x, x)
        torch.cuda.synchronize()
        elapsed_ms = (time.time() - start) * 1000

        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        results["seq_len"].append(seq_len)
        results["time_ms"].append(elapsed_ms)
        results["memory_mb"].append(peak_memory_mb)

        print(f"seq_len={seq_len:>6} | time={elapsed_ms:>8.2f}ms | "
              f"peak_memory={peak_memory_mb:>8.2f}MB")

    return results

def plot_quadratic_growth(results: dict):
    """تصوير النمو التربيعي بصريًا، مع خط مرجعي نظري O(n^2)."""
    seq_lengths = results["seq_len"]
    memory = results["memory_mb"]

    # خط مرجعي نظري (مُطبَّع على أول نقطة)
    theoretical = [memory[0] * (n / seq_lengths[0]) ** 2 for n in seq_lengths]

    plt.figure(figsize=(8, 5))
    plt.plot(seq_lengths, memory, 'o-', label='الذاكرة المُقاسة فعليًا')
    plt.plot(seq_lengths, theoretical, '--', label='النمو النظري O(n²)')
    plt.xlabel('طول التسلسل (n)')
    plt.ylabel('الذاكرة المستهلكة (MB)')
    plt.title('النمو التربيعي لذاكرة Self-Attention')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('attention_quadratic_growth.png', dpi=150)

# مثال تشغيل متوقع (نتائج تقريبية توضيحية):
# seq_len=   512 | time=    2.31ms | peak_memory=   45.20MB
# seq_len=  1024 | time=    6.84ms | peak_memory=  156.40MB   (~3.5x للذاكرة عند 2x للطول)
# seq_len=  2048 | time=   24.15ms | peak_memory=  589.60MB   (~3.8x للذاكرة عند 2x للطول)
# seq_len=  4096 | time=   91.82ms | peak_memory= 2284.10MB   (~3.9x للذاكرة عند 2x للطول)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-13-ch13-transformer-limits.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
