#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التقييم النوعي وتحليل الأخطاء
Mastering Transformers and Large Language Models — Qualitative Evaluation

Book slug : mastering-llms-transformers
Chapter   : ch-45-ch45-qualitative-evaluation (lesson 45)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-45-ch45-qualitative-evaluation.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التقييم النوعي وتحليل الأخطاء
Mastering Transformers and Large Language Models — Qualitative Evaluation

Book slug : mastering-llms-transformers
Chapter   : ch-45-ch45-qualitative-evaluation (lesson 45)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-45-ch45-qualitative-evaluation.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التقييم النوعي وتحليل الأخطاء
Mastering Transformers and Large Language Models — Qualitative Evaluation

Book slug : mastering-llms-transformers
Chapter   : ch-45-ch45-qualitative-evaluation (lesson 45)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-09-evaluation/ch45-qualitative-evaluation.ipynb#cell-1
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
    ("matplotlib", "matplotlib")
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
# [002] content/notebooks/part-09-evaluation/ch45-qualitative-evaluation.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from enum import Enum
from dataclasses import dataclass

class ErrorType(Enum):
    FACTUAL_ERROR = "خطأ واقعي"
    REASONING_ERROR = "خطأ منطقي/استدلالي"
    INSTRUCTION_FOLLOWING = "عدم اتّباع التعليمات"
    FORMATTING_ERROR = "خطأ في التنسيق المطلوب"
    HALLUCINATION = "هلوسة (اختلاق معلومة)"
    REFUSAL_UNWARRANTED = "رفض غير مبرر"
    LANGUAGE_QUALITY = "ضعف في جودة اللغة"
    CONTEXT_MISUNDERSTANDING = "سوء فهم السياق"

@dataclass
class ErrorAnnotation:
    prompt: str
    response: str
    error_types: list
    severity: int  # 1 (طفيف) - 5 (حرج)
    notes: str

def build_error_taxonomy_report(annotated_failures: list) -> dict:
    """تجميع إحصائي لأنماط الفشل — يكشف أين يجب التركيز في التحسين"""
    from collections import Counter
    error_counts = Counter()
    for failure in annotated_failures:
        for error_type in failure.error_types:
            error_counts[error_type.value] += 1

    total = len(annotated_failures)
    report = {
        error_type: {"count": count, "percentage": count / total * 100}
        for error_type, count in error_counts.most_common()
    }
    return report

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-09-evaluation/ch45-qualitative-evaluation.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def slice_based_evaluation(model, test_set: list, slice_functions: dict) -> dict:
    """
    تقييم الأداء ليس فقط إجمالًا، بل عبر "شرائح" محدَّدة من البيانات
    (مثال: حسب طول السؤال، اللغة، الموضوع، صعوبة السؤال)
    """
    results_by_slice = {}

    for slice_name, slice_fn in slice_functions.items():
        slice_examples = [ex for ex in test_set if slice_fn(ex)]
        if not slice_examples:
            continue
        correct = sum(1 for ex in slice_examples if model.predict(ex["input"]) == ex["label"])
        results_by_slice[slice_name] = {
            "accuracy": correct / len(slice_examples),
            "n_samples": len(slice_examples)
        }

    return results_by_slice

# مثال على تعريف شرائح مفيدة للغة العربية تحديدًا
arabic_slices = {
    "فصحى": lambda ex: ex.get("dialect") == "msa",
    "لهجة_خليجية": lambda ex: ex.get("dialect") == "gulf",
    "لهجة_مصرية": lambda ex: ex.get("dialect") == "egyptian",
    "أسئلة_قصيرة": lambda ex: len(ex["input"].split()) < 10,
    "أسئلة_طويلة": lambda ex: len(ex["input"].split()) >= 50,
    "يحتوي_أرقام": lambda ex: any(c.isdigit() for c in ex["input"]),
}

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-09-evaluation/ch45-qualitative-evaluation.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def plot_scaling_curve_analysis(model_sizes: list, benchmark_scores: dict):
    """
    توضيح مفاهيمي: رسم منحنى الأداء مقابل حجم النموذج
    لاكتشاف نقاط "القفزة النوعية" (emergence thresholds)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(benchmark_scores), figsize=(15, 4))
    for idx, (benchmark_name, scores) in enumerate(benchmark_scores.items()):
        ax = axes[idx] if len(benchmark_scores) > 1 else axes
        ax.plot(model_sizes, scores, marker='o')
        ax.set_xscale('log')
        ax.set_xlabel('حجم النموذج (log scale)')
        ax.set_ylabel('الدقة')
        ax.set_title(benchmark_name)
        ax.axhline(y=0.25, color='gray', linestyle='--', label='خط الصدفة العشوائية')
    plt.tight_layout()
    return fig

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-09-evaluation/ch45-qualitative-evaluation.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-45-ch45-qualitative-evaluation.ipynb#cell-1
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
    ("matplotlib", "matplotlib")
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-45-ch45-qualitative-evaluation.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from enum import Enum
from dataclasses import dataclass

class ErrorType(Enum):
    FACTUAL_ERROR = "خطأ واقعي"
    REASONING_ERROR = "خطأ منطقي/استدلالي"
    INSTRUCTION_FOLLOWING = "عدم اتّباع التعليمات"
    FORMATTING_ERROR = "خطأ في التنسيق المطلوب"
    HALLUCINATION = "هلوسة (اختلاق معلومة)"
    REFUSAL_UNWARRANTED = "رفض غير مبرر"
    LANGUAGE_QUALITY = "ضعف في جودة اللغة"
    CONTEXT_MISUNDERSTANDING = "سوء فهم السياق"

@dataclass
class ErrorAnnotation:
    prompt: str
    response: str
    error_types: list
    severity: int  # 1 (طفيف) - 5 (حرج)
    notes: str

def build_error_taxonomy_report(annotated_failures: list) -> dict:
    """تجميع إحصائي لأنماط الفشل — يكشف أين يجب التركيز في التحسين"""
    from collections import Counter
    error_counts = Counter()
    for failure in annotated_failures:
        for error_type in failure.error_types:
            error_counts[error_type.value] += 1

    total = len(annotated_failures)
    report = {
        error_type: {"count": count, "percentage": count / total * 100}
        for error_type, count in error_counts.most_common()
    }
    return report

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-45-ch45-qualitative-evaluation.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def slice_based_evaluation(model, test_set: list, slice_functions: dict) -> dict:
    """
    تقييم الأداء ليس فقط إجمالًا، بل عبر "شرائح" محدَّدة من البيانات
    (مثال: حسب طول السؤال، اللغة، الموضوع، صعوبة السؤال)
    """
    results_by_slice = {}

    for slice_name, slice_fn in slice_functions.items():
        slice_examples = [ex for ex in test_set if slice_fn(ex)]
        if not slice_examples:
            continue
        correct = sum(1 for ex in slice_examples if model.predict(ex["input"]) == ex["label"])
        results_by_slice[slice_name] = {
            "accuracy": correct / len(slice_examples),
            "n_samples": len(slice_examples)
        }

    return results_by_slice

# مثال على تعريف شرائح مفيدة للغة العربية تحديدًا
arabic_slices = {
    "فصحى": lambda ex: ex.get("dialect") == "msa",
    "لهجة_خليجية": lambda ex: ex.get("dialect") == "gulf",
    "لهجة_مصرية": lambda ex: ex.get("dialect") == "egyptian",
    "أسئلة_قصيرة": lambda ex: len(ex["input"].split()) < 10,
    "أسئلة_طويلة": lambda ex: len(ex["input"].split()) >= 50,
    "يحتوي_أرقام": lambda ex: any(c.isdigit() for c in ex["input"]),
}

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-45-ch45-qualitative-evaluation.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def plot_scaling_curve_analysis(model_sizes: list, benchmark_scores: dict):
    """
    توضيح مفاهيمي: رسم منحنى الأداء مقابل حجم النموذج
    لاكتشاف نقاط "القفزة النوعية" (emergence thresholds)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(benchmark_scores), figsize=(15, 4))
    for idx, (benchmark_name, scores) in enumerate(benchmark_scores.items()):
        ax = axes[idx] if len(benchmark_scores) > 1 else axes
        ax.plot(model_sizes, scores, marker='o')
        ax.set_xscale('log')
        ax.set_xlabel('حجم النموذج (log scale)')
        ax.set_ylabel('الدقة')
        ax.set_title(benchmark_name)
        ax.axhline(y=0.25, color='gray', linestyle='--', label='خط الصدفة العشوائية')
    plt.tight_layout()
    return fig

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-45-ch45-qualitative-evaluation.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-45-ch45-qualitative-evaluation.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التقييم النوعي وتحليل الأخطاء
Mastering Transformers and Large Language Models — Qualitative Evaluation

Book slug : mastering-llms-transformers
Chapter   : ch-45-ch45-qualitative-evaluation (lesson 45)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-09-evaluation/ch45-qualitative-evaluation.ipynb#cell-1
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
    ("matplotlib", "matplotlib")
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
# [002] content/notebooks/part-09-evaluation/ch45-qualitative-evaluation.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from enum import Enum
from dataclasses import dataclass

class ErrorType(Enum):
    FACTUAL_ERROR = "خطأ واقعي"
    REASONING_ERROR = "خطأ منطقي/استدلالي"
    INSTRUCTION_FOLLOWING = "عدم اتّباع التعليمات"
    FORMATTING_ERROR = "خطأ في التنسيق المطلوب"
    HALLUCINATION = "هلوسة (اختلاق معلومة)"
    REFUSAL_UNWARRANTED = "رفض غير مبرر"
    LANGUAGE_QUALITY = "ضعف في جودة اللغة"
    CONTEXT_MISUNDERSTANDING = "سوء فهم السياق"

@dataclass
class ErrorAnnotation:
    prompt: str
    response: str
    error_types: list
    severity: int  # 1 (طفيف) - 5 (حرج)
    notes: str

def build_error_taxonomy_report(annotated_failures: list) -> dict:
    """تجميع إحصائي لأنماط الفشل — يكشف أين يجب التركيز في التحسين"""
    from collections import Counter
    error_counts = Counter()
    for failure in annotated_failures:
        for error_type in failure.error_types:
            error_counts[error_type.value] += 1

    total = len(annotated_failures)
    report = {
        error_type: {"count": count, "percentage": count / total * 100}
        for error_type, count in error_counts.most_common()
    }
    return report

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-09-evaluation/ch45-qualitative-evaluation.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def slice_based_evaluation(model, test_set: list, slice_functions: dict) -> dict:
    """
    تقييم الأداء ليس فقط إجمالًا، بل عبر "شرائح" محدَّدة من البيانات
    (مثال: حسب طول السؤال، اللغة، الموضوع، صعوبة السؤال)
    """
    results_by_slice = {}

    for slice_name, slice_fn in slice_functions.items():
        slice_examples = [ex for ex in test_set if slice_fn(ex)]
        if not slice_examples:
            continue
        correct = sum(1 for ex in slice_examples if model.predict(ex["input"]) == ex["label"])
        results_by_slice[slice_name] = {
            "accuracy": correct / len(slice_examples),
            "n_samples": len(slice_examples)
        }

    return results_by_slice

# مثال على تعريف شرائح مفيدة للغة العربية تحديدًا
arabic_slices = {
    "فصحى": lambda ex: ex.get("dialect") == "msa",
    "لهجة_خليجية": lambda ex: ex.get("dialect") == "gulf",
    "لهجة_مصرية": lambda ex: ex.get("dialect") == "egyptian",
    "أسئلة_قصيرة": lambda ex: len(ex["input"].split()) < 10,
    "أسئلة_طويلة": lambda ex: len(ex["input"].split()) >= 50,
    "يحتوي_أرقام": lambda ex: any(c.isdigit() for c in ex["input"]),
}

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-09-evaluation/ch45-qualitative-evaluation.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def plot_scaling_curve_analysis(model_sizes: list, benchmark_scores: dict):
    """
    توضيح مفاهيمي: رسم منحنى الأداء مقابل حجم النموذج
    لاكتشاف نقاط "القفزة النوعية" (emergence thresholds)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(benchmark_scores), figsize=(15, 4))
    for idx, (benchmark_name, scores) in enumerate(benchmark_scores.items()):
        ax = axes[idx] if len(benchmark_scores) > 1 else axes
        ax.plot(model_sizes, scores, marker='o')
        ax.set_xscale('log')
        ax.set_xlabel('حجم النموذج (log scale)')
        ax.set_ylabel('الدقة')
        ax.set_title(benchmark_name)
        ax.axhline(y=0.25, color='gray', linestyle='--', label='خط الصدفة العشوائية')
    plt.tight_layout()
    return fig

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-09-evaluation/ch45-qualitative-evaluation.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-45-ch45-qualitative-evaluation.ipynb#cell-2
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
    ("matplotlib", "matplotlib")
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-45-ch45-qualitative-evaluation.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from enum import Enum
from dataclasses import dataclass

class ErrorType(Enum):
    FACTUAL_ERROR = "خطأ واقعي"
    REASONING_ERROR = "خطأ منطقي/استدلالي"
    INSTRUCTION_FOLLOWING = "عدم اتّباع التعليمات"
    FORMATTING_ERROR = "خطأ في التنسيق المطلوب"
    HALLUCINATION = "هلوسة (اختلاق معلومة)"
    REFUSAL_UNWARRANTED = "رفض غير مبرر"
    LANGUAGE_QUALITY = "ضعف في جودة اللغة"
    CONTEXT_MISUNDERSTANDING = "سوء فهم السياق"

@dataclass
class ErrorAnnotation:
    prompt: str
    response: str
    error_types: list
    severity: int  # 1 (طفيف) - 5 (حرج)
    notes: str

def build_error_taxonomy_report(annotated_failures: list) -> dict:
    """تجميع إحصائي لأنماط الفشل — يكشف أين يجب التركيز في التحسين"""
    from collections import Counter
    error_counts = Counter()
    for failure in annotated_failures:
        for error_type in failure.error_types:
            error_counts[error_type.value] += 1

    total = len(annotated_failures)
    report = {
        error_type: {"count": count, "percentage": count / total * 100}
        for error_type, count in error_counts.most_common()
    }
    return report

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-45-ch45-qualitative-evaluation.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def slice_based_evaluation(model, test_set: list, slice_functions: dict) -> dict:
    """
    تقييم الأداء ليس فقط إجمالًا، بل عبر "شرائح" محدَّدة من البيانات
    (مثال: حسب طول السؤال، اللغة، الموضوع، صعوبة السؤال)
    """
    results_by_slice = {}

    for slice_name, slice_fn in slice_functions.items():
        slice_examples = [ex for ex in test_set if slice_fn(ex)]
        if not slice_examples:
            continue
        correct = sum(1 for ex in slice_examples if model.predict(ex["input"]) == ex["label"])
        results_by_slice[slice_name] = {
            "accuracy": correct / len(slice_examples),
            "n_samples": len(slice_examples)
        }

    return results_by_slice

# مثال على تعريف شرائح مفيدة للغة العربية تحديدًا
arabic_slices = {
    "فصحى": lambda ex: ex.get("dialect") == "msa",
    "لهجة_خليجية": lambda ex: ex.get("dialect") == "gulf",
    "لهجة_مصرية": lambda ex: ex.get("dialect") == "egyptian",
    "أسئلة_قصيرة": lambda ex: len(ex["input"].split()) < 10,
    "أسئلة_طويلة": lambda ex: len(ex["input"].split()) >= 50,
    "يحتوي_أرقام": lambda ex: any(c.isdigit() for c in ex["input"]),
}

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-45-ch45-qualitative-evaluation.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def plot_scaling_curve_analysis(model_sizes: list, benchmark_scores: dict):
    """
    توضيح مفاهيمي: رسم منحنى الأداء مقابل حجم النموذج
    لاكتشاف نقاط "القفزة النوعية" (emergence thresholds)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(benchmark_scores), figsize=(15, 4))
    for idx, (benchmark_name, scores) in enumerate(benchmark_scores.items()):
        ax = axes[idx] if len(benchmark_scores) > 1 else axes
        ax.plot(model_sizes, scores, marker='o')
        ax.set_xscale('log')
        ax.set_xlabel('حجم النموذج (log scale)')
        ax.set_ylabel('الدقة')
        ax.set_title(benchmark_name)
        ax.axhline(y=0.25, color='gray', linestyle='--', label='خط الصدفة العشوائية')
    plt.tight_layout()
    return fig

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-45-ch45-qualitative-evaluation.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
