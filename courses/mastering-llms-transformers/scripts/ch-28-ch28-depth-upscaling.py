#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — توسيع العمق: Depth Upscaling
Mastering Transformers and Large Language Models — Depth Upscaling

Book slug : mastering-llms-transformers
Chapter   : ch-28-ch28-depth-upscaling (lesson 28)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-28-ch28-depth-upscaling.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — توسيع العمق: Depth Upscaling
Mastering Transformers and Large Language Models — Depth Upscaling

Book slug : mastering-llms-transformers
Chapter   : ch-28-ch28-depth-upscaling (lesson 28)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-28-ch28-depth-upscaling.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — توسيع العمق: Depth Upscaling
Mastering Transformers and Large Language Models — Depth Upscaling

Book slug : mastering-llms-transformers
Chapter   : ch-28-ch28-depth-upscaling (lesson 28)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-06-model-reuse/ch28-depth-upscaling.ipynb#cell-1
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
# [002] content/notebooks/part-06-model-reuse/ch28-depth-upscaling.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import copy

class DepthUpscaler:
    """
    تطبيق كامل لمنهجية Depth Upscaling بأسلوب SOLAR 10.7B: بناء نموذج
    أعمق من نسختين من نفس النموذج الأساسي عبر التركيب المباشر
    (Passthrough) للطبقات، تمهيدًا لمرحلة Continued Pretraining.
    """
    def __init__(self, base_model, n_layers_to_remove: int = 8):
        self.base_model = base_model
        self.n_layers_to_remove = n_layers_to_remove
        self.original_n_layers = len(base_model.layers)

    def create_upscaled_model(self):
        """
        بناء النموذج الموسَّع: نسختان من النموذج الأساسي، كل منهما
        تُقلَّم من أحد طرفيها، ثم تُركَّبان معًا.
        """
        copy_a = copy.deepcopy(self.base_model)
        copy_b = copy.deepcopy(self.base_model)

        # النسخة A: إزالة آخر n_layers_to_remove طبقة
        layers_a = copy_a.layers[: self.original_n_layers - self.n_layers_to_remove]

        # النسخة B: إزالة أول n_layers_to_remove طبقة
        layers_b = copy_b.layers[self.n_layers_to_remove:]

        # التركيب المباشر: كل طبقات A يليها كل طبقات B
        upscaled_layers = nn.ModuleList(list(layers_a) + list(layers_b))

        upscaled_model = copy.deepcopy(self.base_model)
        upscaled_model.layers = upscaled_layers

        new_n_layers = len(upscaled_layers)
        print(f"عدد الطبقات: الأصلي={self.original_n_layers} → "
              f"الموسَّع={new_n_layers} "
              f"(تضمين {self.original_n_layers * 2 - new_n_layers} "
              f"طبقة متداخلة مُزالة)")

        return upscaled_model

    def validate_layer_continuity(self, upscaled_model, sample_input):
        """
        فحص سريع قبل CPT: التأكد أن إخراج النموذج الموسَّع "معقول"
        إحصائيًا (ليس NaN أو قيمًا متفجرة)، حتى قبل أي تدريب إضافي -
        هذا ممكن لأن الطبقات المُركَّبة كل منها مُدرَّبة أصلاً جيدًا
        على مدخلات ومخرجات من نفس "النمط" العام، فالنموذج الموسَّع
        غير المُدرَّب لا يزال "يعمل" بشكل خام معقول، وإن لم يكن أمثل.
        """
        with torch.no_grad():
            output = upscaled_model(sample_input)
            has_nan = torch.isnan(output.logits).any().item()
            output_std = output.logits.std().item()

        return {
            "has_nan": has_nan,
            "output_std": output_std,
            "status": "✅ جاهز لـ CPT" if not has_nan else "❌ مشكلة - راجع نقطة الدمج",
        }

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-06-model-reuse/ch28-depth-upscaling.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def estimate_upscaling_cost_savings(
    base_model_params_billions: float,
    target_model_params_billions: float,
    cpt_tokens_billions: float,
    from_scratch_tokens_billions: float,  # وفق Chinchilla (الفصل 20): ≈20×N
) -> dict:
    """
    مقارنة تقديرية للتكلفة الحاسوبية (FLOPs) بين بناء نموذج بالحجم
    المستهدف عبر Depth Upscaling + CPT، مقابل تدريبه من الصفر بالكامل.
    """
    # تقدير FLOPs بقانون Kaplan/Chinchilla المبسّط: FLOPs ≈ 6 × N × D
    flops_upscaling_cpt = 6 * (target_model_params_billions * 1e9) * (cpt_tokens_billions * 1e9)
    flops_from_scratch = 6 * (target_model_params_billions * 1e9) * (from_scratch_tokens_billions * 1e9)

    savings_factor = flops_from_scratch / flops_upscaling_cpt

    return {
        "flops_upscaling_approach": f"{flops_upscaling_cpt:.2e}",
        "flops_from_scratch_approach": f"{flops_from_scratch:.2e}",
        "savings_factor": f"{savings_factor:.1f}×",
        "conclusion": (
            f"بناء نموذج {target_model_params_billions}B عبر Depth Upscaling "
            f"من نموذج {base_model_params_billions}B موجود يوفر تقريبًا "
            f"{savings_factor:.0f} ضعف الحوسبة مقارنة بالتدريب من الصفر"
        ),
    }

result = estimate_upscaling_cost_savings(
    base_model_params_billions=7.0,
    target_model_params_billions=10.7,
    cpt_tokens_billions=3,       # SOLAR استخدمت تدريبًا إضافيًا محدودًا نسبيًا
    from_scratch_tokens_billions=10.7 * 20,  # وفق Chinchilla لنموذج 10.7B
)
print(result)
# النتيجة التقريبية: savings_factor ≈ 71× - توفير هائل جدًا في الحوسبة!

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-06-model-reuse/ch28-depth-upscaling.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-28-ch28-depth-upscaling.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-28-ch28-depth-upscaling.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import copy

class DepthUpscaler:
    """
    تطبيق كامل لمنهجية Depth Upscaling بأسلوب SOLAR 10.7B: بناء نموذج
    أعمق من نسختين من نفس النموذج الأساسي عبر التركيب المباشر
    (Passthrough) للطبقات، تمهيدًا لمرحلة Continued Pretraining.
    """
    def __init__(self, base_model, n_layers_to_remove: int = 8):
        self.base_model = base_model
        self.n_layers_to_remove = n_layers_to_remove
        self.original_n_layers = len(base_model.layers)

    def create_upscaled_model(self):
        """
        بناء النموذج الموسَّع: نسختان من النموذج الأساسي، كل منهما
        تُقلَّم من أحد طرفيها، ثم تُركَّبان معًا.
        """
        copy_a = copy.deepcopy(self.base_model)
        copy_b = copy.deepcopy(self.base_model)

        # النسخة A: إزالة آخر n_layers_to_remove طبقة
        layers_a = copy_a.layers[: self.original_n_layers - self.n_layers_to_remove]

        # النسخة B: إزالة أول n_layers_to_remove طبقة
        layers_b = copy_b.layers[self.n_layers_to_remove:]

        # التركيب المباشر: كل طبقات A يليها كل طبقات B
        upscaled_layers = nn.ModuleList(list(layers_a) + list(layers_b))

        upscaled_model = copy.deepcopy(self.base_model)
        upscaled_model.layers = upscaled_layers

        new_n_layers = len(upscaled_layers)
        print(f"عدد الطبقات: الأصلي={self.original_n_layers} → "
              f"الموسَّع={new_n_layers} "
              f"(تضمين {self.original_n_layers * 2 - new_n_layers} "
              f"طبقة متداخلة مُزالة)")

        return upscaled_model

    def validate_layer_continuity(self, upscaled_model, sample_input):
        """
        فحص سريع قبل CPT: التأكد أن إخراج النموذج الموسَّع "معقول"
        إحصائيًا (ليس NaN أو قيمًا متفجرة)، حتى قبل أي تدريب إضافي -
        هذا ممكن لأن الطبقات المُركَّبة كل منها مُدرَّبة أصلاً جيدًا
        على مدخلات ومخرجات من نفس "النمط" العام، فالنموذج الموسَّع
        غير المُدرَّب لا يزال "يعمل" بشكل خام معقول، وإن لم يكن أمثل.
        """
        with torch.no_grad():
            output = upscaled_model(sample_input)
            has_nan = torch.isnan(output.logits).any().item()
            output_std = output.logits.std().item()

        return {
            "has_nan": has_nan,
            "output_std": output_std,
            "status": "✅ جاهز لـ CPT" if not has_nan else "❌ مشكلة - راجع نقطة الدمج",
        }

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-28-ch28-depth-upscaling.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def estimate_upscaling_cost_savings(
    base_model_params_billions: float,
    target_model_params_billions: float,
    cpt_tokens_billions: float,
    from_scratch_tokens_billions: float,  # وفق Chinchilla (الفصل 20): ≈20×N
) -> dict:
    """
    مقارنة تقديرية للتكلفة الحاسوبية (FLOPs) بين بناء نموذج بالحجم
    المستهدف عبر Depth Upscaling + CPT، مقابل تدريبه من الصفر بالكامل.
    """
    # تقدير FLOPs بقانون Kaplan/Chinchilla المبسّط: FLOPs ≈ 6 × N × D
    flops_upscaling_cpt = 6 * (target_model_params_billions * 1e9) * (cpt_tokens_billions * 1e9)
    flops_from_scratch = 6 * (target_model_params_billions * 1e9) * (from_scratch_tokens_billions * 1e9)

    savings_factor = flops_from_scratch / flops_upscaling_cpt

    return {
        "flops_upscaling_approach": f"{flops_upscaling_cpt:.2e}",
        "flops_from_scratch_approach": f"{flops_from_scratch:.2e}",
        "savings_factor": f"{savings_factor:.1f}×",
        "conclusion": (
            f"بناء نموذج {target_model_params_billions}B عبر Depth Upscaling "
            f"من نموذج {base_model_params_billions}B موجود يوفر تقريبًا "
            f"{savings_factor:.0f} ضعف الحوسبة مقارنة بالتدريب من الصفر"
        ),
    }

result = estimate_upscaling_cost_savings(
    base_model_params_billions=7.0,
    target_model_params_billions=10.7,
    cpt_tokens_billions=3,       # SOLAR استخدمت تدريبًا إضافيًا محدودًا نسبيًا
    from_scratch_tokens_billions=10.7 * 20,  # وفق Chinchilla لنموذج 10.7B
)
print(result)
# النتيجة التقريبية: savings_factor ≈ 71× - توفير هائل جدًا في الحوسبة!

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-28-ch28-depth-upscaling.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-28-ch28-depth-upscaling.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — توسيع العمق: Depth Upscaling
Mastering Transformers and Large Language Models — Depth Upscaling

Book slug : mastering-llms-transformers
Chapter   : ch-28-ch28-depth-upscaling (lesson 28)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-06-model-reuse/ch28-depth-upscaling.ipynb#cell-1
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
# [002] content/notebooks/part-06-model-reuse/ch28-depth-upscaling.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import copy

class DepthUpscaler:
    """
    تطبيق كامل لمنهجية Depth Upscaling بأسلوب SOLAR 10.7B: بناء نموذج
    أعمق من نسختين من نفس النموذج الأساسي عبر التركيب المباشر
    (Passthrough) للطبقات، تمهيدًا لمرحلة Continued Pretraining.
    """
    def __init__(self, base_model, n_layers_to_remove: int = 8):
        self.base_model = base_model
        self.n_layers_to_remove = n_layers_to_remove
        self.original_n_layers = len(base_model.layers)

    def create_upscaled_model(self):
        """
        بناء النموذج الموسَّع: نسختان من النموذج الأساسي، كل منهما
        تُقلَّم من أحد طرفيها، ثم تُركَّبان معًا.
        """
        copy_a = copy.deepcopy(self.base_model)
        copy_b = copy.deepcopy(self.base_model)

        # النسخة A: إزالة آخر n_layers_to_remove طبقة
        layers_a = copy_a.layers[: self.original_n_layers - self.n_layers_to_remove]

        # النسخة B: إزالة أول n_layers_to_remove طبقة
        layers_b = copy_b.layers[self.n_layers_to_remove:]

        # التركيب المباشر: كل طبقات A يليها كل طبقات B
        upscaled_layers = nn.ModuleList(list(layers_a) + list(layers_b))

        upscaled_model = copy.deepcopy(self.base_model)
        upscaled_model.layers = upscaled_layers

        new_n_layers = len(upscaled_layers)
        print(f"عدد الطبقات: الأصلي={self.original_n_layers} → "
              f"الموسَّع={new_n_layers} "
              f"(تضمين {self.original_n_layers * 2 - new_n_layers} "
              f"طبقة متداخلة مُزالة)")

        return upscaled_model

    def validate_layer_continuity(self, upscaled_model, sample_input):
        """
        فحص سريع قبل CPT: التأكد أن إخراج النموذج الموسَّع "معقول"
        إحصائيًا (ليس NaN أو قيمًا متفجرة)، حتى قبل أي تدريب إضافي -
        هذا ممكن لأن الطبقات المُركَّبة كل منها مُدرَّبة أصلاً جيدًا
        على مدخلات ومخرجات من نفس "النمط" العام، فالنموذج الموسَّع
        غير المُدرَّب لا يزال "يعمل" بشكل خام معقول، وإن لم يكن أمثل.
        """
        with torch.no_grad():
            output = upscaled_model(sample_input)
            has_nan = torch.isnan(output.logits).any().item()
            output_std = output.logits.std().item()

        return {
            "has_nan": has_nan,
            "output_std": output_std,
            "status": "✅ جاهز لـ CPT" if not has_nan else "❌ مشكلة - راجع نقطة الدمج",
        }

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-06-model-reuse/ch28-depth-upscaling.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def estimate_upscaling_cost_savings(
    base_model_params_billions: float,
    target_model_params_billions: float,
    cpt_tokens_billions: float,
    from_scratch_tokens_billions: float,  # وفق Chinchilla (الفصل 20): ≈20×N
) -> dict:
    """
    مقارنة تقديرية للتكلفة الحاسوبية (FLOPs) بين بناء نموذج بالحجم
    المستهدف عبر Depth Upscaling + CPT، مقابل تدريبه من الصفر بالكامل.
    """
    # تقدير FLOPs بقانون Kaplan/Chinchilla المبسّط: FLOPs ≈ 6 × N × D
    flops_upscaling_cpt = 6 * (target_model_params_billions * 1e9) * (cpt_tokens_billions * 1e9)
    flops_from_scratch = 6 * (target_model_params_billions * 1e9) * (from_scratch_tokens_billions * 1e9)

    savings_factor = flops_from_scratch / flops_upscaling_cpt

    return {
        "flops_upscaling_approach": f"{flops_upscaling_cpt:.2e}",
        "flops_from_scratch_approach": f"{flops_from_scratch:.2e}",
        "savings_factor": f"{savings_factor:.1f}×",
        "conclusion": (
            f"بناء نموذج {target_model_params_billions}B عبر Depth Upscaling "
            f"من نموذج {base_model_params_billions}B موجود يوفر تقريبًا "
            f"{savings_factor:.0f} ضعف الحوسبة مقارنة بالتدريب من الصفر"
        ),
    }

result = estimate_upscaling_cost_savings(
    base_model_params_billions=7.0,
    target_model_params_billions=10.7,
    cpt_tokens_billions=3,       # SOLAR استخدمت تدريبًا إضافيًا محدودًا نسبيًا
    from_scratch_tokens_billions=10.7 * 20,  # وفق Chinchilla لنموذج 10.7B
)
print(result)
# النتيجة التقريبية: savings_factor ≈ 71× - توفير هائل جدًا في الحوسبة!

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-06-model-reuse/ch28-depth-upscaling.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-28-ch28-depth-upscaling.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-28-ch28-depth-upscaling.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import copy

class DepthUpscaler:
    """
    تطبيق كامل لمنهجية Depth Upscaling بأسلوب SOLAR 10.7B: بناء نموذج
    أعمق من نسختين من نفس النموذج الأساسي عبر التركيب المباشر
    (Passthrough) للطبقات، تمهيدًا لمرحلة Continued Pretraining.
    """
    def __init__(self, base_model, n_layers_to_remove: int = 8):
        self.base_model = base_model
        self.n_layers_to_remove = n_layers_to_remove
        self.original_n_layers = len(base_model.layers)

    def create_upscaled_model(self):
        """
        بناء النموذج الموسَّع: نسختان من النموذج الأساسي، كل منهما
        تُقلَّم من أحد طرفيها، ثم تُركَّبان معًا.
        """
        copy_a = copy.deepcopy(self.base_model)
        copy_b = copy.deepcopy(self.base_model)

        # النسخة A: إزالة آخر n_layers_to_remove طبقة
        layers_a = copy_a.layers[: self.original_n_layers - self.n_layers_to_remove]

        # النسخة B: إزالة أول n_layers_to_remove طبقة
        layers_b = copy_b.layers[self.n_layers_to_remove:]

        # التركيب المباشر: كل طبقات A يليها كل طبقات B
        upscaled_layers = nn.ModuleList(list(layers_a) + list(layers_b))

        upscaled_model = copy.deepcopy(self.base_model)
        upscaled_model.layers = upscaled_layers

        new_n_layers = len(upscaled_layers)
        print(f"عدد الطبقات: الأصلي={self.original_n_layers} → "
              f"الموسَّع={new_n_layers} "
              f"(تضمين {self.original_n_layers * 2 - new_n_layers} "
              f"طبقة متداخلة مُزالة)")

        return upscaled_model

    def validate_layer_continuity(self, upscaled_model, sample_input):
        """
        فحص سريع قبل CPT: التأكد أن إخراج النموذج الموسَّع "معقول"
        إحصائيًا (ليس NaN أو قيمًا متفجرة)، حتى قبل أي تدريب إضافي -
        هذا ممكن لأن الطبقات المُركَّبة كل منها مُدرَّبة أصلاً جيدًا
        على مدخلات ومخرجات من نفس "النمط" العام، فالنموذج الموسَّع
        غير المُدرَّب لا يزال "يعمل" بشكل خام معقول، وإن لم يكن أمثل.
        """
        with torch.no_grad():
            output = upscaled_model(sample_input)
            has_nan = torch.isnan(output.logits).any().item()
            output_std = output.logits.std().item()

        return {
            "has_nan": has_nan,
            "output_std": output_std,
            "status": "✅ جاهز لـ CPT" if not has_nan else "❌ مشكلة - راجع نقطة الدمج",
        }

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-28-ch28-depth-upscaling.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def estimate_upscaling_cost_savings(
    base_model_params_billions: float,
    target_model_params_billions: float,
    cpt_tokens_billions: float,
    from_scratch_tokens_billions: float,  # وفق Chinchilla (الفصل 20): ≈20×N
) -> dict:
    """
    مقارنة تقديرية للتكلفة الحاسوبية (FLOPs) بين بناء نموذج بالحجم
    المستهدف عبر Depth Upscaling + CPT، مقابل تدريبه من الصفر بالكامل.
    """
    # تقدير FLOPs بقانون Kaplan/Chinchilla المبسّط: FLOPs ≈ 6 × N × D
    flops_upscaling_cpt = 6 * (target_model_params_billions * 1e9) * (cpt_tokens_billions * 1e9)
    flops_from_scratch = 6 * (target_model_params_billions * 1e9) * (from_scratch_tokens_billions * 1e9)

    savings_factor = flops_from_scratch / flops_upscaling_cpt

    return {
        "flops_upscaling_approach": f"{flops_upscaling_cpt:.2e}",
        "flops_from_scratch_approach": f"{flops_from_scratch:.2e}",
        "savings_factor": f"{savings_factor:.1f}×",
        "conclusion": (
            f"بناء نموذج {target_model_params_billions}B عبر Depth Upscaling "
            f"من نموذج {base_model_params_billions}B موجود يوفر تقريبًا "
            f"{savings_factor:.0f} ضعف الحوسبة مقارنة بالتدريب من الصفر"
        ),
    }

result = estimate_upscaling_cost_savings(
    base_model_params_billions=7.0,
    target_model_params_billions=10.7,
    cpt_tokens_billions=3,       # SOLAR استخدمت تدريبًا إضافيًا محدودًا نسبيًا
    from_scratch_tokens_billions=10.7 * 20,  # وفق Chinchilla لنموذج 10.7B
)
print(result)
# النتيجة التقريبية: savings_factor ≈ 71× - توفير هائل جدًا في الحوسبة!

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-28-ch28-depth-upscaling.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
