#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الحوسبة الفعالة والذكاء على الحافة
Mastering Transformers and Large Language Models — Edge Computing

Book slug : mastering-llms-transformers
Chapter   : ch-63-ch63-edge-computing (lesson 63)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-63-ch63-edge-computing.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الحوسبة الفعالة والذكاء على الحافة
Mastering Transformers and Large Language Models — Edge Computing

Book slug : mastering-llms-transformers
Chapter   : ch-63-ch63-edge-computing (lesson 63)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-63-ch63-edge-computing.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الحوسبة الفعالة والذكاء على الحافة
Mastering Transformers and Large Language Models — Edge Computing

Book slug : mastering-llms-transformers
Chapter   : ch-63-ch63-edge-computing (lesson 63)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-12-future/ch63-edge-computing.ipynb#cell-1
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
# [002] content/notebooks/part-12-future/ch63-edge-computing.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class BitLinear(nn.Module):
    """
    تطبيق مفاهيمي لطبقة BitLinear من BitNet b1.58: الأوزان تُكمَّم
    إلى ثلاث قيم فقط {-1, 0, +1} أثناء التمرير الأمامي، لكن تُحفَظ
    نسخة كاملة الدقة (FP16/FP32) للأوزان أثناء التدريب (Straight-Through
    Estimator للتدرجات عبر عملية التكميم غير القابلة للاشتقاق).
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # الأوزان الكاملة الدقة (تُحدَّث أثناء التدريب فقط - "Latent Weights")
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

    def quantize_weights_ternary(self, w: torch.Tensor) -> torch.Tensor:
        """
        تكميم الأوزان إلى {-1, 0, +1} باستخدام عتبة = متوسط القيمة المطلقة
        (absmean quantization - الطريقة المستخدمة فعليًا في ورقة BitNet).
        """
        scale = w.abs().mean()  # γ في الورقة الأصلية
        w_scaled = w / (scale + 1e-8)
        # التقريب لأقرب قيمة في {-1, 0, 1} مع القص
        w_quantized = torch.clamp(torch.round(w_scaled), -1, 1)
        return w_quantized, scale

    def quantize_activations_int8(self, x: torch.Tensor) -> torch.Tensor:
        """تكميم التنشيطات إلى 8-bit (absmax quantization لكل صف)."""
        scale = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_quantized = torch.clamp(
            torch.round(x * 127.0 / scale), -128, 127
        )
        return x_quantized, scale / 127.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Straight-Through Estimator: نستخدم القيم المُكمَّمة في التمرير
        # الأمامي لكن نسمح للتدرج بالمرور كأن التكميم لم يحدث (STE trick)
        w_quant, w_scale = self.quantize_weights_ternary(self.weight)
        w_ste = self.weight + (w_quant - self.weight).detach()  # STE

        x_quant, x_scale = self.quantize_activations_int8(x)
        x_ste = x + (x_quant - x).detach()

        # الضرب المصفوفي يصبح نظريًا بدون عمليات ضرب حقيقية على الهاردوير
        # المُخصَّص (فقط جمع/طرح/تجاهل) - هذا مصدر التوفير الهائل في الطاقة
        output = torch.nn.functional.linear(x_ste, w_ste) * w_scale
        return output

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-12-future/ch63-edge-computing.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def estimate_slm_feasibility_on_device(
    model_params_billions: float,
    quantization_bits: int,
    device_ram_gb: float,
    target_tokens_per_second: float,
    device_memory_bandwidth_gbps: float,
) -> dict:
    """
    تقدير جدوى تشغيل نموذج صغير على جهاز حافة محدد، بناءً على قيدين
    جوهريين: الذاكرة المتاحة، وعرض النطاق الترددي للذاكرة (العامل
    الحاكم الفعلي لسرعة الاستدلال في معظم الحالات - وليس قوة المعالج).
    """
    # حجم النموذج التقريبي بعد التكميم
    model_size_gb = (model_params_billions * 1e9 * quantization_bits / 8) / 1e9

    # هامش أمان للذاكرة (KV cache + overhead النظام)
    required_ram_gb = model_size_gb * 1.3

    fits_in_memory = required_ram_gb <= device_ram_gb

    # الاستدلال في نماذج اللغة غالبًا Memory-Bound: السرعة القصوى النظرية
    # = عرض النطاق الترددي للذاكرة / حجم النموذج (لكل رمز يجب قراءة كل الأوزان مرة)
    theoretical_max_tokens_per_sec = (
        device_memory_bandwidth_gbps * 1e9 / (model_size_gb * 1e9)
    )

    meets_speed_target = theoretical_max_tokens_per_sec >= target_tokens_per_second

    return {
        "model_size_gb": round(model_size_gb, 2),
        "required_ram_gb": round(required_ram_gb, 2),
        "fits_in_memory": fits_in_memory,
        "theoretical_max_tokens_per_sec": round(theoretical_max_tokens_per_sec, 1),
        "meets_speed_target": meets_speed_target,
        "recommendation": (
            "✅ قابل للتشغيل بكفاءة على هذا الجهاز"
            if fits_in_memory and meets_speed_target
            else "⚠️ يحتاج تكميم أكثر صرامة أو نموذجًا أصغر"
        ),
    }

# مثال: Phi-3.5-mini (3.8B) مُكمَّم 4-bit على هاتف متوسط
result = estimate_slm_feasibility_on_device(
    model_params_billions=3.8,
    quantization_bits=4,
    device_ram_gb=8.0,
    target_tokens_per_second=15.0,
    device_memory_bandwidth_gbps=25.6,  # نموذجي لهاتف متوسط المدى
)
print(result)
# النتيجة التقريبية: model_size_gb≈1.9, required_ram≈2.47,
# fits_in_memory=True, max_tokens/sec≈13.5 → قد لا يفي بالضبط بالهدف 15

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-12-future/ch63-edge-computing.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class EdgeModelDistillationPipeline:
    """
    خط أنابيب كامل لتقطير نموذج ضخم (مُعلِّم) إلى نموذج حافة صغير
    (تلميذ) جاهز للنشر، يجمع بين: Knowledge Distillation (الفصل 25
    من الجزء السادس - Model Reuse) + Quantization-Aware Training +
    تحسين خاص لمهمة ضيقة (Task-Specific Distillation).
    """
    def __init__(self, teacher_model, student_architecture_config: dict,
                 target_device_constraints: dict):
        self.teacher = teacher_model
        self.student_config = student_architecture_config
        self.constraints = target_device_constraints  # RAM، bandwidth، إلخ

    def generate_distillation_dataset(self, task_prompts: list,
                                       n_samples_per_prompt: int = 5) -> list:
        """
        توليد بيانات تقطير مُخصَّصة للمهمة المستهدفة: استجابات عالية
        الجودة من النموذج المُعلِّم تُستخدم كـ "soft targets" أو
        بيانات SFT مباشرة للتلميذ، مُركَّزة فقط على نطاق المهمة
        المستهدفة (لا حاجة لتغطية عامة واسعة كالنموذج الأصلي).
        """
        dataset = []
        for prompt in task_prompts:
            for _ in range(n_samples_per_prompt):
                teacher_response = self.teacher.generate(
                    prompt, temperature=0.7, max_tokens=512
                )
                dataset.append({"prompt": prompt, "response": teacher_response})
        return dataset

    def distill_with_qat(self, student_model, distillation_dataset: list,
                          quantization_bits: int = 4) -> nn.Module:
        """
        تدريب التلميذ مع Quantization-Aware Training (QAT): محاكاة
        تأثير التكميم أثناء التدريب نفسه (وليس بعده - Post-Training
        Quantization فقط)، لتعويض النموذج عن فقدان الدقة قبل حدوثه.
        """
        from torch.quantization import prepare_qat, get_default_qat_qconfig

        student_model.qconfig = get_default_qat_qconfig('fbgemm')
        student_prepared = prepare_qat(student_model, inplace=False)

        optimizer = torch.optim.AdamW(student_prepared.parameters(), lr=5e-5)

        for batch in self._batchify(distillation_dataset, batch_size=8):
            # دمج Hard-label loss (SFT عادي) مع Soft-label loss (KD -
            # راجع الفصل 25 لتفاصيل compute_distillation_loss الكاملة)
            loss = self._compute_combined_loss(student_prepared, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        from torch.quantization import convert
        student_quantized = convert(student_prepared.eval(), inplace=False)
        return student_quantized

    def validate_against_constraints(self, distilled_model) -> dict:
        """التحقق النهائي من مطابقة النموذج المُقطَّر لقيود الجهاز المستهدف."""
        model_size_mb = sum(
            p.numel() * p.element_size() for p in distilled_model.parameters()
        ) / (1024 ** 2)

        passes_ram_constraint = (
            model_size_mb / 1024 <= self.constraints.get("max_ram_gb", 4.0)
        )
        return {
            "model_size_mb": round(model_size_mb, 1),
            "passes_ram_constraint": passes_ram_constraint,
        }

    def _batchify(self, data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i+batch_size]

    def _compute_combined_loss(self, model, batch):
        raise NotImplementedError("يُطبَّق بحسب واجهة النموذج - راجع الفصل 25")

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-12-future/ch63-edge-computing.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-63-ch63-edge-computing.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-63-ch63-edge-computing.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class BitLinear(nn.Module):
    """
    تطبيق مفاهيمي لطبقة BitLinear من BitNet b1.58: الأوزان تُكمَّم
    إلى ثلاث قيم فقط {-1, 0, +1} أثناء التمرير الأمامي، لكن تُحفَظ
    نسخة كاملة الدقة (FP16/FP32) للأوزان أثناء التدريب (Straight-Through
    Estimator للتدرجات عبر عملية التكميم غير القابلة للاشتقاق).
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # الأوزان الكاملة الدقة (تُحدَّث أثناء التدريب فقط - "Latent Weights")
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

    def quantize_weights_ternary(self, w: torch.Tensor) -> torch.Tensor:
        """
        تكميم الأوزان إلى {-1, 0, +1} باستخدام عتبة = متوسط القيمة المطلقة
        (absmean quantization - الطريقة المستخدمة فعليًا في ورقة BitNet).
        """
        scale = w.abs().mean()  # γ في الورقة الأصلية
        w_scaled = w / (scale + 1e-8)
        # التقريب لأقرب قيمة في {-1, 0, 1} مع القص
        w_quantized = torch.clamp(torch.round(w_scaled), -1, 1)
        return w_quantized, scale

    def quantize_activations_int8(self, x: torch.Tensor) -> torch.Tensor:
        """تكميم التنشيطات إلى 8-bit (absmax quantization لكل صف)."""
        scale = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_quantized = torch.clamp(
            torch.round(x * 127.0 / scale), -128, 127
        )
        return x_quantized, scale / 127.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Straight-Through Estimator: نستخدم القيم المُكمَّمة في التمرير
        # الأمامي لكن نسمح للتدرج بالمرور كأن التكميم لم يحدث (STE trick)
        w_quant, w_scale = self.quantize_weights_ternary(self.weight)
        w_ste = self.weight + (w_quant - self.weight).detach()  # STE

        x_quant, x_scale = self.quantize_activations_int8(x)
        x_ste = x + (x_quant - x).detach()

        # الضرب المصفوفي يصبح نظريًا بدون عمليات ضرب حقيقية على الهاردوير
        # المُخصَّص (فقط جمع/طرح/تجاهل) - هذا مصدر التوفير الهائل في الطاقة
        output = torch.nn.functional.linear(x_ste, w_ste) * w_scale
        return output

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-63-ch63-edge-computing.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def estimate_slm_feasibility_on_device(
    model_params_billions: float,
    quantization_bits: int,
    device_ram_gb: float,
    target_tokens_per_second: float,
    device_memory_bandwidth_gbps: float,
) -> dict:
    """
    تقدير جدوى تشغيل نموذج صغير على جهاز حافة محدد، بناءً على قيدين
    جوهريين: الذاكرة المتاحة، وعرض النطاق الترددي للذاكرة (العامل
    الحاكم الفعلي لسرعة الاستدلال في معظم الحالات - وليس قوة المعالج).
    """
    # حجم النموذج التقريبي بعد التكميم
    model_size_gb = (model_params_billions * 1e9 * quantization_bits / 8) / 1e9

    # هامش أمان للذاكرة (KV cache + overhead النظام)
    required_ram_gb = model_size_gb * 1.3

    fits_in_memory = required_ram_gb <= device_ram_gb

    # الاستدلال في نماذج اللغة غالبًا Memory-Bound: السرعة القصوى النظرية
    # = عرض النطاق الترددي للذاكرة / حجم النموذج (لكل رمز يجب قراءة كل الأوزان مرة)
    theoretical_max_tokens_per_sec = (
        device_memory_bandwidth_gbps * 1e9 / (model_size_gb * 1e9)
    )

    meets_speed_target = theoretical_max_tokens_per_sec >= target_tokens_per_second

    return {
        "model_size_gb": round(model_size_gb, 2),
        "required_ram_gb": round(required_ram_gb, 2),
        "fits_in_memory": fits_in_memory,
        "theoretical_max_tokens_per_sec": round(theoretical_max_tokens_per_sec, 1),
        "meets_speed_target": meets_speed_target,
        "recommendation": (
            "✅ قابل للتشغيل بكفاءة على هذا الجهاز"
            if fits_in_memory and meets_speed_target
            else "⚠️ يحتاج تكميم أكثر صرامة أو نموذجًا أصغر"
        ),
    }

# مثال: Phi-3.5-mini (3.8B) مُكمَّم 4-bit على هاتف متوسط
result = estimate_slm_feasibility_on_device(
    model_params_billions=3.8,
    quantization_bits=4,
    device_ram_gb=8.0,
    target_tokens_per_second=15.0,
    device_memory_bandwidth_gbps=25.6,  # نموذجي لهاتف متوسط المدى
)
print(result)
# النتيجة التقريبية: model_size_gb≈1.9, required_ram≈2.47,
# fits_in_memory=True, max_tokens/sec≈13.5 → قد لا يفي بالضبط بالهدف 15

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-63-ch63-edge-computing.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class EdgeModelDistillationPipeline:
    """
    خط أنابيب كامل لتقطير نموذج ضخم (مُعلِّم) إلى نموذج حافة صغير
    (تلميذ) جاهز للنشر، يجمع بين: Knowledge Distillation (الفصل 25
    من الجزء السادس - Model Reuse) + Quantization-Aware Training +
    تحسين خاص لمهمة ضيقة (Task-Specific Distillation).
    """
    def __init__(self, teacher_model, student_architecture_config: dict,
                 target_device_constraints: dict):
        self.teacher = teacher_model
        self.student_config = student_architecture_config
        self.constraints = target_device_constraints  # RAM، bandwidth، إلخ

    def generate_distillation_dataset(self, task_prompts: list,
                                       n_samples_per_prompt: int = 5) -> list:
        """
        توليد بيانات تقطير مُخصَّصة للمهمة المستهدفة: استجابات عالية
        الجودة من النموذج المُعلِّم تُستخدم كـ "soft targets" أو
        بيانات SFT مباشرة للتلميذ، مُركَّزة فقط على نطاق المهمة
        المستهدفة (لا حاجة لتغطية عامة واسعة كالنموذج الأصلي).
        """
        dataset = []
        for prompt in task_prompts:
            for _ in range(n_samples_per_prompt):
                teacher_response = self.teacher.generate(
                    prompt, temperature=0.7, max_tokens=512
                )
                dataset.append({"prompt": prompt, "response": teacher_response})
        return dataset

    def distill_with_qat(self, student_model, distillation_dataset: list,
                          quantization_bits: int = 4) -> nn.Module:
        """
        تدريب التلميذ مع Quantization-Aware Training (QAT): محاكاة
        تأثير التكميم أثناء التدريب نفسه (وليس بعده - Post-Training
        Quantization فقط)، لتعويض النموذج عن فقدان الدقة قبل حدوثه.
        """
        from torch.quantization import prepare_qat, get_default_qat_qconfig

        student_model.qconfig = get_default_qat_qconfig('fbgemm')
        student_prepared = prepare_qat(student_model, inplace=False)

        optimizer = torch.optim.AdamW(student_prepared.parameters(), lr=5e-5)

        for batch in self._batchify(distillation_dataset, batch_size=8):
            # دمج Hard-label loss (SFT عادي) مع Soft-label loss (KD -
            # راجع الفصل 25 لتفاصيل compute_distillation_loss الكاملة)
            loss = self._compute_combined_loss(student_prepared, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        from torch.quantization import convert
        student_quantized = convert(student_prepared.eval(), inplace=False)
        return student_quantized

    def validate_against_constraints(self, distilled_model) -> dict:
        """التحقق النهائي من مطابقة النموذج المُقطَّر لقيود الجهاز المستهدف."""
        model_size_mb = sum(
            p.numel() * p.element_size() for p in distilled_model.parameters()
        ) / (1024 ** 2)

        passes_ram_constraint = (
            model_size_mb / 1024 <= self.constraints.get("max_ram_gb", 4.0)
        )
        return {
            "model_size_mb": round(model_size_mb, 1),
            "passes_ram_constraint": passes_ram_constraint,
        }

    def _batchify(self, data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i+batch_size]

    def _compute_combined_loss(self, model, batch):
        raise NotImplementedError("يُطبَّق بحسب واجهة النموذج - راجع الفصل 25")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-63-ch63-edge-computing.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-63-ch63-edge-computing.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الحوسبة الفعالة والذكاء على الحافة
Mastering Transformers and Large Language Models — Edge Computing

Book slug : mastering-llms-transformers
Chapter   : ch-63-ch63-edge-computing (lesson 63)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-12-future/ch63-edge-computing.ipynb#cell-1
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
# [002] content/notebooks/part-12-future/ch63-edge-computing.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class BitLinear(nn.Module):
    """
    تطبيق مفاهيمي لطبقة BitLinear من BitNet b1.58: الأوزان تُكمَّم
    إلى ثلاث قيم فقط {-1, 0, +1} أثناء التمرير الأمامي، لكن تُحفَظ
    نسخة كاملة الدقة (FP16/FP32) للأوزان أثناء التدريب (Straight-Through
    Estimator للتدرجات عبر عملية التكميم غير القابلة للاشتقاق).
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # الأوزان الكاملة الدقة (تُحدَّث أثناء التدريب فقط - "Latent Weights")
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

    def quantize_weights_ternary(self, w: torch.Tensor) -> torch.Tensor:
        """
        تكميم الأوزان إلى {-1, 0, +1} باستخدام عتبة = متوسط القيمة المطلقة
        (absmean quantization - الطريقة المستخدمة فعليًا في ورقة BitNet).
        """
        scale = w.abs().mean()  # γ في الورقة الأصلية
        w_scaled = w / (scale + 1e-8)
        # التقريب لأقرب قيمة في {-1, 0, 1} مع القص
        w_quantized = torch.clamp(torch.round(w_scaled), -1, 1)
        return w_quantized, scale

    def quantize_activations_int8(self, x: torch.Tensor) -> torch.Tensor:
        """تكميم التنشيطات إلى 8-bit (absmax quantization لكل صف)."""
        scale = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_quantized = torch.clamp(
            torch.round(x * 127.0 / scale), -128, 127
        )
        return x_quantized, scale / 127.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Straight-Through Estimator: نستخدم القيم المُكمَّمة في التمرير
        # الأمامي لكن نسمح للتدرج بالمرور كأن التكميم لم يحدث (STE trick)
        w_quant, w_scale = self.quantize_weights_ternary(self.weight)
        w_ste = self.weight + (w_quant - self.weight).detach()  # STE

        x_quant, x_scale = self.quantize_activations_int8(x)
        x_ste = x + (x_quant - x).detach()

        # الضرب المصفوفي يصبح نظريًا بدون عمليات ضرب حقيقية على الهاردوير
        # المُخصَّص (فقط جمع/طرح/تجاهل) - هذا مصدر التوفير الهائل في الطاقة
        output = torch.nn.functional.linear(x_ste, w_ste) * w_scale
        return output

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-12-future/ch63-edge-computing.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def estimate_slm_feasibility_on_device(
    model_params_billions: float,
    quantization_bits: int,
    device_ram_gb: float,
    target_tokens_per_second: float,
    device_memory_bandwidth_gbps: float,
) -> dict:
    """
    تقدير جدوى تشغيل نموذج صغير على جهاز حافة محدد، بناءً على قيدين
    جوهريين: الذاكرة المتاحة، وعرض النطاق الترددي للذاكرة (العامل
    الحاكم الفعلي لسرعة الاستدلال في معظم الحالات - وليس قوة المعالج).
    """
    # حجم النموذج التقريبي بعد التكميم
    model_size_gb = (model_params_billions * 1e9 * quantization_bits / 8) / 1e9

    # هامش أمان للذاكرة (KV cache + overhead النظام)
    required_ram_gb = model_size_gb * 1.3

    fits_in_memory = required_ram_gb <= device_ram_gb

    # الاستدلال في نماذج اللغة غالبًا Memory-Bound: السرعة القصوى النظرية
    # = عرض النطاق الترددي للذاكرة / حجم النموذج (لكل رمز يجب قراءة كل الأوزان مرة)
    theoretical_max_tokens_per_sec = (
        device_memory_bandwidth_gbps * 1e9 / (model_size_gb * 1e9)
    )

    meets_speed_target = theoretical_max_tokens_per_sec >= target_tokens_per_second

    return {
        "model_size_gb": round(model_size_gb, 2),
        "required_ram_gb": round(required_ram_gb, 2),
        "fits_in_memory": fits_in_memory,
        "theoretical_max_tokens_per_sec": round(theoretical_max_tokens_per_sec, 1),
        "meets_speed_target": meets_speed_target,
        "recommendation": (
            "✅ قابل للتشغيل بكفاءة على هذا الجهاز"
            if fits_in_memory and meets_speed_target
            else "⚠️ يحتاج تكميم أكثر صرامة أو نموذجًا أصغر"
        ),
    }

# مثال: Phi-3.5-mini (3.8B) مُكمَّم 4-bit على هاتف متوسط
result = estimate_slm_feasibility_on_device(
    model_params_billions=3.8,
    quantization_bits=4,
    device_ram_gb=8.0,
    target_tokens_per_second=15.0,
    device_memory_bandwidth_gbps=25.6,  # نموذجي لهاتف متوسط المدى
)
print(result)
# النتيجة التقريبية: model_size_gb≈1.9, required_ram≈2.47,
# fits_in_memory=True, max_tokens/sec≈13.5 → قد لا يفي بالضبط بالهدف 15

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-12-future/ch63-edge-computing.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class EdgeModelDistillationPipeline:
    """
    خط أنابيب كامل لتقطير نموذج ضخم (مُعلِّم) إلى نموذج حافة صغير
    (تلميذ) جاهز للنشر، يجمع بين: Knowledge Distillation (الفصل 25
    من الجزء السادس - Model Reuse) + Quantization-Aware Training +
    تحسين خاص لمهمة ضيقة (Task-Specific Distillation).
    """
    def __init__(self, teacher_model, student_architecture_config: dict,
                 target_device_constraints: dict):
        self.teacher = teacher_model
        self.student_config = student_architecture_config
        self.constraints = target_device_constraints  # RAM، bandwidth، إلخ

    def generate_distillation_dataset(self, task_prompts: list,
                                       n_samples_per_prompt: int = 5) -> list:
        """
        توليد بيانات تقطير مُخصَّصة للمهمة المستهدفة: استجابات عالية
        الجودة من النموذج المُعلِّم تُستخدم كـ "soft targets" أو
        بيانات SFT مباشرة للتلميذ، مُركَّزة فقط على نطاق المهمة
        المستهدفة (لا حاجة لتغطية عامة واسعة كالنموذج الأصلي).
        """
        dataset = []
        for prompt in task_prompts:
            for _ in range(n_samples_per_prompt):
                teacher_response = self.teacher.generate(
                    prompt, temperature=0.7, max_tokens=512
                )
                dataset.append({"prompt": prompt, "response": teacher_response})
        return dataset

    def distill_with_qat(self, student_model, distillation_dataset: list,
                          quantization_bits: int = 4) -> nn.Module:
        """
        تدريب التلميذ مع Quantization-Aware Training (QAT): محاكاة
        تأثير التكميم أثناء التدريب نفسه (وليس بعده - Post-Training
        Quantization فقط)، لتعويض النموذج عن فقدان الدقة قبل حدوثه.
        """
        from torch.quantization import prepare_qat, get_default_qat_qconfig

        student_model.qconfig = get_default_qat_qconfig('fbgemm')
        student_prepared = prepare_qat(student_model, inplace=False)

        optimizer = torch.optim.AdamW(student_prepared.parameters(), lr=5e-5)

        for batch in self._batchify(distillation_dataset, batch_size=8):
            # دمج Hard-label loss (SFT عادي) مع Soft-label loss (KD -
            # راجع الفصل 25 لتفاصيل compute_distillation_loss الكاملة)
            loss = self._compute_combined_loss(student_prepared, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        from torch.quantization import convert
        student_quantized = convert(student_prepared.eval(), inplace=False)
        return student_quantized

    def validate_against_constraints(self, distilled_model) -> dict:
        """التحقق النهائي من مطابقة النموذج المُقطَّر لقيود الجهاز المستهدف."""
        model_size_mb = sum(
            p.numel() * p.element_size() for p in distilled_model.parameters()
        ) / (1024 ** 2)

        passes_ram_constraint = (
            model_size_mb / 1024 <= self.constraints.get("max_ram_gb", 4.0)
        )
        return {
            "model_size_mb": round(model_size_mb, 1),
            "passes_ram_constraint": passes_ram_constraint,
        }

    def _batchify(self, data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i+batch_size]

    def _compute_combined_loss(self, model, batch):
        raise NotImplementedError("يُطبَّق بحسب واجهة النموذج - راجع الفصل 25")

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-12-future/ch63-edge-computing.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-63-ch63-edge-computing.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-63-ch63-edge-computing.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class BitLinear(nn.Module):
    """
    تطبيق مفاهيمي لطبقة BitLinear من BitNet b1.58: الأوزان تُكمَّم
    إلى ثلاث قيم فقط {-1, 0, +1} أثناء التمرير الأمامي، لكن تُحفَظ
    نسخة كاملة الدقة (FP16/FP32) للأوزان أثناء التدريب (Straight-Through
    Estimator للتدرجات عبر عملية التكميم غير القابلة للاشتقاق).
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # الأوزان الكاملة الدقة (تُحدَّث أثناء التدريب فقط - "Latent Weights")
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

    def quantize_weights_ternary(self, w: torch.Tensor) -> torch.Tensor:
        """
        تكميم الأوزان إلى {-1, 0, +1} باستخدام عتبة = متوسط القيمة المطلقة
        (absmean quantization - الطريقة المستخدمة فعليًا في ورقة BitNet).
        """
        scale = w.abs().mean()  # γ في الورقة الأصلية
        w_scaled = w / (scale + 1e-8)
        # التقريب لأقرب قيمة في {-1, 0, 1} مع القص
        w_quantized = torch.clamp(torch.round(w_scaled), -1, 1)
        return w_quantized, scale

    def quantize_activations_int8(self, x: torch.Tensor) -> torch.Tensor:
        """تكميم التنشيطات إلى 8-bit (absmax quantization لكل صف)."""
        scale = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_quantized = torch.clamp(
            torch.round(x * 127.0 / scale), -128, 127
        )
        return x_quantized, scale / 127.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Straight-Through Estimator: نستخدم القيم المُكمَّمة في التمرير
        # الأمامي لكن نسمح للتدرج بالمرور كأن التكميم لم يحدث (STE trick)
        w_quant, w_scale = self.quantize_weights_ternary(self.weight)
        w_ste = self.weight + (w_quant - self.weight).detach()  # STE

        x_quant, x_scale = self.quantize_activations_int8(x)
        x_ste = x + (x_quant - x).detach()

        # الضرب المصفوفي يصبح نظريًا بدون عمليات ضرب حقيقية على الهاردوير
        # المُخصَّص (فقط جمع/طرح/تجاهل) - هذا مصدر التوفير الهائل في الطاقة
        output = torch.nn.functional.linear(x_ste, w_ste) * w_scale
        return output

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-63-ch63-edge-computing.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def estimate_slm_feasibility_on_device(
    model_params_billions: float,
    quantization_bits: int,
    device_ram_gb: float,
    target_tokens_per_second: float,
    device_memory_bandwidth_gbps: float,
) -> dict:
    """
    تقدير جدوى تشغيل نموذج صغير على جهاز حافة محدد، بناءً على قيدين
    جوهريين: الذاكرة المتاحة، وعرض النطاق الترددي للذاكرة (العامل
    الحاكم الفعلي لسرعة الاستدلال في معظم الحالات - وليس قوة المعالج).
    """
    # حجم النموذج التقريبي بعد التكميم
    model_size_gb = (model_params_billions * 1e9 * quantization_bits / 8) / 1e9

    # هامش أمان للذاكرة (KV cache + overhead النظام)
    required_ram_gb = model_size_gb * 1.3

    fits_in_memory = required_ram_gb <= device_ram_gb

    # الاستدلال في نماذج اللغة غالبًا Memory-Bound: السرعة القصوى النظرية
    # = عرض النطاق الترددي للذاكرة / حجم النموذج (لكل رمز يجب قراءة كل الأوزان مرة)
    theoretical_max_tokens_per_sec = (
        device_memory_bandwidth_gbps * 1e9 / (model_size_gb * 1e9)
    )

    meets_speed_target = theoretical_max_tokens_per_sec >= target_tokens_per_second

    return {
        "model_size_gb": round(model_size_gb, 2),
        "required_ram_gb": round(required_ram_gb, 2),
        "fits_in_memory": fits_in_memory,
        "theoretical_max_tokens_per_sec": round(theoretical_max_tokens_per_sec, 1),
        "meets_speed_target": meets_speed_target,
        "recommendation": (
            "✅ قابل للتشغيل بكفاءة على هذا الجهاز"
            if fits_in_memory and meets_speed_target
            else "⚠️ يحتاج تكميم أكثر صرامة أو نموذجًا أصغر"
        ),
    }

# مثال: Phi-3.5-mini (3.8B) مُكمَّم 4-bit على هاتف متوسط
result = estimate_slm_feasibility_on_device(
    model_params_billions=3.8,
    quantization_bits=4,
    device_ram_gb=8.0,
    target_tokens_per_second=15.0,
    device_memory_bandwidth_gbps=25.6,  # نموذجي لهاتف متوسط المدى
)
print(result)
# النتيجة التقريبية: model_size_gb≈1.9, required_ram≈2.47,
# fits_in_memory=True, max_tokens/sec≈13.5 → قد لا يفي بالضبط بالهدف 15

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-63-ch63-edge-computing.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class EdgeModelDistillationPipeline:
    """
    خط أنابيب كامل لتقطير نموذج ضخم (مُعلِّم) إلى نموذج حافة صغير
    (تلميذ) جاهز للنشر، يجمع بين: Knowledge Distillation (الفصل 25
    من الجزء السادس - Model Reuse) + Quantization-Aware Training +
    تحسين خاص لمهمة ضيقة (Task-Specific Distillation).
    """
    def __init__(self, teacher_model, student_architecture_config: dict,
                 target_device_constraints: dict):
        self.teacher = teacher_model
        self.student_config = student_architecture_config
        self.constraints = target_device_constraints  # RAM، bandwidth، إلخ

    def generate_distillation_dataset(self, task_prompts: list,
                                       n_samples_per_prompt: int = 5) -> list:
        """
        توليد بيانات تقطير مُخصَّصة للمهمة المستهدفة: استجابات عالية
        الجودة من النموذج المُعلِّم تُستخدم كـ "soft targets" أو
        بيانات SFT مباشرة للتلميذ، مُركَّزة فقط على نطاق المهمة
        المستهدفة (لا حاجة لتغطية عامة واسعة كالنموذج الأصلي).
        """
        dataset = []
        for prompt in task_prompts:
            for _ in range(n_samples_per_prompt):
                teacher_response = self.teacher.generate(
                    prompt, temperature=0.7, max_tokens=512
                )
                dataset.append({"prompt": prompt, "response": teacher_response})
        return dataset

    def distill_with_qat(self, student_model, distillation_dataset: list,
                          quantization_bits: int = 4) -> nn.Module:
        """
        تدريب التلميذ مع Quantization-Aware Training (QAT): محاكاة
        تأثير التكميم أثناء التدريب نفسه (وليس بعده - Post-Training
        Quantization فقط)، لتعويض النموذج عن فقدان الدقة قبل حدوثه.
        """
        from torch.quantization import prepare_qat, get_default_qat_qconfig

        student_model.qconfig = get_default_qat_qconfig('fbgemm')
        student_prepared = prepare_qat(student_model, inplace=False)

        optimizer = torch.optim.AdamW(student_prepared.parameters(), lr=5e-5)

        for batch in self._batchify(distillation_dataset, batch_size=8):
            # دمج Hard-label loss (SFT عادي) مع Soft-label loss (KD -
            # راجع الفصل 25 لتفاصيل compute_distillation_loss الكاملة)
            loss = self._compute_combined_loss(student_prepared, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        from torch.quantization import convert
        student_quantized = convert(student_prepared.eval(), inplace=False)
        return student_quantized

    def validate_against_constraints(self, distilled_model) -> dict:
        """التحقق النهائي من مطابقة النموذج المُقطَّر لقيود الجهاز المستهدف."""
        model_size_mb = sum(
            p.numel() * p.element_size() for p in distilled_model.parameters()
        ) / (1024 ** 2)

        passes_ram_constraint = (
            model_size_mb / 1024 <= self.constraints.get("max_ram_gb", 4.0)
        )
        return {
            "model_size_mb": round(model_size_mb, 1),
            "passes_ram_constraint": passes_ram_constraint,
        }

    def _batchify(self, data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i+batch_size]

    def _compute_combined_loss(self, model, batch):
        raise NotImplementedError("يُطبَّق بحسب واجهة النموذج - راجع الفصل 25")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-63-ch63-edge-computing.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
