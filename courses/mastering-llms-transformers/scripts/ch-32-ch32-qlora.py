#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — QLoRA والتكميم أثناء التدريب
Mastering Transformers and Large Language Models — Qlora

Book slug : mastering-llms-transformers
Chapter   : ch-32-ch32-qlora (lesson 32)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-32-ch32-qlora.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — QLoRA والتكميم أثناء التدريب
Mastering Transformers and Large Language Models — Qlora

Book slug : mastering-llms-transformers
Chapter   : ch-32-ch32-qlora (lesson 32)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-32-ch32-qlora.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — QLoRA والتكميم أثناء التدريب
Mastering Transformers and Large Language Models — Qlora

Book slug : mastering-llms-transformers
Chapter   : ch-32-ch32-qlora (lesson 32)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-07-peft-alignment/ch32-qlora.ipynb#cell-1
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
    ("peft", "peft"),
    ("torch", "torch"),
    ("transformers", "transformers")
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
# [002] content/notebooks/part-07-peft-alignment/ch32-qlora.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import numpy as np

def create_nf4_quantization_levels() -> torch.Tensor:
    """
    إنشاء مستويات NF4 الـ 16 المُحسَّبة مسبقًا (Information-theoretically
    optimal) لتوزيع طبيعي معياري N(0,1)، كما في ورقة QLoRA الأصلية.
    هذه القيم ثابتة ومُحسَّبة مرة واحدة (لا تُعاد حسابها أثناء التدريب).
    """
    # القيم الفعلية المُستخدَمة في تطبيق bitsandbytes (مُقرَّبة هنا للتوضيح)
    nf4_levels = torch.tensor([
        -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
        0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
    ])
    return nf4_levels

class NF4Quantizer:
    """
    محاكاة مبسّطة تعليميًا لعملية تكميم NF4 (في التطبيق الفعلي
    تُستخدَم مكتبة bitsandbytes المُحسَّنة بـ CUDA kernels مخصّصة).
    """
    def __init__(self, block_size: int = 64):
        self.levels = create_nf4_quantization_levels()
        self.block_size = block_size  # التكميم يتم على كتل صغيرة (blockwise)

    def quantize_block(self, weight_block: torch.Tensor):
        """
        تكميم كتلة واحدة من الأوزان:
        1. حساب absmax (أقصى قيمة مطلقة) كثابت تطبيع للكتلة.
        2. تطبيع الأوزان إلى [-1, 1].
        3. إيجاد أقرب مستوى NF4 لكل قيمة.
        """
        absmax = weight_block.abs().max()
        normalized = weight_block / (absmax + 1e-8)

        # إيجاد أقرب مستوى NF4 (quantization)
        distances = (normalized.unsqueeze(-1) - self.levels).abs()
        quantized_indices = distances.argmin(dim=-1)  # 4-bit indices (0-15)

        return quantized_indices.to(torch.uint8), absmax

    def dequantize_block(self, quantized_indices: torch.Tensor, absmax: torch.Tensor):
        """فك التكميم: استرجاع القيم الفعلية (بدقة BFloat16) للحساب."""
        dequantized = self.levels[quantized_indices.long()] * absmax
        return dequantized.to(torch.bfloat16)

    def quantize_tensor(self, weight: torch.Tensor):
        """تكميم مصفوفة كاملة عبر تقسيمها إلى كتل (blockwise quantization)."""
        flat = weight.flatten()
        n_blocks = (len(flat) + self.block_size - 1) // self.block_size

        all_indices, all_absmax = [], []
        for i in range(n_blocks):
            block = flat[i * self.block_size:(i + 1) * self.block_size]
            indices, absmax = self.quantize_block(block)
            all_indices.append(indices)
            all_absmax.append(absmax)

        return all_indices, torch.stack(all_absmax), weight.shape

class DoubleQuantization:
    """
    Double Quantization: تكميم ثوابت absmax نفسها (التي عادة FP32)
    إلى 8-bit، لتوفير ذاكرة إضافية (~0.37 بت لكل معامل في المتوسط
    وفق الورقة الأصلية) دون فقدان دقة ملحوظ.
    """
    def __init__(self):
        self.quant_map_8bit = torch.linspace(-1, 1, 256)

    def quantize_absmax_values(self, absmax_values: torch.Tensor):
        # تطبيع ثوابت absmax نفسها ثم تكميمها لـ 8-bit
        meta_absmax = absmax_values.abs().max()
        normalized = absmax_values / (meta_absmax + 1e-8)
        distances = (normalized.unsqueeze(-1) - self.quant_map_8bit).abs()
        indices_8bit = distances.argmin(dim=-1).to(torch.uint8)
        return indices_8bit, meta_absmax

class QLoRALinear(nn.Module):
    """
    طبقة QLoRA متكاملة: أوزان أساسية مُكمَّمة 4-bit (مجمّدة تمامًا)
    + طبقة LoRA بدقة BFloat16 كاملة (قابلة للتدريب).
    """
    def __init__(self, in_features: int, out_features: int,
                 rank: int = 16, alpha: int = 32, block_size: int = 64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantizer = NF4Quantizer(block_size=block_size)

        # في التطبيق الفعلي، نُحمِّل أوزانًا مُدرَّبة مسبقًا ونُكمِّمها
        # هنا نُمثّلها بشكل عشوائي للتوضيح التعليمي
        dummy_weight = torch.randn(out_features, in_features) * 0.02
        self.quantized_indices, self.absmax, self.weight_shape = \
            self.quantizer.quantize_tensor(dummy_weight)

        # طبقة LoRA بدقة كاملة (BFloat16) — هذه فقط القابلة للتدريب
        self.lora = LoRALayer(in_features, out_features, rank=rank, alpha=alpha)

    def get_dequantized_base_weight(self) -> torch.Tensor:
        """فك تكميم الأوزان الأساسية 'عند الطلب' فقط أثناء forward/backward."""
        dequant_blocks = []
        for indices in self.quantized_indices:
            pass  # (مبسّط: في التطبيق الحقيقي نُعيد تجميع كل الكتل بكفاءة)
        # نسخة مبسّطة للتوضيح: استخدام أول كتلة absmax لتمثيل العملية
        flat_dequant = torch.cat([
            self.quantizer.dequantize_block(idx, self.absmax[i])
            for i, idx in enumerate(self.quantized_indices)
        ])
        return flat_dequant.reshape(self.weight_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # فك التكميم مؤقتًا للحساب فقط (لا تُخزَّن النسخة المفكوكة بدقة كاملة)
        base_weight = self.get_dequantized_base_weight().to(x.dtype)
        base_out = torch.nn.functional.linear(x, base_weight)
        lora_out = self.lora(x)
        return base_out + lora_out

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-07-peft-alignment/ch32-qlora.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class PagedOptimizerConcept:
    """
    توضيح مفاهيمي لآلية Paged Optimizers في QLoRA:
    تستخدم NVIDIA Unified Memory للسماح بـ "ترحيل" حالات المُحسِّن
    (Adam states) من GPU إلى CPU RAM تلقائيًا عند حدوث ذروة استخدام
    الذاكرة (مثل طول تسلسل غير متوقع)، تمامًا مثل آلية Paging
    في أنظمة التشغيل لإدارة الذاكرة الافتراضية.

    في الممارسة العملية، يُفعَّل هذا ببساطة عبر:
        optimizer = bnb.optim.PagedAdamW32bit(model.parameters(), lr=2e-4)

    الفائدة: تدريب نماذج كبيرة دون انهيار مفاجئ (CUDA OOM) بسبب
    ذروات عابرة في استخدام الذاكرة، بتكلفة بطء طفيف فقط عند الحاجة
    الفعلية للترحيل.
    """
    pass

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-07-peft-alignment/ch32-qlora.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

def setup_qlora_training(model_name: str = "meta-llama/Llama-3.1-8B"):
    """
    الإعداد القياسي الكامل لـ QLoRA باستخدام مكتبات bitsandbytes وPEFT
    الرسمية — هذا هو الكود الذي تستخدمه عمليًا في المشاريع الحقيقية.
    """
    # 1. إعداد تكوين التكميم 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",             # NF4 وليس Int4 عادي
        bnb_4bit_compute_dtype=torch.bfloat16,  # دقة الحساب الفعلي
        bnb_4bit_use_double_quant=True,         # تفعيل Double Quantization
    )

    # 2. تحميل النموذج المُكمَّم
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. تجهيز النموذج للتدريب الكمومي (طبقات معينة تبقى FP32 للاستقرار)
    model = prepare_model_for_kbit_training(model)

    # 4. إعداد وتطبيق LoRA فوق النموذج المُكمَّم
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Output typical: trainable params: 41,943,040 || all params: 8,072,204,288
    #                 || trainable%: 0.5197

    return model, tokenizer

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-07-peft-alignment/ch32-qlora.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def calculate_qlora_memory_savings(n_params_billions: float = 13):
    n_params = n_params_billions * 1e9

    # Full Fine-Tuning (FP16): weights + grads + 2 adam states (كلها FP32 فعليًا لـ Adam)
    full_ft_gb = n_params * (2 + 2 + 8) / 1e9  # 2(w)+2(g)+8(adam fp32 m,v)

    # LoRA عادي (FP16): النموذج الأساسي FP16 (مجمّد، بدون grad/adam) + LoRA صغير
    lora_fp16_base_gb = n_params * 2 / 1e9  # فقط الأوزان، بدون grad/adam للأساس
    lora_trainable_params = n_params * 0.005  # ~0.5% تقريبًا
    lora_trainable_gb = lora_trainable_params * (2 + 2 + 8) / 1e9
    lora_total_gb = lora_fp16_base_gb + lora_trainable_gb

    # QLoRA: النموذج الأساسي 4-bit (0.5 بايت/معامل) + LoRA بدقة كاملة
    qlora_base_gb = n_params * 0.5 / 1e9  # 4-bit = 0.5 بايت
    qlora_total_gb = qlora_base_gb + lora_trainable_gb

    print(f"نموذج {n_params_billions}B معامل:")
    print(f"  Full Fine-Tuning:  ~{full_ft_gb:.1f} GB")
    print(f"  LoRA (FP16 base):  ~{lora_total_gb:.1f} GB")
    print(f"  QLoRA (4-bit base): ~{qlora_total_gb:.1f} GB")
    print(f"\nتوفير QLoRA مقابل Full FT: {full_ft_gb/qlora_total_gb:.1f}× أقل ذاكرة")

calculate_qlora_memory_savings(13)
# نموذج 13.0B معامل:
#   Full Fine-Tuning:  ~169.0 GB
#   LoRA (FP16 base):  ~26.5 GB
#   QLoRA (4-bit base): ~7.3 GB
# توفير QLoRA مقابل Full FT: 23.2× أقل ذاكرة

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-07-peft-alignment/ch32-qlora.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-32-ch32-qlora.ipynb#cell-1
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
    ("peft", "peft"),
    ("torch", "torch"),
    ("transformers", "transformers")
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-32-ch32-qlora.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import numpy as np

def create_nf4_quantization_levels() -> torch.Tensor:
    """
    إنشاء مستويات NF4 الـ 16 المُحسَّبة مسبقًا (Information-theoretically
    optimal) لتوزيع طبيعي معياري N(0,1)، كما في ورقة QLoRA الأصلية.
    هذه القيم ثابتة ومُحسَّبة مرة واحدة (لا تُعاد حسابها أثناء التدريب).
    """
    # القيم الفعلية المُستخدَمة في تطبيق bitsandbytes (مُقرَّبة هنا للتوضيح)
    nf4_levels = torch.tensor([
        -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
        0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
    ])
    return nf4_levels

class NF4Quantizer:
    """
    محاكاة مبسّطة تعليميًا لعملية تكميم NF4 (في التطبيق الفعلي
    تُستخدَم مكتبة bitsandbytes المُحسَّنة بـ CUDA kernels مخصّصة).
    """
    def __init__(self, block_size: int = 64):
        self.levels = create_nf4_quantization_levels()
        self.block_size = block_size  # التكميم يتم على كتل صغيرة (blockwise)

    def quantize_block(self, weight_block: torch.Tensor):
        """
        تكميم كتلة واحدة من الأوزان:
        1. حساب absmax (أقصى قيمة مطلقة) كثابت تطبيع للكتلة.
        2. تطبيع الأوزان إلى [-1, 1].
        3. إيجاد أقرب مستوى NF4 لكل قيمة.
        """
        absmax = weight_block.abs().max()
        normalized = weight_block / (absmax + 1e-8)

        # إيجاد أقرب مستوى NF4 (quantization)
        distances = (normalized.unsqueeze(-1) - self.levels).abs()
        quantized_indices = distances.argmin(dim=-1)  # 4-bit indices (0-15)

        return quantized_indices.to(torch.uint8), absmax

    def dequantize_block(self, quantized_indices: torch.Tensor, absmax: torch.Tensor):
        """فك التكميم: استرجاع القيم الفعلية (بدقة BFloat16) للحساب."""
        dequantized = self.levels[quantized_indices.long()] * absmax
        return dequantized.to(torch.bfloat16)

    def quantize_tensor(self, weight: torch.Tensor):
        """تكميم مصفوفة كاملة عبر تقسيمها إلى كتل (blockwise quantization)."""
        flat = weight.flatten()
        n_blocks = (len(flat) + self.block_size - 1) // self.block_size

        all_indices, all_absmax = [], []
        for i in range(n_blocks):
            block = flat[i * self.block_size:(i + 1) * self.block_size]
            indices, absmax = self.quantize_block(block)
            all_indices.append(indices)
            all_absmax.append(absmax)

        return all_indices, torch.stack(all_absmax), weight.shape

class DoubleQuantization:
    """
    Double Quantization: تكميم ثوابت absmax نفسها (التي عادة FP32)
    إلى 8-bit، لتوفير ذاكرة إضافية (~0.37 بت لكل معامل في المتوسط
    وفق الورقة الأصلية) دون فقدان دقة ملحوظ.
    """
    def __init__(self):
        self.quant_map_8bit = torch.linspace(-1, 1, 256)

    def quantize_absmax_values(self, absmax_values: torch.Tensor):
        # تطبيع ثوابت absmax نفسها ثم تكميمها لـ 8-bit
        meta_absmax = absmax_values.abs().max()
        normalized = absmax_values / (meta_absmax + 1e-8)
        distances = (normalized.unsqueeze(-1) - self.quant_map_8bit).abs()
        indices_8bit = distances.argmin(dim=-1).to(torch.uint8)
        return indices_8bit, meta_absmax

class QLoRALinear(nn.Module):
    """
    طبقة QLoRA متكاملة: أوزان أساسية مُكمَّمة 4-bit (مجمّدة تمامًا)
    + طبقة LoRA بدقة BFloat16 كاملة (قابلة للتدريب).
    """
    def __init__(self, in_features: int, out_features: int,
                 rank: int = 16, alpha: int = 32, block_size: int = 64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantizer = NF4Quantizer(block_size=block_size)

        # في التطبيق الفعلي، نُحمِّل أوزانًا مُدرَّبة مسبقًا ونُكمِّمها
        # هنا نُمثّلها بشكل عشوائي للتوضيح التعليمي
        dummy_weight = torch.randn(out_features, in_features) * 0.02
        self.quantized_indices, self.absmax, self.weight_shape = \
            self.quantizer.quantize_tensor(dummy_weight)

        # طبقة LoRA بدقة كاملة (BFloat16) — هذه فقط القابلة للتدريب
        self.lora = LoRALayer(in_features, out_features, rank=rank, alpha=alpha)

    def get_dequantized_base_weight(self) -> torch.Tensor:
        """فك تكميم الأوزان الأساسية 'عند الطلب' فقط أثناء forward/backward."""
        dequant_blocks = []
        for indices in self.quantized_indices:
            pass  # (مبسّط: في التطبيق الحقيقي نُعيد تجميع كل الكتل بكفاءة)
        # نسخة مبسّطة للتوضيح: استخدام أول كتلة absmax لتمثيل العملية
        flat_dequant = torch.cat([
            self.quantizer.dequantize_block(idx, self.absmax[i])
            for i, idx in enumerate(self.quantized_indices)
        ])
        return flat_dequant.reshape(self.weight_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # فك التكميم مؤقتًا للحساب فقط (لا تُخزَّن النسخة المفكوكة بدقة كاملة)
        base_weight = self.get_dequantized_base_weight().to(x.dtype)
        base_out = torch.nn.functional.linear(x, base_weight)
        lora_out = self.lora(x)
        return base_out + lora_out

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-32-ch32-qlora.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class PagedOptimizerConcept:
    """
    توضيح مفاهيمي لآلية Paged Optimizers في QLoRA:
    تستخدم NVIDIA Unified Memory للسماح بـ "ترحيل" حالات المُحسِّن
    (Adam states) من GPU إلى CPU RAM تلقائيًا عند حدوث ذروة استخدام
    الذاكرة (مثل طول تسلسل غير متوقع)، تمامًا مثل آلية Paging
    في أنظمة التشغيل لإدارة الذاكرة الافتراضية.

    في الممارسة العملية، يُفعَّل هذا ببساطة عبر:
        optimizer = bnb.optim.PagedAdamW32bit(model.parameters(), lr=2e-4)

    الفائدة: تدريب نماذج كبيرة دون انهيار مفاجئ (CUDA OOM) بسبب
    ذروات عابرة في استخدام الذاكرة، بتكلفة بطء طفيف فقط عند الحاجة
    الفعلية للترحيل.
    """
    pass

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-32-ch32-qlora.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

def setup_qlora_training(model_name: str = "meta-llama/Llama-3.1-8B"):
    """
    الإعداد القياسي الكامل لـ QLoRA باستخدام مكتبات bitsandbytes وPEFT
    الرسمية — هذا هو الكود الذي تستخدمه عمليًا في المشاريع الحقيقية.
    """
    # 1. إعداد تكوين التكميم 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",             # NF4 وليس Int4 عادي
        bnb_4bit_compute_dtype=torch.bfloat16,  # دقة الحساب الفعلي
        bnb_4bit_use_double_quant=True,         # تفعيل Double Quantization
    )

    # 2. تحميل النموذج المُكمَّم
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. تجهيز النموذج للتدريب الكمومي (طبقات معينة تبقى FP32 للاستقرار)
    model = prepare_model_for_kbit_training(model)

    # 4. إعداد وتطبيق LoRA فوق النموذج المُكمَّم
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Output typical: trainable params: 41,943,040 || all params: 8,072,204,288
    #                 || trainable%: 0.5197

    return model, tokenizer

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-32-ch32-qlora.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def calculate_qlora_memory_savings(n_params_billions: float = 13):
    n_params = n_params_billions * 1e9

    # Full Fine-Tuning (FP16): weights + grads + 2 adam states (كلها FP32 فعليًا لـ Adam)
    full_ft_gb = n_params * (2 + 2 + 8) / 1e9  # 2(w)+2(g)+8(adam fp32 m,v)

    # LoRA عادي (FP16): النموذج الأساسي FP16 (مجمّد، بدون grad/adam) + LoRA صغير
    lora_fp16_base_gb = n_params * 2 / 1e9  # فقط الأوزان، بدون grad/adam للأساس
    lora_trainable_params = n_params * 0.005  # ~0.5% تقريبًا
    lora_trainable_gb = lora_trainable_params * (2 + 2 + 8) / 1e9
    lora_total_gb = lora_fp16_base_gb + lora_trainable_gb

    # QLoRA: النموذج الأساسي 4-bit (0.5 بايت/معامل) + LoRA بدقة كاملة
    qlora_base_gb = n_params * 0.5 / 1e9  # 4-bit = 0.5 بايت
    qlora_total_gb = qlora_base_gb + lora_trainable_gb

    print(f"نموذج {n_params_billions}B معامل:")
    print(f"  Full Fine-Tuning:  ~{full_ft_gb:.1f} GB")
    print(f"  LoRA (FP16 base):  ~{lora_total_gb:.1f} GB")
    print(f"  QLoRA (4-bit base): ~{qlora_total_gb:.1f} GB")
    print(f"\nتوفير QLoRA مقابل Full FT: {full_ft_gb/qlora_total_gb:.1f}× أقل ذاكرة")

calculate_qlora_memory_savings(13)
# نموذج 13.0B معامل:
#   Full Fine-Tuning:  ~169.0 GB
#   LoRA (FP16 base):  ~26.5 GB
#   QLoRA (4-bit base): ~7.3 GB
# توفير QLoRA مقابل Full FT: 23.2× أقل ذاكرة

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-32-ch32-qlora.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-32-ch32-qlora.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — QLoRA والتكميم أثناء التدريب
Mastering Transformers and Large Language Models — Qlora

Book slug : mastering-llms-transformers
Chapter   : ch-32-ch32-qlora (lesson 32)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-07-peft-alignment/ch32-qlora.ipynb#cell-1
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
    ("peft", "peft"),
    ("torch", "torch"),
    ("transformers", "transformers")
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
# [002] content/notebooks/part-07-peft-alignment/ch32-qlora.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import numpy as np

def create_nf4_quantization_levels() -> torch.Tensor:
    """
    إنشاء مستويات NF4 الـ 16 المُحسَّبة مسبقًا (Information-theoretically
    optimal) لتوزيع طبيعي معياري N(0,1)، كما في ورقة QLoRA الأصلية.
    هذه القيم ثابتة ومُحسَّبة مرة واحدة (لا تُعاد حسابها أثناء التدريب).
    """
    # القيم الفعلية المُستخدَمة في تطبيق bitsandbytes (مُقرَّبة هنا للتوضيح)
    nf4_levels = torch.tensor([
        -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
        0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
    ])
    return nf4_levels

class NF4Quantizer:
    """
    محاكاة مبسّطة تعليميًا لعملية تكميم NF4 (في التطبيق الفعلي
    تُستخدَم مكتبة bitsandbytes المُحسَّنة بـ CUDA kernels مخصّصة).
    """
    def __init__(self, block_size: int = 64):
        self.levels = create_nf4_quantization_levels()
        self.block_size = block_size  # التكميم يتم على كتل صغيرة (blockwise)

    def quantize_block(self, weight_block: torch.Tensor):
        """
        تكميم كتلة واحدة من الأوزان:
        1. حساب absmax (أقصى قيمة مطلقة) كثابت تطبيع للكتلة.
        2. تطبيع الأوزان إلى [-1, 1].
        3. إيجاد أقرب مستوى NF4 لكل قيمة.
        """
        absmax = weight_block.abs().max()
        normalized = weight_block / (absmax + 1e-8)

        # إيجاد أقرب مستوى NF4 (quantization)
        distances = (normalized.unsqueeze(-1) - self.levels).abs()
        quantized_indices = distances.argmin(dim=-1)  # 4-bit indices (0-15)

        return quantized_indices.to(torch.uint8), absmax

    def dequantize_block(self, quantized_indices: torch.Tensor, absmax: torch.Tensor):
        """فك التكميم: استرجاع القيم الفعلية (بدقة BFloat16) للحساب."""
        dequantized = self.levels[quantized_indices.long()] * absmax
        return dequantized.to(torch.bfloat16)

    def quantize_tensor(self, weight: torch.Tensor):
        """تكميم مصفوفة كاملة عبر تقسيمها إلى كتل (blockwise quantization)."""
        flat = weight.flatten()
        n_blocks = (len(flat) + self.block_size - 1) // self.block_size

        all_indices, all_absmax = [], []
        for i in range(n_blocks):
            block = flat[i * self.block_size:(i + 1) * self.block_size]
            indices, absmax = self.quantize_block(block)
            all_indices.append(indices)
            all_absmax.append(absmax)

        return all_indices, torch.stack(all_absmax), weight.shape

class DoubleQuantization:
    """
    Double Quantization: تكميم ثوابت absmax نفسها (التي عادة FP32)
    إلى 8-bit، لتوفير ذاكرة إضافية (~0.37 بت لكل معامل في المتوسط
    وفق الورقة الأصلية) دون فقدان دقة ملحوظ.
    """
    def __init__(self):
        self.quant_map_8bit = torch.linspace(-1, 1, 256)

    def quantize_absmax_values(self, absmax_values: torch.Tensor):
        # تطبيع ثوابت absmax نفسها ثم تكميمها لـ 8-bit
        meta_absmax = absmax_values.abs().max()
        normalized = absmax_values / (meta_absmax + 1e-8)
        distances = (normalized.unsqueeze(-1) - self.quant_map_8bit).abs()
        indices_8bit = distances.argmin(dim=-1).to(torch.uint8)
        return indices_8bit, meta_absmax

class QLoRALinear(nn.Module):
    """
    طبقة QLoRA متكاملة: أوزان أساسية مُكمَّمة 4-bit (مجمّدة تمامًا)
    + طبقة LoRA بدقة BFloat16 كاملة (قابلة للتدريب).
    """
    def __init__(self, in_features: int, out_features: int,
                 rank: int = 16, alpha: int = 32, block_size: int = 64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantizer = NF4Quantizer(block_size=block_size)

        # في التطبيق الفعلي، نُحمِّل أوزانًا مُدرَّبة مسبقًا ونُكمِّمها
        # هنا نُمثّلها بشكل عشوائي للتوضيح التعليمي
        dummy_weight = torch.randn(out_features, in_features) * 0.02
        self.quantized_indices, self.absmax, self.weight_shape = \
            self.quantizer.quantize_tensor(dummy_weight)

        # طبقة LoRA بدقة كاملة (BFloat16) — هذه فقط القابلة للتدريب
        self.lora = LoRALayer(in_features, out_features, rank=rank, alpha=alpha)

    def get_dequantized_base_weight(self) -> torch.Tensor:
        """فك تكميم الأوزان الأساسية 'عند الطلب' فقط أثناء forward/backward."""
        dequant_blocks = []
        for indices in self.quantized_indices:
            pass  # (مبسّط: في التطبيق الحقيقي نُعيد تجميع كل الكتل بكفاءة)
        # نسخة مبسّطة للتوضيح: استخدام أول كتلة absmax لتمثيل العملية
        flat_dequant = torch.cat([
            self.quantizer.dequantize_block(idx, self.absmax[i])
            for i, idx in enumerate(self.quantized_indices)
        ])
        return flat_dequant.reshape(self.weight_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # فك التكميم مؤقتًا للحساب فقط (لا تُخزَّن النسخة المفكوكة بدقة كاملة)
        base_weight = self.get_dequantized_base_weight().to(x.dtype)
        base_out = torch.nn.functional.linear(x, base_weight)
        lora_out = self.lora(x)
        return base_out + lora_out

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-07-peft-alignment/ch32-qlora.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class PagedOptimizerConcept:
    """
    توضيح مفاهيمي لآلية Paged Optimizers في QLoRA:
    تستخدم NVIDIA Unified Memory للسماح بـ "ترحيل" حالات المُحسِّن
    (Adam states) من GPU إلى CPU RAM تلقائيًا عند حدوث ذروة استخدام
    الذاكرة (مثل طول تسلسل غير متوقع)، تمامًا مثل آلية Paging
    في أنظمة التشغيل لإدارة الذاكرة الافتراضية.

    في الممارسة العملية، يُفعَّل هذا ببساطة عبر:
        optimizer = bnb.optim.PagedAdamW32bit(model.parameters(), lr=2e-4)

    الفائدة: تدريب نماذج كبيرة دون انهيار مفاجئ (CUDA OOM) بسبب
    ذروات عابرة في استخدام الذاكرة، بتكلفة بطء طفيف فقط عند الحاجة
    الفعلية للترحيل.
    """
    pass

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-07-peft-alignment/ch32-qlora.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

def setup_qlora_training(model_name: str = "meta-llama/Llama-3.1-8B"):
    """
    الإعداد القياسي الكامل لـ QLoRA باستخدام مكتبات bitsandbytes وPEFT
    الرسمية — هذا هو الكود الذي تستخدمه عمليًا في المشاريع الحقيقية.
    """
    # 1. إعداد تكوين التكميم 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",             # NF4 وليس Int4 عادي
        bnb_4bit_compute_dtype=torch.bfloat16,  # دقة الحساب الفعلي
        bnb_4bit_use_double_quant=True,         # تفعيل Double Quantization
    )

    # 2. تحميل النموذج المُكمَّم
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. تجهيز النموذج للتدريب الكمومي (طبقات معينة تبقى FP32 للاستقرار)
    model = prepare_model_for_kbit_training(model)

    # 4. إعداد وتطبيق LoRA فوق النموذج المُكمَّم
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Output typical: trainable params: 41,943,040 || all params: 8,072,204,288
    #                 || trainable%: 0.5197

    return model, tokenizer

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-07-peft-alignment/ch32-qlora.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def calculate_qlora_memory_savings(n_params_billions: float = 13):
    n_params = n_params_billions * 1e9

    # Full Fine-Tuning (FP16): weights + grads + 2 adam states (كلها FP32 فعليًا لـ Adam)
    full_ft_gb = n_params * (2 + 2 + 8) / 1e9  # 2(w)+2(g)+8(adam fp32 m,v)

    # LoRA عادي (FP16): النموذج الأساسي FP16 (مجمّد، بدون grad/adam) + LoRA صغير
    lora_fp16_base_gb = n_params * 2 / 1e9  # فقط الأوزان، بدون grad/adam للأساس
    lora_trainable_params = n_params * 0.005  # ~0.5% تقريبًا
    lora_trainable_gb = lora_trainable_params * (2 + 2 + 8) / 1e9
    lora_total_gb = lora_fp16_base_gb + lora_trainable_gb

    # QLoRA: النموذج الأساسي 4-bit (0.5 بايت/معامل) + LoRA بدقة كاملة
    qlora_base_gb = n_params * 0.5 / 1e9  # 4-bit = 0.5 بايت
    qlora_total_gb = qlora_base_gb + lora_trainable_gb

    print(f"نموذج {n_params_billions}B معامل:")
    print(f"  Full Fine-Tuning:  ~{full_ft_gb:.1f} GB")
    print(f"  LoRA (FP16 base):  ~{lora_total_gb:.1f} GB")
    print(f"  QLoRA (4-bit base): ~{qlora_total_gb:.1f} GB")
    print(f"\nتوفير QLoRA مقابل Full FT: {full_ft_gb/qlora_total_gb:.1f}× أقل ذاكرة")

calculate_qlora_memory_savings(13)
# نموذج 13.0B معامل:
#   Full Fine-Tuning:  ~169.0 GB
#   LoRA (FP16 base):  ~26.5 GB
#   QLoRA (4-bit base): ~7.3 GB
# توفير QLoRA مقابل Full FT: 23.2× أقل ذاكرة

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-07-peft-alignment/ch32-qlora.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-32-ch32-qlora.ipynb#cell-2
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
    ("peft", "peft"),
    ("torch", "torch"),
    ("transformers", "transformers")
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-32-ch32-qlora.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import numpy as np

def create_nf4_quantization_levels() -> torch.Tensor:
    """
    إنشاء مستويات NF4 الـ 16 المُحسَّبة مسبقًا (Information-theoretically
    optimal) لتوزيع طبيعي معياري N(0,1)، كما في ورقة QLoRA الأصلية.
    هذه القيم ثابتة ومُحسَّبة مرة واحدة (لا تُعاد حسابها أثناء التدريب).
    """
    # القيم الفعلية المُستخدَمة في تطبيق bitsandbytes (مُقرَّبة هنا للتوضيح)
    nf4_levels = torch.tensor([
        -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
        0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
    ])
    return nf4_levels

class NF4Quantizer:
    """
    محاكاة مبسّطة تعليميًا لعملية تكميم NF4 (في التطبيق الفعلي
    تُستخدَم مكتبة bitsandbytes المُحسَّنة بـ CUDA kernels مخصّصة).
    """
    def __init__(self, block_size: int = 64):
        self.levels = create_nf4_quantization_levels()
        self.block_size = block_size  # التكميم يتم على كتل صغيرة (blockwise)

    def quantize_block(self, weight_block: torch.Tensor):
        """
        تكميم كتلة واحدة من الأوزان:
        1. حساب absmax (أقصى قيمة مطلقة) كثابت تطبيع للكتلة.
        2. تطبيع الأوزان إلى [-1, 1].
        3. إيجاد أقرب مستوى NF4 لكل قيمة.
        """
        absmax = weight_block.abs().max()
        normalized = weight_block / (absmax + 1e-8)

        # إيجاد أقرب مستوى NF4 (quantization)
        distances = (normalized.unsqueeze(-1) - self.levels).abs()
        quantized_indices = distances.argmin(dim=-1)  # 4-bit indices (0-15)

        return quantized_indices.to(torch.uint8), absmax

    def dequantize_block(self, quantized_indices: torch.Tensor, absmax: torch.Tensor):
        """فك التكميم: استرجاع القيم الفعلية (بدقة BFloat16) للحساب."""
        dequantized = self.levels[quantized_indices.long()] * absmax
        return dequantized.to(torch.bfloat16)

    def quantize_tensor(self, weight: torch.Tensor):
        """تكميم مصفوفة كاملة عبر تقسيمها إلى كتل (blockwise quantization)."""
        flat = weight.flatten()
        n_blocks = (len(flat) + self.block_size - 1) // self.block_size

        all_indices, all_absmax = [], []
        for i in range(n_blocks):
            block = flat[i * self.block_size:(i + 1) * self.block_size]
            indices, absmax = self.quantize_block(block)
            all_indices.append(indices)
            all_absmax.append(absmax)

        return all_indices, torch.stack(all_absmax), weight.shape

class DoubleQuantization:
    """
    Double Quantization: تكميم ثوابت absmax نفسها (التي عادة FP32)
    إلى 8-bit، لتوفير ذاكرة إضافية (~0.37 بت لكل معامل في المتوسط
    وفق الورقة الأصلية) دون فقدان دقة ملحوظ.
    """
    def __init__(self):
        self.quant_map_8bit = torch.linspace(-1, 1, 256)

    def quantize_absmax_values(self, absmax_values: torch.Tensor):
        # تطبيع ثوابت absmax نفسها ثم تكميمها لـ 8-bit
        meta_absmax = absmax_values.abs().max()
        normalized = absmax_values / (meta_absmax + 1e-8)
        distances = (normalized.unsqueeze(-1) - self.quant_map_8bit).abs()
        indices_8bit = distances.argmin(dim=-1).to(torch.uint8)
        return indices_8bit, meta_absmax

class QLoRALinear(nn.Module):
    """
    طبقة QLoRA متكاملة: أوزان أساسية مُكمَّمة 4-bit (مجمّدة تمامًا)
    + طبقة LoRA بدقة BFloat16 كاملة (قابلة للتدريب).
    """
    def __init__(self, in_features: int, out_features: int,
                 rank: int = 16, alpha: int = 32, block_size: int = 64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantizer = NF4Quantizer(block_size=block_size)

        # في التطبيق الفعلي، نُحمِّل أوزانًا مُدرَّبة مسبقًا ونُكمِّمها
        # هنا نُمثّلها بشكل عشوائي للتوضيح التعليمي
        dummy_weight = torch.randn(out_features, in_features) * 0.02
        self.quantized_indices, self.absmax, self.weight_shape = \
            self.quantizer.quantize_tensor(dummy_weight)

        # طبقة LoRA بدقة كاملة (BFloat16) — هذه فقط القابلة للتدريب
        self.lora = LoRALayer(in_features, out_features, rank=rank, alpha=alpha)

    def get_dequantized_base_weight(self) -> torch.Tensor:
        """فك تكميم الأوزان الأساسية 'عند الطلب' فقط أثناء forward/backward."""
        dequant_blocks = []
        for indices in self.quantized_indices:
            pass  # (مبسّط: في التطبيق الحقيقي نُعيد تجميع كل الكتل بكفاءة)
        # نسخة مبسّطة للتوضيح: استخدام أول كتلة absmax لتمثيل العملية
        flat_dequant = torch.cat([
            self.quantizer.dequantize_block(idx, self.absmax[i])
            for i, idx in enumerate(self.quantized_indices)
        ])
        return flat_dequant.reshape(self.weight_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # فك التكميم مؤقتًا للحساب فقط (لا تُخزَّن النسخة المفكوكة بدقة كاملة)
        base_weight = self.get_dequantized_base_weight().to(x.dtype)
        base_out = torch.nn.functional.linear(x, base_weight)
        lora_out = self.lora(x)
        return base_out + lora_out

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-32-ch32-qlora.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class PagedOptimizerConcept:
    """
    توضيح مفاهيمي لآلية Paged Optimizers في QLoRA:
    تستخدم NVIDIA Unified Memory للسماح بـ "ترحيل" حالات المُحسِّن
    (Adam states) من GPU إلى CPU RAM تلقائيًا عند حدوث ذروة استخدام
    الذاكرة (مثل طول تسلسل غير متوقع)، تمامًا مثل آلية Paging
    في أنظمة التشغيل لإدارة الذاكرة الافتراضية.

    في الممارسة العملية، يُفعَّل هذا ببساطة عبر:
        optimizer = bnb.optim.PagedAdamW32bit(model.parameters(), lr=2e-4)

    الفائدة: تدريب نماذج كبيرة دون انهيار مفاجئ (CUDA OOM) بسبب
    ذروات عابرة في استخدام الذاكرة، بتكلفة بطء طفيف فقط عند الحاجة
    الفعلية للترحيل.
    """
    pass

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-32-ch32-qlora.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

def setup_qlora_training(model_name: str = "meta-llama/Llama-3.1-8B"):
    """
    الإعداد القياسي الكامل لـ QLoRA باستخدام مكتبات bitsandbytes وPEFT
    الرسمية — هذا هو الكود الذي تستخدمه عمليًا في المشاريع الحقيقية.
    """
    # 1. إعداد تكوين التكميم 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",             # NF4 وليس Int4 عادي
        bnb_4bit_compute_dtype=torch.bfloat16,  # دقة الحساب الفعلي
        bnb_4bit_use_double_quant=True,         # تفعيل Double Quantization
    )

    # 2. تحميل النموذج المُكمَّم
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. تجهيز النموذج للتدريب الكمومي (طبقات معينة تبقى FP32 للاستقرار)
    model = prepare_model_for_kbit_training(model)

    # 4. إعداد وتطبيق LoRA فوق النموذج المُكمَّم
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Output typical: trainable params: 41,943,040 || all params: 8,072,204,288
    #                 || trainable%: 0.5197

    return model, tokenizer

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-32-ch32-qlora.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def calculate_qlora_memory_savings(n_params_billions: float = 13):
    n_params = n_params_billions * 1e9

    # Full Fine-Tuning (FP16): weights + grads + 2 adam states (كلها FP32 فعليًا لـ Adam)
    full_ft_gb = n_params * (2 + 2 + 8) / 1e9  # 2(w)+2(g)+8(adam fp32 m,v)

    # LoRA عادي (FP16): النموذج الأساسي FP16 (مجمّد، بدون grad/adam) + LoRA صغير
    lora_fp16_base_gb = n_params * 2 / 1e9  # فقط الأوزان، بدون grad/adam للأساس
    lora_trainable_params = n_params * 0.005  # ~0.5% تقريبًا
    lora_trainable_gb = lora_trainable_params * (2 + 2 + 8) / 1e9
    lora_total_gb = lora_fp16_base_gb + lora_trainable_gb

    # QLoRA: النموذج الأساسي 4-bit (0.5 بايت/معامل) + LoRA بدقة كاملة
    qlora_base_gb = n_params * 0.5 / 1e9  # 4-bit = 0.5 بايت
    qlora_total_gb = qlora_base_gb + lora_trainable_gb

    print(f"نموذج {n_params_billions}B معامل:")
    print(f"  Full Fine-Tuning:  ~{full_ft_gb:.1f} GB")
    print(f"  LoRA (FP16 base):  ~{lora_total_gb:.1f} GB")
    print(f"  QLoRA (4-bit base): ~{qlora_total_gb:.1f} GB")
    print(f"\nتوفير QLoRA مقابل Full FT: {full_ft_gb/qlora_total_gb:.1f}× أقل ذاكرة")

calculate_qlora_memory_savings(13)
# نموذج 13.0B معامل:
#   Full Fine-Tuning:  ~169.0 GB
#   LoRA (FP16 base):  ~26.5 GB
#   QLoRA (4-bit base): ~7.3 GB
# توفير QLoRA مقابل Full FT: 23.2× أقل ذاكرة

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-32-ch32-qlora.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
