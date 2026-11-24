#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الكفاءة كاتجاه بحثي — النماذج اللغوية الصغيرة (SLMs: Phi, Gemma, Qwen-Small)
Large Language Model Engineering — SLMs

Book slug : llm-engineering-69-chapters
Chapter   : ch-68-chapter-68 (lesson 68)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-68-chapter-68.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الكفاءة كاتجاه بحثي — النماذج اللغوية الصغيرة (SLMs: Phi, Gemma, Qwen-Small)
Large Language Model Engineering — SLMs

Book slug : llm-engineering-69-chapters
Chapter   : ch-68-chapter-68 (lesson 68)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-68-chapter-68.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الكفاءة كاتجاه بحثي — النماذج اللغوية الصغيرة (SLMs: Phi, Gemma, Qwen-Small)
Large Language Model Engineering — SLMs

Book slug : llm-engineering-69-chapters
Chapter   : ch-68-chapter-68 (lesson 68)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_68.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 68: الفصل 68: الكفاءة كاتجاه بحثي — النماذج اللغوية الصغيرة (SLMs: Phi, Gemma, Qwen-Small)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_slm() -> None:
    teacher = [0.65, 0.25, 0.1]
    student = [0.58, 0.3, 0.12]
    divergence = sum((p * math.log(p / q) for p, q in zip(teacher, student)))
    model_sizes = {'teacher_params': 70000000000, 'student_params': 3000000000}
    print('teacher/student KL:', round(divergence, 5))
    print('parameter reduction:', round(1 - model_sizes['student_params'] / model_sizes['teacher_params'], 3))
    print('measure quality and latency on the target task before choosing.')

if __name__ == "__main__":
    print("Chapter 68 lab: Compare a compact student with a large teacher")
    run_slm()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-12-frontier-trends/chapter-68.md
# ────────────────────────────────────────────────────────────────────────
"""
data_quality_simulation.py
محاكاة تعليمية مبسّطة توضّح فكرة: بيانات أقل لكن أنظف
قد تُنتج تعلّمًا أفضل من بيانات أكثر لكن أكثر ضوضاءً،
في مهام تتطلب إشارة واضحة (signal) بدلًا من تنوع واسع.

هذا ليس تدريب نموذج لغوي حقيقي، بل توضيح مُبسّط جدًا
للفكرة الإحصائية الكامنة خلف "Textbooks Are All You Need".
"""
import numpy as np


def generate_noisy_dataset(n_samples: int, noise_level: float) -> tuple[np.ndarray, np.ndarray]:
    """توليد بيانات تعليمية بعلاقة خطية بسيطة + ضوضاء بمستوى متحكَّم به"""
    rng = np.random.default_rng(42)
    x = rng.uniform(-5, 5, n_samples)
    true_relationship = 2.0 * x + 3.0
    noise = rng.normal(0, noise_level, n_samples)
    y = true_relationship + noise
    return x, y


def fit_simple_linear_model(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """نموذج خطي بسيط عبر المربعات الصغرى (يمثل "تعلّم" العلاقة من البيانات)"""
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope, intercept


def evaluate_on_clean_test_set(slope: float, intercept: float) -> float:
    """تقييم الدقة على بيانات اختبار نظيفة (الحقيقة المعروفة: slope=2, intercept=3)"""
    error = abs(slope - 2.0) + abs(intercept - 3.0)
    return error


# سيناريو 1: كمية بيانات كبيرة جدًا، لكن ضوضاء عالية (يحاكي بيانات الويب الخام)
x_large_noisy, y_large_noisy = generate_noisy_dataset(n_samples=10_000, noise_level=15.0)
slope_1, intercept_1 = fit_simple_linear_model(x_large_noisy, y_large_noisy)
error_1 = evaluate_on_clean_test_set(slope_1, intercept_1)

# سيناريو 2: كمية بيانات أصغر بكثير، لكن ضوضاء منخفضة جدًا (يحاكي بيانات "textbook-quality")
x_small_clean, y_small_clean = generate_noisy_dataset(n_samples=200, noise_level=0.5)
slope_2, intercept_2 = fit_simple_linear_model(x_small_clean, y_small_clean)
error_2 = evaluate_on_clean_test_set(slope_2, intercept_2)

print(f"سيناريو 1 (بيانات كثيرة وصاخبة):  حجم={10_000}, خطأ النموذج={error_1:.4f}")
print(f"سيناريو 2 (بيانات قليلة ونظيفة): حجم={200}, خطأ النموذج={error_2:.4f}")
print(f"\n{'✅ البيانات النظيفة الأقل تفوّقت!' if error_2 < error_1 else '⚠️ البيانات الكثيرة تفوّقت رغم الضوضاء'}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-12-frontier-trends/chapter-68.md
# ────────────────────────────────────────────────────────────────────────
"""
grouped_query_attention.py
تطبيق تعليمي واضح لآلية Grouped-Query Attention (GQA) من الصفر،
مقارنةً بـ Multi-Head Attention (MHA) التقليدي، لتوضيح وفورات الذاكرة.
"""
import numpy as np


def compute_kv_cache_size(
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    sequence_length: int,
    bytes_per_param: int = 2,  # FP16
) -> float:
    """
    حساب حجم KV Cache بالميغابايت.
    في GQA، عدد رؤوس K,V المُخزَّنة أقل من عدد رؤوس الاستعلام،
    مما يقلل حجم الكاش مباشرة بنفس النسبة.
    """
    # الكاش يُخزَّن فقط K وV، وليس Q (Q يُحسب لحظيًا لكل توكن جديد)
    kv_cache_elements = 2 * num_kv_heads * head_dim * sequence_length  # ×2 لـ K و V
    kv_cache_bytes = kv_cache_elements * bytes_per_param
    return kv_cache_bytes / (1024 ** 2)  # تحويل لميغابايت


# مقارنة: نموذج بـ 32 رأس استعلام، head_dim=128، سياق 8192 توكن
num_query_heads = 32
head_dim = 128
sequence_length = 8192

configurations = {
    "MHA التقليدي (32 رأس K,V منفصل)": 32,
    "GQA مع 8 مجموعات (Gemma/Llama-style)": 8,
    "GQA مع 4 مجموعات (أكثر عدوانية)": 4,
    "MQA (رأس K,V واحد مشترك)": 1,
}

print(f"{'الإعداد':<40}{'رؤوس K,V':<12}{'حجم KV Cache (MB)':<20}")
print("-" * 72)
for config_name, num_kv_heads in configurations.items():
    cache_size = compute_kv_cache_size(
        num_query_heads, num_kv_heads, head_dim, sequence_length
    )
    print(f"{config_name:<40}{num_kv_heads:<12}{cache_size:<20.1f}")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-12-frontier-trends/chapter-68.md
# ────────────────────────────────────────────────────────────────────────
class GroupedQueryAttention:
    """
    تطبيق مُبسّط وقابل للتشغيل فعليًا لـ GQA باستخدام NumPy،
    لأغراض تعليمية (في الإنتاج الحقيقي تُستخدم PyTorch/JAX مع kernels محسّنة).
    """

    def __init__(self, num_query_heads: int, num_kv_heads: int, head_dim: int):
        assert num_query_heads % num_kv_heads == 0, \
            "عدد رؤوس الاستعلام يجب أن يقبل القسمة على عدد مجموعات K,V"

        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = num_query_heads // num_kv_heads  # كم رأس استعلام لكل مجموعة K,V

    def forward(
        self, queries: np.ndarray, keys: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        """
        queries: شكل (num_query_heads, seq_len, head_dim)
        keys, values: شكل (num_kv_heads, seq_len, head_dim)  <- أقل من عدد رؤوس الاستعلام!
        """
        seq_len = queries.shape[1]
        output = np.zeros_like(queries)

        for q_head_idx in range(self.num_query_heads):
            # كل رأس استعلام "يشارك" مجموعة K,V محددة بناءً على موقعه
            kv_group_idx = q_head_idx // self.group_size

            q = queries[q_head_idx]            # (seq_len, head_dim)
            k = keys[kv_group_idx]              # (seq_len, head_dim) — مُشتركة بين عدة رؤوس Q
            v = values[kv_group_idx]

            scores = (q @ k.T) / np.sqrt(self.head_dim)
            # قناع سببي بسيط (causal mask) لمحاكاة التوليد التتابعي
            causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
            scores[causal_mask] = -np.inf

            attn_weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn_weights /= attn_weights.sum(axis=-1, keepdims=True)

            output[q_head_idx] = attn_weights @ v

        return output


# اختبار عملي
np.random.seed(0)
num_query_heads, num_kv_heads, head_dim, seq_len = 8, 2, 16, 5

gqa = GroupedQueryAttention(num_query_heads, num_kv_heads, head_dim)
queries = np.random.randn(num_query_heads, seq_len, head_dim)
keys = np.random.randn(num_kv_heads, seq_len, head_dim)
values = np.random.randn(num_kv_heads, seq_len, head_dim)

output = gqa.forward(queries, keys, values)
print(f"\nشكل مخرجات GQA: {output.shape}")
print(f"عدد رؤوس K,V الفعلية المُخزَّنة: {num_kv_heads} بدلاً من {num_query_heads}")
print(f"نسبة توفير KV Cache: {(1 - num_kv_heads/num_query_heads) * 100:.0f}%")

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-12-frontier-trends/chapter-68.md
# ────────────────────────────────────────────────────────────────────────
"""
model_selection_advisor.py
أداة مساعدة لاتخاذ قرار: متى تختار SLM ومن أي عائلة،
بناءً على قيود المشروع الفعلية.
"""
from dataclasses import dataclass
from enum import Enum


class ConstraintPriority(Enum):
    LATENCY_CRITICAL = "latency_critical"       # يجب أن يعمل محليًا/بسرعة فائقة
    MULTILINGUAL_HEAVY = "multilingual_heavy"   # دعم لغات متعددة أساسي
    CODE_SPECIALIZED = "code_specialized"       # مهمة برمجية متخصصة
    MATH_REASONING = "math_reasoning"           # استدلال رياضي/منطقي
    GENERAL_PURPOSE = "general_purpose"         # مهام عامة متنوعة
    BUDGET_EXTREME = "budget_extreme"           # ميزانية حوسبة شديدة المحدودية (IoT مثلًا)


@dataclass
class ModelRecommendation:
    family: str
    suggested_size: str
    rationale: str


MODEL_ADVISOR_TABLE: dict[ConstraintPriority, ModelRecommendation] = {
    ConstraintPriority.LATENCY_CRITICAL: ModelRecommendation(
        family="Phi-3-mini / Gemma-2B",
        suggested_size="2B-3.8B",
        rationale="أحجام مصممة خصيصًا للتشغيل على الحافة (edge) والهواتف، "
                   "مع دعم رسمي لـ ONNX Runtime وتهيئات مُحسَّنة للموبايل.",
    ),
    ConstraintPriority.MULTILINGUAL_HEAVY: ModelRecommendation(
        family="Qwen2.5",
        suggested_size="3B-7B",
        rationale="تدريب مكثف على بيانات متعددة اللغات متنوعة، "
                   "أداء موثّق أقوى في اللغات غير الإنجليزية مقارنة بمنافسيها في نفس الفئة.",
    ),
    ConstraintPriority.CODE_SPECIALIZED: ModelRecommendation(
        family="Qwen2.5-Coder أو Phi نسخة مُدرَّبة على كود",
        suggested_size="7B-14B",
        rationale="إصدارات متخصصة مُدرَّبة إضافيًا على كميات هائلة من الكود عالي الجودة.",
    ),
    ConstraintPriority.MATH_REASONING: ModelRecommendation(
        family="Phi-4 أو Qwen2.5-Math",
        suggested_size="7B-14B",
        rationale="تركيز تدريبي صريح على بيانات استدلال رياضي ومنطقي تركيبية عالية الجودة.",
    ),
    ConstraintPriority.GENERAL_PURPOSE: ModelRecommendation(
        family="Gemma 2 أو Qwen2.5",
        suggested_size="7B-9B",
        rationale="توازن جيد بين الأداء العام والتكلفة، مجتمع دعم واسع ومصادر تدريب غنية.",
    ),
    ConstraintPriority.BUDGET_EXTREME: ModelRecommendation(
        family="Qwen2.5-0.5B أو Gemma-2B مُكمَّم (quantized)",
        suggested_size="0.5B-2B",
        rationale="أصغر الأحجام المتاحة مع دعم تكميم فعّال (4-bit/8-bit) "
                   "للعمل على أجهزة IoT أو متصفحات (WebGPU).",
    ),
}


def recommend_model(priority: ConstraintPriority) -> ModelRecommendation:
    return MODEL_ADVISOR_TABLE[priority]


# مثال استخدام
recommendation = recommend_model(ConstraintPriority.LATENCY_CRITICAL)
print(f"العائلة المقترحة: {recommendation.family}")
print(f"الحجم المقترح   : {recommendation.suggested_size}")
print(f"السبب            : {recommendation.rationale}")

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-12-frontier-trends/chapter-68.md
# ────────────────────────────────────────────────────────────────────────
"""
specialized_vs_general_benchmark.py
محاكاة تعليمية توضح منهجية المقارنة بين SLM مضبوط دقيقًا
مقابل LLM عام ضخم، على مهمة ضيقة محددة (تصنيف فواتير).
"""
from dataclasses import dataclass


@dataclass
class ModelBenchmarkResult:
    model_name: str
    accuracy_on_narrow_task: float
    latency_ms: float
    cost_per_1000_requests_usd: float
    requires_internet: bool


def simulate_benchmark_results() -> list[ModelBenchmarkResult]:
    """
    نتائج محاكاة واقعية الشكل (مبنية على أنماط حقيقية موثقة في الأدبيات،
    وليست أرقامًا رسمية من أي شركة محددة) لتوضيح الاتجاه العام.
    """
    return [
        ModelBenchmarkResult(
            model_name="GPT-4 class (عام، عبر API)",
            accuracy_on_narrow_task=0.91,
            latency_ms=1800,
            cost_per_1000_requests_usd=15.0,
            requires_internet=True,
        ),
        ModelBenchmarkResult(
            model_name="Phi-3-mini (3.8B) + LoRA مضبوط على المهمة",
            accuracy_on_narrow_task=0.94,
            latency_ms=85,
            cost_per_1000_requests_usd=0.3,
            requires_internet=False,
        ),
        ModelBenchmarkResult(
            model_name="Phi-3-mini (3.8B) بدون ضبط دقيق (zero-shot)",
            accuracy_on_narrow_task=0.72,
            latency_ms=85,
            cost_per_1000_requests_usd=0.3,
            requires_internet=False,
        ),
    ]


def print_comparison_report(results: list[ModelBenchmarkResult]) -> None:
    print(f"{'النموذج':<45}{'الدقة':<10}{'الكمون(ms)':<12}{'التكلفة/1000':<15}{'يحتاج إنترنت'}")
    print("-" * 100)
    for r in results:
        internet_flag = "نعم" if r.requires_internet else "لا"
        print(
            f"{r.model_name:<45}{r.accuracy_on_narrow_task:<10.2f}"
            f"{r.latency_ms:<12.0f}${r.cost_per_1000_requests_usd:<14.2f}{internet_flag}"
        )


results = simulate_benchmark_results()
print_comparison_report(results)

print("\n📊 الاستنتاج الرئيسي:")
print("النموذج الصغير المضبوط دقيقًا تفوّق على النموذج الضخم العام في الدقة")
print("على هذه المهمة الضيقة تحديدًا، بينما كان أسرع بـ 21 مرة وأرخص بـ 50 مرة،")
print("وقابل للعمل محليًا بدون اتصال إنترنت.")
print("\n⚠️ تنبيه مهم: هذا لا يعني SLM 'أفضل مطلقًا' —")
print("بدون الضبط الدقيق، أداؤه انخفض بشكل حاد (من 0.94 إلى 0.72)،")
print("والنموذج الضخم العام يبقى الخيار الأفضل لمهام متنوعة غير محددة مسبقًا.")

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-12-frontier-trends/chapter-68.md
# ────────────────────────────────────────────────────────────────────────
"""
invoice_classifier_slm_pipeline.py
نظام End-to-End كامل: تحضير بيانات، ضبط دقيق (LoRA) لنموذج SLM صغير
(Phi-3-mini كمثال)، وتقييم النتيجة على بيانات اختبار محجوزة.

يستخدم مكتبات PyTorch + Hugging Face transformers + peft (LoRA)،
وهي المكدس المعياري الحديث (2024-2025) لهذا النوع من المهام.

ملاحظة: تشغيل هذا الكود فعليًا يتطلب تثبيت
transformers, peft, torch, datasets, bitsandbytes
وتحميل أوزان نموذج حقيقي (مثل microsoft/Phi-3-mini-4k-instruct)،
وقد يتطلب GPU للتدريب الفعلي. البنية والمنطق أدناه كاملان وصحيحان
ويمثلان تطبيقًا إنتاجيًا حقيقيًا، وليسا محاكاة.
"""

from dataclasses import dataclass
import json


# ---------- المرحلة 1: تحضير بيانات الضبط الدقيق ----------

@dataclass
class InvoiceExample:
    raw_text: str
    category: str   # مثل: "مطاعم", "نقل", "مستلزمات مكتبية", "اشتراكات"
    amount: float
    date: str


def build_training_dataset() -> list[InvoiceExample]:
    """بناء مجموعة بيانات تدريب صغيرة ومُنسّقة (في الواقع، قد تحتاج آلاف الأمثلة)"""
    return [
        InvoiceExample(
            raw_text="مطعم السلطان - فاتورة بتاريخ 2025-03-14 بقيمة 145.50 جنيه",
            category="مطاعم", amount=145.50, date="2025-03-14",
        ),
        InvoiceExample(
            raw_text="اشتراك شهري Netflix - 2025-03-01 - 199 جنيه",
            category="اشتراكات", amount=199.0, date="2025-03-01",
        ),
        InvoiceExample(
            raw_text="أوبر - رحلة بتاريخ 15 مارس 2025 - المبلغ 68 جنيه",
            category="نقل", amount=68.0, date="2025-03-15",
        ),
        InvoiceExample(
            raw_text="مكتبة الأهرام - أدوات مكتبية - 2025-03-10 - 320 جنيه",
            category="مستلزمات مكتبية", amount=320.0, date="2025-03-10",
        ),
        # في مشروع حقيقي: مئات أو آلاف الأمثلة المتنوعة والممثّلة لتوزيع البيانات الفعلي
    ]


def format_as_instruction_pairs(examples: list[InvoiceExample]) -> list[dict]:
    """
    تحويل الأمثلة لصيغة instruction-tuning القياسية
    (نفس الصيغة المستخدمة في الفصول السابقة لـ SFT).
    """
    formatted = []
    for ex in examples:
        instruction = (
            "استخرج من نص الفاتورة التالي: الفئة، المبلغ، والتاريخ، "
            "وأرجع النتيجة بصيغة JSON فقط."
        )
        output_json = json.dumps({
            "category": ex.category, "amount": ex.amount, "date": ex.date
        }, ensure_ascii=False)

        formatted.append({
            "instruction": instruction,
            "input": ex.raw_text,
            "output": output_json,
        })
    return formatted


# ---------- المرحلة 2: إعداد الضبط الدقيق بـ LoRA ----------

def setup_lora_finetuning_config():
    """
    إعداد كامل وصحيح لضبط Phi-3-mini بـ LoRA.
    هذا الكود يمثل البنية الفعلية المستخدمة في الإنتاج
    (نفس الأدوات والمنهجية من فصل LoRA/PEFT السابق في الكتاب).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    import torch

    model_id = "microsoft/Phi-3-mini-4k-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                          # رتبة مصفوفات LoRA
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    # عادة تُظهر هذه الطباعة أن أقل من 1% من المعاملات فعليًا قابلة للتدريب

    training_args = TrainingArguments(
        output_dir="./phi3-invoice-classifier-lora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    return tokenizer, peft_model, training_args


# ---------- المرحلة 3: دالة التدريب الكاملة ----------

def run_finetuning_pipeline():
    """
    تنفيذ كامل لدورة الضبط الدقيق، من تحضير البيانات حتى حفظ النموذج.
    (يتطلب بيئة GPU فعلية للتشغيل الحقيقي)
    """
    from datasets import Dataset
    from trl import SFTTrainer

    raw_examples = build_training_dataset()
    formatted_data = format_as_instruction_pairs(raw_examples)

    def format_prompt(example: dict) -> str:
        return (
            f"<|user|>\n{example['instruction']}\n\n{example['input']}<|end|>\n"
            f"<|assistant|>\n{example['output']}<|end|>"
        )

    dataset = Dataset.from_list(
        [{"text": format_prompt(ex)} for ex in formatted_data]
    )

    tokenizer, peft_model, training_args = setup_lora_finetuning_config()

    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=512,
    )

    trainer.train()
    trainer.save_model("./phi3-invoice-classifier-final")
    print("✅ تم حفظ النموذج المضبوط دقيقًا بنجاح.")


# ---------- المرحلة 4: التقييم على بيانات اختبار محجوزة ----------

def evaluate_finetuned_model(
    model, tokenizer, test_examples: list[InvoiceExample]
) -> float:
    """حساب دقة التصنيف على بيانات اختبار لم يرها النموذج أثناء التدريب"""
    correct = 0

    for example in test_examples:
        prompt = (
            f"<|user|>\nاستخرج من نص الفاتورة التالي: الفئة، المبلغ، والتاريخ، "
            f"وأرجع النتيجة بصيغة JSON فقط.\n\n{example.raw_text}<|end|>\n<|assistant|>\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.0)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            parsed = json.loads(generated_text.split("<|assistant|>")[-1].strip())
            if parsed.get("category") == example.category:
                correct += 1
        except (json.JSONDecodeError, IndexError):
            pass  # فشل في تحليل المخرجات = تصنيف خاطئ

    return correct / len(test_examples)


print("البنية الكاملة لخط أنابيب الضبط الدقيق جاهزة.")
print("لتشغيلها فعليًا: pip install transformers peft trl datasets bitsandbytes torch")
print("ثم: run_finetuning_pipeline()")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-68-chapter-68.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 68: الفصل 68: الكفاءة كاتجاه بحثي — النماذج اللغوية الصغيرة (SLMs: Phi, Gemma, Qwen-Small)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_slm() -> None:
    teacher = [0.65, 0.25, 0.1]
    student = [0.58, 0.3, 0.12]
    divergence = sum((p * math.log(p / q) for p, q in zip(teacher, student)))
    model_sizes = {'teacher_params': 70000000000, 'student_params': 3000000000}
    print('teacher/student KL:', round(divergence, 5))
    print('parameter reduction:', round(1 - model_sizes['student_params'] / model_sizes['teacher_params'], 3))
    print('measure quality and latency on the target task before choosing.')

if __name__ == "__main__":
    print("Chapter 68 lab: Compare a compact student with a large teacher")
    run_slm()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-68-chapter-68.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
data_quality_simulation.py
محاكاة تعليمية مبسّطة توضّح فكرة: بيانات أقل لكن أنظف
قد تُنتج تعلّمًا أفضل من بيانات أكثر لكن أكثر ضوضاءً،
في مهام تتطلب إشارة واضحة (signal) بدلًا من تنوع واسع.

هذا ليس تدريب نموذج لغوي حقيقي، بل توضيح مُبسّط جدًا
للفكرة الإحصائية الكامنة خلف "Textbooks Are All You Need".
"""
import numpy as np


def generate_noisy_dataset(n_samples: int, noise_level: float) -> tuple[np.ndarray, np.ndarray]:
    """توليد بيانات تعليمية بعلاقة خطية بسيطة + ضوضاء بمستوى متحكَّم به"""
    rng = np.random.default_rng(42)
    x = rng.uniform(-5, 5, n_samples)
    true_relationship = 2.0 * x + 3.0
    noise = rng.normal(0, noise_level, n_samples)
    y = true_relationship + noise
    return x, y


def fit_simple_linear_model(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """نموذج خطي بسيط عبر المربعات الصغرى (يمثل "تعلّم" العلاقة من البيانات)"""
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope, intercept


def evaluate_on_clean_test_set(slope: float, intercept: float) -> float:
    """تقييم الدقة على بيانات اختبار نظيفة (الحقيقة المعروفة: slope=2, intercept=3)"""
    error = abs(slope - 2.0) + abs(intercept - 3.0)
    return error


# سيناريو 1: كمية بيانات كبيرة جدًا، لكن ضوضاء عالية (يحاكي بيانات الويب الخام)
x_large_noisy, y_large_noisy = generate_noisy_dataset(n_samples=10_000, noise_level=15.0)
slope_1, intercept_1 = fit_simple_linear_model(x_large_noisy, y_large_noisy)
error_1 = evaluate_on_clean_test_set(slope_1, intercept_1)

# سيناريو 2: كمية بيانات أصغر بكثير، لكن ضوضاء منخفضة جدًا (يحاكي بيانات "textbook-quality")
x_small_clean, y_small_clean = generate_noisy_dataset(n_samples=200, noise_level=0.5)
slope_2, intercept_2 = fit_simple_linear_model(x_small_clean, y_small_clean)
error_2 = evaluate_on_clean_test_set(slope_2, intercept_2)

print(f"سيناريو 1 (بيانات كثيرة وصاخبة):  حجم={10_000}, خطأ النموذج={error_1:.4f}")
print(f"سيناريو 2 (بيانات قليلة ونظيفة): حجم={200}, خطأ النموذج={error_2:.4f}")
print(f"\n{'✅ البيانات النظيفة الأقل تفوّقت!' if error_2 < error_1 else '⚠️ البيانات الكثيرة تفوّقت رغم الضوضاء'}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-68-chapter-68.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
grouped_query_attention.py
تطبيق تعليمي واضح لآلية Grouped-Query Attention (GQA) من الصفر،
مقارنةً بـ Multi-Head Attention (MHA) التقليدي، لتوضيح وفورات الذاكرة.
"""
import numpy as np


def compute_kv_cache_size(
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    sequence_length: int,
    bytes_per_param: int = 2,  # FP16
) -> float:
    """
    حساب حجم KV Cache بالميغابايت.
    في GQA، عدد رؤوس K,V المُخزَّنة أقل من عدد رؤوس الاستعلام،
    مما يقلل حجم الكاش مباشرة بنفس النسبة.
    """
    # الكاش يُخزَّن فقط K وV، وليس Q (Q يُحسب لحظيًا لكل توكن جديد)
    kv_cache_elements = 2 * num_kv_heads * head_dim * sequence_length  # ×2 لـ K و V
    kv_cache_bytes = kv_cache_elements * bytes_per_param
    return kv_cache_bytes / (1024 ** 2)  # تحويل لميغابايت


# مقارنة: نموذج بـ 32 رأس استعلام، head_dim=128، سياق 8192 توكن
num_query_heads = 32
head_dim = 128
sequence_length = 8192

configurations = {
    "MHA التقليدي (32 رأس K,V منفصل)": 32,
    "GQA مع 8 مجموعات (Gemma/Llama-style)": 8,
    "GQA مع 4 مجموعات (أكثر عدوانية)": 4,
    "MQA (رأس K,V واحد مشترك)": 1,
}

print(f"{'الإعداد':<40}{'رؤوس K,V':<12}{'حجم KV Cache (MB)':<20}")
print("-" * 72)
for config_name, num_kv_heads in configurations.items():
    cache_size = compute_kv_cache_size(
        num_query_heads, num_kv_heads, head_dim, sequence_length
    )
    print(f"{config_name:<40}{num_kv_heads:<12}{cache_size:<20.1f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-68-chapter-68.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class GroupedQueryAttention:
    """
    تطبيق مُبسّط وقابل للتشغيل فعليًا لـ GQA باستخدام NumPy،
    لأغراض تعليمية (في الإنتاج الحقيقي تُستخدم PyTorch/JAX مع kernels محسّنة).
    """

    def __init__(self, num_query_heads: int, num_kv_heads: int, head_dim: int):
        assert num_query_heads % num_kv_heads == 0, \
            "عدد رؤوس الاستعلام يجب أن يقبل القسمة على عدد مجموعات K,V"

        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = num_query_heads // num_kv_heads  # كم رأس استعلام لكل مجموعة K,V

    def forward(
        self, queries: np.ndarray, keys: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        """
        queries: شكل (num_query_heads, seq_len, head_dim)
        keys, values: شكل (num_kv_heads, seq_len, head_dim)  <- أقل من عدد رؤوس الاستعلام!
        """
        seq_len = queries.shape[1]
        output = np.zeros_like(queries)

        for q_head_idx in range(self.num_query_heads):
            # كل رأس استعلام "يشارك" مجموعة K,V محددة بناءً على موقعه
            kv_group_idx = q_head_idx // self.group_size

            q = queries[q_head_idx]            # (seq_len, head_dim)
            k = keys[kv_group_idx]              # (seq_len, head_dim) — مُشتركة بين عدة رؤوس Q
            v = values[kv_group_idx]

            scores = (q @ k.T) / np.sqrt(self.head_dim)
            # قناع سببي بسيط (causal mask) لمحاكاة التوليد التتابعي
            causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
            scores[causal_mask] = -np.inf

            attn_weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn_weights /= attn_weights.sum(axis=-1, keepdims=True)

            output[q_head_idx] = attn_weights @ v

        return output


# اختبار عملي
np.random.seed(0)
num_query_heads, num_kv_heads, head_dim, seq_len = 8, 2, 16, 5

gqa = GroupedQueryAttention(num_query_heads, num_kv_heads, head_dim)
queries = np.random.randn(num_query_heads, seq_len, head_dim)
keys = np.random.randn(num_kv_heads, seq_len, head_dim)
values = np.random.randn(num_kv_heads, seq_len, head_dim)

output = gqa.forward(queries, keys, values)
print(f"\nشكل مخرجات GQA: {output.shape}")
print(f"عدد رؤوس K,V الفعلية المُخزَّنة: {num_kv_heads} بدلاً من {num_query_heads}")
print(f"نسبة توفير KV Cache: {(1 - num_kv_heads/num_query_heads) * 100:.0f}%")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-68-chapter-68.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
"""
model_selection_advisor.py
أداة مساعدة لاتخاذ قرار: متى تختار SLM ومن أي عائلة،
بناءً على قيود المشروع الفعلية.
"""
from dataclasses import dataclass
from enum import Enum


class ConstraintPriority(Enum):
    LATENCY_CRITICAL = "latency_critical"       # يجب أن يعمل محليًا/بسرعة فائقة
    MULTILINGUAL_HEAVY = "multilingual_heavy"   # دعم لغات متعددة أساسي
    CODE_SPECIALIZED = "code_specialized"       # مهمة برمجية متخصصة
    MATH_REASONING = "math_reasoning"           # استدلال رياضي/منطقي
    GENERAL_PURPOSE = "general_purpose"         # مهام عامة متنوعة
    BUDGET_EXTREME = "budget_extreme"           # ميزانية حوسبة شديدة المحدودية (IoT مثلًا)


@dataclass
class ModelRecommendation:
    family: str
    suggested_size: str
    rationale: str


MODEL_ADVISOR_TABLE: dict[ConstraintPriority, ModelRecommendation] = {
    ConstraintPriority.LATENCY_CRITICAL: ModelRecommendation(
        family="Phi-3-mini / Gemma-2B",
        suggested_size="2B-3.8B",
        rationale="أحجام مصممة خصيصًا للتشغيل على الحافة (edge) والهواتف، "
                   "مع دعم رسمي لـ ONNX Runtime وتهيئات مُحسَّنة للموبايل.",
    ),
    ConstraintPriority.MULTILINGUAL_HEAVY: ModelRecommendation(
        family="Qwen2.5",
        suggested_size="3B-7B",
        rationale="تدريب مكثف على بيانات متعددة اللغات متنوعة، "
                   "أداء موثّق أقوى في اللغات غير الإنجليزية مقارنة بمنافسيها في نفس الفئة.",
    ),
    ConstraintPriority.CODE_SPECIALIZED: ModelRecommendation(
        family="Qwen2.5-Coder أو Phi نسخة مُدرَّبة على كود",
        suggested_size="7B-14B",
        rationale="إصدارات متخصصة مُدرَّبة إضافيًا على كميات هائلة من الكود عالي الجودة.",
    ),
    ConstraintPriority.MATH_REASONING: ModelRecommendation(
        family="Phi-4 أو Qwen2.5-Math",
        suggested_size="7B-14B",
        rationale="تركيز تدريبي صريح على بيانات استدلال رياضي ومنطقي تركيبية عالية الجودة.",
    ),
    ConstraintPriority.GENERAL_PURPOSE: ModelRecommendation(
        family="Gemma 2 أو Qwen2.5",
        suggested_size="7B-9B",
        rationale="توازن جيد بين الأداء العام والتكلفة، مجتمع دعم واسع ومصادر تدريب غنية.",
    ),
    ConstraintPriority.BUDGET_EXTREME: ModelRecommendation(
        family="Qwen2.5-0.5B أو Gemma-2B مُكمَّم (quantized)",
        suggested_size="0.5B-2B",
        rationale="أصغر الأحجام المتاحة مع دعم تكميم فعّال (4-bit/8-bit) "
                   "للعمل على أجهزة IoT أو متصفحات (WebGPU).",
    ),
}


def recommend_model(priority: ConstraintPriority) -> ModelRecommendation:
    return MODEL_ADVISOR_TABLE[priority]


# مثال استخدام
recommendation = recommend_model(ConstraintPriority.LATENCY_CRITICAL)
print(f"العائلة المقترحة: {recommendation.family}")
print(f"الحجم المقترح   : {recommendation.suggested_size}")
print(f"السبب            : {recommendation.rationale}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-68-chapter-68.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
"""
specialized_vs_general_benchmark.py
محاكاة تعليمية توضح منهجية المقارنة بين SLM مضبوط دقيقًا
مقابل LLM عام ضخم، على مهمة ضيقة محددة (تصنيف فواتير).
"""
from dataclasses import dataclass


@dataclass
class ModelBenchmarkResult:
    model_name: str
    accuracy_on_narrow_task: float
    latency_ms: float
    cost_per_1000_requests_usd: float
    requires_internet: bool


def simulate_benchmark_results() -> list[ModelBenchmarkResult]:
    """
    نتائج محاكاة واقعية الشكل (مبنية على أنماط حقيقية موثقة في الأدبيات،
    وليست أرقامًا رسمية من أي شركة محددة) لتوضيح الاتجاه العام.
    """
    return [
        ModelBenchmarkResult(
            model_name="GPT-4 class (عام، عبر API)",
            accuracy_on_narrow_task=0.91,
            latency_ms=1800,
            cost_per_1000_requests_usd=15.0,
            requires_internet=True,
        ),
        ModelBenchmarkResult(
            model_name="Phi-3-mini (3.8B) + LoRA مضبوط على المهمة",
            accuracy_on_narrow_task=0.94,
            latency_ms=85,
            cost_per_1000_requests_usd=0.3,
            requires_internet=False,
        ),
        ModelBenchmarkResult(
            model_name="Phi-3-mini (3.8B) بدون ضبط دقيق (zero-shot)",
            accuracy_on_narrow_task=0.72,
            latency_ms=85,
            cost_per_1000_requests_usd=0.3,
            requires_internet=False,
        ),
    ]


def print_comparison_report(results: list[ModelBenchmarkResult]) -> None:
    print(f"{'النموذج':<45}{'الدقة':<10}{'الكمون(ms)':<12}{'التكلفة/1000':<15}{'يحتاج إنترنت'}")
    print("-" * 100)
    for r in results:
        internet_flag = "نعم" if r.requires_internet else "لا"
        print(
            f"{r.model_name:<45}{r.accuracy_on_narrow_task:<10.2f}"
            f"{r.latency_ms:<12.0f}${r.cost_per_1000_requests_usd:<14.2f}{internet_flag}"
        )


results = simulate_benchmark_results()
print_comparison_report(results)

print("\n📊 الاستنتاج الرئيسي:")
print("النموذج الصغير المضبوط دقيقًا تفوّق على النموذج الضخم العام في الدقة")
print("على هذه المهمة الضيقة تحديدًا، بينما كان أسرع بـ 21 مرة وأرخص بـ 50 مرة،")
print("وقابل للعمل محليًا بدون اتصال إنترنت.")
print("\n⚠️ تنبيه مهم: هذا لا يعني SLM 'أفضل مطلقًا' —")
print("بدون الضبط الدقيق، أداؤه انخفض بشكل حاد (من 0.94 إلى 0.72)،")
print("والنموذج الضخم العام يبقى الخيار الأفضل لمهام متنوعة غير محددة مسبقًا.")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-68-chapter-68.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
"""
invoice_classifier_slm_pipeline.py
نظام End-to-End كامل: تحضير بيانات، ضبط دقيق (LoRA) لنموذج SLM صغير
(Phi-3-mini كمثال)، وتقييم النتيجة على بيانات اختبار محجوزة.

يستخدم مكتبات PyTorch + Hugging Face transformers + peft (LoRA)،
وهي المكدس المعياري الحديث (2024-2025) لهذا النوع من المهام.

ملاحظة: تشغيل هذا الكود فعليًا يتطلب تثبيت
transformers, peft, torch, datasets, bitsandbytes
وتحميل أوزان نموذج حقيقي (مثل microsoft/Phi-3-mini-4k-instruct)،
وقد يتطلب GPU للتدريب الفعلي. البنية والمنطق أدناه كاملان وصحيحان
ويمثلان تطبيقًا إنتاجيًا حقيقيًا، وليسا محاكاة.
"""

from dataclasses import dataclass
import json


# ---------- المرحلة 1: تحضير بيانات الضبط الدقيق ----------

@dataclass
class InvoiceExample:
    raw_text: str
    category: str   # مثل: "مطاعم", "نقل", "مستلزمات مكتبية", "اشتراكات"
    amount: float
    date: str


def build_training_dataset() -> list[InvoiceExample]:
    """بناء مجموعة بيانات تدريب صغيرة ومُنسّقة (في الواقع، قد تحتاج آلاف الأمثلة)"""
    return [
        InvoiceExample(
            raw_text="مطعم السلطان - فاتورة بتاريخ 2025-03-14 بقيمة 145.50 جنيه",
            category="مطاعم", amount=145.50, date="2025-03-14",
        ),
        InvoiceExample(
            raw_text="اشتراك شهري Netflix - 2025-03-01 - 199 جنيه",
            category="اشتراكات", amount=199.0, date="2025-03-01",
        ),
        InvoiceExample(
            raw_text="أوبر - رحلة بتاريخ 15 مارس 2025 - المبلغ 68 جنيه",
            category="نقل", amount=68.0, date="2025-03-15",
        ),
        InvoiceExample(
            raw_text="مكتبة الأهرام - أدوات مكتبية - 2025-03-10 - 320 جنيه",
            category="مستلزمات مكتبية", amount=320.0, date="2025-03-10",
        ),
        # في مشروع حقيقي: مئات أو آلاف الأمثلة المتنوعة والممثّلة لتوزيع البيانات الفعلي
    ]


def format_as_instruction_pairs(examples: list[InvoiceExample]) -> list[dict]:
    """
    تحويل الأمثلة لصيغة instruction-tuning القياسية
    (نفس الصيغة المستخدمة في الفصول السابقة لـ SFT).
    """
    formatted = []
    for ex in examples:
        instruction = (
            "استخرج من نص الفاتورة التالي: الفئة، المبلغ، والتاريخ، "
            "وأرجع النتيجة بصيغة JSON فقط."
        )
        output_json = json.dumps({
            "category": ex.category, "amount": ex.amount, "date": ex.date
        }, ensure_ascii=False)

        formatted.append({
            "instruction": instruction,
            "input": ex.raw_text,
            "output": output_json,
        })
    return formatted


# ---------- المرحلة 2: إعداد الضبط الدقيق بـ LoRA ----------

def setup_lora_finetuning_config():
    """
    إعداد كامل وصحيح لضبط Phi-3-mini بـ LoRA.
    هذا الكود يمثل البنية الفعلية المستخدمة في الإنتاج
    (نفس الأدوات والمنهجية من فصل LoRA/PEFT السابق في الكتاب).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    import torch

    model_id = "microsoft/Phi-3-mini-4k-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                          # رتبة مصفوفات LoRA
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    # عادة تُظهر هذه الطباعة أن أقل من 1% من المعاملات فعليًا قابلة للتدريب

    training_args = TrainingArguments(
        output_dir="./phi3-invoice-classifier-lora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    return tokenizer, peft_model, training_args


# ---------- المرحلة 3: دالة التدريب الكاملة ----------

def run_finetuning_pipeline():
    """
    تنفيذ كامل لدورة الضبط الدقيق، من تحضير البيانات حتى حفظ النموذج.
    (يتطلب بيئة GPU فعلية للتشغيل الحقيقي)
    """
    from datasets import Dataset
    from trl import SFTTrainer

    raw_examples = build_training_dataset()
    formatted_data = format_as_instruction_pairs(raw_examples)

    def format_prompt(example: dict) -> str:
        return (
            f"<|user|>\n{example['instruction']}\n\n{example['input']}<|end|>\n"
            f"<|assistant|>\n{example['output']}<|end|>"
        )

    dataset = Dataset.from_list(
        [{"text": format_prompt(ex)} for ex in formatted_data]
    )

    tokenizer, peft_model, training_args = setup_lora_finetuning_config()

    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=512,
    )

    trainer.train()
    trainer.save_model("./phi3-invoice-classifier-final")
    print("✅ تم حفظ النموذج المضبوط دقيقًا بنجاح.")


# ---------- المرحلة 4: التقييم على بيانات اختبار محجوزة ----------

def evaluate_finetuned_model(
    model, tokenizer, test_examples: list[InvoiceExample]
) -> float:
    """حساب دقة التصنيف على بيانات اختبار لم يرها النموذج أثناء التدريب"""
    correct = 0

    for example in test_examples:
        prompt = (
            f"<|user|>\nاستخرج من نص الفاتورة التالي: الفئة، المبلغ، والتاريخ، "
            f"وأرجع النتيجة بصيغة JSON فقط.\n\n{example.raw_text}<|end|>\n<|assistant|>\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.0)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            parsed = json.loads(generated_text.split("<|assistant|>")[-1].strip())
            if parsed.get("category") == example.category:
                correct += 1
        except (json.JSONDecodeError, IndexError):
            pass  # فشل في تحليل المخرجات = تصنيف خاطئ

    return correct / len(test_examples)


print("البنية الكاملة لخط أنابيب الضبط الدقيق جاهزة.")
print("لتشغيلها فعليًا: pip install transformers peft trl datasets bitsandbytes torch")
print("ثم: run_finetuning_pipeline()")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-68-chapter-68.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الكفاءة كاتجاه بحثي — النماذج اللغوية الصغيرة (SLMs: Phi, Gemma, Qwen-Small)
Large Language Model Engineering — SLMs

Book slug : llm-engineering-69-chapters
Chapter   : ch-68-chapter-68 (lesson 68)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_68.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 68: الفصل 68: الكفاءة كاتجاه بحثي — النماذج اللغوية الصغيرة (SLMs: Phi, Gemma, Qwen-Small)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_slm() -> None:
    teacher = [0.65, 0.25, 0.1]
    student = [0.58, 0.3, 0.12]
    divergence = sum((p * math.log(p / q) for p, q in zip(teacher, student)))
    model_sizes = {'teacher_params': 70000000000, 'student_params': 3000000000}
    print('teacher/student KL:', round(divergence, 5))
    print('parameter reduction:', round(1 - model_sizes['student_params'] / model_sizes['teacher_params'], 3))
    print('measure quality and latency on the target task before choosing.')

if __name__ == "__main__":
    print("Chapter 68 lab: Compare a compact student with a large teacher")
    run_slm()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-12-frontier-trends/chapter-68.md
# ────────────────────────────────────────────────────────────────────────
"""
data_quality_simulation.py
محاكاة تعليمية مبسّطة توضّح فكرة: بيانات أقل لكن أنظف
قد تُنتج تعلّمًا أفضل من بيانات أكثر لكن أكثر ضوضاءً،
في مهام تتطلب إشارة واضحة (signal) بدلًا من تنوع واسع.

هذا ليس تدريب نموذج لغوي حقيقي، بل توضيح مُبسّط جدًا
للفكرة الإحصائية الكامنة خلف "Textbooks Are All You Need".
"""
import numpy as np


def generate_noisy_dataset(n_samples: int, noise_level: float) -> tuple[np.ndarray, np.ndarray]:
    """توليد بيانات تعليمية بعلاقة خطية بسيطة + ضوضاء بمستوى متحكَّم به"""
    rng = np.random.default_rng(42)
    x = rng.uniform(-5, 5, n_samples)
    true_relationship = 2.0 * x + 3.0
    noise = rng.normal(0, noise_level, n_samples)
    y = true_relationship + noise
    return x, y


def fit_simple_linear_model(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """نموذج خطي بسيط عبر المربعات الصغرى (يمثل "تعلّم" العلاقة من البيانات)"""
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope, intercept


def evaluate_on_clean_test_set(slope: float, intercept: float) -> float:
    """تقييم الدقة على بيانات اختبار نظيفة (الحقيقة المعروفة: slope=2, intercept=3)"""
    error = abs(slope - 2.0) + abs(intercept - 3.0)
    return error


# سيناريو 1: كمية بيانات كبيرة جدًا، لكن ضوضاء عالية (يحاكي بيانات الويب الخام)
x_large_noisy, y_large_noisy = generate_noisy_dataset(n_samples=10_000, noise_level=15.0)
slope_1, intercept_1 = fit_simple_linear_model(x_large_noisy, y_large_noisy)
error_1 = evaluate_on_clean_test_set(slope_1, intercept_1)

# سيناريو 2: كمية بيانات أصغر بكثير، لكن ضوضاء منخفضة جدًا (يحاكي بيانات "textbook-quality")
x_small_clean, y_small_clean = generate_noisy_dataset(n_samples=200, noise_level=0.5)
slope_2, intercept_2 = fit_simple_linear_model(x_small_clean, y_small_clean)
error_2 = evaluate_on_clean_test_set(slope_2, intercept_2)

print(f"سيناريو 1 (بيانات كثيرة وصاخبة):  حجم={10_000}, خطأ النموذج={error_1:.4f}")
print(f"سيناريو 2 (بيانات قليلة ونظيفة): حجم={200}, خطأ النموذج={error_2:.4f}")
print(f"\n{'✅ البيانات النظيفة الأقل تفوّقت!' if error_2 < error_1 else '⚠️ البيانات الكثيرة تفوّقت رغم الضوضاء'}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-12-frontier-trends/chapter-68.md
# ────────────────────────────────────────────────────────────────────────
"""
grouped_query_attention.py
تطبيق تعليمي واضح لآلية Grouped-Query Attention (GQA) من الصفر،
مقارنةً بـ Multi-Head Attention (MHA) التقليدي، لتوضيح وفورات الذاكرة.
"""
import numpy as np


def compute_kv_cache_size(
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    sequence_length: int,
    bytes_per_param: int = 2,  # FP16
) -> float:
    """
    حساب حجم KV Cache بالميغابايت.
    في GQA، عدد رؤوس K,V المُخزَّنة أقل من عدد رؤوس الاستعلام،
    مما يقلل حجم الكاش مباشرة بنفس النسبة.
    """
    # الكاش يُخزَّن فقط K وV، وليس Q (Q يُحسب لحظيًا لكل توكن جديد)
    kv_cache_elements = 2 * num_kv_heads * head_dim * sequence_length  # ×2 لـ K و V
    kv_cache_bytes = kv_cache_elements * bytes_per_param
    return kv_cache_bytes / (1024 ** 2)  # تحويل لميغابايت


# مقارنة: نموذج بـ 32 رأس استعلام، head_dim=128، سياق 8192 توكن
num_query_heads = 32
head_dim = 128
sequence_length = 8192

configurations = {
    "MHA التقليدي (32 رأس K,V منفصل)": 32,
    "GQA مع 8 مجموعات (Gemma/Llama-style)": 8,
    "GQA مع 4 مجموعات (أكثر عدوانية)": 4,
    "MQA (رأس K,V واحد مشترك)": 1,
}

print(f"{'الإعداد':<40}{'رؤوس K,V':<12}{'حجم KV Cache (MB)':<20}")
print("-" * 72)
for config_name, num_kv_heads in configurations.items():
    cache_size = compute_kv_cache_size(
        num_query_heads, num_kv_heads, head_dim, sequence_length
    )
    print(f"{config_name:<40}{num_kv_heads:<12}{cache_size:<20.1f}")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-12-frontier-trends/chapter-68.md
# ────────────────────────────────────────────────────────────────────────
class GroupedQueryAttention:
    """
    تطبيق مُبسّط وقابل للتشغيل فعليًا لـ GQA باستخدام NumPy،
    لأغراض تعليمية (في الإنتاج الحقيقي تُستخدم PyTorch/JAX مع kernels محسّنة).
    """

    def __init__(self, num_query_heads: int, num_kv_heads: int, head_dim: int):
        assert num_query_heads % num_kv_heads == 0, \
            "عدد رؤوس الاستعلام يجب أن يقبل القسمة على عدد مجموعات K,V"

        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = num_query_heads // num_kv_heads  # كم رأس استعلام لكل مجموعة K,V

    def forward(
        self, queries: np.ndarray, keys: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        """
        queries: شكل (num_query_heads, seq_len, head_dim)
        keys, values: شكل (num_kv_heads, seq_len, head_dim)  <- أقل من عدد رؤوس الاستعلام!
        """
        seq_len = queries.shape[1]
        output = np.zeros_like(queries)

        for q_head_idx in range(self.num_query_heads):
            # كل رأس استعلام "يشارك" مجموعة K,V محددة بناءً على موقعه
            kv_group_idx = q_head_idx // self.group_size

            q = queries[q_head_idx]            # (seq_len, head_dim)
            k = keys[kv_group_idx]              # (seq_len, head_dim) — مُشتركة بين عدة رؤوس Q
            v = values[kv_group_idx]

            scores = (q @ k.T) / np.sqrt(self.head_dim)
            # قناع سببي بسيط (causal mask) لمحاكاة التوليد التتابعي
            causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
            scores[causal_mask] = -np.inf

            attn_weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn_weights /= attn_weights.sum(axis=-1, keepdims=True)

            output[q_head_idx] = attn_weights @ v

        return output


# اختبار عملي
np.random.seed(0)
num_query_heads, num_kv_heads, head_dim, seq_len = 8, 2, 16, 5

gqa = GroupedQueryAttention(num_query_heads, num_kv_heads, head_dim)
queries = np.random.randn(num_query_heads, seq_len, head_dim)
keys = np.random.randn(num_kv_heads, seq_len, head_dim)
values = np.random.randn(num_kv_heads, seq_len, head_dim)

output = gqa.forward(queries, keys, values)
print(f"\nشكل مخرجات GQA: {output.shape}")
print(f"عدد رؤوس K,V الفعلية المُخزَّنة: {num_kv_heads} بدلاً من {num_query_heads}")
print(f"نسبة توفير KV Cache: {(1 - num_kv_heads/num_query_heads) * 100:.0f}%")

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-12-frontier-trends/chapter-68.md
# ────────────────────────────────────────────────────────────────────────
"""
model_selection_advisor.py
أداة مساعدة لاتخاذ قرار: متى تختار SLM ومن أي عائلة،
بناءً على قيود المشروع الفعلية.
"""
from dataclasses import dataclass
from enum import Enum


class ConstraintPriority(Enum):
    LATENCY_CRITICAL = "latency_critical"       # يجب أن يعمل محليًا/بسرعة فائقة
    MULTILINGUAL_HEAVY = "multilingual_heavy"   # دعم لغات متعددة أساسي
    CODE_SPECIALIZED = "code_specialized"       # مهمة برمجية متخصصة
    MATH_REASONING = "math_reasoning"           # استدلال رياضي/منطقي
    GENERAL_PURPOSE = "general_purpose"         # مهام عامة متنوعة
    BUDGET_EXTREME = "budget_extreme"           # ميزانية حوسبة شديدة المحدودية (IoT مثلًا)


@dataclass
class ModelRecommendation:
    family: str
    suggested_size: str
    rationale: str


MODEL_ADVISOR_TABLE: dict[ConstraintPriority, ModelRecommendation] = {
    ConstraintPriority.LATENCY_CRITICAL: ModelRecommendation(
        family="Phi-3-mini / Gemma-2B",
        suggested_size="2B-3.8B",
        rationale="أحجام مصممة خصيصًا للتشغيل على الحافة (edge) والهواتف، "
                   "مع دعم رسمي لـ ONNX Runtime وتهيئات مُحسَّنة للموبايل.",
    ),
    ConstraintPriority.MULTILINGUAL_HEAVY: ModelRecommendation(
        family="Qwen2.5",
        suggested_size="3B-7B",
        rationale="تدريب مكثف على بيانات متعددة اللغات متنوعة، "
                   "أداء موثّق أقوى في اللغات غير الإنجليزية مقارنة بمنافسيها في نفس الفئة.",
    ),
    ConstraintPriority.CODE_SPECIALIZED: ModelRecommendation(
        family="Qwen2.5-Coder أو Phi نسخة مُدرَّبة على كود",
        suggested_size="7B-14B",
        rationale="إصدارات متخصصة مُدرَّبة إضافيًا على كميات هائلة من الكود عالي الجودة.",
    ),
    ConstraintPriority.MATH_REASONING: ModelRecommendation(
        family="Phi-4 أو Qwen2.5-Math",
        suggested_size="7B-14B",
        rationale="تركيز تدريبي صريح على بيانات استدلال رياضي ومنطقي تركيبية عالية الجودة.",
    ),
    ConstraintPriority.GENERAL_PURPOSE: ModelRecommendation(
        family="Gemma 2 أو Qwen2.5",
        suggested_size="7B-9B",
        rationale="توازن جيد بين الأداء العام والتكلفة، مجتمع دعم واسع ومصادر تدريب غنية.",
    ),
    ConstraintPriority.BUDGET_EXTREME: ModelRecommendation(
        family="Qwen2.5-0.5B أو Gemma-2B مُكمَّم (quantized)",
        suggested_size="0.5B-2B",
        rationale="أصغر الأحجام المتاحة مع دعم تكميم فعّال (4-bit/8-bit) "
                   "للعمل على أجهزة IoT أو متصفحات (WebGPU).",
    ),
}


def recommend_model(priority: ConstraintPriority) -> ModelRecommendation:
    return MODEL_ADVISOR_TABLE[priority]


# مثال استخدام
recommendation = recommend_model(ConstraintPriority.LATENCY_CRITICAL)
print(f"العائلة المقترحة: {recommendation.family}")
print(f"الحجم المقترح   : {recommendation.suggested_size}")
print(f"السبب            : {recommendation.rationale}")

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-12-frontier-trends/chapter-68.md
# ────────────────────────────────────────────────────────────────────────
"""
specialized_vs_general_benchmark.py
محاكاة تعليمية توضح منهجية المقارنة بين SLM مضبوط دقيقًا
مقابل LLM عام ضخم، على مهمة ضيقة محددة (تصنيف فواتير).
"""
from dataclasses import dataclass


@dataclass
class ModelBenchmarkResult:
    model_name: str
    accuracy_on_narrow_task: float
    latency_ms: float
    cost_per_1000_requests_usd: float
    requires_internet: bool


def simulate_benchmark_results() -> list[ModelBenchmarkResult]:
    """
    نتائج محاكاة واقعية الشكل (مبنية على أنماط حقيقية موثقة في الأدبيات،
    وليست أرقامًا رسمية من أي شركة محددة) لتوضيح الاتجاه العام.
    """
    return [
        ModelBenchmarkResult(
            model_name="GPT-4 class (عام، عبر API)",
            accuracy_on_narrow_task=0.91,
            latency_ms=1800,
            cost_per_1000_requests_usd=15.0,
            requires_internet=True,
        ),
        ModelBenchmarkResult(
            model_name="Phi-3-mini (3.8B) + LoRA مضبوط على المهمة",
            accuracy_on_narrow_task=0.94,
            latency_ms=85,
            cost_per_1000_requests_usd=0.3,
            requires_internet=False,
        ),
        ModelBenchmarkResult(
            model_name="Phi-3-mini (3.8B) بدون ضبط دقيق (zero-shot)",
            accuracy_on_narrow_task=0.72,
            latency_ms=85,
            cost_per_1000_requests_usd=0.3,
            requires_internet=False,
        ),
    ]


def print_comparison_report(results: list[ModelBenchmarkResult]) -> None:
    print(f"{'النموذج':<45}{'الدقة':<10}{'الكمون(ms)':<12}{'التكلفة/1000':<15}{'يحتاج إنترنت'}")
    print("-" * 100)
    for r in results:
        internet_flag = "نعم" if r.requires_internet else "لا"
        print(
            f"{r.model_name:<45}{r.accuracy_on_narrow_task:<10.2f}"
            f"{r.latency_ms:<12.0f}${r.cost_per_1000_requests_usd:<14.2f}{internet_flag}"
        )


results = simulate_benchmark_results()
print_comparison_report(results)

print("\n📊 الاستنتاج الرئيسي:")
print("النموذج الصغير المضبوط دقيقًا تفوّق على النموذج الضخم العام في الدقة")
print("على هذه المهمة الضيقة تحديدًا، بينما كان أسرع بـ 21 مرة وأرخص بـ 50 مرة،")
print("وقابل للعمل محليًا بدون اتصال إنترنت.")
print("\n⚠️ تنبيه مهم: هذا لا يعني SLM 'أفضل مطلقًا' —")
print("بدون الضبط الدقيق، أداؤه انخفض بشكل حاد (من 0.94 إلى 0.72)،")
print("والنموذج الضخم العام يبقى الخيار الأفضل لمهام متنوعة غير محددة مسبقًا.")

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-12-frontier-trends/chapter-68.md
# ────────────────────────────────────────────────────────────────────────
"""
invoice_classifier_slm_pipeline.py
نظام End-to-End كامل: تحضير بيانات، ضبط دقيق (LoRA) لنموذج SLM صغير
(Phi-3-mini كمثال)، وتقييم النتيجة على بيانات اختبار محجوزة.

يستخدم مكتبات PyTorch + Hugging Face transformers + peft (LoRA)،
وهي المكدس المعياري الحديث (2024-2025) لهذا النوع من المهام.

ملاحظة: تشغيل هذا الكود فعليًا يتطلب تثبيت
transformers, peft, torch, datasets, bitsandbytes
وتحميل أوزان نموذج حقيقي (مثل microsoft/Phi-3-mini-4k-instruct)،
وقد يتطلب GPU للتدريب الفعلي. البنية والمنطق أدناه كاملان وصحيحان
ويمثلان تطبيقًا إنتاجيًا حقيقيًا، وليسا محاكاة.
"""

from dataclasses import dataclass
import json


# ---------- المرحلة 1: تحضير بيانات الضبط الدقيق ----------

@dataclass
class InvoiceExample:
    raw_text: str
    category: str   # مثل: "مطاعم", "نقل", "مستلزمات مكتبية", "اشتراكات"
    amount: float
    date: str


def build_training_dataset() -> list[InvoiceExample]:
    """بناء مجموعة بيانات تدريب صغيرة ومُنسّقة (في الواقع، قد تحتاج آلاف الأمثلة)"""
    return [
        InvoiceExample(
            raw_text="مطعم السلطان - فاتورة بتاريخ 2025-03-14 بقيمة 145.50 جنيه",
            category="مطاعم", amount=145.50, date="2025-03-14",
        ),
        InvoiceExample(
            raw_text="اشتراك شهري Netflix - 2025-03-01 - 199 جنيه",
            category="اشتراكات", amount=199.0, date="2025-03-01",
        ),
        InvoiceExample(
            raw_text="أوبر - رحلة بتاريخ 15 مارس 2025 - المبلغ 68 جنيه",
            category="نقل", amount=68.0, date="2025-03-15",
        ),
        InvoiceExample(
            raw_text="مكتبة الأهرام - أدوات مكتبية - 2025-03-10 - 320 جنيه",
            category="مستلزمات مكتبية", amount=320.0, date="2025-03-10",
        ),
        # في مشروع حقيقي: مئات أو آلاف الأمثلة المتنوعة والممثّلة لتوزيع البيانات الفعلي
    ]


def format_as_instruction_pairs(examples: list[InvoiceExample]) -> list[dict]:
    """
    تحويل الأمثلة لصيغة instruction-tuning القياسية
    (نفس الصيغة المستخدمة في الفصول السابقة لـ SFT).
    """
    formatted = []
    for ex in examples:
        instruction = (
            "استخرج من نص الفاتورة التالي: الفئة، المبلغ، والتاريخ، "
            "وأرجع النتيجة بصيغة JSON فقط."
        )
        output_json = json.dumps({
            "category": ex.category, "amount": ex.amount, "date": ex.date
        }, ensure_ascii=False)

        formatted.append({
            "instruction": instruction,
            "input": ex.raw_text,
            "output": output_json,
        })
    return formatted


# ---------- المرحلة 2: إعداد الضبط الدقيق بـ LoRA ----------

def setup_lora_finetuning_config():
    """
    إعداد كامل وصحيح لضبط Phi-3-mini بـ LoRA.
    هذا الكود يمثل البنية الفعلية المستخدمة في الإنتاج
    (نفس الأدوات والمنهجية من فصل LoRA/PEFT السابق في الكتاب).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    import torch

    model_id = "microsoft/Phi-3-mini-4k-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                          # رتبة مصفوفات LoRA
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    # عادة تُظهر هذه الطباعة أن أقل من 1% من المعاملات فعليًا قابلة للتدريب

    training_args = TrainingArguments(
        output_dir="./phi3-invoice-classifier-lora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    return tokenizer, peft_model, training_args


# ---------- المرحلة 3: دالة التدريب الكاملة ----------

def run_finetuning_pipeline():
    """
    تنفيذ كامل لدورة الضبط الدقيق، من تحضير البيانات حتى حفظ النموذج.
    (يتطلب بيئة GPU فعلية للتشغيل الحقيقي)
    """
    from datasets import Dataset
    from trl import SFTTrainer

    raw_examples = build_training_dataset()
    formatted_data = format_as_instruction_pairs(raw_examples)

    def format_prompt(example: dict) -> str:
        return (
            f"<|user|>\n{example['instruction']}\n\n{example['input']}<|end|>\n"
            f"<|assistant|>\n{example['output']}<|end|>"
        )

    dataset = Dataset.from_list(
        [{"text": format_prompt(ex)} for ex in formatted_data]
    )

    tokenizer, peft_model, training_args = setup_lora_finetuning_config()

    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=512,
    )

    trainer.train()
    trainer.save_model("./phi3-invoice-classifier-final")
    print("✅ تم حفظ النموذج المضبوط دقيقًا بنجاح.")


# ---------- المرحلة 4: التقييم على بيانات اختبار محجوزة ----------

def evaluate_finetuned_model(
    model, tokenizer, test_examples: list[InvoiceExample]
) -> float:
    """حساب دقة التصنيف على بيانات اختبار لم يرها النموذج أثناء التدريب"""
    correct = 0

    for example in test_examples:
        prompt = (
            f"<|user|>\nاستخرج من نص الفاتورة التالي: الفئة، المبلغ، والتاريخ، "
            f"وأرجع النتيجة بصيغة JSON فقط.\n\n{example.raw_text}<|end|>\n<|assistant|>\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.0)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            parsed = json.loads(generated_text.split("<|assistant|>")[-1].strip())
            if parsed.get("category") == example.category:
                correct += 1
        except (json.JSONDecodeError, IndexError):
            pass  # فشل في تحليل المخرجات = تصنيف خاطئ

    return correct / len(test_examples)


print("البنية الكاملة لخط أنابيب الضبط الدقيق جاهزة.")
print("لتشغيلها فعليًا: pip install transformers peft trl datasets bitsandbytes torch")
print("ثم: run_finetuning_pipeline()")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-68-chapter-68.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 68: الفصل 68: الكفاءة كاتجاه بحثي — النماذج اللغوية الصغيرة (SLMs: Phi, Gemma, Qwen-Small)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_slm() -> None:
    teacher = [0.65, 0.25, 0.1]
    student = [0.58, 0.3, 0.12]
    divergence = sum((p * math.log(p / q) for p, q in zip(teacher, student)))
    model_sizes = {'teacher_params': 70000000000, 'student_params': 3000000000}
    print('teacher/student KL:', round(divergence, 5))
    print('parameter reduction:', round(1 - model_sizes['student_params'] / model_sizes['teacher_params'], 3))
    print('measure quality and latency on the target task before choosing.')

if __name__ == "__main__":
    print("Chapter 68 lab: Compare a compact student with a large teacher")
    run_slm()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-68-chapter-68.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
data_quality_simulation.py
محاكاة تعليمية مبسّطة توضّح فكرة: بيانات أقل لكن أنظف
قد تُنتج تعلّمًا أفضل من بيانات أكثر لكن أكثر ضوضاءً،
في مهام تتطلب إشارة واضحة (signal) بدلًا من تنوع واسع.

هذا ليس تدريب نموذج لغوي حقيقي، بل توضيح مُبسّط جدًا
للفكرة الإحصائية الكامنة خلف "Textbooks Are All You Need".
"""
import numpy as np


def generate_noisy_dataset(n_samples: int, noise_level: float) -> tuple[np.ndarray, np.ndarray]:
    """توليد بيانات تعليمية بعلاقة خطية بسيطة + ضوضاء بمستوى متحكَّم به"""
    rng = np.random.default_rng(42)
    x = rng.uniform(-5, 5, n_samples)
    true_relationship = 2.0 * x + 3.0
    noise = rng.normal(0, noise_level, n_samples)
    y = true_relationship + noise
    return x, y


def fit_simple_linear_model(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """نموذج خطي بسيط عبر المربعات الصغرى (يمثل "تعلّم" العلاقة من البيانات)"""
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope, intercept


def evaluate_on_clean_test_set(slope: float, intercept: float) -> float:
    """تقييم الدقة على بيانات اختبار نظيفة (الحقيقة المعروفة: slope=2, intercept=3)"""
    error = abs(slope - 2.0) + abs(intercept - 3.0)
    return error


# سيناريو 1: كمية بيانات كبيرة جدًا، لكن ضوضاء عالية (يحاكي بيانات الويب الخام)
x_large_noisy, y_large_noisy = generate_noisy_dataset(n_samples=10_000, noise_level=15.0)
slope_1, intercept_1 = fit_simple_linear_model(x_large_noisy, y_large_noisy)
error_1 = evaluate_on_clean_test_set(slope_1, intercept_1)

# سيناريو 2: كمية بيانات أصغر بكثير، لكن ضوضاء منخفضة جدًا (يحاكي بيانات "textbook-quality")
x_small_clean, y_small_clean = generate_noisy_dataset(n_samples=200, noise_level=0.5)
slope_2, intercept_2 = fit_simple_linear_model(x_small_clean, y_small_clean)
error_2 = evaluate_on_clean_test_set(slope_2, intercept_2)

print(f"سيناريو 1 (بيانات كثيرة وصاخبة):  حجم={10_000}, خطأ النموذج={error_1:.4f}")
print(f"سيناريو 2 (بيانات قليلة ونظيفة): حجم={200}, خطأ النموذج={error_2:.4f}")
print(f"\n{'✅ البيانات النظيفة الأقل تفوّقت!' if error_2 < error_1 else '⚠️ البيانات الكثيرة تفوّقت رغم الضوضاء'}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-68-chapter-68.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
grouped_query_attention.py
تطبيق تعليمي واضح لآلية Grouped-Query Attention (GQA) من الصفر،
مقارنةً بـ Multi-Head Attention (MHA) التقليدي، لتوضيح وفورات الذاكرة.
"""
import numpy as np


def compute_kv_cache_size(
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    sequence_length: int,
    bytes_per_param: int = 2,  # FP16
) -> float:
    """
    حساب حجم KV Cache بالميغابايت.
    في GQA، عدد رؤوس K,V المُخزَّنة أقل من عدد رؤوس الاستعلام،
    مما يقلل حجم الكاش مباشرة بنفس النسبة.
    """
    # الكاش يُخزَّن فقط K وV، وليس Q (Q يُحسب لحظيًا لكل توكن جديد)
    kv_cache_elements = 2 * num_kv_heads * head_dim * sequence_length  # ×2 لـ K و V
    kv_cache_bytes = kv_cache_elements * bytes_per_param
    return kv_cache_bytes / (1024 ** 2)  # تحويل لميغابايت


# مقارنة: نموذج بـ 32 رأس استعلام، head_dim=128، سياق 8192 توكن
num_query_heads = 32
head_dim = 128
sequence_length = 8192

configurations = {
    "MHA التقليدي (32 رأس K,V منفصل)": 32,
    "GQA مع 8 مجموعات (Gemma/Llama-style)": 8,
    "GQA مع 4 مجموعات (أكثر عدوانية)": 4,
    "MQA (رأس K,V واحد مشترك)": 1,
}

print(f"{'الإعداد':<40}{'رؤوس K,V':<12}{'حجم KV Cache (MB)':<20}")
print("-" * 72)
for config_name, num_kv_heads in configurations.items():
    cache_size = compute_kv_cache_size(
        num_query_heads, num_kv_heads, head_dim, sequence_length
    )
    print(f"{config_name:<40}{num_kv_heads:<12}{cache_size:<20.1f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-68-chapter-68.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class GroupedQueryAttention:
    """
    تطبيق مُبسّط وقابل للتشغيل فعليًا لـ GQA باستخدام NumPy،
    لأغراض تعليمية (في الإنتاج الحقيقي تُستخدم PyTorch/JAX مع kernels محسّنة).
    """

    def __init__(self, num_query_heads: int, num_kv_heads: int, head_dim: int):
        assert num_query_heads % num_kv_heads == 0, \
            "عدد رؤوس الاستعلام يجب أن يقبل القسمة على عدد مجموعات K,V"

        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = num_query_heads // num_kv_heads  # كم رأس استعلام لكل مجموعة K,V

    def forward(
        self, queries: np.ndarray, keys: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        """
        queries: شكل (num_query_heads, seq_len, head_dim)
        keys, values: شكل (num_kv_heads, seq_len, head_dim)  <- أقل من عدد رؤوس الاستعلام!
        """
        seq_len = queries.shape[1]
        output = np.zeros_like(queries)

        for q_head_idx in range(self.num_query_heads):
            # كل رأس استعلام "يشارك" مجموعة K,V محددة بناءً على موقعه
            kv_group_idx = q_head_idx // self.group_size

            q = queries[q_head_idx]            # (seq_len, head_dim)
            k = keys[kv_group_idx]              # (seq_len, head_dim) — مُشتركة بين عدة رؤوس Q
            v = values[kv_group_idx]

            scores = (q @ k.T) / np.sqrt(self.head_dim)
            # قناع سببي بسيط (causal mask) لمحاكاة التوليد التتابعي
            causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
            scores[causal_mask] = -np.inf

            attn_weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn_weights /= attn_weights.sum(axis=-1, keepdims=True)

            output[q_head_idx] = attn_weights @ v

        return output


# اختبار عملي
np.random.seed(0)
num_query_heads, num_kv_heads, head_dim, seq_len = 8, 2, 16, 5

gqa = GroupedQueryAttention(num_query_heads, num_kv_heads, head_dim)
queries = np.random.randn(num_query_heads, seq_len, head_dim)
keys = np.random.randn(num_kv_heads, seq_len, head_dim)
values = np.random.randn(num_kv_heads, seq_len, head_dim)

output = gqa.forward(queries, keys, values)
print(f"\nشكل مخرجات GQA: {output.shape}")
print(f"عدد رؤوس K,V الفعلية المُخزَّنة: {num_kv_heads} بدلاً من {num_query_heads}")
print(f"نسبة توفير KV Cache: {(1 - num_kv_heads/num_query_heads) * 100:.0f}%")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-68-chapter-68.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
"""
model_selection_advisor.py
أداة مساعدة لاتخاذ قرار: متى تختار SLM ومن أي عائلة،
بناءً على قيود المشروع الفعلية.
"""
from dataclasses import dataclass
from enum import Enum


class ConstraintPriority(Enum):
    LATENCY_CRITICAL = "latency_critical"       # يجب أن يعمل محليًا/بسرعة فائقة
    MULTILINGUAL_HEAVY = "multilingual_heavy"   # دعم لغات متعددة أساسي
    CODE_SPECIALIZED = "code_specialized"       # مهمة برمجية متخصصة
    MATH_REASONING = "math_reasoning"           # استدلال رياضي/منطقي
    GENERAL_PURPOSE = "general_purpose"         # مهام عامة متنوعة
    BUDGET_EXTREME = "budget_extreme"           # ميزانية حوسبة شديدة المحدودية (IoT مثلًا)


@dataclass
class ModelRecommendation:
    family: str
    suggested_size: str
    rationale: str


MODEL_ADVISOR_TABLE: dict[ConstraintPriority, ModelRecommendation] = {
    ConstraintPriority.LATENCY_CRITICAL: ModelRecommendation(
        family="Phi-3-mini / Gemma-2B",
        suggested_size="2B-3.8B",
        rationale="أحجام مصممة خصيصًا للتشغيل على الحافة (edge) والهواتف، "
                   "مع دعم رسمي لـ ONNX Runtime وتهيئات مُحسَّنة للموبايل.",
    ),
    ConstraintPriority.MULTILINGUAL_HEAVY: ModelRecommendation(
        family="Qwen2.5",
        suggested_size="3B-7B",
        rationale="تدريب مكثف على بيانات متعددة اللغات متنوعة، "
                   "أداء موثّق أقوى في اللغات غير الإنجليزية مقارنة بمنافسيها في نفس الفئة.",
    ),
    ConstraintPriority.CODE_SPECIALIZED: ModelRecommendation(
        family="Qwen2.5-Coder أو Phi نسخة مُدرَّبة على كود",
        suggested_size="7B-14B",
        rationale="إصدارات متخصصة مُدرَّبة إضافيًا على كميات هائلة من الكود عالي الجودة.",
    ),
    ConstraintPriority.MATH_REASONING: ModelRecommendation(
        family="Phi-4 أو Qwen2.5-Math",
        suggested_size="7B-14B",
        rationale="تركيز تدريبي صريح على بيانات استدلال رياضي ومنطقي تركيبية عالية الجودة.",
    ),
    ConstraintPriority.GENERAL_PURPOSE: ModelRecommendation(
        family="Gemma 2 أو Qwen2.5",
        suggested_size="7B-9B",
        rationale="توازن جيد بين الأداء العام والتكلفة، مجتمع دعم واسع ومصادر تدريب غنية.",
    ),
    ConstraintPriority.BUDGET_EXTREME: ModelRecommendation(
        family="Qwen2.5-0.5B أو Gemma-2B مُكمَّم (quantized)",
        suggested_size="0.5B-2B",
        rationale="أصغر الأحجام المتاحة مع دعم تكميم فعّال (4-bit/8-bit) "
                   "للعمل على أجهزة IoT أو متصفحات (WebGPU).",
    ),
}


def recommend_model(priority: ConstraintPriority) -> ModelRecommendation:
    return MODEL_ADVISOR_TABLE[priority]


# مثال استخدام
recommendation = recommend_model(ConstraintPriority.LATENCY_CRITICAL)
print(f"العائلة المقترحة: {recommendation.family}")
print(f"الحجم المقترح   : {recommendation.suggested_size}")
print(f"السبب            : {recommendation.rationale}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-68-chapter-68.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
"""
specialized_vs_general_benchmark.py
محاكاة تعليمية توضح منهجية المقارنة بين SLM مضبوط دقيقًا
مقابل LLM عام ضخم، على مهمة ضيقة محددة (تصنيف فواتير).
"""
from dataclasses import dataclass


@dataclass
class ModelBenchmarkResult:
    model_name: str
    accuracy_on_narrow_task: float
    latency_ms: float
    cost_per_1000_requests_usd: float
    requires_internet: bool


def simulate_benchmark_results() -> list[ModelBenchmarkResult]:
    """
    نتائج محاكاة واقعية الشكل (مبنية على أنماط حقيقية موثقة في الأدبيات،
    وليست أرقامًا رسمية من أي شركة محددة) لتوضيح الاتجاه العام.
    """
    return [
        ModelBenchmarkResult(
            model_name="GPT-4 class (عام، عبر API)",
            accuracy_on_narrow_task=0.91,
            latency_ms=1800,
            cost_per_1000_requests_usd=15.0,
            requires_internet=True,
        ),
        ModelBenchmarkResult(
            model_name="Phi-3-mini (3.8B) + LoRA مضبوط على المهمة",
            accuracy_on_narrow_task=0.94,
            latency_ms=85,
            cost_per_1000_requests_usd=0.3,
            requires_internet=False,
        ),
        ModelBenchmarkResult(
            model_name="Phi-3-mini (3.8B) بدون ضبط دقيق (zero-shot)",
            accuracy_on_narrow_task=0.72,
            latency_ms=85,
            cost_per_1000_requests_usd=0.3,
            requires_internet=False,
        ),
    ]


def print_comparison_report(results: list[ModelBenchmarkResult]) -> None:
    print(f"{'النموذج':<45}{'الدقة':<10}{'الكمون(ms)':<12}{'التكلفة/1000':<15}{'يحتاج إنترنت'}")
    print("-" * 100)
    for r in results:
        internet_flag = "نعم" if r.requires_internet else "لا"
        print(
            f"{r.model_name:<45}{r.accuracy_on_narrow_task:<10.2f}"
            f"{r.latency_ms:<12.0f}${r.cost_per_1000_requests_usd:<14.2f}{internet_flag}"
        )


results = simulate_benchmark_results()
print_comparison_report(results)

print("\n📊 الاستنتاج الرئيسي:")
print("النموذج الصغير المضبوط دقيقًا تفوّق على النموذج الضخم العام في الدقة")
print("على هذه المهمة الضيقة تحديدًا، بينما كان أسرع بـ 21 مرة وأرخص بـ 50 مرة،")
print("وقابل للعمل محليًا بدون اتصال إنترنت.")
print("\n⚠️ تنبيه مهم: هذا لا يعني SLM 'أفضل مطلقًا' —")
print("بدون الضبط الدقيق، أداؤه انخفض بشكل حاد (من 0.94 إلى 0.72)،")
print("والنموذج الضخم العام يبقى الخيار الأفضل لمهام متنوعة غير محددة مسبقًا.")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-68-chapter-68.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
"""
invoice_classifier_slm_pipeline.py
نظام End-to-End كامل: تحضير بيانات، ضبط دقيق (LoRA) لنموذج SLM صغير
(Phi-3-mini كمثال)، وتقييم النتيجة على بيانات اختبار محجوزة.

يستخدم مكتبات PyTorch + Hugging Face transformers + peft (LoRA)،
وهي المكدس المعياري الحديث (2024-2025) لهذا النوع من المهام.

ملاحظة: تشغيل هذا الكود فعليًا يتطلب تثبيت
transformers, peft, torch, datasets, bitsandbytes
وتحميل أوزان نموذج حقيقي (مثل microsoft/Phi-3-mini-4k-instruct)،
وقد يتطلب GPU للتدريب الفعلي. البنية والمنطق أدناه كاملان وصحيحان
ويمثلان تطبيقًا إنتاجيًا حقيقيًا، وليسا محاكاة.
"""

from dataclasses import dataclass
import json


# ---------- المرحلة 1: تحضير بيانات الضبط الدقيق ----------

@dataclass
class InvoiceExample:
    raw_text: str
    category: str   # مثل: "مطاعم", "نقل", "مستلزمات مكتبية", "اشتراكات"
    amount: float
    date: str


def build_training_dataset() -> list[InvoiceExample]:
    """بناء مجموعة بيانات تدريب صغيرة ومُنسّقة (في الواقع، قد تحتاج آلاف الأمثلة)"""
    return [
        InvoiceExample(
            raw_text="مطعم السلطان - فاتورة بتاريخ 2025-03-14 بقيمة 145.50 جنيه",
            category="مطاعم", amount=145.50, date="2025-03-14",
        ),
        InvoiceExample(
            raw_text="اشتراك شهري Netflix - 2025-03-01 - 199 جنيه",
            category="اشتراكات", amount=199.0, date="2025-03-01",
        ),
        InvoiceExample(
            raw_text="أوبر - رحلة بتاريخ 15 مارس 2025 - المبلغ 68 جنيه",
            category="نقل", amount=68.0, date="2025-03-15",
        ),
        InvoiceExample(
            raw_text="مكتبة الأهرام - أدوات مكتبية - 2025-03-10 - 320 جنيه",
            category="مستلزمات مكتبية", amount=320.0, date="2025-03-10",
        ),
        # في مشروع حقيقي: مئات أو آلاف الأمثلة المتنوعة والممثّلة لتوزيع البيانات الفعلي
    ]


def format_as_instruction_pairs(examples: list[InvoiceExample]) -> list[dict]:
    """
    تحويل الأمثلة لصيغة instruction-tuning القياسية
    (نفس الصيغة المستخدمة في الفصول السابقة لـ SFT).
    """
    formatted = []
    for ex in examples:
        instruction = (
            "استخرج من نص الفاتورة التالي: الفئة، المبلغ، والتاريخ، "
            "وأرجع النتيجة بصيغة JSON فقط."
        )
        output_json = json.dumps({
            "category": ex.category, "amount": ex.amount, "date": ex.date
        }, ensure_ascii=False)

        formatted.append({
            "instruction": instruction,
            "input": ex.raw_text,
            "output": output_json,
        })
    return formatted


# ---------- المرحلة 2: إعداد الضبط الدقيق بـ LoRA ----------

def setup_lora_finetuning_config():
    """
    إعداد كامل وصحيح لضبط Phi-3-mini بـ LoRA.
    هذا الكود يمثل البنية الفعلية المستخدمة في الإنتاج
    (نفس الأدوات والمنهجية من فصل LoRA/PEFT السابق في الكتاب).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    import torch

    model_id = "microsoft/Phi-3-mini-4k-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                          # رتبة مصفوفات LoRA
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    # عادة تُظهر هذه الطباعة أن أقل من 1% من المعاملات فعليًا قابلة للتدريب

    training_args = TrainingArguments(
        output_dir="./phi3-invoice-classifier-lora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    return tokenizer, peft_model, training_args


# ---------- المرحلة 3: دالة التدريب الكاملة ----------

def run_finetuning_pipeline():
    """
    تنفيذ كامل لدورة الضبط الدقيق، من تحضير البيانات حتى حفظ النموذج.
    (يتطلب بيئة GPU فعلية للتشغيل الحقيقي)
    """
    from datasets import Dataset
    from trl import SFTTrainer

    raw_examples = build_training_dataset()
    formatted_data = format_as_instruction_pairs(raw_examples)

    def format_prompt(example: dict) -> str:
        return (
            f"<|user|>\n{example['instruction']}\n\n{example['input']}<|end|>\n"
            f"<|assistant|>\n{example['output']}<|end|>"
        )

    dataset = Dataset.from_list(
        [{"text": format_prompt(ex)} for ex in formatted_data]
    )

    tokenizer, peft_model, training_args = setup_lora_finetuning_config()

    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=512,
    )

    trainer.train()
    trainer.save_model("./phi3-invoice-classifier-final")
    print("✅ تم حفظ النموذج المضبوط دقيقًا بنجاح.")


# ---------- المرحلة 4: التقييم على بيانات اختبار محجوزة ----------

def evaluate_finetuned_model(
    model, tokenizer, test_examples: list[InvoiceExample]
) -> float:
    """حساب دقة التصنيف على بيانات اختبار لم يرها النموذج أثناء التدريب"""
    correct = 0

    for example in test_examples:
        prompt = (
            f"<|user|>\nاستخرج من نص الفاتورة التالي: الفئة، المبلغ، والتاريخ، "
            f"وأرجع النتيجة بصيغة JSON فقط.\n\n{example.raw_text}<|end|>\n<|assistant|>\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.0)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            parsed = json.loads(generated_text.split("<|assistant|>")[-1].strip())
            if parsed.get("category") == example.category:
                correct += 1
        except (json.JSONDecodeError, IndexError):
            pass  # فشل في تحليل المخرجات = تصنيف خاطئ

    return correct / len(test_examples)


print("البنية الكاملة لخط أنابيب الضبط الدقيق جاهزة.")
print("لتشغيلها فعليًا: pip install transformers peft trl datasets bitsandbytes torch")
print("ثم: run_finetuning_pipeline()")
