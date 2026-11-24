#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الرابع عشر: تحليل التعقيد الحسابي والذاكرة — فهم Transformer بلغة FLOPs والبايتات
Large Language Model Engineering — Transformer

Book slug : llm-engineering-69-chapters
Chapter   : ch-14-chapter-14 (lesson 14)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-14-chapter-14.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الرابع عشر: تحليل التعقيد الحسابي والذاكرة — فهم Transformer بلغة FLOPs والبايتات
Large Language Model Engineering — Transformer

Book slug : llm-engineering-69-chapters
Chapter   : ch-14-chapter-14 (lesson 14)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-14-chapter-14.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الرابع عشر: تحليل التعقيد الحسابي والذاكرة — فهم Transformer بلغة FLOPs والبايتات
Large Language Model Engineering — Transformer

Book slug : llm-engineering-69-chapters
Chapter   : ch-14-chapter-14 (lesson 14)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_14.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 14: الفصل الرابع عشر: تحليل التعقيد الحسابي والذاكرة — فهم Transformer بلغة FLOPs والبايتات."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_complexity() -> None:
    batch, length, hidden, layers = (2, 1024, 2048, 24)
    attention = 4 * batch * length ** 2 * hidden
    projections = 8 * batch * length * hidden ** 2
    parameters = 12 * layers * hidden ** 2
    print('attention FLOPs (approx):', f'{attention:,}')
    print('projection/MLP FLOPs (approx):', f'{projections * layers:,}')
    print('parameter count proxy:', f'{parameters:,}')
    print('FP16 weight memory (GiB):', round(parameters * 2 / 2 ** 30, 2))

if __name__ == "__main__":
    print("Chapter 14 lab: Transformer FLOPs and activation memory")
    run_complexity()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-03-transformers-core/chapter-14.md
# ────────────────────────────────────────────────────────────────────────
# flops_memory_calculator.py
"""
حاسبة شاملة لـ FLOPs والذاكرة لنماذج Transformer.
Python 3.13+
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    vocab_size: int
    seq_len: int


def count_parameters(cfg: TransformerConfig) -> int:
    """حساب دقيق لعدد المعاملات (وليس التقريب 12Ld² فقط)."""
    d, L, d_ff, V = cfg.d_model, cfg.n_layers, cfg.d_ff, cfg.vocab_size

    # QKV + Output projection لكل طبقة
    attn_params_per_layer = 4 * d * d
    # FFN: W1 (d -> d_ff) + W2 (d_ff -> d)
    ffn_params_per_layer = 2 * d * d_ff
    # LayerNorm (مُهمَل عمليًا، صغير جدًا: 2d لكل LN × 2 LN لكل طبقة)
    ln_params_per_layer = 4 * d

    per_layer = attn_params_per_layer + ffn_params_per_layer + ln_params_per_layer
    transformer_params = L * per_layer

    # التضمين (مُشترَك غالبًا مع رأس الإخراج - weight tying)
    embedding_params = V * d

    return transformer_params + embedding_params


def flops_attention_layer(cfg: TransformerConfig) -> int:
    n, d = cfg.seq_len, cfg.d_model
    qkv = 6 * n * d * d
    qk_t = 2 * n * n * d
    attn_v = 2 * n * n * d
    out_proj = 2 * n * d * d
    return qkv + qk_t + attn_v + out_proj


def flops_ffn_layer(cfg: TransformerConfig) -> int:
    n, d, d_ff = cfg.seq_len, cfg.d_model, cfg.d_ff
    return 4 * n * d * d_ff


def flops_forward_pass(cfg: TransformerConfig) -> int:
    per_layer = flops_attention_layer(cfg) + flops_ffn_layer(cfg)
    return cfg.n_layers * per_layer


def flops_training_6ND(n_params: int, n_tokens: int) -> int:
    """القاعدة الذهبية: FLOPs التدريب الكلية تقريبًا 6ND."""
    return 6 * n_params * n_tokens


def memory_training_bytes(n_params: int) -> dict[str, float]:
    """تحليل الذاكرة بالجيجابايت لتدريب Mixed Precision + Adam."""
    gb = 1024 ** 3
    weights_fp16 = 2 * n_params
    gradients_fp16 = 2 * n_params
    master_weights_fp32 = 4 * n_params
    adam_states_fp32 = 8 * n_params
    total = weights_fp16 + gradients_fp16 + master_weights_fp32 + adam_states_fp32

    return {
        "weights_fp16_GB": weights_fp16 / gb,
        "gradients_fp16_GB": gradients_fp16 / gb,
        "master_weights_fp32_GB": master_weights_fp32 / gb,
        "adam_states_fp32_GB": adam_states_fp32 / gb,
        "total_GB": total / gb,
    }


def memory_inference_bytes(n_params: int, cfg: TransformerConfig, batch: int = 1) -> dict[str, float]:
    gb = 1024 ** 3
    weights_fp16 = 2 * n_params
    # KV cache: 2 (K و V) × L × n × d × 2 بايت (FP16) × batch
    kv_cache = 2 * cfg.n_layers * cfg.seq_len * cfg.d_model * 2 * batch
    return {
        "weights_fp16_GB": weights_fp16 / gb,
        "kv_cache_GB": kv_cache / gb,
        "total_GB": (weights_fp16 + kv_cache) / gb,
    }


if __name__ == "__main__":
    # تهيئة شبيهة بـ GPT-2 Medium تقريبًا
    cfg = TransformerConfig(
        d_model=1024, n_layers=24, n_heads=16, d_ff=4096,
        vocab_size=50257, seq_len=2048,
    )

    n_params = count_parameters(cfg)
    print(f"عدد المعاملات التقريبي: {n_params:,} (≈ {n_params/1e6:.0f}M)\n")

    attn_flops = flops_attention_layer(cfg)
    ffn_flops = flops_ffn_layer(cfg)
    print(f"FLOPs لطبقة Self-Attention واحدة : {attn_flops:,}")
    print(f"FLOPs لطبقة FFN واحدة             : {ffn_flops:,}")
    print(f"النسبة FFN/Attention              : {ffn_flops/attn_flops:.2f}x\n")

    # نقطة التعادل النظرية n = 4d
    print(f"نقطة التعادل النظرية بين Attention وFFN: n = 4d = {4*cfg.d_model}")
    print(f"طول السياق الحالي n = {cfg.seq_len}  →  "
          f"{'FFN مهيمنة' if cfg.seq_len < 4*cfg.d_model else 'Attention مهيمنة'}\n")

    # القاعدة الذهبية 6ND لتدريب على 300B توكن (مشابه لنماذج صغيرة حديثة)
    n_tokens = 300_000_000_000
    training_flops = flops_training_6ND(n_params, n_tokens)
    print(f"FLOPs الكلية للتدريب على {n_tokens:,} توكن: {training_flops:.3e}\n")

    # تقدير الوقت على GPU واحد (مثال: A100 بقدرة ~312 TFLOPs بـ BF16 فعليًا)
    gpu_flops_per_sec = 312e12 * 0.4  # نفترض كفاءة فعلية 40% (MFU واقعي)
    seconds = training_flops / gpu_flops_per_sec
    days = seconds / (3600 * 24)
    print(f"الوقت التقريبي على GPU A100 واحد (MFU=40%): {days:.1f} يوم\n")

    # تحليل الذاكرة للتدريب
    mem_train = memory_training_bytes(n_params)
    print("تحليل الذاكرة للتدريب (Mixed Precision + Adam):")
    for k, v in mem_train.items():
        print(f"   {k:28s}: {v:7.2f} GB")

    print()
    mem_infer = memory_inference_bytes(n_params, cfg, batch=1)
    print("تحليل الذاكرة للاستدلال (batch=1):")
    for k, v in mem_infer.items():
        print(f"   {k:28s}: {v:7.2f} GB")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-03-transformers-core/chapter-14.md
# ────────────────────────────────────────────────────────────────────────
# ❌ استنتاج متسرّع: "الانتباه تربيعي، إذن هو عنق الزجاجة دومًا"
if seq_len_increases:
    optimize_attention_first()  # قد يكون قرارًا خاطئًا لسياقات قصيرة!

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-03-transformers-core/chapter-14.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: احسب فعليًا قبل اتخاذ قرار التحسين
if cfg.seq_len > 4 * cfg.d_model:
    prioritize_attention_optimization()
else:
    prioritize_ffn_optimization()  # أو انظر لكليهما بناءً على القياس الفعلي

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-03-transformers-core/chapter-14.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع جدًا عند التخطيط لتدريب
gpu_memory_needed = n_params * 2  # بافتراض FP16 فقط — يتجاهل المحسِّن والتدرّجات!

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-03-transformers-core/chapter-14.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح
gpu_memory_needed = n_params * 16  # تقدير شامل تقريبي (Mixed Precision + Adam)

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-03-transformers-core/chapter-14.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: افتراض كفاءة 100% عند حساب الوقت المتوقع
estimated_time = total_flops / gpu_peak_flops  # تجاهل MFU الواقعي تمامًا!

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-03-transformers-core/chapter-14.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: ضع دائمًا عامل كفاءة واقعي (MFU)
realistic_mfu = 0.35  # قيمة نموذجية واقعية، تحقّق منها تجريبيًا لعتادك
estimated_time = total_flops / (gpu_peak_flops * realistic_mfu)

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-03-transformers-core/chapter-14.md
# ────────────────────────────────────────────────────────────────────────
# mini_project_training_budget_planner.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingBudgetPlan:
    n_params: int
    n_tokens: int
    n_gpus: int
    gpu_peak_flops: float   # FLOPs/ثانية لكل GPU
    mfu: float              # كفاءة فعلية متوقعة (0.0 - 1.0)
    cost_per_gpu_hour_usd: float


def plan_training(plan: TrainingBudgetPlan) -> dict[str, float]:
    # TODO:
    # 1) احسب FLOPs الكلية بقاعدة 6ND
    # 2) احسب القدرة الكلية الفعلية للعتاد = n_gpus × gpu_peak_flops × mfu
    # 3) احسب الزمن بالثواني ثم الأيام
    # 4) احسب التكلفة الكلية بالدولار = عدد ساعات كل GPU × عدد GPUs × السعر بالساعة
    # 5) احسب الذاكرة المطلوبة لكل GPU تحت Data Parallelism البسيط (16N لكل نسخة كاملة)
    # 6) أعد كل هذه القيم في قاموس منظّم
    raise NotImplementedError("أكمل هذه الدالة")


if __name__ == "__main__":
    plan = TrainingBudgetPlan(
        n_params=7_000_000_000,
        n_tokens=1_000_000_000_000,   # 1 تريليون توكن (مشابه لنماذج 7B حديثة)
        n_gpus=256,
        gpu_peak_flops=989e12,         # H100 تقريبًا (BF16)
        mfu=0.45,
        cost_per_gpu_hour_usd=2.5,
    )
    result = plan_training(plan)
    for k, v in result.items():
        print(f"{k}: {v:,.2f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-14-chapter-14.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 14: الفصل الرابع عشر: تحليل التعقيد الحسابي والذاكرة — فهم Transformer بلغة FLOPs والبايتات."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_complexity() -> None:
    batch, length, hidden, layers = (2, 1024, 2048, 24)
    attention = 4 * batch * length ** 2 * hidden
    projections = 8 * batch * length * hidden ** 2
    parameters = 12 * layers * hidden ** 2
    print('attention FLOPs (approx):', f'{attention:,}')
    print('projection/MLP FLOPs (approx):', f'{projections * layers:,}')
    print('parameter count proxy:', f'{parameters:,}')
    print('FP16 weight memory (GiB):', round(parameters * 2 / 2 ** 30, 2))

if __name__ == "__main__":
    print("Chapter 14 lab: Transformer FLOPs and activation memory")
    run_complexity()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-14-chapter-14.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# flops_memory_calculator.py
"""
حاسبة شاملة لـ FLOPs والذاكرة لنماذج Transformer.
Python 3.13+
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    vocab_size: int
    seq_len: int


def count_parameters(cfg: TransformerConfig) -> int:
    """حساب دقيق لعدد المعاملات (وليس التقريب 12Ld² فقط)."""
    d, L, d_ff, V = cfg.d_model, cfg.n_layers, cfg.d_ff, cfg.vocab_size

    # QKV + Output projection لكل طبقة
    attn_params_per_layer = 4 * d * d
    # FFN: W1 (d -> d_ff) + W2 (d_ff -> d)
    ffn_params_per_layer = 2 * d * d_ff
    # LayerNorm (مُهمَل عمليًا، صغير جدًا: 2d لكل LN × 2 LN لكل طبقة)
    ln_params_per_layer = 4 * d

    per_layer = attn_params_per_layer + ffn_params_per_layer + ln_params_per_layer
    transformer_params = L * per_layer

    # التضمين (مُشترَك غالبًا مع رأس الإخراج - weight tying)
    embedding_params = V * d

    return transformer_params + embedding_params


def flops_attention_layer(cfg: TransformerConfig) -> int:
    n, d = cfg.seq_len, cfg.d_model
    qkv = 6 * n * d * d
    qk_t = 2 * n * n * d
    attn_v = 2 * n * n * d
    out_proj = 2 * n * d * d
    return qkv + qk_t + attn_v + out_proj


def flops_ffn_layer(cfg: TransformerConfig) -> int:
    n, d, d_ff = cfg.seq_len, cfg.d_model, cfg.d_ff
    return 4 * n * d * d_ff


def flops_forward_pass(cfg: TransformerConfig) -> int:
    per_layer = flops_attention_layer(cfg) + flops_ffn_layer(cfg)
    return cfg.n_layers * per_layer


def flops_training_6ND(n_params: int, n_tokens: int) -> int:
    """القاعدة الذهبية: FLOPs التدريب الكلية تقريبًا 6ND."""
    return 6 * n_params * n_tokens


def memory_training_bytes(n_params: int) -> dict[str, float]:
    """تحليل الذاكرة بالجيجابايت لتدريب Mixed Precision + Adam."""
    gb = 1024 ** 3
    weights_fp16 = 2 * n_params
    gradients_fp16 = 2 * n_params
    master_weights_fp32 = 4 * n_params
    adam_states_fp32 = 8 * n_params
    total = weights_fp16 + gradients_fp16 + master_weights_fp32 + adam_states_fp32

    return {
        "weights_fp16_GB": weights_fp16 / gb,
        "gradients_fp16_GB": gradients_fp16 / gb,
        "master_weights_fp32_GB": master_weights_fp32 / gb,
        "adam_states_fp32_GB": adam_states_fp32 / gb,
        "total_GB": total / gb,
    }


def memory_inference_bytes(n_params: int, cfg: TransformerConfig, batch: int = 1) -> dict[str, float]:
    gb = 1024 ** 3
    weights_fp16 = 2 * n_params
    # KV cache: 2 (K و V) × L × n × d × 2 بايت (FP16) × batch
    kv_cache = 2 * cfg.n_layers * cfg.seq_len * cfg.d_model * 2 * batch
    return {
        "weights_fp16_GB": weights_fp16 / gb,
        "kv_cache_GB": kv_cache / gb,
        "total_GB": (weights_fp16 + kv_cache) / gb,
    }


if __name__ == "__main__":
    # تهيئة شبيهة بـ GPT-2 Medium تقريبًا
    cfg = TransformerConfig(
        d_model=1024, n_layers=24, n_heads=16, d_ff=4096,
        vocab_size=50257, seq_len=2048,
    )

    n_params = count_parameters(cfg)
    print(f"عدد المعاملات التقريبي: {n_params:,} (≈ {n_params/1e6:.0f}M)\n")

    attn_flops = flops_attention_layer(cfg)
    ffn_flops = flops_ffn_layer(cfg)
    print(f"FLOPs لطبقة Self-Attention واحدة : {attn_flops:,}")
    print(f"FLOPs لطبقة FFN واحدة             : {ffn_flops:,}")
    print(f"النسبة FFN/Attention              : {ffn_flops/attn_flops:.2f}x\n")

    # نقطة التعادل النظرية n = 4d
    print(f"نقطة التعادل النظرية بين Attention وFFN: n = 4d = {4*cfg.d_model}")
    print(f"طول السياق الحالي n = {cfg.seq_len}  →  "
          f"{'FFN مهيمنة' if cfg.seq_len < 4*cfg.d_model else 'Attention مهيمنة'}\n")

    # القاعدة الذهبية 6ND لتدريب على 300B توكن (مشابه لنماذج صغيرة حديثة)
    n_tokens = 300_000_000_000
    training_flops = flops_training_6ND(n_params, n_tokens)
    print(f"FLOPs الكلية للتدريب على {n_tokens:,} توكن: {training_flops:.3e}\n")

    # تقدير الوقت على GPU واحد (مثال: A100 بقدرة ~312 TFLOPs بـ BF16 فعليًا)
    gpu_flops_per_sec = 312e12 * 0.4  # نفترض كفاءة فعلية 40% (MFU واقعي)
    seconds = training_flops / gpu_flops_per_sec
    days = seconds / (3600 * 24)
    print(f"الوقت التقريبي على GPU A100 واحد (MFU=40%): {days:.1f} يوم\n")

    # تحليل الذاكرة للتدريب
    mem_train = memory_training_bytes(n_params)
    print("تحليل الذاكرة للتدريب (Mixed Precision + Adam):")
    for k, v in mem_train.items():
        print(f"   {k:28s}: {v:7.2f} GB")

    print()
    mem_infer = memory_inference_bytes(n_params, cfg, batch=1)
    print("تحليل الذاكرة للاستدلال (batch=1):")
    for k, v in mem_infer.items():
        print(f"   {k:28s}: {v:7.2f} GB")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-14-chapter-14.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ❌ استنتاج متسرّع: "الانتباه تربيعي، إذن هو عنق الزجاجة دومًا"
if seq_len_increases:
    optimize_attention_first()  # قد يكون قرارًا خاطئًا لسياقات قصيرة!

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-14-chapter-14.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: احسب فعليًا قبل اتخاذ قرار التحسين
if cfg.seq_len > 4 * cfg.d_model:
    prioritize_attention_optimization()
else:
    prioritize_ffn_optimization()  # أو انظر لكليهما بناءً على القياس الفعلي

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-14-chapter-14.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع جدًا عند التخطيط لتدريب
gpu_memory_needed = n_params * 2  # بافتراض FP16 فقط — يتجاهل المحسِّن والتدرّجات!

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-14-chapter-14.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح
gpu_memory_needed = n_params * 16  # تقدير شامل تقريبي (Mixed Precision + Adam)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-14-chapter-14.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: افتراض كفاءة 100% عند حساب الوقت المتوقع
estimated_time = total_flops / gpu_peak_flops  # تجاهل MFU الواقعي تمامًا!

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-14-chapter-14.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: ضع دائمًا عامل كفاءة واقعي (MFU)
realistic_mfu = 0.35  # قيمة نموذجية واقعية، تحقّق منها تجريبيًا لعتادك
estimated_time = total_flops / (gpu_peak_flops * realistic_mfu)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-14-chapter-14.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# mini_project_training_budget_planner.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingBudgetPlan:
    n_params: int
    n_tokens: int
    n_gpus: int
    gpu_peak_flops: float   # FLOPs/ثانية لكل GPU
    mfu: float              # كفاءة فعلية متوقعة (0.0 - 1.0)
    cost_per_gpu_hour_usd: float


def plan_training(plan: TrainingBudgetPlan) -> dict[str, float]:
    # TODO:
    # 1) احسب FLOPs الكلية بقاعدة 6ND
    # 2) احسب القدرة الكلية الفعلية للعتاد = n_gpus × gpu_peak_flops × mfu
    # 3) احسب الزمن بالثواني ثم الأيام
    # 4) احسب التكلفة الكلية بالدولار = عدد ساعات كل GPU × عدد GPUs × السعر بالساعة
    # 5) احسب الذاكرة المطلوبة لكل GPU تحت Data Parallelism البسيط (16N لكل نسخة كاملة)
    # 6) أعد كل هذه القيم في قاموس منظّم
    raise NotImplementedError("أكمل هذه الدالة")


if __name__ == "__main__":
    plan = TrainingBudgetPlan(
        n_params=7_000_000_000,
        n_tokens=1_000_000_000_000,   # 1 تريليون توكن (مشابه لنماذج 7B حديثة)
        n_gpus=256,
        gpu_peak_flops=989e12,         # H100 تقريبًا (BF16)
        mfu=0.45,
        cost_per_gpu_hour_usd=2.5,
    )
    result = plan_training(plan)
    for k, v in result.items():
        print(f"{k}: {v:,.2f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-14-chapter-14.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الرابع عشر: تحليل التعقيد الحسابي والذاكرة — فهم Transformer بلغة FLOPs والبايتات
Large Language Model Engineering — Transformer

Book slug : llm-engineering-69-chapters
Chapter   : ch-14-chapter-14 (lesson 14)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_14.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 14: الفصل الرابع عشر: تحليل التعقيد الحسابي والذاكرة — فهم Transformer بلغة FLOPs والبايتات."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_complexity() -> None:
    batch, length, hidden, layers = (2, 1024, 2048, 24)
    attention = 4 * batch * length ** 2 * hidden
    projections = 8 * batch * length * hidden ** 2
    parameters = 12 * layers * hidden ** 2
    print('attention FLOPs (approx):', f'{attention:,}')
    print('projection/MLP FLOPs (approx):', f'{projections * layers:,}')
    print('parameter count proxy:', f'{parameters:,}')
    print('FP16 weight memory (GiB):', round(parameters * 2 / 2 ** 30, 2))

if __name__ == "__main__":
    print("Chapter 14 lab: Transformer FLOPs and activation memory")
    run_complexity()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-03-transformers-core/chapter-14.md
# ────────────────────────────────────────────────────────────────────────
# flops_memory_calculator.py
"""
حاسبة شاملة لـ FLOPs والذاكرة لنماذج Transformer.
Python 3.13+
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    vocab_size: int
    seq_len: int


def count_parameters(cfg: TransformerConfig) -> int:
    """حساب دقيق لعدد المعاملات (وليس التقريب 12Ld² فقط)."""
    d, L, d_ff, V = cfg.d_model, cfg.n_layers, cfg.d_ff, cfg.vocab_size

    # QKV + Output projection لكل طبقة
    attn_params_per_layer = 4 * d * d
    # FFN: W1 (d -> d_ff) + W2 (d_ff -> d)
    ffn_params_per_layer = 2 * d * d_ff
    # LayerNorm (مُهمَل عمليًا، صغير جدًا: 2d لكل LN × 2 LN لكل طبقة)
    ln_params_per_layer = 4 * d

    per_layer = attn_params_per_layer + ffn_params_per_layer + ln_params_per_layer
    transformer_params = L * per_layer

    # التضمين (مُشترَك غالبًا مع رأس الإخراج - weight tying)
    embedding_params = V * d

    return transformer_params + embedding_params


def flops_attention_layer(cfg: TransformerConfig) -> int:
    n, d = cfg.seq_len, cfg.d_model
    qkv = 6 * n * d * d
    qk_t = 2 * n * n * d
    attn_v = 2 * n * n * d
    out_proj = 2 * n * d * d
    return qkv + qk_t + attn_v + out_proj


def flops_ffn_layer(cfg: TransformerConfig) -> int:
    n, d, d_ff = cfg.seq_len, cfg.d_model, cfg.d_ff
    return 4 * n * d * d_ff


def flops_forward_pass(cfg: TransformerConfig) -> int:
    per_layer = flops_attention_layer(cfg) + flops_ffn_layer(cfg)
    return cfg.n_layers * per_layer


def flops_training_6ND(n_params: int, n_tokens: int) -> int:
    """القاعدة الذهبية: FLOPs التدريب الكلية تقريبًا 6ND."""
    return 6 * n_params * n_tokens


def memory_training_bytes(n_params: int) -> dict[str, float]:
    """تحليل الذاكرة بالجيجابايت لتدريب Mixed Precision + Adam."""
    gb = 1024 ** 3
    weights_fp16 = 2 * n_params
    gradients_fp16 = 2 * n_params
    master_weights_fp32 = 4 * n_params
    adam_states_fp32 = 8 * n_params
    total = weights_fp16 + gradients_fp16 + master_weights_fp32 + adam_states_fp32

    return {
        "weights_fp16_GB": weights_fp16 / gb,
        "gradients_fp16_GB": gradients_fp16 / gb,
        "master_weights_fp32_GB": master_weights_fp32 / gb,
        "adam_states_fp32_GB": adam_states_fp32 / gb,
        "total_GB": total / gb,
    }


def memory_inference_bytes(n_params: int, cfg: TransformerConfig, batch: int = 1) -> dict[str, float]:
    gb = 1024 ** 3
    weights_fp16 = 2 * n_params
    # KV cache: 2 (K و V) × L × n × d × 2 بايت (FP16) × batch
    kv_cache = 2 * cfg.n_layers * cfg.seq_len * cfg.d_model * 2 * batch
    return {
        "weights_fp16_GB": weights_fp16 / gb,
        "kv_cache_GB": kv_cache / gb,
        "total_GB": (weights_fp16 + kv_cache) / gb,
    }


if __name__ == "__main__":
    # تهيئة شبيهة بـ GPT-2 Medium تقريبًا
    cfg = TransformerConfig(
        d_model=1024, n_layers=24, n_heads=16, d_ff=4096,
        vocab_size=50257, seq_len=2048,
    )

    n_params = count_parameters(cfg)
    print(f"عدد المعاملات التقريبي: {n_params:,} (≈ {n_params/1e6:.0f}M)\n")

    attn_flops = flops_attention_layer(cfg)
    ffn_flops = flops_ffn_layer(cfg)
    print(f"FLOPs لطبقة Self-Attention واحدة : {attn_flops:,}")
    print(f"FLOPs لطبقة FFN واحدة             : {ffn_flops:,}")
    print(f"النسبة FFN/Attention              : {ffn_flops/attn_flops:.2f}x\n")

    # نقطة التعادل النظرية n = 4d
    print(f"نقطة التعادل النظرية بين Attention وFFN: n = 4d = {4*cfg.d_model}")
    print(f"طول السياق الحالي n = {cfg.seq_len}  →  "
          f"{'FFN مهيمنة' if cfg.seq_len < 4*cfg.d_model else 'Attention مهيمنة'}\n")

    # القاعدة الذهبية 6ND لتدريب على 300B توكن (مشابه لنماذج صغيرة حديثة)
    n_tokens = 300_000_000_000
    training_flops = flops_training_6ND(n_params, n_tokens)
    print(f"FLOPs الكلية للتدريب على {n_tokens:,} توكن: {training_flops:.3e}\n")

    # تقدير الوقت على GPU واحد (مثال: A100 بقدرة ~312 TFLOPs بـ BF16 فعليًا)
    gpu_flops_per_sec = 312e12 * 0.4  # نفترض كفاءة فعلية 40% (MFU واقعي)
    seconds = training_flops / gpu_flops_per_sec
    days = seconds / (3600 * 24)
    print(f"الوقت التقريبي على GPU A100 واحد (MFU=40%): {days:.1f} يوم\n")

    # تحليل الذاكرة للتدريب
    mem_train = memory_training_bytes(n_params)
    print("تحليل الذاكرة للتدريب (Mixed Precision + Adam):")
    for k, v in mem_train.items():
        print(f"   {k:28s}: {v:7.2f} GB")

    print()
    mem_infer = memory_inference_bytes(n_params, cfg, batch=1)
    print("تحليل الذاكرة للاستدلال (batch=1):")
    for k, v in mem_infer.items():
        print(f"   {k:28s}: {v:7.2f} GB")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-03-transformers-core/chapter-14.md
# ────────────────────────────────────────────────────────────────────────
# ❌ استنتاج متسرّع: "الانتباه تربيعي، إذن هو عنق الزجاجة دومًا"
if seq_len_increases:
    optimize_attention_first()  # قد يكون قرارًا خاطئًا لسياقات قصيرة!

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-03-transformers-core/chapter-14.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: احسب فعليًا قبل اتخاذ قرار التحسين
if cfg.seq_len > 4 * cfg.d_model:
    prioritize_attention_optimization()
else:
    prioritize_ffn_optimization()  # أو انظر لكليهما بناءً على القياس الفعلي

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-03-transformers-core/chapter-14.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع جدًا عند التخطيط لتدريب
gpu_memory_needed = n_params * 2  # بافتراض FP16 فقط — يتجاهل المحسِّن والتدرّجات!

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-03-transformers-core/chapter-14.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح
gpu_memory_needed = n_params * 16  # تقدير شامل تقريبي (Mixed Precision + Adam)

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-03-transformers-core/chapter-14.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: افتراض كفاءة 100% عند حساب الوقت المتوقع
estimated_time = total_flops / gpu_peak_flops  # تجاهل MFU الواقعي تمامًا!

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-03-transformers-core/chapter-14.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: ضع دائمًا عامل كفاءة واقعي (MFU)
realistic_mfu = 0.35  # قيمة نموذجية واقعية، تحقّق منها تجريبيًا لعتادك
estimated_time = total_flops / (gpu_peak_flops * realistic_mfu)

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-03-transformers-core/chapter-14.md
# ────────────────────────────────────────────────────────────────────────
# mini_project_training_budget_planner.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingBudgetPlan:
    n_params: int
    n_tokens: int
    n_gpus: int
    gpu_peak_flops: float   # FLOPs/ثانية لكل GPU
    mfu: float              # كفاءة فعلية متوقعة (0.0 - 1.0)
    cost_per_gpu_hour_usd: float


def plan_training(plan: TrainingBudgetPlan) -> dict[str, float]:
    # TODO:
    # 1) احسب FLOPs الكلية بقاعدة 6ND
    # 2) احسب القدرة الكلية الفعلية للعتاد = n_gpus × gpu_peak_flops × mfu
    # 3) احسب الزمن بالثواني ثم الأيام
    # 4) احسب التكلفة الكلية بالدولار = عدد ساعات كل GPU × عدد GPUs × السعر بالساعة
    # 5) احسب الذاكرة المطلوبة لكل GPU تحت Data Parallelism البسيط (16N لكل نسخة كاملة)
    # 6) أعد كل هذه القيم في قاموس منظّم
    raise NotImplementedError("أكمل هذه الدالة")


if __name__ == "__main__":
    plan = TrainingBudgetPlan(
        n_params=7_000_000_000,
        n_tokens=1_000_000_000_000,   # 1 تريليون توكن (مشابه لنماذج 7B حديثة)
        n_gpus=256,
        gpu_peak_flops=989e12,         # H100 تقريبًا (BF16)
        mfu=0.45,
        cost_per_gpu_hour_usd=2.5,
    )
    result = plan_training(plan)
    for k, v in result.items():
        print(f"{k}: {v:,.2f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-14-chapter-14.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 14: الفصل الرابع عشر: تحليل التعقيد الحسابي والذاكرة — فهم Transformer بلغة FLOPs والبايتات."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_complexity() -> None:
    batch, length, hidden, layers = (2, 1024, 2048, 24)
    attention = 4 * batch * length ** 2 * hidden
    projections = 8 * batch * length * hidden ** 2
    parameters = 12 * layers * hidden ** 2
    print('attention FLOPs (approx):', f'{attention:,}')
    print('projection/MLP FLOPs (approx):', f'{projections * layers:,}')
    print('parameter count proxy:', f'{parameters:,}')
    print('FP16 weight memory (GiB):', round(parameters * 2 / 2 ** 30, 2))

if __name__ == "__main__":
    print("Chapter 14 lab: Transformer FLOPs and activation memory")
    run_complexity()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-14-chapter-14.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# flops_memory_calculator.py
"""
حاسبة شاملة لـ FLOPs والذاكرة لنماذج Transformer.
Python 3.13+
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    vocab_size: int
    seq_len: int


def count_parameters(cfg: TransformerConfig) -> int:
    """حساب دقيق لعدد المعاملات (وليس التقريب 12Ld² فقط)."""
    d, L, d_ff, V = cfg.d_model, cfg.n_layers, cfg.d_ff, cfg.vocab_size

    # QKV + Output projection لكل طبقة
    attn_params_per_layer = 4 * d * d
    # FFN: W1 (d -> d_ff) + W2 (d_ff -> d)
    ffn_params_per_layer = 2 * d * d_ff
    # LayerNorm (مُهمَل عمليًا، صغير جدًا: 2d لكل LN × 2 LN لكل طبقة)
    ln_params_per_layer = 4 * d

    per_layer = attn_params_per_layer + ffn_params_per_layer + ln_params_per_layer
    transformer_params = L * per_layer

    # التضمين (مُشترَك غالبًا مع رأس الإخراج - weight tying)
    embedding_params = V * d

    return transformer_params + embedding_params


def flops_attention_layer(cfg: TransformerConfig) -> int:
    n, d = cfg.seq_len, cfg.d_model
    qkv = 6 * n * d * d
    qk_t = 2 * n * n * d
    attn_v = 2 * n * n * d
    out_proj = 2 * n * d * d
    return qkv + qk_t + attn_v + out_proj


def flops_ffn_layer(cfg: TransformerConfig) -> int:
    n, d, d_ff = cfg.seq_len, cfg.d_model, cfg.d_ff
    return 4 * n * d * d_ff


def flops_forward_pass(cfg: TransformerConfig) -> int:
    per_layer = flops_attention_layer(cfg) + flops_ffn_layer(cfg)
    return cfg.n_layers * per_layer


def flops_training_6ND(n_params: int, n_tokens: int) -> int:
    """القاعدة الذهبية: FLOPs التدريب الكلية تقريبًا 6ND."""
    return 6 * n_params * n_tokens


def memory_training_bytes(n_params: int) -> dict[str, float]:
    """تحليل الذاكرة بالجيجابايت لتدريب Mixed Precision + Adam."""
    gb = 1024 ** 3
    weights_fp16 = 2 * n_params
    gradients_fp16 = 2 * n_params
    master_weights_fp32 = 4 * n_params
    adam_states_fp32 = 8 * n_params
    total = weights_fp16 + gradients_fp16 + master_weights_fp32 + adam_states_fp32

    return {
        "weights_fp16_GB": weights_fp16 / gb,
        "gradients_fp16_GB": gradients_fp16 / gb,
        "master_weights_fp32_GB": master_weights_fp32 / gb,
        "adam_states_fp32_GB": adam_states_fp32 / gb,
        "total_GB": total / gb,
    }


def memory_inference_bytes(n_params: int, cfg: TransformerConfig, batch: int = 1) -> dict[str, float]:
    gb = 1024 ** 3
    weights_fp16 = 2 * n_params
    # KV cache: 2 (K و V) × L × n × d × 2 بايت (FP16) × batch
    kv_cache = 2 * cfg.n_layers * cfg.seq_len * cfg.d_model * 2 * batch
    return {
        "weights_fp16_GB": weights_fp16 / gb,
        "kv_cache_GB": kv_cache / gb,
        "total_GB": (weights_fp16 + kv_cache) / gb,
    }


if __name__ == "__main__":
    # تهيئة شبيهة بـ GPT-2 Medium تقريبًا
    cfg = TransformerConfig(
        d_model=1024, n_layers=24, n_heads=16, d_ff=4096,
        vocab_size=50257, seq_len=2048,
    )

    n_params = count_parameters(cfg)
    print(f"عدد المعاملات التقريبي: {n_params:,} (≈ {n_params/1e6:.0f}M)\n")

    attn_flops = flops_attention_layer(cfg)
    ffn_flops = flops_ffn_layer(cfg)
    print(f"FLOPs لطبقة Self-Attention واحدة : {attn_flops:,}")
    print(f"FLOPs لطبقة FFN واحدة             : {ffn_flops:,}")
    print(f"النسبة FFN/Attention              : {ffn_flops/attn_flops:.2f}x\n")

    # نقطة التعادل النظرية n = 4d
    print(f"نقطة التعادل النظرية بين Attention وFFN: n = 4d = {4*cfg.d_model}")
    print(f"طول السياق الحالي n = {cfg.seq_len}  →  "
          f"{'FFN مهيمنة' if cfg.seq_len < 4*cfg.d_model else 'Attention مهيمنة'}\n")

    # القاعدة الذهبية 6ND لتدريب على 300B توكن (مشابه لنماذج صغيرة حديثة)
    n_tokens = 300_000_000_000
    training_flops = flops_training_6ND(n_params, n_tokens)
    print(f"FLOPs الكلية للتدريب على {n_tokens:,} توكن: {training_flops:.3e}\n")

    # تقدير الوقت على GPU واحد (مثال: A100 بقدرة ~312 TFLOPs بـ BF16 فعليًا)
    gpu_flops_per_sec = 312e12 * 0.4  # نفترض كفاءة فعلية 40% (MFU واقعي)
    seconds = training_flops / gpu_flops_per_sec
    days = seconds / (3600 * 24)
    print(f"الوقت التقريبي على GPU A100 واحد (MFU=40%): {days:.1f} يوم\n")

    # تحليل الذاكرة للتدريب
    mem_train = memory_training_bytes(n_params)
    print("تحليل الذاكرة للتدريب (Mixed Precision + Adam):")
    for k, v in mem_train.items():
        print(f"   {k:28s}: {v:7.2f} GB")

    print()
    mem_infer = memory_inference_bytes(n_params, cfg, batch=1)
    print("تحليل الذاكرة للاستدلال (batch=1):")
    for k, v in mem_infer.items():
        print(f"   {k:28s}: {v:7.2f} GB")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-14-chapter-14.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ❌ استنتاج متسرّع: "الانتباه تربيعي، إذن هو عنق الزجاجة دومًا"
if seq_len_increases:
    optimize_attention_first()  # قد يكون قرارًا خاطئًا لسياقات قصيرة!

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-14-chapter-14.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: احسب فعليًا قبل اتخاذ قرار التحسين
if cfg.seq_len > 4 * cfg.d_model:
    prioritize_attention_optimization()
else:
    prioritize_ffn_optimization()  # أو انظر لكليهما بناءً على القياس الفعلي

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-14-chapter-14.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع جدًا عند التخطيط لتدريب
gpu_memory_needed = n_params * 2  # بافتراض FP16 فقط — يتجاهل المحسِّن والتدرّجات!

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-14-chapter-14.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح
gpu_memory_needed = n_params * 16  # تقدير شامل تقريبي (Mixed Precision + Adam)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-14-chapter-14.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: افتراض كفاءة 100% عند حساب الوقت المتوقع
estimated_time = total_flops / gpu_peak_flops  # تجاهل MFU الواقعي تمامًا!

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-14-chapter-14.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: ضع دائمًا عامل كفاءة واقعي (MFU)
realistic_mfu = 0.35  # قيمة نموذجية واقعية، تحقّق منها تجريبيًا لعتادك
estimated_time = total_flops / (gpu_peak_flops * realistic_mfu)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-14-chapter-14.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# mini_project_training_budget_planner.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingBudgetPlan:
    n_params: int
    n_tokens: int
    n_gpus: int
    gpu_peak_flops: float   # FLOPs/ثانية لكل GPU
    mfu: float              # كفاءة فعلية متوقعة (0.0 - 1.0)
    cost_per_gpu_hour_usd: float


def plan_training(plan: TrainingBudgetPlan) -> dict[str, float]:
    # TODO:
    # 1) احسب FLOPs الكلية بقاعدة 6ND
    # 2) احسب القدرة الكلية الفعلية للعتاد = n_gpus × gpu_peak_flops × mfu
    # 3) احسب الزمن بالثواني ثم الأيام
    # 4) احسب التكلفة الكلية بالدولار = عدد ساعات كل GPU × عدد GPUs × السعر بالساعة
    # 5) احسب الذاكرة المطلوبة لكل GPU تحت Data Parallelism البسيط (16N لكل نسخة كاملة)
    # 6) أعد كل هذه القيم في قاموس منظّم
    raise NotImplementedError("أكمل هذه الدالة")


if __name__ == "__main__":
    plan = TrainingBudgetPlan(
        n_params=7_000_000_000,
        n_tokens=1_000_000_000_000,   # 1 تريليون توكن (مشابه لنماذج 7B حديثة)
        n_gpus=256,
        gpu_peak_flops=989e12,         # H100 تقريبًا (BF16)
        mfu=0.45,
        cost_per_gpu_hour_usd=2.5,
    )
    result = plan_training(plan)
    for k, v in result.items():
        print(f"{k}: {v:,.2f}")
