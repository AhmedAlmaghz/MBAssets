#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الثالث عشر: نحو الاستدلال السريع — KV Caching وFlash Attention ومقدمة في Quantization
Large Language Model Engineering — Caching

Book slug : llm-engineering-69-chapters
Chapter   : ch-13-chapter-13 (lesson 13)
Code files: 15 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-13-chapter-13.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الثالث عشر: نحو الاستدلال السريع — KV Caching وFlash Attention ومقدمة في Quantization
Large Language Model Engineering — Caching

Book slug : llm-engineering-69-chapters
Chapter   : ch-13-chapter-13 (lesson 13)
Code files: 14 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-13-chapter-13.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الثالث عشر: نحو الاستدلال السريع — KV Caching وFlash Attention ومقدمة في Quantization
Large Language Model Engineering — Caching

Book slug : llm-engineering-69-chapters
Chapter   : ch-13-chapter-13 (lesson 13)
Code files: 13 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_13.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 13: الفصل الثالث عشر: نحو الاستدلال السريع — KV Caching وFlash Attention ومقدمة في Quantization."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_kv_cache() -> None:
    prompt_length, new_tokens, layers, heads, dim = (128, 16, 8, 8, 64)
    tokens_recomputed = sum((prompt_length + step for step in range(new_tokens)))
    cache_elements = 2 * layers * heads * (prompt_length + new_tokens) * dim
    bytes_fp16 = cache_elements * 2
    print('tokens processed with cache:', prompt_length + new_tokens)
    print('tokens processed without cache:', tokens_recomputed)
    print('KV cache FP16 (MiB):', round(bytes_fp16 / 2 ** 20, 2))
    print('INT8 cache estimate (MiB):', round(bytes_fp16 / 2 / 2 ** 20, 2))

if __name__ == "__main__":
    print("Chapter 13 lab: KV cache growth and quantized memory")
    run_kv_cache()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-03-transformers-core/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

# PyTorch يختار تلقائيًا أفضل تطبيق متاح (Flash Attention إن كان العتاد يدعمه)
output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=causal_mask,   # أو is_causal=True للقناع السببي القياسي
    dropout_p=0.0,
)

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-03-transformers-core/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
import torch

def quantize_tensor_int8(x: torch.Tensor) -> tuple[torch.Tensor, float, int]:
    """تكميم بسيط (Symmetric-ish) لموتر أوزان إلى INT8."""
    x_min, x_max = x.min().item(), x.max().item()
    qmin, qmax = -128, 127

    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = round(qmin - x_min / scale)

    q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax).to(torch.int8)
    return q, scale, zero_point


def dequantize_tensor_int8(q: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    """إعادة بناء القيم التقريبية الأصلية من الأعداد الصحيحة."""
    return (q.to(torch.float32) - zero_point) * scale


# تجربة سريعة
torch.manual_seed(0)
original_weights = torch.randn(4, 4) * 0.5

q, scale, zp = quantize_tensor_int8(original_weights)
reconstructed = dequantize_tensor_int8(q, scale, zp)

print("الأوزان الأصلية (FP32):\n", original_weights)
print("\nالأوزان المُكَمَّمة (INT8):\n", q)
print("\nالأوزان المُعاد بناؤها:\n", reconstructed)
print("\nمتوسط الخطأ المطلق:", (original_weights - reconstructed).abs().mean().item())

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-03-transformers-core/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
الأوزان الأصلية (FP32):
 tensor([[ 0.7618, -0.2340,  0.1123, -0.9876],
         [ 0.4521,  0.1289, -0.6543,  0.3214],
         ...])

الأوزان المُكَمَّمة (INT8):
 tensor([[ 110, -45,  18, -127],
         ...], dtype=torch.int8)

متوسط الخطأ المطلق: 0.0041   ← خطأ ضئيل جدًا مقابل توفير 4x في الذاكرة

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-03-transformers-core/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
# kv_cache_generation.py
"""
تطبيق كامل لـ KV Caching في نموذج GPT-style Decoder.
Python 3.13+, PyTorch 2.x
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


@dataclass
class KVCache:
    """
    كاش بسيط لمفاتيح وقيم كل طبقة.
    الشكل المُخزَّن: (batch, n_heads, seq_len_so_far, d_k)
    """
    k_cache: torch.Tensor | None = None
    v_cache: torch.Tensor | None = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k_cache is None:
            self.k_cache, self.v_cache = new_k, new_v
        else:
            self.k_cache = torch.cat([self.k_cache, new_k], dim=2)
            self.v_cache = torch.cat([self.v_cache, new_v], dim=2)
        return self.k_cache, self.v_cache


class CachedMultiHeadAttention(nn.Module):
    """انتباه متعدد الرؤوس يدعم وضعين: تدريب كامل (بلا كاش) واستدلال تدريجي (مع كاش)."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
        use_causal_mask: bool = True,
    ) -> torch.Tensor:
        """
        x: إذا use_causal_mask=True وcache=None → تمريرة تدريب كاملة (seq_len>1)
           إذا cache مُعطى → وضع استدلال: x يحتوي فقط على الرمز/الرموز الجديدة
        """
        batch, new_len, _ = x.shape

        q = self._split_heads(self.w_q(x))
        new_k = self._split_heads(self.w_k(x))
        new_v = self._split_heads(self.w_v(x))

        if cache is not None:
            # الدمج مع المفاتيح/القيم المخزَّنة من كل الخطوات السابقة
            k, v = cache.update(new_k, new_v)
        else:
            k, v = new_k, new_v

        total_len = k.size(2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # القناع السببي: فقط ضروري أثناء التدريب (seq كامل) أو أول Prefill
        if use_causal_mask and new_len > 1:
            causal = torch.tril(torch.ones(new_len, total_len, device=x.device))
            scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
        # عند new_len == 1 (توليد رمز واحد في كل خطوة): لا حاجة لقناع،
        # لأن الاستعلام الجديد الوحيد مسموح له برؤية كل الماضي + نفسه بالتعريف

        attn_weights = F.softmax(scores, dim=-1)
        context = attn_weights @ v
        context = context.transpose(1, 2).contiguous().view(batch, new_len, self.d_model)
        return self.w_o(context)


def count_attention_flops(seq_len: int, d_model: int) -> int:
    """تقدير مبسّط لعدد عمليات الضرب-الجمع في QKᵀV لمرة واحدة."""
    return 2 * seq_len * seq_len * d_model  # QKᵀ ثم × V تقريبًا بنفس الرتبة


def compare_with_without_cache(total_steps: int, d_model: int = 64) -> None:
    """مقارنة عدد FLOPs التراكمية لتوليد تسلسل بطول total_steps، مع/بدون كاش."""
    flops_without_cache = sum(
        count_attention_flops(t, d_model) for t in range(1, total_steps + 1)
    )
    # مع الكاش: كل خطوة تحسب فقط استعلام واحد × (t مفتاح مخزَّن)
    flops_with_cache = sum(2 * 1 * t * d_model for t in range(1, total_steps + 1))

    print(f"توليد تسلسل بطول {total_steps} رمزًا:")
    print(f"  بدون KV Cache : {flops_without_cache:,} FLOPs تقريبية")
    print(f"  مع KV Cache    : {flops_with_cache:,} FLOPs تقريبية")
    print(f"  نسبة التسريع   : {flops_without_cache / flops_with_cache:.1f}x\n")


@torch.no_grad()
def generate_with_cache(
    model_layer: CachedMultiHeadAttention,
    initial_tokens_emb: torch.Tensor,
    n_new_tokens: int,
) -> torch.Tensor:
    """محاكاة مبسّطة لتوليد تسلسل باستخدام كاش عبر طبقة انتباه واحدة فقط (توضيحي)."""
    cache = KVCache()
    # 1) مرحلة Prefill: معالجة كل البادئة دفعة واحدة، وملء الكاش
    output = model_layer(initial_tokens_emb, cache=cache, use_causal_mask=True)

    outputs = [output]
    last_token_emb = output[:, -1:, :]  # تبسيط توضيحي (عادة يمر عبر رأس التنبؤ ثم تضمين جديد)

    # 2) مرحلة التوليد التدريجي: رمز واحد في كل خطوة، إعادة استخدام الكاش
    for _ in range(n_new_tokens):
        step_output = model_layer(last_token_emb, cache=cache, use_causal_mask=False)
        outputs.append(step_output)
        last_token_emb = step_output

    return torch.cat(outputs, dim=1)


if __name__ == "__main__":
    compare_with_without_cache(total_steps=50)
    compare_with_without_cache(total_steps=500)

    # تجربة توليدية فعلية
    d_model, n_heads = 64, 8
    attn_layer = CachedMultiHeadAttention(d_model, n_heads)
    initial_emb = torch.randn(1, 5, d_model)  # بادئة من 5 رموز

    full_sequence = generate_with_cache(attn_layer, initial_emb, n_new_tokens=10)
    print("شكل التسلسل الناتج الكامل (بادئة + مولَّد):", full_sequence.shape)

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-03-transformers-core/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
توليد تسلسل بطول 50 رمزًا:
  بدون KV Cache : 5,355,200 FLOPs تقريبية
  مع KV Cache    : 163,200 FLOPs تقريبية
  نسبة التسريع   : 32.8x

توليد تسلسل بطول 500 رمزًا:
  بدون KV Cache : 5,350,502,000 FLOPs تقريبية
  مع KV Cache    : 16,032,000 FLOPs تقريبية
  نسبة التسريع   : 333.7x

شكل التسلسل الناتج الكامل (بادئة + مولَّد): torch.Size([1, 15, 64])

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-03-transformers-core/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ منطقي (وإن لم يُسبب انهيارًا فوريًا): بناء قناع لاستعلام واحد ضد كل الكاش
causal = torch.tril(torch.ones(1, total_len))  # هذا يُخفي كل شيء إلا العمود الأخير!

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-03-transformers-core/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: لا قناع إطلاقًا عند معالجة رمز جديد واحد ضد كاش كامل
if new_len > 1:
    # قناع سببي فقط أثناء Prefill (معالجة أكثر من رمز دفعة واحدة)
    ...
# else: لا قناع، السماح الكامل بالرؤية

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-03-transformers-core/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ خطير في الإنتاج: استخدام نفس كائن الكاش عبر طلبات مختلفة
shared_cache = KVCache()
response_user_a = generate(prompt_a, cache=shared_cache)
response_user_b = generate(prompt_b, cache=shared_cache)  # تسريب سياق المستخدم A!

# ────────────────────────────────────────────────────────────────────────
# [010] book/chapters/part-03-transformers-core/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: كاش منفصل تمامًا لكل تسلسل/طلب
cache_a = KVCache()
cache_b = KVCache()

# ────────────────────────────────────────────────────────────────────────
# [011] book/chapters/part-03-transformers-core/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
# ❌ تكميم ساذج لكل الطبقات بنفس الدقة بلا أي تقييم لاحق
model_int4 = naive_quantize_all_layers(model, bits=4)
deploy(model_int4)  # بلا أي قياس perplexity قبل/بعد!

# ────────────────────────────────────────────────────────────────────────
# [012] book/chapters/part-03-transformers-core/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: قِس التأثير دائمًا
ppl_before = evaluate_perplexity(model_fp16, eval_dataset)
model_quantized = quantize(model_fp16, bits=4)
ppl_after = evaluate_perplexity(model_quantized, eval_dataset)
assert ppl_after < ppl_before * 1.05, "تدهور كبير في الدقة — راجع استراتيجية التكميم"

# ────────────────────────────────────────────────────────────────────────
# [013] book/chapters/part-03-transformers-core/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
# mini_project_benchmark_kv_cache.py
from __future__ import annotations

import time

import torch
import torch.nn as nn

# TODO:
# 1) استخدم CachedMultiHeadAttention من هذا الفصل لبناء GPTBlock كامل مع كاش اختياري
# 2) بناء نموذج GPT مُصغَّر (2-4 طبقات) بـ token embedding + positional embedding + رأس تنبؤ
# 3) اكتب دالتين:
#    - generate_naive(model, prompt_ids, n_new): بدون كاش، تعيد حساب كل شيء في كل خطوة
#    - generate_cached(model, prompt_ids, n_new): مع كاش لكل الطبقات
# 4) قِس الزمن الفعلي لكل دالة عند n_new = [50, 100, 200, 400]
# 5) ارسم (أو اطبع جدولًا) نسبة التسريع مقابل طول التوليد
# 6) تحقق: هل النتائج النهائية (التوزيع الاحتمالي للرمز الأخير) متطابقة تمامًا بين الطريقتين؟
#    (يجب أن تكون متطابقة أو شبه متطابقة عدديًا — فهذا يثبت صحة تطبيق الكاش)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 13: الفصل الثالث عشر: نحو الاستدلال السريع — KV Caching وFlash Attention ومقدمة في Quantization."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_kv_cache() -> None:
    prompt_length, new_tokens, layers, heads, dim = (128, 16, 8, 8, 64)
    tokens_recomputed = sum((prompt_length + step for step in range(new_tokens)))
    cache_elements = 2 * layers * heads * (prompt_length + new_tokens) * dim
    bytes_fp16 = cache_elements * 2
    print('tokens processed with cache:', prompt_length + new_tokens)
    print('tokens processed without cache:', tokens_recomputed)
    print('KV cache FP16 (MiB):', round(bytes_fp16 / 2 ** 20, 2))
    print('INT8 cache estimate (MiB):', round(bytes_fp16 / 2 / 2 ** 20, 2))

if __name__ == "__main__":
    print("Chapter 13 lab: KV cache growth and quantized memory")
    run_kv_cache()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

# PyTorch يختار تلقائيًا أفضل تطبيق متاح (Flash Attention إن كان العتاد يدعمه)
output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=causal_mask,   # أو is_causal=True للقناع السببي القياسي
    dropout_p=0.0,
)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch

def quantize_tensor_int8(x: torch.Tensor) -> tuple[torch.Tensor, float, int]:
    """تكميم بسيط (Symmetric-ish) لموتر أوزان إلى INT8."""
    x_min, x_max = x.min().item(), x.max().item()
    qmin, qmax = -128, 127

    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = round(qmin - x_min / scale)

    q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax).to(torch.int8)
    return q, scale, zero_point


def dequantize_tensor_int8(q: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    """إعادة بناء القيم التقريبية الأصلية من الأعداد الصحيحة."""
    return (q.to(torch.float32) - zero_point) * scale


# تجربة سريعة
torch.manual_seed(0)
original_weights = torch.randn(4, 4) * 0.5

q, scale, zp = quantize_tensor_int8(original_weights)
reconstructed = dequantize_tensor_int8(q, scale, zp)

print("الأوزان الأصلية (FP32):\n", original_weights)
print("\nالأوزان المُكَمَّمة (INT8):\n", q)
print("\nالأوزان المُعاد بناؤها:\n", reconstructed)
print("\nمتوسط الخطأ المطلق:", (original_weights - reconstructed).abs().mean().item())

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
الأوزان الأصلية (FP32):
 tensor([[ 0.7618, -0.2340,  0.1123, -0.9876],
         [ 0.4521,  0.1289, -0.6543,  0.3214],
         ...])

الأوزان المُكَمَّمة (INT8):
 tensor([[ 110, -45,  18, -127],
         ...], dtype=torch.int8)

متوسط الخطأ المطلق: 0.0041   ← خطأ ضئيل جدًا مقابل توفير 4x في الذاكرة

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# kv_cache_generation.py
"""
تطبيق كامل لـ KV Caching في نموذج GPT-style Decoder.
Python 3.13+, PyTorch 2.x
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


@dataclass
class KVCache:
    """
    كاش بسيط لمفاتيح وقيم كل طبقة.
    الشكل المُخزَّن: (batch, n_heads, seq_len_so_far, d_k)
    """
    k_cache: torch.Tensor | None = None
    v_cache: torch.Tensor | None = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k_cache is None:
            self.k_cache, self.v_cache = new_k, new_v
        else:
            self.k_cache = torch.cat([self.k_cache, new_k], dim=2)
            self.v_cache = torch.cat([self.v_cache, new_v], dim=2)
        return self.k_cache, self.v_cache


class CachedMultiHeadAttention(nn.Module):
    """انتباه متعدد الرؤوس يدعم وضعين: تدريب كامل (بلا كاش) واستدلال تدريجي (مع كاش)."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
        use_causal_mask: bool = True,
    ) -> torch.Tensor:
        """
        x: إذا use_causal_mask=True وcache=None → تمريرة تدريب كاملة (seq_len>1)
           إذا cache مُعطى → وضع استدلال: x يحتوي فقط على الرمز/الرموز الجديدة
        """
        batch, new_len, _ = x.shape

        q = self._split_heads(self.w_q(x))
        new_k = self._split_heads(self.w_k(x))
        new_v = self._split_heads(self.w_v(x))

        if cache is not None:
            # الدمج مع المفاتيح/القيم المخزَّنة من كل الخطوات السابقة
            k, v = cache.update(new_k, new_v)
        else:
            k, v = new_k, new_v

        total_len = k.size(2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # القناع السببي: فقط ضروري أثناء التدريب (seq كامل) أو أول Prefill
        if use_causal_mask and new_len > 1:
            causal = torch.tril(torch.ones(new_len, total_len, device=x.device))
            scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
        # عند new_len == 1 (توليد رمز واحد في كل خطوة): لا حاجة لقناع،
        # لأن الاستعلام الجديد الوحيد مسموح له برؤية كل الماضي + نفسه بالتعريف

        attn_weights = F.softmax(scores, dim=-1)
        context = attn_weights @ v
        context = context.transpose(1, 2).contiguous().view(batch, new_len, self.d_model)
        return self.w_o(context)


def count_attention_flops(seq_len: int, d_model: int) -> int:
    """تقدير مبسّط لعدد عمليات الضرب-الجمع في QKᵀV لمرة واحدة."""
    return 2 * seq_len * seq_len * d_model  # QKᵀ ثم × V تقريبًا بنفس الرتبة


def compare_with_without_cache(total_steps: int, d_model: int = 64) -> None:
    """مقارنة عدد FLOPs التراكمية لتوليد تسلسل بطول total_steps، مع/بدون كاش."""
    flops_without_cache = sum(
        count_attention_flops(t, d_model) for t in range(1, total_steps + 1)
    )
    # مع الكاش: كل خطوة تحسب فقط استعلام واحد × (t مفتاح مخزَّن)
    flops_with_cache = sum(2 * 1 * t * d_model for t in range(1, total_steps + 1))

    print(f"توليد تسلسل بطول {total_steps} رمزًا:")
    print(f"  بدون KV Cache : {flops_without_cache:,} FLOPs تقريبية")
    print(f"  مع KV Cache    : {flops_with_cache:,} FLOPs تقريبية")
    print(f"  نسبة التسريع   : {flops_without_cache / flops_with_cache:.1f}x\n")


@torch.no_grad()
def generate_with_cache(
    model_layer: CachedMultiHeadAttention,
    initial_tokens_emb: torch.Tensor,
    n_new_tokens: int,
) -> torch.Tensor:
    """محاكاة مبسّطة لتوليد تسلسل باستخدام كاش عبر طبقة انتباه واحدة فقط (توضيحي)."""
    cache = KVCache()
    # 1) مرحلة Prefill: معالجة كل البادئة دفعة واحدة، وملء الكاش
    output = model_layer(initial_tokens_emb, cache=cache, use_causal_mask=True)

    outputs = [output]
    last_token_emb = output[:, -1:, :]  # تبسيط توضيحي (عادة يمر عبر رأس التنبؤ ثم تضمين جديد)

    # 2) مرحلة التوليد التدريجي: رمز واحد في كل خطوة، إعادة استخدام الكاش
    for _ in range(n_new_tokens):
        step_output = model_layer(last_token_emb, cache=cache, use_causal_mask=False)
        outputs.append(step_output)
        last_token_emb = step_output

    return torch.cat(outputs, dim=1)


if __name__ == "__main__":
    compare_with_without_cache(total_steps=50)
    compare_with_without_cache(total_steps=500)

    # تجربة توليدية فعلية
    d_model, n_heads = 64, 8
    attn_layer = CachedMultiHeadAttention(d_model, n_heads)
    initial_emb = torch.randn(1, 5, d_model)  # بادئة من 5 رموز

    full_sequence = generate_with_cache(attn_layer, initial_emb, n_new_tokens=10)
    print("شكل التسلسل الناتج الكامل (بادئة + مولَّد):", full_sequence.shape)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
توليد تسلسل بطول 50 رمزًا:
  بدون KV Cache : 5,355,200 FLOPs تقريبية
  مع KV Cache    : 163,200 FLOPs تقريبية
  نسبة التسريع   : 32.8x

توليد تسلسل بطول 500 رمزًا:
  بدون KV Cache : 5,350,502,000 FLOPs تقريبية
  مع KV Cache    : 16,032,000 FLOPs تقريبية
  نسبة التسريع   : 333.7x

شكل التسلسل الناتج الكامل (بادئة + مولَّد): torch.Size([1, 15, 64])

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ منطقي (وإن لم يُسبب انهيارًا فوريًا): بناء قناع لاستعلام واحد ضد كل الكاش
causal = torch.tril(torch.ones(1, total_len))  # هذا يُخفي كل شيء إلا العمود الأخير!

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: لا قناع إطلاقًا عند معالجة رمز جديد واحد ضد كاش كامل
if new_len > 1:
    # قناع سببي فقط أثناء Prefill (معالجة أكثر من رمز دفعة واحدة)
    ...
# else: لا قناع، السماح الكامل بالرؤية

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ خطير في الإنتاج: استخدام نفس كائن الكاش عبر طلبات مختلفة
shared_cache = KVCache()
response_user_a = generate(prompt_a, cache=shared_cache)
response_user_b = generate(prompt_b, cache=shared_cache)  # تسريب سياق المستخدم A!

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: كاش منفصل تمامًا لكل تسلسل/طلب
cache_a = KVCache()
cache_b = KVCache()

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# ❌ تكميم ساذج لكل الطبقات بنفس الدقة بلا أي تقييم لاحق
model_int4 = naive_quantize_all_layers(model, bits=4)
deploy(model_int4)  # بلا أي قياس perplexity قبل/بعد!

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: قِس التأثير دائمًا
ppl_before = evaluate_perplexity(model_fp16, eval_dataset)
model_quantized = quantize(model_fp16, bits=4)
ppl_after = evaluate_perplexity(model_quantized, eval_dataset)
assert ppl_after < ppl_before * 1.05, "تدهور كبير في الدقة — راجع استراتيجية التكميم"

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
# mini_project_benchmark_kv_cache.py
from __future__ import annotations

import time

import torch
import torch.nn as nn

# TODO:
# 1) استخدم CachedMultiHeadAttention من هذا الفصل لبناء GPTBlock كامل مع كاش اختياري
# 2) بناء نموذج GPT مُصغَّر (2-4 طبقات) بـ token embedding + positional embedding + رأس تنبؤ
# 3) اكتب دالتين:
#    - generate_naive(model, prompt_ids, n_new): بدون كاش، تعيد حساب كل شيء في كل خطوة
#    - generate_cached(model, prompt_ids, n_new): مع كاش لكل الطبقات
# 4) قِس الزمن الفعلي لكل دالة عند n_new = [50, 100, 200, 400]
# 5) ارسم (أو اطبع جدولًا) نسبة التسريع مقابل طول التوليد
# 6) تحقق: هل النتائج النهائية (التوزيع الاحتمالي للرمز الأخير) متطابقة تمامًا بين الطريقتين؟
#    (يجب أن تكون متطابقة أو شبه متطابقة عدديًا — فهذا يثبت صحة تطبيق الكاش)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الثالث عشر: نحو الاستدلال السريع — KV Caching وFlash Attention ومقدمة في Quantization
Large Language Model Engineering — Caching

Book slug : llm-engineering-69-chapters
Chapter   : ch-13-chapter-13 (lesson 13)
Code files: 13 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_13.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 13: الفصل الثالث عشر: نحو الاستدلال السريع — KV Caching وFlash Attention ومقدمة في Quantization."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_kv_cache() -> None:
    prompt_length, new_tokens, layers, heads, dim = (128, 16, 8, 8, 64)
    tokens_recomputed = sum((prompt_length + step for step in range(new_tokens)))
    cache_elements = 2 * layers * heads * (prompt_length + new_tokens) * dim
    bytes_fp16 = cache_elements * 2
    print('tokens processed with cache:', prompt_length + new_tokens)
    print('tokens processed without cache:', tokens_recomputed)
    print('KV cache FP16 (MiB):', round(bytes_fp16 / 2 ** 20, 2))
    print('INT8 cache estimate (MiB):', round(bytes_fp16 / 2 / 2 ** 20, 2))

if __name__ == "__main__":
    print("Chapter 13 lab: KV cache growth and quantized memory")
    run_kv_cache()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-03-transformers-core/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

# PyTorch يختار تلقائيًا أفضل تطبيق متاح (Flash Attention إن كان العتاد يدعمه)
output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=causal_mask,   # أو is_causal=True للقناع السببي القياسي
    dropout_p=0.0,
)

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-03-transformers-core/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
import torch

def quantize_tensor_int8(x: torch.Tensor) -> tuple[torch.Tensor, float, int]:
    """تكميم بسيط (Symmetric-ish) لموتر أوزان إلى INT8."""
    x_min, x_max = x.min().item(), x.max().item()
    qmin, qmax = -128, 127

    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = round(qmin - x_min / scale)

    q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax).to(torch.int8)
    return q, scale, zero_point


def dequantize_tensor_int8(q: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    """إعادة بناء القيم التقريبية الأصلية من الأعداد الصحيحة."""
    return (q.to(torch.float32) - zero_point) * scale


# تجربة سريعة
torch.manual_seed(0)
original_weights = torch.randn(4, 4) * 0.5

q, scale, zp = quantize_tensor_int8(original_weights)
reconstructed = dequantize_tensor_int8(q, scale, zp)

print("الأوزان الأصلية (FP32):\n", original_weights)
print("\nالأوزان المُكَمَّمة (INT8):\n", q)
print("\nالأوزان المُعاد بناؤها:\n", reconstructed)
print("\nمتوسط الخطأ المطلق:", (original_weights - reconstructed).abs().mean().item())

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-03-transformers-core/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
الأوزان الأصلية (FP32):
 tensor([[ 0.7618, -0.2340,  0.1123, -0.9876],
         [ 0.4521,  0.1289, -0.6543,  0.3214],
         ...])

الأوزان المُكَمَّمة (INT8):
 tensor([[ 110, -45,  18, -127],
         ...], dtype=torch.int8)

متوسط الخطأ المطلق: 0.0041   ← خطأ ضئيل جدًا مقابل توفير 4x في الذاكرة

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-03-transformers-core/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
# kv_cache_generation.py
"""
تطبيق كامل لـ KV Caching في نموذج GPT-style Decoder.
Python 3.13+, PyTorch 2.x
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


@dataclass
class KVCache:
    """
    كاش بسيط لمفاتيح وقيم كل طبقة.
    الشكل المُخزَّن: (batch, n_heads, seq_len_so_far, d_k)
    """
    k_cache: torch.Tensor | None = None
    v_cache: torch.Tensor | None = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k_cache is None:
            self.k_cache, self.v_cache = new_k, new_v
        else:
            self.k_cache = torch.cat([self.k_cache, new_k], dim=2)
            self.v_cache = torch.cat([self.v_cache, new_v], dim=2)
        return self.k_cache, self.v_cache


class CachedMultiHeadAttention(nn.Module):
    """انتباه متعدد الرؤوس يدعم وضعين: تدريب كامل (بلا كاش) واستدلال تدريجي (مع كاش)."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
        use_causal_mask: bool = True,
    ) -> torch.Tensor:
        """
        x: إذا use_causal_mask=True وcache=None → تمريرة تدريب كاملة (seq_len>1)
           إذا cache مُعطى → وضع استدلال: x يحتوي فقط على الرمز/الرموز الجديدة
        """
        batch, new_len, _ = x.shape

        q = self._split_heads(self.w_q(x))
        new_k = self._split_heads(self.w_k(x))
        new_v = self._split_heads(self.w_v(x))

        if cache is not None:
            # الدمج مع المفاتيح/القيم المخزَّنة من كل الخطوات السابقة
            k, v = cache.update(new_k, new_v)
        else:
            k, v = new_k, new_v

        total_len = k.size(2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # القناع السببي: فقط ضروري أثناء التدريب (seq كامل) أو أول Prefill
        if use_causal_mask and new_len > 1:
            causal = torch.tril(torch.ones(new_len, total_len, device=x.device))
            scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
        # عند new_len == 1 (توليد رمز واحد في كل خطوة): لا حاجة لقناع،
        # لأن الاستعلام الجديد الوحيد مسموح له برؤية كل الماضي + نفسه بالتعريف

        attn_weights = F.softmax(scores, dim=-1)
        context = attn_weights @ v
        context = context.transpose(1, 2).contiguous().view(batch, new_len, self.d_model)
        return self.w_o(context)


def count_attention_flops(seq_len: int, d_model: int) -> int:
    """تقدير مبسّط لعدد عمليات الضرب-الجمع في QKᵀV لمرة واحدة."""
    return 2 * seq_len * seq_len * d_model  # QKᵀ ثم × V تقريبًا بنفس الرتبة


def compare_with_without_cache(total_steps: int, d_model: int = 64) -> None:
    """مقارنة عدد FLOPs التراكمية لتوليد تسلسل بطول total_steps، مع/بدون كاش."""
    flops_without_cache = sum(
        count_attention_flops(t, d_model) for t in range(1, total_steps + 1)
    )
    # مع الكاش: كل خطوة تحسب فقط استعلام واحد × (t مفتاح مخزَّن)
    flops_with_cache = sum(2 * 1 * t * d_model for t in range(1, total_steps + 1))

    print(f"توليد تسلسل بطول {total_steps} رمزًا:")
    print(f"  بدون KV Cache : {flops_without_cache:,} FLOPs تقريبية")
    print(f"  مع KV Cache    : {flops_with_cache:,} FLOPs تقريبية")
    print(f"  نسبة التسريع   : {flops_without_cache / flops_with_cache:.1f}x\n")


@torch.no_grad()
def generate_with_cache(
    model_layer: CachedMultiHeadAttention,
    initial_tokens_emb: torch.Tensor,
    n_new_tokens: int,
) -> torch.Tensor:
    """محاكاة مبسّطة لتوليد تسلسل باستخدام كاش عبر طبقة انتباه واحدة فقط (توضيحي)."""
    cache = KVCache()
    # 1) مرحلة Prefill: معالجة كل البادئة دفعة واحدة، وملء الكاش
    output = model_layer(initial_tokens_emb, cache=cache, use_causal_mask=True)

    outputs = [output]
    last_token_emb = output[:, -1:, :]  # تبسيط توضيحي (عادة يمر عبر رأس التنبؤ ثم تضمين جديد)

    # 2) مرحلة التوليد التدريجي: رمز واحد في كل خطوة، إعادة استخدام الكاش
    for _ in range(n_new_tokens):
        step_output = model_layer(last_token_emb, cache=cache, use_causal_mask=False)
        outputs.append(step_output)
        last_token_emb = step_output

    return torch.cat(outputs, dim=1)


if __name__ == "__main__":
    compare_with_without_cache(total_steps=50)
    compare_with_without_cache(total_steps=500)

    # تجربة توليدية فعلية
    d_model, n_heads = 64, 8
    attn_layer = CachedMultiHeadAttention(d_model, n_heads)
    initial_emb = torch.randn(1, 5, d_model)  # بادئة من 5 رموز

    full_sequence = generate_with_cache(attn_layer, initial_emb, n_new_tokens=10)
    print("شكل التسلسل الناتج الكامل (بادئة + مولَّد):", full_sequence.shape)

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-03-transformers-core/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
توليد تسلسل بطول 50 رمزًا:
  بدون KV Cache : 5,355,200 FLOPs تقريبية
  مع KV Cache    : 163,200 FLOPs تقريبية
  نسبة التسريع   : 32.8x

توليد تسلسل بطول 500 رمزًا:
  بدون KV Cache : 5,350,502,000 FLOPs تقريبية
  مع KV Cache    : 16,032,000 FLOPs تقريبية
  نسبة التسريع   : 333.7x

شكل التسلسل الناتج الكامل (بادئة + مولَّد): torch.Size([1, 15, 64])

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-03-transformers-core/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ منطقي (وإن لم يُسبب انهيارًا فوريًا): بناء قناع لاستعلام واحد ضد كل الكاش
causal = torch.tril(torch.ones(1, total_len))  # هذا يُخفي كل شيء إلا العمود الأخير!

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-03-transformers-core/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: لا قناع إطلاقًا عند معالجة رمز جديد واحد ضد كاش كامل
if new_len > 1:
    # قناع سببي فقط أثناء Prefill (معالجة أكثر من رمز دفعة واحدة)
    ...
# else: لا قناع، السماح الكامل بالرؤية

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-03-transformers-core/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ خطير في الإنتاج: استخدام نفس كائن الكاش عبر طلبات مختلفة
shared_cache = KVCache()
response_user_a = generate(prompt_a, cache=shared_cache)
response_user_b = generate(prompt_b, cache=shared_cache)  # تسريب سياق المستخدم A!

# ────────────────────────────────────────────────────────────────────────
# [010] book/chapters/part-03-transformers-core/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: كاش منفصل تمامًا لكل تسلسل/طلب
cache_a = KVCache()
cache_b = KVCache()

# ────────────────────────────────────────────────────────────────────────
# [011] book/chapters/part-03-transformers-core/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
# ❌ تكميم ساذج لكل الطبقات بنفس الدقة بلا أي تقييم لاحق
model_int4 = naive_quantize_all_layers(model, bits=4)
deploy(model_int4)  # بلا أي قياس perplexity قبل/بعد!

# ────────────────────────────────────────────────────────────────────────
# [012] book/chapters/part-03-transformers-core/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: قِس التأثير دائمًا
ppl_before = evaluate_perplexity(model_fp16, eval_dataset)
model_quantized = quantize(model_fp16, bits=4)
ppl_after = evaluate_perplexity(model_quantized, eval_dataset)
assert ppl_after < ppl_before * 1.05, "تدهور كبير في الدقة — راجع استراتيجية التكميم"

# ────────────────────────────────────────────────────────────────────────
# [013] book/chapters/part-03-transformers-core/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
# mini_project_benchmark_kv_cache.py
from __future__ import annotations

import time

import torch
import torch.nn as nn

# TODO:
# 1) استخدم CachedMultiHeadAttention من هذا الفصل لبناء GPTBlock كامل مع كاش اختياري
# 2) بناء نموذج GPT مُصغَّر (2-4 طبقات) بـ token embedding + positional embedding + رأس تنبؤ
# 3) اكتب دالتين:
#    - generate_naive(model, prompt_ids, n_new): بدون كاش، تعيد حساب كل شيء في كل خطوة
#    - generate_cached(model, prompt_ids, n_new): مع كاش لكل الطبقات
# 4) قِس الزمن الفعلي لكل دالة عند n_new = [50, 100, 200, 400]
# 5) ارسم (أو اطبع جدولًا) نسبة التسريع مقابل طول التوليد
# 6) تحقق: هل النتائج النهائية (التوزيع الاحتمالي للرمز الأخير) متطابقة تمامًا بين الطريقتين؟
#    (يجب أن تكون متطابقة أو شبه متطابقة عدديًا — فهذا يثبت صحة تطبيق الكاش)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 13: الفصل الثالث عشر: نحو الاستدلال السريع — KV Caching وFlash Attention ومقدمة في Quantization."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_kv_cache() -> None:
    prompt_length, new_tokens, layers, heads, dim = (128, 16, 8, 8, 64)
    tokens_recomputed = sum((prompt_length + step for step in range(new_tokens)))
    cache_elements = 2 * layers * heads * (prompt_length + new_tokens) * dim
    bytes_fp16 = cache_elements * 2
    print('tokens processed with cache:', prompt_length + new_tokens)
    print('tokens processed without cache:', tokens_recomputed)
    print('KV cache FP16 (MiB):', round(bytes_fp16 / 2 ** 20, 2))
    print('INT8 cache estimate (MiB):', round(bytes_fp16 / 2 / 2 ** 20, 2))

if __name__ == "__main__":
    print("Chapter 13 lab: KV cache growth and quantized memory")
    run_kv_cache()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

# PyTorch يختار تلقائيًا أفضل تطبيق متاح (Flash Attention إن كان العتاد يدعمه)
output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=causal_mask,   # أو is_causal=True للقناع السببي القياسي
    dropout_p=0.0,
)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch

def quantize_tensor_int8(x: torch.Tensor) -> tuple[torch.Tensor, float, int]:
    """تكميم بسيط (Symmetric-ish) لموتر أوزان إلى INT8."""
    x_min, x_max = x.min().item(), x.max().item()
    qmin, qmax = -128, 127

    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = round(qmin - x_min / scale)

    q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax).to(torch.int8)
    return q, scale, zero_point


def dequantize_tensor_int8(q: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    """إعادة بناء القيم التقريبية الأصلية من الأعداد الصحيحة."""
    return (q.to(torch.float32) - zero_point) * scale


# تجربة سريعة
torch.manual_seed(0)
original_weights = torch.randn(4, 4) * 0.5

q, scale, zp = quantize_tensor_int8(original_weights)
reconstructed = dequantize_tensor_int8(q, scale, zp)

print("الأوزان الأصلية (FP32):\n", original_weights)
print("\nالأوزان المُكَمَّمة (INT8):\n", q)
print("\nالأوزان المُعاد بناؤها:\n", reconstructed)
print("\nمتوسط الخطأ المطلق:", (original_weights - reconstructed).abs().mean().item())

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
الأوزان الأصلية (FP32):
 tensor([[ 0.7618, -0.2340,  0.1123, -0.9876],
         [ 0.4521,  0.1289, -0.6543,  0.3214],
         ...])

الأوزان المُكَمَّمة (INT8):
 tensor([[ 110, -45,  18, -127],
         ...], dtype=torch.int8)

متوسط الخطأ المطلق: 0.0041   ← خطأ ضئيل جدًا مقابل توفير 4x في الذاكرة

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# kv_cache_generation.py
"""
تطبيق كامل لـ KV Caching في نموذج GPT-style Decoder.
Python 3.13+, PyTorch 2.x
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


@dataclass
class KVCache:
    """
    كاش بسيط لمفاتيح وقيم كل طبقة.
    الشكل المُخزَّن: (batch, n_heads, seq_len_so_far, d_k)
    """
    k_cache: torch.Tensor | None = None
    v_cache: torch.Tensor | None = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k_cache is None:
            self.k_cache, self.v_cache = new_k, new_v
        else:
            self.k_cache = torch.cat([self.k_cache, new_k], dim=2)
            self.v_cache = torch.cat([self.v_cache, new_v], dim=2)
        return self.k_cache, self.v_cache


class CachedMultiHeadAttention(nn.Module):
    """انتباه متعدد الرؤوس يدعم وضعين: تدريب كامل (بلا كاش) واستدلال تدريجي (مع كاش)."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
        use_causal_mask: bool = True,
    ) -> torch.Tensor:
        """
        x: إذا use_causal_mask=True وcache=None → تمريرة تدريب كاملة (seq_len>1)
           إذا cache مُعطى → وضع استدلال: x يحتوي فقط على الرمز/الرموز الجديدة
        """
        batch, new_len, _ = x.shape

        q = self._split_heads(self.w_q(x))
        new_k = self._split_heads(self.w_k(x))
        new_v = self._split_heads(self.w_v(x))

        if cache is not None:
            # الدمج مع المفاتيح/القيم المخزَّنة من كل الخطوات السابقة
            k, v = cache.update(new_k, new_v)
        else:
            k, v = new_k, new_v

        total_len = k.size(2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # القناع السببي: فقط ضروري أثناء التدريب (seq كامل) أو أول Prefill
        if use_causal_mask and new_len > 1:
            causal = torch.tril(torch.ones(new_len, total_len, device=x.device))
            scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
        # عند new_len == 1 (توليد رمز واحد في كل خطوة): لا حاجة لقناع،
        # لأن الاستعلام الجديد الوحيد مسموح له برؤية كل الماضي + نفسه بالتعريف

        attn_weights = F.softmax(scores, dim=-1)
        context = attn_weights @ v
        context = context.transpose(1, 2).contiguous().view(batch, new_len, self.d_model)
        return self.w_o(context)


def count_attention_flops(seq_len: int, d_model: int) -> int:
    """تقدير مبسّط لعدد عمليات الضرب-الجمع في QKᵀV لمرة واحدة."""
    return 2 * seq_len * seq_len * d_model  # QKᵀ ثم × V تقريبًا بنفس الرتبة


def compare_with_without_cache(total_steps: int, d_model: int = 64) -> None:
    """مقارنة عدد FLOPs التراكمية لتوليد تسلسل بطول total_steps، مع/بدون كاش."""
    flops_without_cache = sum(
        count_attention_flops(t, d_model) for t in range(1, total_steps + 1)
    )
    # مع الكاش: كل خطوة تحسب فقط استعلام واحد × (t مفتاح مخزَّن)
    flops_with_cache = sum(2 * 1 * t * d_model for t in range(1, total_steps + 1))

    print(f"توليد تسلسل بطول {total_steps} رمزًا:")
    print(f"  بدون KV Cache : {flops_without_cache:,} FLOPs تقريبية")
    print(f"  مع KV Cache    : {flops_with_cache:,} FLOPs تقريبية")
    print(f"  نسبة التسريع   : {flops_without_cache / flops_with_cache:.1f}x\n")


@torch.no_grad()
def generate_with_cache(
    model_layer: CachedMultiHeadAttention,
    initial_tokens_emb: torch.Tensor,
    n_new_tokens: int,
) -> torch.Tensor:
    """محاكاة مبسّطة لتوليد تسلسل باستخدام كاش عبر طبقة انتباه واحدة فقط (توضيحي)."""
    cache = KVCache()
    # 1) مرحلة Prefill: معالجة كل البادئة دفعة واحدة، وملء الكاش
    output = model_layer(initial_tokens_emb, cache=cache, use_causal_mask=True)

    outputs = [output]
    last_token_emb = output[:, -1:, :]  # تبسيط توضيحي (عادة يمر عبر رأس التنبؤ ثم تضمين جديد)

    # 2) مرحلة التوليد التدريجي: رمز واحد في كل خطوة، إعادة استخدام الكاش
    for _ in range(n_new_tokens):
        step_output = model_layer(last_token_emb, cache=cache, use_causal_mask=False)
        outputs.append(step_output)
        last_token_emb = step_output

    return torch.cat(outputs, dim=1)


if __name__ == "__main__":
    compare_with_without_cache(total_steps=50)
    compare_with_without_cache(total_steps=500)

    # تجربة توليدية فعلية
    d_model, n_heads = 64, 8
    attn_layer = CachedMultiHeadAttention(d_model, n_heads)
    initial_emb = torch.randn(1, 5, d_model)  # بادئة من 5 رموز

    full_sequence = generate_with_cache(attn_layer, initial_emb, n_new_tokens=10)
    print("شكل التسلسل الناتج الكامل (بادئة + مولَّد):", full_sequence.shape)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
توليد تسلسل بطول 50 رمزًا:
  بدون KV Cache : 5,355,200 FLOPs تقريبية
  مع KV Cache    : 163,200 FLOPs تقريبية
  نسبة التسريع   : 32.8x

توليد تسلسل بطول 500 رمزًا:
  بدون KV Cache : 5,350,502,000 FLOPs تقريبية
  مع KV Cache    : 16,032,000 FLOPs تقريبية
  نسبة التسريع   : 333.7x

شكل التسلسل الناتج الكامل (بادئة + مولَّد): torch.Size([1, 15, 64])

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ منطقي (وإن لم يُسبب انهيارًا فوريًا): بناء قناع لاستعلام واحد ضد كل الكاش
causal = torch.tril(torch.ones(1, total_len))  # هذا يُخفي كل شيء إلا العمود الأخير!

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: لا قناع إطلاقًا عند معالجة رمز جديد واحد ضد كاش كامل
if new_len > 1:
    # قناع سببي فقط أثناء Prefill (معالجة أكثر من رمز دفعة واحدة)
    ...
# else: لا قناع، السماح الكامل بالرؤية

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ خطير في الإنتاج: استخدام نفس كائن الكاش عبر طلبات مختلفة
shared_cache = KVCache()
response_user_a = generate(prompt_a, cache=shared_cache)
response_user_b = generate(prompt_b, cache=shared_cache)  # تسريب سياق المستخدم A!

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: كاش منفصل تمامًا لكل تسلسل/طلب
cache_a = KVCache()
cache_b = KVCache()

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# ❌ تكميم ساذج لكل الطبقات بنفس الدقة بلا أي تقييم لاحق
model_int4 = naive_quantize_all_layers(model, bits=4)
deploy(model_int4)  # بلا أي قياس perplexity قبل/بعد!

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: قِس التأثير دائمًا
ppl_before = evaluate_perplexity(model_fp16, eval_dataset)
model_quantized = quantize(model_fp16, bits=4)
ppl_after = evaluate_perplexity(model_quantized, eval_dataset)
assert ppl_after < ppl_before * 1.05, "تدهور كبير في الدقة — راجع استراتيجية التكميم"

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-13-chapter-13.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
# mini_project_benchmark_kv_cache.py
from __future__ import annotations

import time

import torch
import torch.nn as nn

# TODO:
# 1) استخدم CachedMultiHeadAttention من هذا الفصل لبناء GPTBlock كامل مع كاش اختياري
# 2) بناء نموذج GPT مُصغَّر (2-4 طبقات) بـ token embedding + positional embedding + رأس تنبؤ
# 3) اكتب دالتين:
#    - generate_naive(model, prompt_ids, n_new): بدون كاش، تعيد حساب كل شيء في كل خطوة
#    - generate_cached(model, prompt_ids, n_new): مع كاش لكل الطبقات
# 4) قِس الزمن الفعلي لكل دالة عند n_new = [50, 100, 200, 400]
# 5) ارسم (أو اطبع جدولًا) نسبة التسريع مقابل طول التوليد
# 6) تحقق: هل النتائج النهائية (التوزيع الاحتمالي للرمز الأخير) متطابقة تمامًا بين الطريقتين؟
#    (يجب أن تكون متطابقة أو شبه متطابقة عدديًا — فهذا يثبت صحة تطبيق الكاش)
