#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الخامس عشر: تحسينات الانتباه الحديثة — RoPE، ALiBi، MQA/GQA، وSliding Window Attention
Large Language Model Engineering — RoPE

Book slug : llm-engineering-69-chapters
Chapter   : ch-15-chapter-15 (lesson 15)
Code files: 13 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-15-chapter-15.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الخامس عشر: تحسينات الانتباه الحديثة — RoPE، ALiBi، MQA/GQA، وSliding Window Attention
Large Language Model Engineering — RoPE

Book slug : llm-engineering-69-chapters
Chapter   : ch-15-chapter-15 (lesson 15)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-15-chapter-15.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الخامس عشر: تحسينات الانتباه الحديثة — RoPE، ALiBi، MQA/GQA، وSliding Window Attention
Large Language Model Engineering — RoPE

Book slug : llm-engineering-69-chapters
Chapter   : ch-15-chapter-15 (lesson 15)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_15.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 15: الفصل الخامس عشر: تحسينات الانتباه الحديثة — RoPE، ALiBi، MQA/GQA، وSliding Window Attention."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_modern_attention() -> None:
    vector = [1.0, 0.0, -1.0, 0.5]
    angle = 0.25
    rotated = [vector[0] * math.cos(angle) - vector[1] * math.sin(angle), vector[0] * math.sin(angle) + vector[1] * math.cos(angle), vector[2] * math.cos(angle) - vector[3] * math.sin(angle), vector[2] * math.sin(angle) + vector[3] * math.cos(angle)]
    length, window, kv_heads, query_heads = (2048, 256, 4, 16)
    print('RoPE-rotated vector:', [round(x, 3) for x in rotated])
    print('sliding-window visible tokens:', min(length, window))
    print('MQA/GQA KV heads:', kv_heads, 'for', query_heads, 'query heads')

if __name__ == "__main__":
    print("Chapter 15 lab: Rotary positions and sliding-window attention")
    run_modern_attention()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-04-modern-architectures/chapter-15.md
# ────────────────────────────────────────────────────────────────────────
# rope_implementation.py
from __future__ import annotations

import math
import torch


def precompute_rope_frequencies(d_k: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """حساب الترددات θᵢ ومصفوفة الزوايا لكل موضع، مسبقًا (Precompute)."""
    assert d_k % 2 == 0, "RoPE يتطلب بُعدًا زوجيًا"
    inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))  # (d_k/2,)
    positions = torch.arange(max_seq_len).float()                       # (max_seq_len,)
    angles = torch.outer(positions, inv_freq)                           # (max_seq_len, d_k/2)
    return angles


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    تطبيق الدوران على x بشكل زوجي.
    x: (batch, n_heads, seq_len, d_k)
    angles: (seq_len, d_k/2)
    """
    batch, n_heads, seq_len, d_k = x.shape
    x1 = x[..., 0::2]   # الأبعاد الزوجية
    x2 = x[..., 1::2]   # الأبعاد الفردية

    cos = angles.cos()[None, None, :, :]  # (1,1,seq_len,d_k/2)
    sin = angles.sin()[None, None, :, :]

    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    # إعادة التجميع بالتناوب الأصلي
    out = torch.zeros_like(x)
    out[..., 0::2] = rotated_x1
    out[..., 1::2] = rotated_x2
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    d_k, seq_len, n_heads, batch = 8, 6, 2, 1

    q = torch.randn(batch, n_heads, seq_len, d_k)
    k = torch.randn(batch, n_heads, seq_len, d_k)

    angles = precompute_rope_frequencies(d_k, seq_len)
    q_rot = apply_rope(q, angles)
    k_rot = apply_rope(k, angles)

    # تحقق من الخاصية الأساسية: النتيجة تعتمد على الفرق m-n فقط
    score_0_2 = (q_rot[0, 0, 0] @ k_rot[0, 0, 2])  # m=0, n=2 → فرق=2
    score_1_3 = (q_rot[0, 0, 1] @ k_rot[0, 0, 3])  # m=1, n=3 → فرق=2 أيضًا (قبل التدوير q,k مختلفان لكن الزاوية النسبية نفسها)
    print(f"score(m=0,n=2) = {score_0_2.item():.4f}")
    print(f"score(m=1,n=3) = {score_1_3.item():.4f}")
    print("(القيمتان مختلفتان لأن q,k أنفسهما مختلفان بين الموضعين،"
          " لكن كلاهما يحمل نفس الإزاحة النسبية=2 بشكل متسق رياضيًا)")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-04-modern-architectures/chapter-15.md
# ────────────────────────────────────────────────────────────────────────
def compute_alibi_bias(n_heads: int, seq_len: int) -> torch.Tensor:
    """
    حساب انحياز ALiBi لكل رأس انتباه.
    المنحدرات m تُحسب كسلسلة هندسية: m_h = 2^(-8h/n_heads)
    """
    def get_slopes(n: int) -> list[float]:
        start = 2 ** (-8 / n)
        return [start ** (i + 1) for i in range(n)]

    slopes = torch.tensor(get_slopes(n_heads))  # (n_heads,)
    positions = torch.arange(seq_len)
    relative_pos = positions[None, :] - positions[:, None]   # (seq_len, seq_len)
    distance = relative_pos.abs().float()

    # انحياز سالب يتناسب مع المسافة، مختلف لكل رأس
    bias = -slopes[:, None, None] * distance[None, :, :]      # (n_heads, seq_len, seq_len)
    return bias

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-04-modern-architectures/chapter-15.md
# ────────────────────────────────────────────────────────────────────────
# gqa_implementation.py
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """
    تنفيذ Grouped-Query Attention.
    n_q_heads: عدد رؤوس الاستعلام (مثلاً 32)
    n_kv_heads: عدد مجموعات K,V (مثلاً 8) — يجب أن يقبل n_q_heads القسمة عليه
    عندما n_kv_heads == n_q_heads → يُصبح MHA القياسي
    عندما n_kv_heads == 1          → يُصبح MQA
    """

    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int) -> None:
        super().__init__()
        assert n_q_heads % n_kv_heads == 0, "عدد رؤوس Q يجب أن يقبل القسمة على عدد مجموعات KV"
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_q_heads // n_kv_heads     # كم رأس Q يُشارك كل مجموعة KV
        self.d_k = d_model // n_q_heads

        self.w_q = nn.Linear(d_model, n_q_heads * self.d_k, bias=False)
        # ملاحظة جوهرية: K,V بحجم أصغر بكثير! فقط n_kv_heads بدلاً من n_q_heads
        self.w_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.w_o = nn.Linear(n_q_heads * self.d_k, d_model, bias=False)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.w_q(x).view(batch, seq_len, self.n_q_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)

        # الخطوة الحاسمة: تكرار K,V لتُطابق عدد رؤوس Q (دون تخزين نسخ فعلية إضافية في الكاش!)
        k = k.repeat_interleave(self.n_rep, dim=1)   # (batch, n_q_heads, seq_len, d_k)
        v = v.repeat_interleave(self.n_rep, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        context = attn_weights @ v
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.w_o(context)


if __name__ == "__main__":
    torch.manual_seed(0)
    d_model, seq_len, batch = 512, 10, 2

    # مقارنة حجم الكاش: MHA (32 رأس KV) مقابل GQA (8 مجموعات) مقابل MQA (1 مجموعة)
    for n_kv_heads, name in [(32, "MHA"), (8, "GQA"), (1, "MQA")]:
        layer = GroupedQueryAttention(d_model, n_q_heads=32, n_kv_heads=n_kv_heads)
        x = torch.randn(batch, seq_len, d_model)
        out = layer(x)
        kv_cache_size_per_layer = 2 * n_kv_heads * seq_len * (d_model // 32) * batch
        print(f"{name:5s} (n_kv_heads={n_kv_heads:2d}): "
              f"شكل الإخراج {tuple(out.shape)}, "
              f"عناصر كاش KV تقريبية لكل طبقة: {kv_cache_size_per_layer}")

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-04-modern-architectures/chapter-15.md
# ────────────────────────────────────────────────────────────────────────
# attention_variants_comparison.py
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

torch.manual_seed(7)


def standard_causal_attention(q, k, v) -> torch.Tensor:
    d_k = q.size(-1)
    seq_len = q.size(-2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    causal = torch.tril(torch.ones(seq_len, seq_len))
    scores = scores.masked_fill(causal == 0, float("-inf"))
    return F.softmax(scores, dim=-1) @ v


def sliding_window_attention(q, k, v, window: int) -> torch.Tensor:
    d_k = q.size(-1)
    seq_len = q.size(-2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

    idx = torch.arange(seq_len)
    distance = idx[:, None] - idx[None, :]
    allowed = (distance >= 0) & (distance < window)   # سببي + ضمن النافذة فقط
    scores = scores.masked_fill(~allowed, float("-inf"))
    return F.softmax(scores, dim=-1) @ v


if __name__ == "__main__":
    seq_len, d_k = 10, 16
    q = torch.randn(1, seq_len, d_k)
    k = torch.randn(1, seq_len, d_k)
    v = torch.randn(1, seq_len, d_k)

    out_full = standard_causal_attention(q, k, v)
    out_window = sliding_window_attention(q, k, v, window=3)

    print("شكل إخراج الانتباه الكامل      :", out_full.shape)
    print("شكل إخراج Sliding Window (w=3) :", out_window.shape)

    diff = (out_full - out_window).abs().mean().item()
    print(f"\nمتوسط الفرق المطلق بين الطريقتين: {diff:.4f}")
    print("(الفرق متوقع وطبيعي — كل طريقة ترى معلومات سياقية مختلفة الحجم)")

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-04-modern-architectures/chapter-15.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: تطبيق RoPE بعد دمج الرؤوس بدلاً من قبل تقسيمها لكل رأس
q_combined = self.w_q(x)          # (batch, seq, d_model)
q_rotated = apply_rope(q_combined, angles)   # خطأ في الشكل والدلالة!

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-04-modern-architectures/chapter-15.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: قسّم الرؤوس أولًا، ثم طبّق RoPE على كل رأس بشكل مستقل
q = split_heads(self.w_q(x))      # (batch, n_heads, seq, d_k)
q_rotated = apply_rope(q, angles)  # لكل رأس بشكل منفصل، صحيح دلاليًا

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-04-modern-architectures/chapter-15.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: استخدام repeat العادي يُنتج ترتيبًا خاطئًا للمجموعات
k_wrong = k.repeat(1, self.n_rep, 1, 1)   # يُكرر التسلسل الكامل، وليس كل رأس

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-04-modern-architectures/chapter-15.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: repeat_interleave يُكرر كل رأس على حدة بالترتيب الصحيح
k_correct = k.repeat_interleave(self.n_rep, dim=1)

# ────────────────────────────────────────────────────────────────────────
# [010] book/chapters/part-04-modern-architectures/chapter-15.md
# ────────────────────────────────────────────────────────────────────────
# ✅ تحقق دائمًا: الحقل الاستقبالي الفعلي يكفي للمسافة المطلوبة في مهمتك؟
effective_receptive_field = n_layers * window_size
assert effective_receptive_field >= expected_max_dependency_distance

# ────────────────────────────────────────────────────────────────────────
# [011] book/chapters/part-04-modern-architectures/chapter-15.md
# ────────────────────────────────────────────────────────────────────────
# mini_project_length_extrapolation.py
from __future__ import annotations

import torch
import torch.nn as nn

# TODO:
# 1) بناء ثلاثة نماذج GPT-style صغيرة متطابقة تمامًا في كل شيء
#    إلا آلية الترميز الموضعي: (أ) PE الجيبي المُضاف، (ب) RoPE، (ج) ALiBi
# 2) درّب الثلاثة على نفس مجموعة بيانات بسيطة (مثلاً نص عشوائي أو TinyShakespeare)
#    بطول تسلسل ثابت = 128 رمزًا فقط أثناء التدريب
# 3) قِس Perplexity على بيانات اختبار بأطوال: 128 (داخل التدريب)، 256، 512، 1024 (خارج التدريب)
# 4) ارسم جدولًا: Perplexity لكل نموذج عند كل طول
# 5) النتيجة المتوقعة: PE الجيبي يتدهور بشدة خارج 128، RoPE يتدهور بشكل أقل،
#    ALiBi يُحافظ على أداء أكثر استقرارًا نسبيًا عند الأطوال الأكبر

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-15-chapter-15.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 15: الفصل الخامس عشر: تحسينات الانتباه الحديثة — RoPE، ALiBi، MQA/GQA، وSliding Window Attention."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_modern_attention() -> None:
    vector = [1.0, 0.0, -1.0, 0.5]
    angle = 0.25
    rotated = [vector[0] * math.cos(angle) - vector[1] * math.sin(angle), vector[0] * math.sin(angle) + vector[1] * math.cos(angle), vector[2] * math.cos(angle) - vector[3] * math.sin(angle), vector[2] * math.sin(angle) + vector[3] * math.cos(angle)]
    length, window, kv_heads, query_heads = (2048, 256, 4, 16)
    print('RoPE-rotated vector:', [round(x, 3) for x in rotated])
    print('sliding-window visible tokens:', min(length, window))
    print('MQA/GQA KV heads:', kv_heads, 'for', query_heads, 'query heads')

if __name__ == "__main__":
    print("Chapter 15 lab: Rotary positions and sliding-window attention")
    run_modern_attention()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-15-chapter-15.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# rope_implementation.py
from __future__ import annotations

import math
import torch


def precompute_rope_frequencies(d_k: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """حساب الترددات θᵢ ومصفوفة الزوايا لكل موضع، مسبقًا (Precompute)."""
    assert d_k % 2 == 0, "RoPE يتطلب بُعدًا زوجيًا"
    inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))  # (d_k/2,)
    positions = torch.arange(max_seq_len).float()                       # (max_seq_len,)
    angles = torch.outer(positions, inv_freq)                           # (max_seq_len, d_k/2)
    return angles


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    تطبيق الدوران على x بشكل زوجي.
    x: (batch, n_heads, seq_len, d_k)
    angles: (seq_len, d_k/2)
    """
    batch, n_heads, seq_len, d_k = x.shape
    x1 = x[..., 0::2]   # الأبعاد الزوجية
    x2 = x[..., 1::2]   # الأبعاد الفردية

    cos = angles.cos()[None, None, :, :]  # (1,1,seq_len,d_k/2)
    sin = angles.sin()[None, None, :, :]

    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    # إعادة التجميع بالتناوب الأصلي
    out = torch.zeros_like(x)
    out[..., 0::2] = rotated_x1
    out[..., 1::2] = rotated_x2
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    d_k, seq_len, n_heads, batch = 8, 6, 2, 1

    q = torch.randn(batch, n_heads, seq_len, d_k)
    k = torch.randn(batch, n_heads, seq_len, d_k)

    angles = precompute_rope_frequencies(d_k, seq_len)
    q_rot = apply_rope(q, angles)
    k_rot = apply_rope(k, angles)

    # تحقق من الخاصية الأساسية: النتيجة تعتمد على الفرق m-n فقط
    score_0_2 = (q_rot[0, 0, 0] @ k_rot[0, 0, 2])  # m=0, n=2 → فرق=2
    score_1_3 = (q_rot[0, 0, 1] @ k_rot[0, 0, 3])  # m=1, n=3 → فرق=2 أيضًا (قبل التدوير q,k مختلفان لكن الزاوية النسبية نفسها)
    print(f"score(m=0,n=2) = {score_0_2.item():.4f}")
    print(f"score(m=1,n=3) = {score_1_3.item():.4f}")
    print("(القيمتان مختلفتان لأن q,k أنفسهما مختلفان بين الموضعين،"
          " لكن كلاهما يحمل نفس الإزاحة النسبية=2 بشكل متسق رياضيًا)")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-15-chapter-15.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def compute_alibi_bias(n_heads: int, seq_len: int) -> torch.Tensor:
    """
    حساب انحياز ALiBi لكل رأس انتباه.
    المنحدرات m تُحسب كسلسلة هندسية: m_h = 2^(-8h/n_heads)
    """
    def get_slopes(n: int) -> list[float]:
        start = 2 ** (-8 / n)
        return [start ** (i + 1) for i in range(n)]

    slopes = torch.tensor(get_slopes(n_heads))  # (n_heads,)
    positions = torch.arange(seq_len)
    relative_pos = positions[None, :] - positions[:, None]   # (seq_len, seq_len)
    distance = relative_pos.abs().float()

    # انحياز سالب يتناسب مع المسافة، مختلف لكل رأس
    bias = -slopes[:, None, None] * distance[None, :, :]      # (n_heads, seq_len, seq_len)
    return bias

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-15-chapter-15.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# gqa_implementation.py
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """
    تنفيذ Grouped-Query Attention.
    n_q_heads: عدد رؤوس الاستعلام (مثلاً 32)
    n_kv_heads: عدد مجموعات K,V (مثلاً 8) — يجب أن يقبل n_q_heads القسمة عليه
    عندما n_kv_heads == n_q_heads → يُصبح MHA القياسي
    عندما n_kv_heads == 1          → يُصبح MQA
    """

    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int) -> None:
        super().__init__()
        assert n_q_heads % n_kv_heads == 0, "عدد رؤوس Q يجب أن يقبل القسمة على عدد مجموعات KV"
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_q_heads // n_kv_heads     # كم رأس Q يُشارك كل مجموعة KV
        self.d_k = d_model // n_q_heads

        self.w_q = nn.Linear(d_model, n_q_heads * self.d_k, bias=False)
        # ملاحظة جوهرية: K,V بحجم أصغر بكثير! فقط n_kv_heads بدلاً من n_q_heads
        self.w_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.w_o = nn.Linear(n_q_heads * self.d_k, d_model, bias=False)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.w_q(x).view(batch, seq_len, self.n_q_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)

        # الخطوة الحاسمة: تكرار K,V لتُطابق عدد رؤوس Q (دون تخزين نسخ فعلية إضافية في الكاش!)
        k = k.repeat_interleave(self.n_rep, dim=1)   # (batch, n_q_heads, seq_len, d_k)
        v = v.repeat_interleave(self.n_rep, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        context = attn_weights @ v
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.w_o(context)


if __name__ == "__main__":
    torch.manual_seed(0)
    d_model, seq_len, batch = 512, 10, 2

    # مقارنة حجم الكاش: MHA (32 رأس KV) مقابل GQA (8 مجموعات) مقابل MQA (1 مجموعة)
    for n_kv_heads, name in [(32, "MHA"), (8, "GQA"), (1, "MQA")]:
        layer = GroupedQueryAttention(d_model, n_q_heads=32, n_kv_heads=n_kv_heads)
        x = torch.randn(batch, seq_len, d_model)
        out = layer(x)
        kv_cache_size_per_layer = 2 * n_kv_heads * seq_len * (d_model // 32) * batch
        print(f"{name:5s} (n_kv_heads={n_kv_heads:2d}): "
              f"شكل الإخراج {tuple(out.shape)}, "
              f"عناصر كاش KV تقريبية لكل طبقة: {kv_cache_size_per_layer}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-15-chapter-15.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# attention_variants_comparison.py
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

torch.manual_seed(7)


def standard_causal_attention(q, k, v) -> torch.Tensor:
    d_k = q.size(-1)
    seq_len = q.size(-2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    causal = torch.tril(torch.ones(seq_len, seq_len))
    scores = scores.masked_fill(causal == 0, float("-inf"))
    return F.softmax(scores, dim=-1) @ v


def sliding_window_attention(q, k, v, window: int) -> torch.Tensor:
    d_k = q.size(-1)
    seq_len = q.size(-2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

    idx = torch.arange(seq_len)
    distance = idx[:, None] - idx[None, :]
    allowed = (distance >= 0) & (distance < window)   # سببي + ضمن النافذة فقط
    scores = scores.masked_fill(~allowed, float("-inf"))
    return F.softmax(scores, dim=-1) @ v


if __name__ == "__main__":
    seq_len, d_k = 10, 16
    q = torch.randn(1, seq_len, d_k)
    k = torch.randn(1, seq_len, d_k)
    v = torch.randn(1, seq_len, d_k)

    out_full = standard_causal_attention(q, k, v)
    out_window = sliding_window_attention(q, k, v, window=3)

    print("شكل إخراج الانتباه الكامل      :", out_full.shape)
    print("شكل إخراج Sliding Window (w=3) :", out_window.shape)

    diff = (out_full - out_window).abs().mean().item()
    print(f"\nمتوسط الفرق المطلق بين الطريقتين: {diff:.4f}")
    print("(الفرق متوقع وطبيعي — كل طريقة ترى معلومات سياقية مختلفة الحجم)")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-15-chapter-15.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: تطبيق RoPE بعد دمج الرؤوس بدلاً من قبل تقسيمها لكل رأس
q_combined = self.w_q(x)          # (batch, seq, d_model)
q_rotated = apply_rope(q_combined, angles)   # خطأ في الشكل والدلالة!

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-15-chapter-15.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: قسّم الرؤوس أولًا، ثم طبّق RoPE على كل رأس بشكل مستقل
q = split_heads(self.w_q(x))      # (batch, n_heads, seq, d_k)
q_rotated = apply_rope(q, angles)  # لكل رأس بشكل منفصل، صحيح دلاليًا

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-15-chapter-15.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: استخدام repeat العادي يُنتج ترتيبًا خاطئًا للمجموعات
k_wrong = k.repeat(1, self.n_rep, 1, 1)   # يُكرر التسلسل الكامل، وليس كل رأس

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-15-chapter-15.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: repeat_interleave يُكرر كل رأس على حدة بالترتيب الصحيح
k_correct = k.repeat_interleave(self.n_rep, dim=1)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-15-chapter-15.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ✅ تحقق دائمًا: الحقل الاستقبالي الفعلي يكفي للمسافة المطلوبة في مهمتك؟
effective_receptive_field = n_layers * window_size
assert effective_receptive_field >= expected_max_dependency_distance

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-15-chapter-15.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# mini_project_length_extrapolation.py
from __future__ import annotations

import torch
import torch.nn as nn

# TODO:
# 1) بناء ثلاثة نماذج GPT-style صغيرة متطابقة تمامًا في كل شيء
#    إلا آلية الترميز الموضعي: (أ) PE الجيبي المُضاف، (ب) RoPE، (ج) ALiBi
# 2) درّب الثلاثة على نفس مجموعة بيانات بسيطة (مثلاً نص عشوائي أو TinyShakespeare)
#    بطول تسلسل ثابت = 128 رمزًا فقط أثناء التدريب
# 3) قِس Perplexity على بيانات اختبار بأطوال: 128 (داخل التدريب)، 256، 512، 1024 (خارج التدريب)
# 4) ارسم جدولًا: Perplexity لكل نموذج عند كل طول
# 5) النتيجة المتوقعة: PE الجيبي يتدهور بشدة خارج 128، RoPE يتدهور بشكل أقل،
#    ALiBi يُحافظ على أداء أكثر استقرارًا نسبيًا عند الأطوال الأكبر

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-15-chapter-15.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الخامس عشر: تحسينات الانتباه الحديثة — RoPE، ALiBi، MQA/GQA، وSliding Window Attention
Large Language Model Engineering — RoPE

Book slug : llm-engineering-69-chapters
Chapter   : ch-15-chapter-15 (lesson 15)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_15.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 15: الفصل الخامس عشر: تحسينات الانتباه الحديثة — RoPE، ALiBi، MQA/GQA، وSliding Window Attention."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_modern_attention() -> None:
    vector = [1.0, 0.0, -1.0, 0.5]
    angle = 0.25
    rotated = [vector[0] * math.cos(angle) - vector[1] * math.sin(angle), vector[0] * math.sin(angle) + vector[1] * math.cos(angle), vector[2] * math.cos(angle) - vector[3] * math.sin(angle), vector[2] * math.sin(angle) + vector[3] * math.cos(angle)]
    length, window, kv_heads, query_heads = (2048, 256, 4, 16)
    print('RoPE-rotated vector:', [round(x, 3) for x in rotated])
    print('sliding-window visible tokens:', min(length, window))
    print('MQA/GQA KV heads:', kv_heads, 'for', query_heads, 'query heads')

if __name__ == "__main__":
    print("Chapter 15 lab: Rotary positions and sliding-window attention")
    run_modern_attention()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-04-modern-architectures/chapter-15.md
# ────────────────────────────────────────────────────────────────────────
# rope_implementation.py
from __future__ import annotations

import math
import torch


def precompute_rope_frequencies(d_k: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """حساب الترددات θᵢ ومصفوفة الزوايا لكل موضع، مسبقًا (Precompute)."""
    assert d_k % 2 == 0, "RoPE يتطلب بُعدًا زوجيًا"
    inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))  # (d_k/2,)
    positions = torch.arange(max_seq_len).float()                       # (max_seq_len,)
    angles = torch.outer(positions, inv_freq)                           # (max_seq_len, d_k/2)
    return angles


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    تطبيق الدوران على x بشكل زوجي.
    x: (batch, n_heads, seq_len, d_k)
    angles: (seq_len, d_k/2)
    """
    batch, n_heads, seq_len, d_k = x.shape
    x1 = x[..., 0::2]   # الأبعاد الزوجية
    x2 = x[..., 1::2]   # الأبعاد الفردية

    cos = angles.cos()[None, None, :, :]  # (1,1,seq_len,d_k/2)
    sin = angles.sin()[None, None, :, :]

    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    # إعادة التجميع بالتناوب الأصلي
    out = torch.zeros_like(x)
    out[..., 0::2] = rotated_x1
    out[..., 1::2] = rotated_x2
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    d_k, seq_len, n_heads, batch = 8, 6, 2, 1

    q = torch.randn(batch, n_heads, seq_len, d_k)
    k = torch.randn(batch, n_heads, seq_len, d_k)

    angles = precompute_rope_frequencies(d_k, seq_len)
    q_rot = apply_rope(q, angles)
    k_rot = apply_rope(k, angles)

    # تحقق من الخاصية الأساسية: النتيجة تعتمد على الفرق m-n فقط
    score_0_2 = (q_rot[0, 0, 0] @ k_rot[0, 0, 2])  # m=0, n=2 → فرق=2
    score_1_3 = (q_rot[0, 0, 1] @ k_rot[0, 0, 3])  # m=1, n=3 → فرق=2 أيضًا (قبل التدوير q,k مختلفان لكن الزاوية النسبية نفسها)
    print(f"score(m=0,n=2) = {score_0_2.item():.4f}")
    print(f"score(m=1,n=3) = {score_1_3.item():.4f}")
    print("(القيمتان مختلفتان لأن q,k أنفسهما مختلفان بين الموضعين،"
          " لكن كلاهما يحمل نفس الإزاحة النسبية=2 بشكل متسق رياضيًا)")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-04-modern-architectures/chapter-15.md
# ────────────────────────────────────────────────────────────────────────
def compute_alibi_bias(n_heads: int, seq_len: int) -> torch.Tensor:
    """
    حساب انحياز ALiBi لكل رأس انتباه.
    المنحدرات m تُحسب كسلسلة هندسية: m_h = 2^(-8h/n_heads)
    """
    def get_slopes(n: int) -> list[float]:
        start = 2 ** (-8 / n)
        return [start ** (i + 1) for i in range(n)]

    slopes = torch.tensor(get_slopes(n_heads))  # (n_heads,)
    positions = torch.arange(seq_len)
    relative_pos = positions[None, :] - positions[:, None]   # (seq_len, seq_len)
    distance = relative_pos.abs().float()

    # انحياز سالب يتناسب مع المسافة، مختلف لكل رأس
    bias = -slopes[:, None, None] * distance[None, :, :]      # (n_heads, seq_len, seq_len)
    return bias

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-04-modern-architectures/chapter-15.md
# ────────────────────────────────────────────────────────────────────────
# gqa_implementation.py
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """
    تنفيذ Grouped-Query Attention.
    n_q_heads: عدد رؤوس الاستعلام (مثلاً 32)
    n_kv_heads: عدد مجموعات K,V (مثلاً 8) — يجب أن يقبل n_q_heads القسمة عليه
    عندما n_kv_heads == n_q_heads → يُصبح MHA القياسي
    عندما n_kv_heads == 1          → يُصبح MQA
    """

    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int) -> None:
        super().__init__()
        assert n_q_heads % n_kv_heads == 0, "عدد رؤوس Q يجب أن يقبل القسمة على عدد مجموعات KV"
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_q_heads // n_kv_heads     # كم رأس Q يُشارك كل مجموعة KV
        self.d_k = d_model // n_q_heads

        self.w_q = nn.Linear(d_model, n_q_heads * self.d_k, bias=False)
        # ملاحظة جوهرية: K,V بحجم أصغر بكثير! فقط n_kv_heads بدلاً من n_q_heads
        self.w_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.w_o = nn.Linear(n_q_heads * self.d_k, d_model, bias=False)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.w_q(x).view(batch, seq_len, self.n_q_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)

        # الخطوة الحاسمة: تكرار K,V لتُطابق عدد رؤوس Q (دون تخزين نسخ فعلية إضافية في الكاش!)
        k = k.repeat_interleave(self.n_rep, dim=1)   # (batch, n_q_heads, seq_len, d_k)
        v = v.repeat_interleave(self.n_rep, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        context = attn_weights @ v
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.w_o(context)


if __name__ == "__main__":
    torch.manual_seed(0)
    d_model, seq_len, batch = 512, 10, 2

    # مقارنة حجم الكاش: MHA (32 رأس KV) مقابل GQA (8 مجموعات) مقابل MQA (1 مجموعة)
    for n_kv_heads, name in [(32, "MHA"), (8, "GQA"), (1, "MQA")]:
        layer = GroupedQueryAttention(d_model, n_q_heads=32, n_kv_heads=n_kv_heads)
        x = torch.randn(batch, seq_len, d_model)
        out = layer(x)
        kv_cache_size_per_layer = 2 * n_kv_heads * seq_len * (d_model // 32) * batch
        print(f"{name:5s} (n_kv_heads={n_kv_heads:2d}): "
              f"شكل الإخراج {tuple(out.shape)}, "
              f"عناصر كاش KV تقريبية لكل طبقة: {kv_cache_size_per_layer}")

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-04-modern-architectures/chapter-15.md
# ────────────────────────────────────────────────────────────────────────
# attention_variants_comparison.py
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

torch.manual_seed(7)


def standard_causal_attention(q, k, v) -> torch.Tensor:
    d_k = q.size(-1)
    seq_len = q.size(-2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    causal = torch.tril(torch.ones(seq_len, seq_len))
    scores = scores.masked_fill(causal == 0, float("-inf"))
    return F.softmax(scores, dim=-1) @ v


def sliding_window_attention(q, k, v, window: int) -> torch.Tensor:
    d_k = q.size(-1)
    seq_len = q.size(-2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

    idx = torch.arange(seq_len)
    distance = idx[:, None] - idx[None, :]
    allowed = (distance >= 0) & (distance < window)   # سببي + ضمن النافذة فقط
    scores = scores.masked_fill(~allowed, float("-inf"))
    return F.softmax(scores, dim=-1) @ v


if __name__ == "__main__":
    seq_len, d_k = 10, 16
    q = torch.randn(1, seq_len, d_k)
    k = torch.randn(1, seq_len, d_k)
    v = torch.randn(1, seq_len, d_k)

    out_full = standard_causal_attention(q, k, v)
    out_window = sliding_window_attention(q, k, v, window=3)

    print("شكل إخراج الانتباه الكامل      :", out_full.shape)
    print("شكل إخراج Sliding Window (w=3) :", out_window.shape)

    diff = (out_full - out_window).abs().mean().item()
    print(f"\nمتوسط الفرق المطلق بين الطريقتين: {diff:.4f}")
    print("(الفرق متوقع وطبيعي — كل طريقة ترى معلومات سياقية مختلفة الحجم)")

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-04-modern-architectures/chapter-15.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: تطبيق RoPE بعد دمج الرؤوس بدلاً من قبل تقسيمها لكل رأس
q_combined = self.w_q(x)          # (batch, seq, d_model)
q_rotated = apply_rope(q_combined, angles)   # خطأ في الشكل والدلالة!

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-04-modern-architectures/chapter-15.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: قسّم الرؤوس أولًا، ثم طبّق RoPE على كل رأس بشكل مستقل
q = split_heads(self.w_q(x))      # (batch, n_heads, seq, d_k)
q_rotated = apply_rope(q, angles)  # لكل رأس بشكل منفصل، صحيح دلاليًا

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-04-modern-architectures/chapter-15.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: استخدام repeat العادي يُنتج ترتيبًا خاطئًا للمجموعات
k_wrong = k.repeat(1, self.n_rep, 1, 1)   # يُكرر التسلسل الكامل، وليس كل رأس

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-04-modern-architectures/chapter-15.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: repeat_interleave يُكرر كل رأس على حدة بالترتيب الصحيح
k_correct = k.repeat_interleave(self.n_rep, dim=1)

# ────────────────────────────────────────────────────────────────────────
# [010] book/chapters/part-04-modern-architectures/chapter-15.md
# ────────────────────────────────────────────────────────────────────────
# ✅ تحقق دائمًا: الحقل الاستقبالي الفعلي يكفي للمسافة المطلوبة في مهمتك؟
effective_receptive_field = n_layers * window_size
assert effective_receptive_field >= expected_max_dependency_distance

# ────────────────────────────────────────────────────────────────────────
# [011] book/chapters/part-04-modern-architectures/chapter-15.md
# ────────────────────────────────────────────────────────────────────────
# mini_project_length_extrapolation.py
from __future__ import annotations

import torch
import torch.nn as nn

# TODO:
# 1) بناء ثلاثة نماذج GPT-style صغيرة متطابقة تمامًا في كل شيء
#    إلا آلية الترميز الموضعي: (أ) PE الجيبي المُضاف، (ب) RoPE، (ج) ALiBi
# 2) درّب الثلاثة على نفس مجموعة بيانات بسيطة (مثلاً نص عشوائي أو TinyShakespeare)
#    بطول تسلسل ثابت = 128 رمزًا فقط أثناء التدريب
# 3) قِس Perplexity على بيانات اختبار بأطوال: 128 (داخل التدريب)، 256، 512، 1024 (خارج التدريب)
# 4) ارسم جدولًا: Perplexity لكل نموذج عند كل طول
# 5) النتيجة المتوقعة: PE الجيبي يتدهور بشدة خارج 128، RoPE يتدهور بشكل أقل،
#    ALiBi يُحافظ على أداء أكثر استقرارًا نسبيًا عند الأطوال الأكبر

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-15-chapter-15.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 15: الفصل الخامس عشر: تحسينات الانتباه الحديثة — RoPE، ALiBi، MQA/GQA، وSliding Window Attention."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_modern_attention() -> None:
    vector = [1.0, 0.0, -1.0, 0.5]
    angle = 0.25
    rotated = [vector[0] * math.cos(angle) - vector[1] * math.sin(angle), vector[0] * math.sin(angle) + vector[1] * math.cos(angle), vector[2] * math.cos(angle) - vector[3] * math.sin(angle), vector[2] * math.sin(angle) + vector[3] * math.cos(angle)]
    length, window, kv_heads, query_heads = (2048, 256, 4, 16)
    print('RoPE-rotated vector:', [round(x, 3) for x in rotated])
    print('sliding-window visible tokens:', min(length, window))
    print('MQA/GQA KV heads:', kv_heads, 'for', query_heads, 'query heads')

if __name__ == "__main__":
    print("Chapter 15 lab: Rotary positions and sliding-window attention")
    run_modern_attention()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-15-chapter-15.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# rope_implementation.py
from __future__ import annotations

import math
import torch


def precompute_rope_frequencies(d_k: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """حساب الترددات θᵢ ومصفوفة الزوايا لكل موضع، مسبقًا (Precompute)."""
    assert d_k % 2 == 0, "RoPE يتطلب بُعدًا زوجيًا"
    inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))  # (d_k/2,)
    positions = torch.arange(max_seq_len).float()                       # (max_seq_len,)
    angles = torch.outer(positions, inv_freq)                           # (max_seq_len, d_k/2)
    return angles


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    تطبيق الدوران على x بشكل زوجي.
    x: (batch, n_heads, seq_len, d_k)
    angles: (seq_len, d_k/2)
    """
    batch, n_heads, seq_len, d_k = x.shape
    x1 = x[..., 0::2]   # الأبعاد الزوجية
    x2 = x[..., 1::2]   # الأبعاد الفردية

    cos = angles.cos()[None, None, :, :]  # (1,1,seq_len,d_k/2)
    sin = angles.sin()[None, None, :, :]

    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    # إعادة التجميع بالتناوب الأصلي
    out = torch.zeros_like(x)
    out[..., 0::2] = rotated_x1
    out[..., 1::2] = rotated_x2
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    d_k, seq_len, n_heads, batch = 8, 6, 2, 1

    q = torch.randn(batch, n_heads, seq_len, d_k)
    k = torch.randn(batch, n_heads, seq_len, d_k)

    angles = precompute_rope_frequencies(d_k, seq_len)
    q_rot = apply_rope(q, angles)
    k_rot = apply_rope(k, angles)

    # تحقق من الخاصية الأساسية: النتيجة تعتمد على الفرق m-n فقط
    score_0_2 = (q_rot[0, 0, 0] @ k_rot[0, 0, 2])  # m=0, n=2 → فرق=2
    score_1_3 = (q_rot[0, 0, 1] @ k_rot[0, 0, 3])  # m=1, n=3 → فرق=2 أيضًا (قبل التدوير q,k مختلفان لكن الزاوية النسبية نفسها)
    print(f"score(m=0,n=2) = {score_0_2.item():.4f}")
    print(f"score(m=1,n=3) = {score_1_3.item():.4f}")
    print("(القيمتان مختلفتان لأن q,k أنفسهما مختلفان بين الموضعين،"
          " لكن كلاهما يحمل نفس الإزاحة النسبية=2 بشكل متسق رياضيًا)")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-15-chapter-15.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def compute_alibi_bias(n_heads: int, seq_len: int) -> torch.Tensor:
    """
    حساب انحياز ALiBi لكل رأس انتباه.
    المنحدرات m تُحسب كسلسلة هندسية: m_h = 2^(-8h/n_heads)
    """
    def get_slopes(n: int) -> list[float]:
        start = 2 ** (-8 / n)
        return [start ** (i + 1) for i in range(n)]

    slopes = torch.tensor(get_slopes(n_heads))  # (n_heads,)
    positions = torch.arange(seq_len)
    relative_pos = positions[None, :] - positions[:, None]   # (seq_len, seq_len)
    distance = relative_pos.abs().float()

    # انحياز سالب يتناسب مع المسافة، مختلف لكل رأس
    bias = -slopes[:, None, None] * distance[None, :, :]      # (n_heads, seq_len, seq_len)
    return bias

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-15-chapter-15.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# gqa_implementation.py
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """
    تنفيذ Grouped-Query Attention.
    n_q_heads: عدد رؤوس الاستعلام (مثلاً 32)
    n_kv_heads: عدد مجموعات K,V (مثلاً 8) — يجب أن يقبل n_q_heads القسمة عليه
    عندما n_kv_heads == n_q_heads → يُصبح MHA القياسي
    عندما n_kv_heads == 1          → يُصبح MQA
    """

    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int) -> None:
        super().__init__()
        assert n_q_heads % n_kv_heads == 0, "عدد رؤوس Q يجب أن يقبل القسمة على عدد مجموعات KV"
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_q_heads // n_kv_heads     # كم رأس Q يُشارك كل مجموعة KV
        self.d_k = d_model // n_q_heads

        self.w_q = nn.Linear(d_model, n_q_heads * self.d_k, bias=False)
        # ملاحظة جوهرية: K,V بحجم أصغر بكثير! فقط n_kv_heads بدلاً من n_q_heads
        self.w_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.w_o = nn.Linear(n_q_heads * self.d_k, d_model, bias=False)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.w_q(x).view(batch, seq_len, self.n_q_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)

        # الخطوة الحاسمة: تكرار K,V لتُطابق عدد رؤوس Q (دون تخزين نسخ فعلية إضافية في الكاش!)
        k = k.repeat_interleave(self.n_rep, dim=1)   # (batch, n_q_heads, seq_len, d_k)
        v = v.repeat_interleave(self.n_rep, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        context = attn_weights @ v
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.w_o(context)


if __name__ == "__main__":
    torch.manual_seed(0)
    d_model, seq_len, batch = 512, 10, 2

    # مقارنة حجم الكاش: MHA (32 رأس KV) مقابل GQA (8 مجموعات) مقابل MQA (1 مجموعة)
    for n_kv_heads, name in [(32, "MHA"), (8, "GQA"), (1, "MQA")]:
        layer = GroupedQueryAttention(d_model, n_q_heads=32, n_kv_heads=n_kv_heads)
        x = torch.randn(batch, seq_len, d_model)
        out = layer(x)
        kv_cache_size_per_layer = 2 * n_kv_heads * seq_len * (d_model // 32) * batch
        print(f"{name:5s} (n_kv_heads={n_kv_heads:2d}): "
              f"شكل الإخراج {tuple(out.shape)}, "
              f"عناصر كاش KV تقريبية لكل طبقة: {kv_cache_size_per_layer}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-15-chapter-15.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# attention_variants_comparison.py
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

torch.manual_seed(7)


def standard_causal_attention(q, k, v) -> torch.Tensor:
    d_k = q.size(-1)
    seq_len = q.size(-2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    causal = torch.tril(torch.ones(seq_len, seq_len))
    scores = scores.masked_fill(causal == 0, float("-inf"))
    return F.softmax(scores, dim=-1) @ v


def sliding_window_attention(q, k, v, window: int) -> torch.Tensor:
    d_k = q.size(-1)
    seq_len = q.size(-2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

    idx = torch.arange(seq_len)
    distance = idx[:, None] - idx[None, :]
    allowed = (distance >= 0) & (distance < window)   # سببي + ضمن النافذة فقط
    scores = scores.masked_fill(~allowed, float("-inf"))
    return F.softmax(scores, dim=-1) @ v


if __name__ == "__main__":
    seq_len, d_k = 10, 16
    q = torch.randn(1, seq_len, d_k)
    k = torch.randn(1, seq_len, d_k)
    v = torch.randn(1, seq_len, d_k)

    out_full = standard_causal_attention(q, k, v)
    out_window = sliding_window_attention(q, k, v, window=3)

    print("شكل إخراج الانتباه الكامل      :", out_full.shape)
    print("شكل إخراج Sliding Window (w=3) :", out_window.shape)

    diff = (out_full - out_window).abs().mean().item()
    print(f"\nمتوسط الفرق المطلق بين الطريقتين: {diff:.4f}")
    print("(الفرق متوقع وطبيعي — كل طريقة ترى معلومات سياقية مختلفة الحجم)")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-15-chapter-15.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: تطبيق RoPE بعد دمج الرؤوس بدلاً من قبل تقسيمها لكل رأس
q_combined = self.w_q(x)          # (batch, seq, d_model)
q_rotated = apply_rope(q_combined, angles)   # خطأ في الشكل والدلالة!

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-15-chapter-15.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: قسّم الرؤوس أولًا، ثم طبّق RoPE على كل رأس بشكل مستقل
q = split_heads(self.w_q(x))      # (batch, n_heads, seq, d_k)
q_rotated = apply_rope(q, angles)  # لكل رأس بشكل منفصل، صحيح دلاليًا

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-15-chapter-15.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: استخدام repeat العادي يُنتج ترتيبًا خاطئًا للمجموعات
k_wrong = k.repeat(1, self.n_rep, 1, 1)   # يُكرر التسلسل الكامل، وليس كل رأس

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-15-chapter-15.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: repeat_interleave يُكرر كل رأس على حدة بالترتيب الصحيح
k_correct = k.repeat_interleave(self.n_rep, dim=1)

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-15-chapter-15.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# ✅ تحقق دائمًا: الحقل الاستقبالي الفعلي يكفي للمسافة المطلوبة في مهمتك؟
effective_receptive_field = n_layers * window_size
assert effective_receptive_field >= expected_max_dependency_distance

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-15-chapter-15.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# mini_project_length_extrapolation.py
from __future__ import annotations

import torch
import torch.nn as nn

# TODO:
# 1) بناء ثلاثة نماذج GPT-style صغيرة متطابقة تمامًا في كل شيء
#    إلا آلية الترميز الموضعي: (أ) PE الجيبي المُضاف، (ب) RoPE، (ج) ALiBi
# 2) درّب الثلاثة على نفس مجموعة بيانات بسيطة (مثلاً نص عشوائي أو TinyShakespeare)
#    بطول تسلسل ثابت = 128 رمزًا فقط أثناء التدريب
# 3) قِس Perplexity على بيانات اختبار بأطوال: 128 (داخل التدريب)، 256، 512، 1024 (خارج التدريب)
# 4) ارسم جدولًا: Perplexity لكل نموذج عند كل طول
# 5) النتيجة المتوقعة: PE الجيبي يتدهور بشدة خارج 128، RoPE يتدهور بشكل أقل،
#    ALiBi يُحافظ على أداء أكثر استقرارًا نسبيًا عند الأطوال الأكبر
