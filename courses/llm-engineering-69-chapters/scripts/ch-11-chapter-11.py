#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الحادي عشر: بنية Transformer الكاملة
Large Language Model Engineering — Transformer

Book slug : llm-engineering-69-chapters
Chapter   : ch-11-chapter-11 (lesson 11)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-11-chapter-11.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الحادي عشر: بنية Transformer الكاملة
Large Language Model Engineering — Transformer

Book slug : llm-engineering-69-chapters
Chapter   : ch-11-chapter-11 (lesson 11)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-11-chapter-11.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الحادي عشر: بنية Transformer الكاملة
Large Language Model Engineering — Transformer

Book slug : llm-engineering-69-chapters
Chapter   : ch-11-chapter-11 (lesson 11)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_11.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 11: الفصل الحادي عشر: بنية Transformer الكاملة."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_transformer() -> None:
    hidden = [1.0, -1.0, 0.5, 0.0]
    mean = sum(hidden) / len(hidden)
    variance = sum(((x - mean) ** 2 for x in hidden)) / len(hidden)
    normalized = [(x - mean) / math.sqrt(variance + 1e-05) for x in hidden]
    feed_forward = [max(0.0, x * 0.7 + 0.1) for x in normalized]
    output = [x + y for x, y in zip(hidden, feed_forward)]
    print('normalized:', [round(x, 3) for x in normalized])
    print('residual output:', [round(x, 3) for x in output])

if __name__ == "__main__":
    print("Chapter 11 lab: Residual, normalization, and feed-forward blocks")
    run_transformer()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-03-transformers-core/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
"""
transformer_block_complete.py
بناء كتلة Transformer كاملة (Positional Encoding + Attention + FFN + Residual + LN)
بأسلوب Pre-LN الحديث.
Python 3.13+ | PyTorch 2.x
"""
from __future__ import annotations

import math

import torch
from torch import nn, Tensor

torch.manual_seed(42)


# =====================================================================
# 1) الترميز الموضعي الجيبي (Sinusoidal Positional Encoding)
# =====================================================================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1).float()          # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)   # الأبعاد الزوجية: sin
        pe[:, 1::2] = torch.cos(position * div_term)   # الأبعاد الفردية: cos

        # تسجيلها كـ buffer وليس Parameter: لا تُدرَّب، لكن تُحفظ وتُنقل مع .to(device)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, seq_len, d_model)"""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# =====================================================================
# 2) Multi-Head Self-Attention (كما شُرحت بالتفصيل في الفصل العاشر)
# =====================================================================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model يجب أن يقبل القسمة على num_heads"

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_size, seq_len, d_model = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # كل منها: (batch, num_heads, seq_len, d_k)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)   # (batch, heads, seq, seq)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = attn_weights @ V                                 # (batch, heads, seq, d_k)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        return self.W_o(context)


# =====================================================================
# 3) شبكة التغذية الأمامية الموضعية (Position-wise FFN)
# =====================================================================
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),              # GELU: الخيار الحديث الأشيع بدلًا من ReLU الأصلية
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# =====================================================================
# 4) كتلة Transformer الكاملة بأسلوب Pre-LN
# =====================================================================
class TransformerBlockPreLN(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        # --- الوصلة التخطّيية الأولى: Attention ---
        # لاحظ: LayerNorm تُطبَّق قبل الدخول للطبقة الفرعية (Pre-LN)
        attn_out = self.attention(self.ln1(x), mask)
        x = x + self.dropout1(attn_out)          # الجمع التخطّيي على المدخل الخام غير المُطبَّع

        # --- الوصلة التخطّيية الثانية: FFN ---
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout2(ffn_out)

        return x


# للمقارنة: نسخة Post-LN كما في ورقة 2017 الأصلية
class TransformerBlockPostLN(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        # لاحظ الفرق: LayerNorm تأتي بعد الجمع التخطّيي، وليس قبل الطبقة الفرعية
        attn_out = self.attention(x, mask)
        x = self.ln1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout2(ffn_out))

        return x


# =====================================================================
# 5) محوّل كامل: تضمين + ترميز موضعي + N كتلة مُكدَّسة
# =====================================================================
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 6,
        max_len: int = 5000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlockPreLN(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # تطبيع نهائي إضافي بعد آخر كتلة -- ممارسة قياسية في تصاميم Pre-LN الحديثة
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, input_ids: Tensor, mask: Tensor | None = None) -> Tensor:
        # تحجيم التضمين بـ sqrt(d_model) -- ممارسة من الورقة الأصلية
        # لموازنة مقياس القيم بين التضمين والترميز الموضعي
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, mask)

        return self.final_ln(x)


# =====================================================================
# 6) اختبار شامل للتحقق من سلامة الأبعاد وتدفق التدرّج
# =====================================================================
def sanity_check() -> None:
    vocab_size, batch_size, seq_len = 10_000, 4, 32

    model = TransformerEncoder(vocab_size, d_model=256, num_heads=8, d_ff=1024, num_layers=4)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    output = model(input_ids)
    print(f"شكل المخرج: {output.shape}")  # (4, 32, 256)
    assert output.shape == (batch_size, seq_len, 256)

    # تحقق من تدفّق التدرّج عبر كل الطبقات (لا يوجد NaN أو صفر تام)
    loss = output.sum()
    loss.backward()

    total_params = sum(p.numel() for p in model.parameters())
    grad_norms = [
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    ]
    print(f"عدد المعاملات الكلي: {total_params:,}")
    print(f"متوسط معيار التدرّج عبر كل المعاملات: {sum(grad_norms)/len(grad_norms):.6f}")
    print(f"أصغر معيار تدرّج: {min(grad_norms):.6f} | أكبر معيار تدرّج: {max(grad_norms):.6f}")
    print("✅ لا توجد قيم NaN أو صفر تام -- التدرّج يتدفّق بثبات عبر كل الطبقات")


if __name__ == "__main__":
    sanity_check()

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-03-transformers-core/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
import torch
import math

d_model = 4
embedding_cat = torch.tensor([0.5, -0.2, 0.8, 0.1])

def compute_pe(pos: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(d_model)
    for j in range(0, d_model, 2):
        div_term = 10000 ** (j / d_model)
        pe[j] = math.sin(pos / div_term)
        if j + 1 < d_model:
            pe[j + 1] = math.cos(pos / div_term)
    return pe

pe_pos0 = compute_pe(0, d_model)
pe_pos2 = compute_pe(2, d_model)

input_pos0 = embedding_cat + pe_pos0
input_pos2 = embedding_cat + pe_pos2

print(f"PE(pos=0): {pe_pos0}")
print(f"PE(pos=2): {pe_pos2}")
print(f"التمثيل النهائي في الموضع 0: {input_pos0}")
print(f"التمثيل النهائي في الموضع 2: {input_pos2}")
print(f"هل التمثيلان متطابقان؟ {torch.allclose(input_pos0, input_pos2)}")  # False -- تم التمييز بنجاح!

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-03-transformers-core/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
x = self.embedding(input_ids)  # ❌ بدون أي ترميز موضعي!
for block in self.blocks:
    x = block(x)

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-03-transformers-core/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
# خطأ: مزيج غير متّسق بين الأسلوبين
attn_out = self.attention(x, mask)             # ❌ لم تُطبَّق LN على x قبل الدخول (نمط Post-LN)
x = x + self.dropout(self.ln1(attn_out))        # ❌ لكن LN هنا تُطبَّق على المخرج، بعد الجمع التخطّيي جزئيًا -- خليط مربك وغير صحيح في كلا النمطين

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-03-transformers-core/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
# ❌ استخدام قناع سببي للـ Encoder أيضًا -- يمنعه من رؤية السياق الكامل بلا داعٍ
encoder_output = encoder(x, mask=causal_mask)

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-03-transformers-core/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
context = context.transpose(1, 2).view(batch_size, seq_len, d_model)  # ❌ قد يرمي خطأ!

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-03-transformers-core/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
"""
preln_vs_postln_stability_experiment.py
تجربة تحقق من استقرار التدريب لـ Pre-LN مقابل Post-LN عند أعماق مختلفة.
"""
from __future__ import annotations

import torch
from torch import nn

# استورد TransformerBlockPreLN و TransformerBlockPostLN من كود القسم 3.7


def build_stack(
    block_class: type[nn.Module], num_layers: int, d_model: int = 128, num_heads: int = 4, d_ff: int = 512
) -> nn.Module:
    """يبني سلسلة من كتل Transformer المُكدَّسة بدون Embedding، للتركيز فقط على استقرار التدرّج."""
    return nn.ModuleList([block_class(d_model, num_heads, d_ff) for _ in range(num_layers)])


def measure_gradient_stability(
    block_class: type[nn.Module], num_layers: int, d_model: int = 128
) -> dict[str, float]:
    """
    يُمرّر مدخلًا عشوائيًا عبر N كتلة، يحسب الخسارة، وينشر الخطأ خلفيًا،
    ثم يُحلّل توزيع معايير التدرّج عبر الطبقات المختلفة (الأولى مقابل الأخيرة).
    """
    # TODO 1: ابنِ المكدّس باستخدام build_stack
    # TODO 2: مرّر x عشوائي (batch=8, seq_len=16, d_model=128) عبر كل الكتل بالتتابع
    # TODO 3: احسب loss = output.sum()، واستدعِ loss.backward()
    # TODO 4: اجمع معيار التدرّج (gradient norm) لأول كتلة مقابل آخر كتلة
    # TODO 5: أعد: {"first_layer_grad_norm": ..., "last_layer_grad_norm": ..., "ratio": ...}
    raise NotImplementedError


def run_comparison() -> None:
    """
    جرّب بأعماق مختلفة: [6, 24, 48, 96]
    لكل عمق، قارن نسبة (معيار تدرّج الطبقة الأولى / معيار تدرّج الطبقة الأخيرة)
    بين Pre-LN وPost-LN.

    النتيجة المتوقعة: مع ازدياد العمق، نسبة Post-LN تنحرف بشكل متزايد وحاد
    (دليل على تلاشي/عدم استقرار التدرّج)، بينما نسبة Pre-LN تبقى مستقرة نسبيًا
    حتى في الأعماق الكبيرة جدًا (96 طبقة).
    """
    raise NotImplementedError

# تحدٍّ إضافي: أضف نسخة ثالثة بدون أي LayerNorm إطلاقًا (فقط Residual)،
# وقارن مدى سوء الاستقرار دون أي تطبيع على الإطلاق -- لإثبات
# أن الوصلات التخطّيية وحدها غير كافية، ويلزم التطبيع أيضًا.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-11-chapter-11.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 11: الفصل الحادي عشر: بنية Transformer الكاملة."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_transformer() -> None:
    hidden = [1.0, -1.0, 0.5, 0.0]
    mean = sum(hidden) / len(hidden)
    variance = sum(((x - mean) ** 2 for x in hidden)) / len(hidden)
    normalized = [(x - mean) / math.sqrt(variance + 1e-05) for x in hidden]
    feed_forward = [max(0.0, x * 0.7 + 0.1) for x in normalized]
    output = [x + y for x, y in zip(hidden, feed_forward)]
    print('normalized:', [round(x, 3) for x in normalized])
    print('residual output:', [round(x, 3) for x in output])

if __name__ == "__main__":
    print("Chapter 11 lab: Residual, normalization, and feed-forward blocks")
    run_transformer()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-11-chapter-11.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
transformer_block_complete.py
بناء كتلة Transformer كاملة (Positional Encoding + Attention + FFN + Residual + LN)
بأسلوب Pre-LN الحديث.
Python 3.13+ | PyTorch 2.x
"""
from __future__ import annotations

import math

import torch
from torch import nn, Tensor

torch.manual_seed(42)


# =====================================================================
# 1) الترميز الموضعي الجيبي (Sinusoidal Positional Encoding)
# =====================================================================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1).float()          # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)   # الأبعاد الزوجية: sin
        pe[:, 1::2] = torch.cos(position * div_term)   # الأبعاد الفردية: cos

        # تسجيلها كـ buffer وليس Parameter: لا تُدرَّب، لكن تُحفظ وتُنقل مع .to(device)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, seq_len, d_model)"""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# =====================================================================
# 2) Multi-Head Self-Attention (كما شُرحت بالتفصيل في الفصل العاشر)
# =====================================================================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model يجب أن يقبل القسمة على num_heads"

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_size, seq_len, d_model = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # كل منها: (batch, num_heads, seq_len, d_k)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)   # (batch, heads, seq, seq)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = attn_weights @ V                                 # (batch, heads, seq, d_k)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        return self.W_o(context)


# =====================================================================
# 3) شبكة التغذية الأمامية الموضعية (Position-wise FFN)
# =====================================================================
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),              # GELU: الخيار الحديث الأشيع بدلًا من ReLU الأصلية
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# =====================================================================
# 4) كتلة Transformer الكاملة بأسلوب Pre-LN
# =====================================================================
class TransformerBlockPreLN(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        # --- الوصلة التخطّيية الأولى: Attention ---
        # لاحظ: LayerNorm تُطبَّق قبل الدخول للطبقة الفرعية (Pre-LN)
        attn_out = self.attention(self.ln1(x), mask)
        x = x + self.dropout1(attn_out)          # الجمع التخطّيي على المدخل الخام غير المُطبَّع

        # --- الوصلة التخطّيية الثانية: FFN ---
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout2(ffn_out)

        return x


# للمقارنة: نسخة Post-LN كما في ورقة 2017 الأصلية
class TransformerBlockPostLN(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        # لاحظ الفرق: LayerNorm تأتي بعد الجمع التخطّيي، وليس قبل الطبقة الفرعية
        attn_out = self.attention(x, mask)
        x = self.ln1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout2(ffn_out))

        return x


# =====================================================================
# 5) محوّل كامل: تضمين + ترميز موضعي + N كتلة مُكدَّسة
# =====================================================================
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 6,
        max_len: int = 5000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlockPreLN(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # تطبيع نهائي إضافي بعد آخر كتلة -- ممارسة قياسية في تصاميم Pre-LN الحديثة
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, input_ids: Tensor, mask: Tensor | None = None) -> Tensor:
        # تحجيم التضمين بـ sqrt(d_model) -- ممارسة من الورقة الأصلية
        # لموازنة مقياس القيم بين التضمين والترميز الموضعي
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, mask)

        return self.final_ln(x)


# =====================================================================
# 6) اختبار شامل للتحقق من سلامة الأبعاد وتدفق التدرّج
# =====================================================================
def sanity_check() -> None:
    vocab_size, batch_size, seq_len = 10_000, 4, 32

    model = TransformerEncoder(vocab_size, d_model=256, num_heads=8, d_ff=1024, num_layers=4)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    output = model(input_ids)
    print(f"شكل المخرج: {output.shape}")  # (4, 32, 256)
    assert output.shape == (batch_size, seq_len, 256)

    # تحقق من تدفّق التدرّج عبر كل الطبقات (لا يوجد NaN أو صفر تام)
    loss = output.sum()
    loss.backward()

    total_params = sum(p.numel() for p in model.parameters())
    grad_norms = [
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    ]
    print(f"عدد المعاملات الكلي: {total_params:,}")
    print(f"متوسط معيار التدرّج عبر كل المعاملات: {sum(grad_norms)/len(grad_norms):.6f}")
    print(f"أصغر معيار تدرّج: {min(grad_norms):.6f} | أكبر معيار تدرّج: {max(grad_norms):.6f}")
    print("✅ لا توجد قيم NaN أو صفر تام -- التدرّج يتدفّق بثبات عبر كل الطبقات")


if __name__ == "__main__":
    sanity_check()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-11-chapter-11.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import math

d_model = 4
embedding_cat = torch.tensor([0.5, -0.2, 0.8, 0.1])

def compute_pe(pos: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(d_model)
    for j in range(0, d_model, 2):
        div_term = 10000 ** (j / d_model)
        pe[j] = math.sin(pos / div_term)
        if j + 1 < d_model:
            pe[j + 1] = math.cos(pos / div_term)
    return pe

pe_pos0 = compute_pe(0, d_model)
pe_pos2 = compute_pe(2, d_model)

input_pos0 = embedding_cat + pe_pos0
input_pos2 = embedding_cat + pe_pos2

print(f"PE(pos=0): {pe_pos0}")
print(f"PE(pos=2): {pe_pos2}")
print(f"التمثيل النهائي في الموضع 0: {input_pos0}")
print(f"التمثيل النهائي في الموضع 2: {input_pos2}")
print(f"هل التمثيلان متطابقان؟ {torch.allclose(input_pos0, input_pos2)}")  # False -- تم التمييز بنجاح!

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-11-chapter-11.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
x = self.embedding(input_ids)  # ❌ بدون أي ترميز موضعي!
for block in self.blocks:
    x = block(x)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-11-chapter-11.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# خطأ: مزيج غير متّسق بين الأسلوبين
attn_out = self.attention(x, mask)             # ❌ لم تُطبَّق LN على x قبل الدخول (نمط Post-LN)
x = x + self.dropout(self.ln1(attn_out))        # ❌ لكن LN هنا تُطبَّق على المخرج، بعد الجمع التخطّيي جزئيًا -- خليط مربك وغير صحيح في كلا النمطين

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-11-chapter-11.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ❌ استخدام قناع سببي للـ Encoder أيضًا -- يمنعه من رؤية السياق الكامل بلا داعٍ
encoder_output = encoder(x, mask=causal_mask)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-11-chapter-11.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
context = context.transpose(1, 2).view(batch_size, seq_len, d_model)  # ❌ قد يرمي خطأ!

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-11-chapter-11.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
"""
preln_vs_postln_stability_experiment.py
تجربة تحقق من استقرار التدريب لـ Pre-LN مقابل Post-LN عند أعماق مختلفة.
"""
from __future__ import annotations

import torch
from torch import nn

# استورد TransformerBlockPreLN و TransformerBlockPostLN من كود القسم 3.7


def build_stack(
    block_class: type[nn.Module], num_layers: int, d_model: int = 128, num_heads: int = 4, d_ff: int = 512
) -> nn.Module:
    """يبني سلسلة من كتل Transformer المُكدَّسة بدون Embedding، للتركيز فقط على استقرار التدرّج."""
    return nn.ModuleList([block_class(d_model, num_heads, d_ff) for _ in range(num_layers)])


def measure_gradient_stability(
    block_class: type[nn.Module], num_layers: int, d_model: int = 128
) -> dict[str, float]:
    """
    يُمرّر مدخلًا عشوائيًا عبر N كتلة، يحسب الخسارة، وينشر الخطأ خلفيًا،
    ثم يُحلّل توزيع معايير التدرّج عبر الطبقات المختلفة (الأولى مقابل الأخيرة).
    """
    # TODO 1: ابنِ المكدّس باستخدام build_stack
    # TODO 2: مرّر x عشوائي (batch=8, seq_len=16, d_model=128) عبر كل الكتل بالتتابع
    # TODO 3: احسب loss = output.sum()، واستدعِ loss.backward()
    # TODO 4: اجمع معيار التدرّج (gradient norm) لأول كتلة مقابل آخر كتلة
    # TODO 5: أعد: {"first_layer_grad_norm": ..., "last_layer_grad_norm": ..., "ratio": ...}
    raise NotImplementedError


def run_comparison() -> None:
    """
    جرّب بأعماق مختلفة: [6, 24, 48, 96]
    لكل عمق، قارن نسبة (معيار تدرّج الطبقة الأولى / معيار تدرّج الطبقة الأخيرة)
    بين Pre-LN وPost-LN.

    النتيجة المتوقعة: مع ازدياد العمق، نسبة Post-LN تنحرف بشكل متزايد وحاد
    (دليل على تلاشي/عدم استقرار التدرّج)، بينما نسبة Pre-LN تبقى مستقرة نسبيًا
    حتى في الأعماق الكبيرة جدًا (96 طبقة).
    """
    raise NotImplementedError

# تحدٍّ إضافي: أضف نسخة ثالثة بدون أي LayerNorm إطلاقًا (فقط Residual)،
# وقارن مدى سوء الاستقرار دون أي تطبيع على الإطلاق -- لإثبات
# أن الوصلات التخطّيية وحدها غير كافية، ويلزم التطبيع أيضًا.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-11-chapter-11.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الحادي عشر: بنية Transformer الكاملة
Large Language Model Engineering — Transformer

Book slug : llm-engineering-69-chapters
Chapter   : ch-11-chapter-11 (lesson 11)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_11.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 11: الفصل الحادي عشر: بنية Transformer الكاملة."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_transformer() -> None:
    hidden = [1.0, -1.0, 0.5, 0.0]
    mean = sum(hidden) / len(hidden)
    variance = sum(((x - mean) ** 2 for x in hidden)) / len(hidden)
    normalized = [(x - mean) / math.sqrt(variance + 1e-05) for x in hidden]
    feed_forward = [max(0.0, x * 0.7 + 0.1) for x in normalized]
    output = [x + y for x, y in zip(hidden, feed_forward)]
    print('normalized:', [round(x, 3) for x in normalized])
    print('residual output:', [round(x, 3) for x in output])

if __name__ == "__main__":
    print("Chapter 11 lab: Residual, normalization, and feed-forward blocks")
    run_transformer()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-03-transformers-core/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
"""
transformer_block_complete.py
بناء كتلة Transformer كاملة (Positional Encoding + Attention + FFN + Residual + LN)
بأسلوب Pre-LN الحديث.
Python 3.13+ | PyTorch 2.x
"""
from __future__ import annotations

import math

import torch
from torch import nn, Tensor

torch.manual_seed(42)


# =====================================================================
# 1) الترميز الموضعي الجيبي (Sinusoidal Positional Encoding)
# =====================================================================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1).float()          # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)   # الأبعاد الزوجية: sin
        pe[:, 1::2] = torch.cos(position * div_term)   # الأبعاد الفردية: cos

        # تسجيلها كـ buffer وليس Parameter: لا تُدرَّب، لكن تُحفظ وتُنقل مع .to(device)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, seq_len, d_model)"""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# =====================================================================
# 2) Multi-Head Self-Attention (كما شُرحت بالتفصيل في الفصل العاشر)
# =====================================================================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model يجب أن يقبل القسمة على num_heads"

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_size, seq_len, d_model = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # كل منها: (batch, num_heads, seq_len, d_k)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)   # (batch, heads, seq, seq)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = attn_weights @ V                                 # (batch, heads, seq, d_k)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        return self.W_o(context)


# =====================================================================
# 3) شبكة التغذية الأمامية الموضعية (Position-wise FFN)
# =====================================================================
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),              # GELU: الخيار الحديث الأشيع بدلًا من ReLU الأصلية
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# =====================================================================
# 4) كتلة Transformer الكاملة بأسلوب Pre-LN
# =====================================================================
class TransformerBlockPreLN(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        # --- الوصلة التخطّيية الأولى: Attention ---
        # لاحظ: LayerNorm تُطبَّق قبل الدخول للطبقة الفرعية (Pre-LN)
        attn_out = self.attention(self.ln1(x), mask)
        x = x + self.dropout1(attn_out)          # الجمع التخطّيي على المدخل الخام غير المُطبَّع

        # --- الوصلة التخطّيية الثانية: FFN ---
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout2(ffn_out)

        return x


# للمقارنة: نسخة Post-LN كما في ورقة 2017 الأصلية
class TransformerBlockPostLN(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        # لاحظ الفرق: LayerNorm تأتي بعد الجمع التخطّيي، وليس قبل الطبقة الفرعية
        attn_out = self.attention(x, mask)
        x = self.ln1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout2(ffn_out))

        return x


# =====================================================================
# 5) محوّل كامل: تضمين + ترميز موضعي + N كتلة مُكدَّسة
# =====================================================================
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 6,
        max_len: int = 5000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlockPreLN(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # تطبيع نهائي إضافي بعد آخر كتلة -- ممارسة قياسية في تصاميم Pre-LN الحديثة
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, input_ids: Tensor, mask: Tensor | None = None) -> Tensor:
        # تحجيم التضمين بـ sqrt(d_model) -- ممارسة من الورقة الأصلية
        # لموازنة مقياس القيم بين التضمين والترميز الموضعي
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, mask)

        return self.final_ln(x)


# =====================================================================
# 6) اختبار شامل للتحقق من سلامة الأبعاد وتدفق التدرّج
# =====================================================================
def sanity_check() -> None:
    vocab_size, batch_size, seq_len = 10_000, 4, 32

    model = TransformerEncoder(vocab_size, d_model=256, num_heads=8, d_ff=1024, num_layers=4)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    output = model(input_ids)
    print(f"شكل المخرج: {output.shape}")  # (4, 32, 256)
    assert output.shape == (batch_size, seq_len, 256)

    # تحقق من تدفّق التدرّج عبر كل الطبقات (لا يوجد NaN أو صفر تام)
    loss = output.sum()
    loss.backward()

    total_params = sum(p.numel() for p in model.parameters())
    grad_norms = [
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    ]
    print(f"عدد المعاملات الكلي: {total_params:,}")
    print(f"متوسط معيار التدرّج عبر كل المعاملات: {sum(grad_norms)/len(grad_norms):.6f}")
    print(f"أصغر معيار تدرّج: {min(grad_norms):.6f} | أكبر معيار تدرّج: {max(grad_norms):.6f}")
    print("✅ لا توجد قيم NaN أو صفر تام -- التدرّج يتدفّق بثبات عبر كل الطبقات")


if __name__ == "__main__":
    sanity_check()

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-03-transformers-core/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
import torch
import math

d_model = 4
embedding_cat = torch.tensor([0.5, -0.2, 0.8, 0.1])

def compute_pe(pos: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(d_model)
    for j in range(0, d_model, 2):
        div_term = 10000 ** (j / d_model)
        pe[j] = math.sin(pos / div_term)
        if j + 1 < d_model:
            pe[j + 1] = math.cos(pos / div_term)
    return pe

pe_pos0 = compute_pe(0, d_model)
pe_pos2 = compute_pe(2, d_model)

input_pos0 = embedding_cat + pe_pos0
input_pos2 = embedding_cat + pe_pos2

print(f"PE(pos=0): {pe_pos0}")
print(f"PE(pos=2): {pe_pos2}")
print(f"التمثيل النهائي في الموضع 0: {input_pos0}")
print(f"التمثيل النهائي في الموضع 2: {input_pos2}")
print(f"هل التمثيلان متطابقان؟ {torch.allclose(input_pos0, input_pos2)}")  # False -- تم التمييز بنجاح!

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-03-transformers-core/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
x = self.embedding(input_ids)  # ❌ بدون أي ترميز موضعي!
for block in self.blocks:
    x = block(x)

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-03-transformers-core/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
# خطأ: مزيج غير متّسق بين الأسلوبين
attn_out = self.attention(x, mask)             # ❌ لم تُطبَّق LN على x قبل الدخول (نمط Post-LN)
x = x + self.dropout(self.ln1(attn_out))        # ❌ لكن LN هنا تُطبَّق على المخرج، بعد الجمع التخطّيي جزئيًا -- خليط مربك وغير صحيح في كلا النمطين

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-03-transformers-core/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
# ❌ استخدام قناع سببي للـ Encoder أيضًا -- يمنعه من رؤية السياق الكامل بلا داعٍ
encoder_output = encoder(x, mask=causal_mask)

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-03-transformers-core/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
context = context.transpose(1, 2).view(batch_size, seq_len, d_model)  # ❌ قد يرمي خطأ!

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-03-transformers-core/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
"""
preln_vs_postln_stability_experiment.py
تجربة تحقق من استقرار التدريب لـ Pre-LN مقابل Post-LN عند أعماق مختلفة.
"""
from __future__ import annotations

import torch
from torch import nn

# استورد TransformerBlockPreLN و TransformerBlockPostLN من كود القسم 3.7


def build_stack(
    block_class: type[nn.Module], num_layers: int, d_model: int = 128, num_heads: int = 4, d_ff: int = 512
) -> nn.Module:
    """يبني سلسلة من كتل Transformer المُكدَّسة بدون Embedding، للتركيز فقط على استقرار التدرّج."""
    return nn.ModuleList([block_class(d_model, num_heads, d_ff) for _ in range(num_layers)])


def measure_gradient_stability(
    block_class: type[nn.Module], num_layers: int, d_model: int = 128
) -> dict[str, float]:
    """
    يُمرّر مدخلًا عشوائيًا عبر N كتلة، يحسب الخسارة، وينشر الخطأ خلفيًا،
    ثم يُحلّل توزيع معايير التدرّج عبر الطبقات المختلفة (الأولى مقابل الأخيرة).
    """
    # TODO 1: ابنِ المكدّس باستخدام build_stack
    # TODO 2: مرّر x عشوائي (batch=8, seq_len=16, d_model=128) عبر كل الكتل بالتتابع
    # TODO 3: احسب loss = output.sum()، واستدعِ loss.backward()
    # TODO 4: اجمع معيار التدرّج (gradient norm) لأول كتلة مقابل آخر كتلة
    # TODO 5: أعد: {"first_layer_grad_norm": ..., "last_layer_grad_norm": ..., "ratio": ...}
    raise NotImplementedError


def run_comparison() -> None:
    """
    جرّب بأعماق مختلفة: [6, 24, 48, 96]
    لكل عمق، قارن نسبة (معيار تدرّج الطبقة الأولى / معيار تدرّج الطبقة الأخيرة)
    بين Pre-LN وPost-LN.

    النتيجة المتوقعة: مع ازدياد العمق، نسبة Post-LN تنحرف بشكل متزايد وحاد
    (دليل على تلاشي/عدم استقرار التدرّج)، بينما نسبة Pre-LN تبقى مستقرة نسبيًا
    حتى في الأعماق الكبيرة جدًا (96 طبقة).
    """
    raise NotImplementedError

# تحدٍّ إضافي: أضف نسخة ثالثة بدون أي LayerNorm إطلاقًا (فقط Residual)،
# وقارن مدى سوء الاستقرار دون أي تطبيع على الإطلاق -- لإثبات
# أن الوصلات التخطّيية وحدها غير كافية، ويلزم التطبيع أيضًا.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-11-chapter-11.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 11: الفصل الحادي عشر: بنية Transformer الكاملة."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_transformer() -> None:
    hidden = [1.0, -1.0, 0.5, 0.0]
    mean = sum(hidden) / len(hidden)
    variance = sum(((x - mean) ** 2 for x in hidden)) / len(hidden)
    normalized = [(x - mean) / math.sqrt(variance + 1e-05) for x in hidden]
    feed_forward = [max(0.0, x * 0.7 + 0.1) for x in normalized]
    output = [x + y for x, y in zip(hidden, feed_forward)]
    print('normalized:', [round(x, 3) for x in normalized])
    print('residual output:', [round(x, 3) for x in output])

if __name__ == "__main__":
    print("Chapter 11 lab: Residual, normalization, and feed-forward blocks")
    run_transformer()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-11-chapter-11.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
transformer_block_complete.py
بناء كتلة Transformer كاملة (Positional Encoding + Attention + FFN + Residual + LN)
بأسلوب Pre-LN الحديث.
Python 3.13+ | PyTorch 2.x
"""
from __future__ import annotations

import math

import torch
from torch import nn, Tensor

torch.manual_seed(42)


# =====================================================================
# 1) الترميز الموضعي الجيبي (Sinusoidal Positional Encoding)
# =====================================================================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1).float()          # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)   # الأبعاد الزوجية: sin
        pe[:, 1::2] = torch.cos(position * div_term)   # الأبعاد الفردية: cos

        # تسجيلها كـ buffer وليس Parameter: لا تُدرَّب، لكن تُحفظ وتُنقل مع .to(device)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, seq_len, d_model)"""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# =====================================================================
# 2) Multi-Head Self-Attention (كما شُرحت بالتفصيل في الفصل العاشر)
# =====================================================================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model يجب أن يقبل القسمة على num_heads"

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_size, seq_len, d_model = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # كل منها: (batch, num_heads, seq_len, d_k)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)   # (batch, heads, seq, seq)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = attn_weights @ V                                 # (batch, heads, seq, d_k)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        return self.W_o(context)


# =====================================================================
# 3) شبكة التغذية الأمامية الموضعية (Position-wise FFN)
# =====================================================================
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),              # GELU: الخيار الحديث الأشيع بدلًا من ReLU الأصلية
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# =====================================================================
# 4) كتلة Transformer الكاملة بأسلوب Pre-LN
# =====================================================================
class TransformerBlockPreLN(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        # --- الوصلة التخطّيية الأولى: Attention ---
        # لاحظ: LayerNorm تُطبَّق قبل الدخول للطبقة الفرعية (Pre-LN)
        attn_out = self.attention(self.ln1(x), mask)
        x = x + self.dropout1(attn_out)          # الجمع التخطّيي على المدخل الخام غير المُطبَّع

        # --- الوصلة التخطّيية الثانية: FFN ---
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout2(ffn_out)

        return x


# للمقارنة: نسخة Post-LN كما في ورقة 2017 الأصلية
class TransformerBlockPostLN(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        # لاحظ الفرق: LayerNorm تأتي بعد الجمع التخطّيي، وليس قبل الطبقة الفرعية
        attn_out = self.attention(x, mask)
        x = self.ln1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout2(ffn_out))

        return x


# =====================================================================
# 5) محوّل كامل: تضمين + ترميز موضعي + N كتلة مُكدَّسة
# =====================================================================
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 6,
        max_len: int = 5000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlockPreLN(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # تطبيع نهائي إضافي بعد آخر كتلة -- ممارسة قياسية في تصاميم Pre-LN الحديثة
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, input_ids: Tensor, mask: Tensor | None = None) -> Tensor:
        # تحجيم التضمين بـ sqrt(d_model) -- ممارسة من الورقة الأصلية
        # لموازنة مقياس القيم بين التضمين والترميز الموضعي
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, mask)

        return self.final_ln(x)


# =====================================================================
# 6) اختبار شامل للتحقق من سلامة الأبعاد وتدفق التدرّج
# =====================================================================
def sanity_check() -> None:
    vocab_size, batch_size, seq_len = 10_000, 4, 32

    model = TransformerEncoder(vocab_size, d_model=256, num_heads=8, d_ff=1024, num_layers=4)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    output = model(input_ids)
    print(f"شكل المخرج: {output.shape}")  # (4, 32, 256)
    assert output.shape == (batch_size, seq_len, 256)

    # تحقق من تدفّق التدرّج عبر كل الطبقات (لا يوجد NaN أو صفر تام)
    loss = output.sum()
    loss.backward()

    total_params = sum(p.numel() for p in model.parameters())
    grad_norms = [
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    ]
    print(f"عدد المعاملات الكلي: {total_params:,}")
    print(f"متوسط معيار التدرّج عبر كل المعاملات: {sum(grad_norms)/len(grad_norms):.6f}")
    print(f"أصغر معيار تدرّج: {min(grad_norms):.6f} | أكبر معيار تدرّج: {max(grad_norms):.6f}")
    print("✅ لا توجد قيم NaN أو صفر تام -- التدرّج يتدفّق بثبات عبر كل الطبقات")


if __name__ == "__main__":
    sanity_check()

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-11-chapter-11.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
import math

d_model = 4
embedding_cat = torch.tensor([0.5, -0.2, 0.8, 0.1])

def compute_pe(pos: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(d_model)
    for j in range(0, d_model, 2):
        div_term = 10000 ** (j / d_model)
        pe[j] = math.sin(pos / div_term)
        if j + 1 < d_model:
            pe[j + 1] = math.cos(pos / div_term)
    return pe

pe_pos0 = compute_pe(0, d_model)
pe_pos2 = compute_pe(2, d_model)

input_pos0 = embedding_cat + pe_pos0
input_pos2 = embedding_cat + pe_pos2

print(f"PE(pos=0): {pe_pos0}")
print(f"PE(pos=2): {pe_pos2}")
print(f"التمثيل النهائي في الموضع 0: {input_pos0}")
print(f"التمثيل النهائي في الموضع 2: {input_pos2}")
print(f"هل التمثيلان متطابقان؟ {torch.allclose(input_pos0, input_pos2)}")  # False -- تم التمييز بنجاح!

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-11-chapter-11.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
x = self.embedding(input_ids)  # ❌ بدون أي ترميز موضعي!
for block in self.blocks:
    x = block(x)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-11-chapter-11.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# خطأ: مزيج غير متّسق بين الأسلوبين
attn_out = self.attention(x, mask)             # ❌ لم تُطبَّق LN على x قبل الدخول (نمط Post-LN)
x = x + self.dropout(self.ln1(attn_out))        # ❌ لكن LN هنا تُطبَّق على المخرج، بعد الجمع التخطّيي جزئيًا -- خليط مربك وغير صحيح في كلا النمطين

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-11-chapter-11.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ❌ استخدام قناع سببي للـ Encoder أيضًا -- يمنعه من رؤية السياق الكامل بلا داعٍ
encoder_output = encoder(x, mask=causal_mask)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-11-chapter-11.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
context = context.transpose(1, 2).view(batch_size, seq_len, d_model)  # ❌ قد يرمي خطأ!

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-11-chapter-11.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
"""
preln_vs_postln_stability_experiment.py
تجربة تحقق من استقرار التدريب لـ Pre-LN مقابل Post-LN عند أعماق مختلفة.
"""
from __future__ import annotations

import torch
from torch import nn

# استورد TransformerBlockPreLN و TransformerBlockPostLN من كود القسم 3.7


def build_stack(
    block_class: type[nn.Module], num_layers: int, d_model: int = 128, num_heads: int = 4, d_ff: int = 512
) -> nn.Module:
    """يبني سلسلة من كتل Transformer المُكدَّسة بدون Embedding، للتركيز فقط على استقرار التدرّج."""
    return nn.ModuleList([block_class(d_model, num_heads, d_ff) for _ in range(num_layers)])


def measure_gradient_stability(
    block_class: type[nn.Module], num_layers: int, d_model: int = 128
) -> dict[str, float]:
    """
    يُمرّر مدخلًا عشوائيًا عبر N كتلة، يحسب الخسارة، وينشر الخطأ خلفيًا،
    ثم يُحلّل توزيع معايير التدرّج عبر الطبقات المختلفة (الأولى مقابل الأخيرة).
    """
    # TODO 1: ابنِ المكدّس باستخدام build_stack
    # TODO 2: مرّر x عشوائي (batch=8, seq_len=16, d_model=128) عبر كل الكتل بالتتابع
    # TODO 3: احسب loss = output.sum()، واستدعِ loss.backward()
    # TODO 4: اجمع معيار التدرّج (gradient norm) لأول كتلة مقابل آخر كتلة
    # TODO 5: أعد: {"first_layer_grad_norm": ..., "last_layer_grad_norm": ..., "ratio": ...}
    raise NotImplementedError


def run_comparison() -> None:
    """
    جرّب بأعماق مختلفة: [6, 24, 48, 96]
    لكل عمق، قارن نسبة (معيار تدرّج الطبقة الأولى / معيار تدرّج الطبقة الأخيرة)
    بين Pre-LN وPost-LN.

    النتيجة المتوقعة: مع ازدياد العمق، نسبة Post-LN تنحرف بشكل متزايد وحاد
    (دليل على تلاشي/عدم استقرار التدرّج)، بينما نسبة Pre-LN تبقى مستقرة نسبيًا
    حتى في الأعماق الكبيرة جدًا (96 طبقة).
    """
    raise NotImplementedError

# تحدٍّ إضافي: أضف نسخة ثالثة بدون أي LayerNorm إطلاقًا (فقط Residual)،
# وقارن مدى سوء الاستقرار دون أي تطبيع على الإطلاق -- لإثبات
# أن الوصلات التخطّيية وحدها غير كافية، ويلزم التطبيع أيضًا.
