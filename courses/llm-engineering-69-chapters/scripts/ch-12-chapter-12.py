#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الثاني عشر: التنويعات المعمارية — Encoder-Only، Decoder-Only، Encoder-Decoder
Large Language Model Engineering — Encoder-Only

Book slug : llm-engineering-69-chapters
Chapter   : ch-12-chapter-12 (lesson 12)
Code files: 13 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-12-chapter-12.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الثاني عشر: التنويعات المعمارية — Encoder-Only، Decoder-Only، Encoder-Decoder
Large Language Model Engineering — Encoder-Only

Book slug : llm-engineering-69-chapters
Chapter   : ch-12-chapter-12 (lesson 12)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-12-chapter-12.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الثاني عشر: التنويعات المعمارية — Encoder-Only، Decoder-Only، Encoder-Decoder
Large Language Model Engineering — Encoder-Only

Book slug : llm-engineering-69-chapters
Chapter   : ch-12-chapter-12 (lesson 12)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_12.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 12: الفصل الثاني عشر: التنويعات المعمارية — Encoder-Only، Decoder-Only، Encoder-Decoder."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_masks() -> None:
    length = 5
    causal = [[int(key <= query) for key in range(length)] for query in range(length)]
    padding = [1, 1, 1, 0, 0]
    print('causal mask:')
    for row in causal:
        print(' ', row)
    print('padding-valid positions:', padding)
    print('encoder-decoder mask shape:', (length, 3))

if __name__ == "__main__":
    print("Chapter 12 lab: Encoder, decoder, and cross-attention masks")
    run_masks()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-03-transformers-core/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
# architecture_variants.py
"""
تنفيذ كامل للتنويعات المعمارية الثلاث على Transformer:
Encoder-Only (BERT-style), Decoder-Only (GPT-style), Encoder-Decoder (T5-style)
Python 3.13+, PyTorch 2.x
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


# ---------------------------------------------------------------------------
# 1) اللبنات المشتركة (معاد استخدامها من الفصل 10/11 مع توسيع للـ Cross-Attention)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    انتباه متعدد الرؤوس عام يدعم:
      - Self-Attention (عندما kv_input is None)
      - Cross-Attention (عندما kv_input يأتي من مصدر آخر، مثل مخرجات Encoder)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model يجب أن يقبل القسمة على n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, n_heads, seq_len, d_k)

    def forward(
        self,
        query_input: torch.Tensor,
        kv_input: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if kv_input is None:
            kv_input = query_input  # Self-Attention

        batch, q_len, _ = query_input.shape
        q = self._split_heads(self.w_q(query_input))
        k = self._split_heads(self.w_k(kv_input))
        v = self._split_heads(self.w_v(kv_input))

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        context = attn_weights @ v  # (batch, n_heads, q_len, d_k)

        context = context.transpose(1, 2).contiguous().view(batch, q_len, self.d_model)
        return self.w_o(context)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """قناع سببي: مثلث سفلي = 1 (مسموح)، مثلث علوي = 0 (ممنوع)."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len) للبث على batch/heads


def make_padding_mask(pad_mask_1d: torch.Tensor) -> torch.Tensor:
    """
    pad_mask_1d: (batch, seq_len) حيث 1 = رمز حقيقي، 0 = حشو (padding)
    نعيد (batch, 1, 1, seq_len) للبث على رؤوس الانتباه وكل استعلام.
    """
    return pad_mask_1d.unsqueeze(1).unsqueeze(2)


# ---------------------------------------------------------------------------
# 2) BERT-style Block (Encoder-Only): بلا قناع سببي إطلاقًا
# ---------------------------------------------------------------------------

class BERTBlock(nn.Module):
    """كتلة ثنائية الاتجاه بالكامل — Pre-LN كما اتفقنا في الفصل 11."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # ملاحظة جوهرية: لا يوجد أي causal mask هنا — فقط padding_mask إن وُجد
        attn_out = self.self_attn(self.ln1(x), mask=padding_mask)
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)
        return x


# ---------------------------------------------------------------------------
# 3) GPT-style Block (Decoder-Only): قناع سببي إلزامي
# ---------------------------------------------------------------------------

class GPTBlock(nn.Module):
    """كتلة أحادية الاتجاه — القناع السببي هو الفارق الوحيد بنيويًا عن BERTBlock."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.causal_self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.causal_self_attn(self.ln1(x), mask=causal_mask)
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)
        return x


# ---------------------------------------------------------------------------
# 4) T5-style Block (Encoder-Decoder): ثلاث طبقات انتباه فعليًا في الـ Decoder
# ---------------------------------------------------------------------------

class T5EncoderBlock(nn.Module):
    """مطابقة تمامًا لـ BERTBlock — الـ Encoder في T5 ثنائي الاتجاه كذلك."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = BERTBlock(d_model, n_heads, d_ff, dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.block(x, padding_mask)


class T5DecoderBlock(nn.Module):
    """
    ثلاث طبقات فرعية:
      1. Masked Self-Attention (سببي، على مخرجات الـ Decoder حتى الآن)
      2. Cross-Attention (Q من الـ Decoder، K/V من مخرجات الـ Encoder)
      3. Feed-Forward
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.masked_self_attn = MultiHeadAttention(d_model, n_heads, dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)

        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        causal_mask: torch.Tensor,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1) Masked self-attention داخل الـ decoder فقط
        self_attn_out = self.masked_self_attn(self.ln1(x), mask=causal_mask)
        x = x + self.dropout(self_attn_out)

        # 2) Cross-attention: الاستعلامات من x (decoder)، K/V من الـ encoder
        cross_attn_out = self.cross_attn(
            query_input=self.ln2(x), kv_input=encoder_output, mask=cross_mask
        )
        x = x + self.dropout(cross_attn_out)

        # 3) FFN كالمعتاد
        ffn_out = self.ffn(self.ln3(x))
        x = x + self.dropout(ffn_out)
        return x


# ---------------------------------------------------------------------------
# 5) تحقّق سريع: هل القناع فعليًا يمنع رؤية المستقبل؟
# ---------------------------------------------------------------------------

def sanity_check_masks() -> None:
    device = torch.device("cpu")
    seq_len = 5
    causal = make_causal_mask(seq_len, device).squeeze()
    print("قناع GPT السببي (1=مسموح، 0=ممنوع):")
    print(causal.int())
    # تحقق: العنصر [0, 4] يجب أن يكون 0 (الرمز الأول لا يرى الرمز الخامس)
    assert causal[0, 4].item() == 0, "خطأ: الرمز الأول يرى المستقبل!"
    assert causal[4, 0].item() == 1, "خطأ: الرمز الأخير يجب أن يرى الماضي كله"
    print("✅ القناع السببي يعمل بشكل صحيح: لا تسريب من المستقبل.\n")


if __name__ == "__main__":
    sanity_check_masks()

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-03-transformers-core/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
قناع GPT السببي (1=مسموح، 0=ممنوع):
tensor([[1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1]], dtype=torch.int32)
✅ القناع السببي يعمل بشكل صحيح: لا تسريب من المستقبل.

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-03-transformers-core/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
# training_objectives_demo.py
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(seed=7)

sentence = ["اذهب", "إلى", "المتجر", "واشترِ", "خمسة", "تفاحات"]


def bert_mlm_example(tokens: list[str]) -> None:
    """محاكاة MLM: اختر 15% من الرموز، طبّق قاعدة 80/10/10."""
    n_to_mask = max(1, round(0.15 * len(tokens)))
    mask_indices = rng.choice(len(tokens), size=n_to_mask, replace=False)

    masked_tokens = tokens.copy()
    for idx in mask_indices:
        roll = rng.random()
        if roll < 0.8:
            masked_tokens[idx] = "[MASK]"
        elif roll < 0.9:
            masked_tokens[idx] = rng.choice(tokens)  # استبدال عشوائي
        # وإلا: يبقى كما هو (10% المتبقية)

    print("BERT — مدخل MLM     :", " ".join(masked_tokens))
    print("BERT — الهدف (كامل) :", " ".join(tokens), "\n")


def gpt_causal_lm_example(tokens: list[str]) -> None:
    """محاكاة Next-Token Prediction: كل موضع يتنبأ بالتالي مباشرة."""
    print("GPT — تسلسل التدريب (مدخل → هدف لكل خطوة):")
    for t in range(len(tokens) - 1):
        context = " ".join(tokens[: t + 1])
        target = tokens[t + 1]
        print(f"   السياق: '{context}'  →  الهدف التالي: '{target}'")
    print()


def t5_span_corruption_example(tokens: list[str]) -> None:
    """محاكاة Span Corruption: إتلاف نطاق متجاور واحد أو أكثر."""
    span_start, span_len = 2, 2  # إتلاف "المتجر واشترِ"
    sentinel = "<X>"

    corrupted_input = (
        tokens[:span_start] + [sentinel] + tokens[span_start + span_len :]
    )
    target_output = [sentinel] + tokens[span_start : span_start + span_len] + ["<Z>"]

    print("T5 — مدخل Encoder (مُتلَف):", " ".join(corrupted_input))
    print("T5 — هدف Decoder (للاستعادة):", " ".join(target_output), "\n")


if __name__ == "__main__":
    bert_mlm_example(sentence)
    gpt_causal_lm_example(sentence)
    t5_span_corruption_example(sentence)

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-03-transformers-core/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ مفاهيمي: BERT غير مُصمم للتوليد التسلسلي الحر
output = bert_model.generate(prompt)  # لا توجد دالة generate منطقية هنا أصلاً

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-03-transformers-core/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: استخدم GPT-style أو T5-style decoder للتوليد
output_ids = gpt_model.generate(input_ids, max_new_tokens=50)

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-03-transformers-core/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ خطير جدًا وصامت: لا قناع إطلاقًا
attn_out = self.self_attn(x)  # بلا mask — تسريب كامل من المستقبل!

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-03-transformers-core/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح
causal_mask = make_causal_mask(seq_len, x.device)
attn_out = self.causal_self_attn(x, mask=causal_mask)

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-03-transformers-core/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: استخدام الـ decoder نفسه كمصدر K/V بدلاً من مخرجات الـ encoder
cross_attn_out = self.cross_attn(query_input=x, kv_input=x)  # هذا Self-Attention وليس Cross!

# ────────────────────────────────────────────────────────────────────────
# [010] book/chapters/part-03-transformers-core/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح
cross_attn_out = self.cross_attn(query_input=x, kv_input=encoder_output)

# ────────────────────────────────────────────────────────────────────────
# [011] book/chapters/part-03-transformers-core/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
# mini_project_toy_t5.py
from __future__ import annotations

import torch
import torch.nn as nn

from architecture_variants import (
    T5EncoderBlock,
    T5DecoderBlock,
    make_causal_mask,
)

VOCAB_SIZE = 12          # أرقام 0-9 + <BOS> + <EOS>
D_MODEL = 32
N_HEADS = 4
D_FF = 64
N_LAYERS = 2
MAX_LEN = 8


class ToyT5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(MAX_LEN, D_MODEL)

        self.encoder_blocks = nn.ModuleList(
            [T5EncoderBlock(D_MODEL, N_HEADS, D_FF) for _ in range(N_LAYERS)]
        )
        self.decoder_blocks = nn.ModuleList(
            [T5DecoderBlock(D_MODEL, N_HEADS, D_FF) for _ in range(N_LAYERS)]
        )
        self.output_head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(ids.size(1), device=ids.device)
        return self.token_emb(ids) + self.pos_emb(positions)

    def encode(self, src_ids: torch.Tensor) -> torch.Tensor:
        x = self._embed(src_ids)
        for block in self.encoder_blocks:
            x = block(x)
        return x

    def decode(self, tgt_ids: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        x = self._embed(tgt_ids)
        causal_mask = make_causal_mask(tgt_ids.size(1), tgt_ids.device)
        for block in self.decoder_blocks:
            x = block(x, encoder_output, causal_mask)
        return self.output_head(x)

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        encoder_output = self.encode(src_ids)
        return self.decode(tgt_ids, encoder_output)


# --- TODO لإكمال التمرين ---
# 1) اكتب دالة توليد بيانات: تسلسل عشوائي من الأرقام 0-9 وعكسه (أضف <BOS>=10, <EOS>=11)
# 2) اكتب حلقة تدريب باستخدام CrossEntropyLoss (تجاهل الحشو إن وُجد)
# 3) اكتب دالة استدلال Autoregressive: ابدأ بـ <BOS>، وَلِّد رمزًا رمزًا حتى <EOS>
# 4) قيّم الدقة: هل ينجح النموذج في عكس تسلسلات لم يرها أثناء التدريب؟
# 5) تجربة إضافية: احذف Cross-Attention بالكامل (استبدلها بتمرير صفري) وشاهد كيف ينهار الأداء تمامًا

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-12-chapter-12.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 12: الفصل الثاني عشر: التنويعات المعمارية — Encoder-Only، Decoder-Only، Encoder-Decoder."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_masks() -> None:
    length = 5
    causal = [[int(key <= query) for key in range(length)] for query in range(length)]
    padding = [1, 1, 1, 0, 0]
    print('causal mask:')
    for row in causal:
        print(' ', row)
    print('padding-valid positions:', padding)
    print('encoder-decoder mask shape:', (length, 3))

if __name__ == "__main__":
    print("Chapter 12 lab: Encoder, decoder, and cross-attention masks")
    run_masks()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-12-chapter-12.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# architecture_variants.py
"""
تنفيذ كامل للتنويعات المعمارية الثلاث على Transformer:
Encoder-Only (BERT-style), Decoder-Only (GPT-style), Encoder-Decoder (T5-style)
Python 3.13+, PyTorch 2.x
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


# ---------------------------------------------------------------------------
# 1) اللبنات المشتركة (معاد استخدامها من الفصل 10/11 مع توسيع للـ Cross-Attention)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    انتباه متعدد الرؤوس عام يدعم:
      - Self-Attention (عندما kv_input is None)
      - Cross-Attention (عندما kv_input يأتي من مصدر آخر، مثل مخرجات Encoder)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model يجب أن يقبل القسمة على n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, n_heads, seq_len, d_k)

    def forward(
        self,
        query_input: torch.Tensor,
        kv_input: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if kv_input is None:
            kv_input = query_input  # Self-Attention

        batch, q_len, _ = query_input.shape
        q = self._split_heads(self.w_q(query_input))
        k = self._split_heads(self.w_k(kv_input))
        v = self._split_heads(self.w_v(kv_input))

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        context = attn_weights @ v  # (batch, n_heads, q_len, d_k)

        context = context.transpose(1, 2).contiguous().view(batch, q_len, self.d_model)
        return self.w_o(context)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """قناع سببي: مثلث سفلي = 1 (مسموح)، مثلث علوي = 0 (ممنوع)."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len) للبث على batch/heads


def make_padding_mask(pad_mask_1d: torch.Tensor) -> torch.Tensor:
    """
    pad_mask_1d: (batch, seq_len) حيث 1 = رمز حقيقي، 0 = حشو (padding)
    نعيد (batch, 1, 1, seq_len) للبث على رؤوس الانتباه وكل استعلام.
    """
    return pad_mask_1d.unsqueeze(1).unsqueeze(2)


# ---------------------------------------------------------------------------
# 2) BERT-style Block (Encoder-Only): بلا قناع سببي إطلاقًا
# ---------------------------------------------------------------------------

class BERTBlock(nn.Module):
    """كتلة ثنائية الاتجاه بالكامل — Pre-LN كما اتفقنا في الفصل 11."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # ملاحظة جوهرية: لا يوجد أي causal mask هنا — فقط padding_mask إن وُجد
        attn_out = self.self_attn(self.ln1(x), mask=padding_mask)
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)
        return x


# ---------------------------------------------------------------------------
# 3) GPT-style Block (Decoder-Only): قناع سببي إلزامي
# ---------------------------------------------------------------------------

class GPTBlock(nn.Module):
    """كتلة أحادية الاتجاه — القناع السببي هو الفارق الوحيد بنيويًا عن BERTBlock."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.causal_self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.causal_self_attn(self.ln1(x), mask=causal_mask)
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)
        return x


# ---------------------------------------------------------------------------
# 4) T5-style Block (Encoder-Decoder): ثلاث طبقات انتباه فعليًا في الـ Decoder
# ---------------------------------------------------------------------------

class T5EncoderBlock(nn.Module):
    """مطابقة تمامًا لـ BERTBlock — الـ Encoder في T5 ثنائي الاتجاه كذلك."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = BERTBlock(d_model, n_heads, d_ff, dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.block(x, padding_mask)


class T5DecoderBlock(nn.Module):
    """
    ثلاث طبقات فرعية:
      1. Masked Self-Attention (سببي، على مخرجات الـ Decoder حتى الآن)
      2. Cross-Attention (Q من الـ Decoder، K/V من مخرجات الـ Encoder)
      3. Feed-Forward
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.masked_self_attn = MultiHeadAttention(d_model, n_heads, dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)

        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        causal_mask: torch.Tensor,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1) Masked self-attention داخل الـ decoder فقط
        self_attn_out = self.masked_self_attn(self.ln1(x), mask=causal_mask)
        x = x + self.dropout(self_attn_out)

        # 2) Cross-attention: الاستعلامات من x (decoder)، K/V من الـ encoder
        cross_attn_out = self.cross_attn(
            query_input=self.ln2(x), kv_input=encoder_output, mask=cross_mask
        )
        x = x + self.dropout(cross_attn_out)

        # 3) FFN كالمعتاد
        ffn_out = self.ffn(self.ln3(x))
        x = x + self.dropout(ffn_out)
        return x


# ---------------------------------------------------------------------------
# 5) تحقّق سريع: هل القناع فعليًا يمنع رؤية المستقبل؟
# ---------------------------------------------------------------------------

def sanity_check_masks() -> None:
    device = torch.device("cpu")
    seq_len = 5
    causal = make_causal_mask(seq_len, device).squeeze()
    print("قناع GPT السببي (1=مسموح، 0=ممنوع):")
    print(causal.int())
    # تحقق: العنصر [0, 4] يجب أن يكون 0 (الرمز الأول لا يرى الرمز الخامس)
    assert causal[0, 4].item() == 0, "خطأ: الرمز الأول يرى المستقبل!"
    assert causal[4, 0].item() == 1, "خطأ: الرمز الأخير يجب أن يرى الماضي كله"
    print("✅ القناع السببي يعمل بشكل صحيح: لا تسريب من المستقبل.\n")


if __name__ == "__main__":
    sanity_check_masks()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-12-chapter-12.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
قناع GPT السببي (1=مسموح، 0=ممنوع):
tensor([[1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1]], dtype=torch.int32)
✅ القناع السببي يعمل بشكل صحيح: لا تسريب من المستقبل.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-12-chapter-12.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# training_objectives_demo.py
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(seed=7)

sentence = ["اذهب", "إلى", "المتجر", "واشترِ", "خمسة", "تفاحات"]


def bert_mlm_example(tokens: list[str]) -> None:
    """محاكاة MLM: اختر 15% من الرموز، طبّق قاعدة 80/10/10."""
    n_to_mask = max(1, round(0.15 * len(tokens)))
    mask_indices = rng.choice(len(tokens), size=n_to_mask, replace=False)

    masked_tokens = tokens.copy()
    for idx in mask_indices:
        roll = rng.random()
        if roll < 0.8:
            masked_tokens[idx] = "[MASK]"
        elif roll < 0.9:
            masked_tokens[idx] = rng.choice(tokens)  # استبدال عشوائي
        # وإلا: يبقى كما هو (10% المتبقية)

    print("BERT — مدخل MLM     :", " ".join(masked_tokens))
    print("BERT — الهدف (كامل) :", " ".join(tokens), "\n")


def gpt_causal_lm_example(tokens: list[str]) -> None:
    """محاكاة Next-Token Prediction: كل موضع يتنبأ بالتالي مباشرة."""
    print("GPT — تسلسل التدريب (مدخل → هدف لكل خطوة):")
    for t in range(len(tokens) - 1):
        context = " ".join(tokens[: t + 1])
        target = tokens[t + 1]
        print(f"   السياق: '{context}'  →  الهدف التالي: '{target}'")
    print()


def t5_span_corruption_example(tokens: list[str]) -> None:
    """محاكاة Span Corruption: إتلاف نطاق متجاور واحد أو أكثر."""
    span_start, span_len = 2, 2  # إتلاف "المتجر واشترِ"
    sentinel = "<X>"

    corrupted_input = (
        tokens[:span_start] + [sentinel] + tokens[span_start + span_len :]
    )
    target_output = [sentinel] + tokens[span_start : span_start + span_len] + ["<Z>"]

    print("T5 — مدخل Encoder (مُتلَف):", " ".join(corrupted_input))
    print("T5 — هدف Decoder (للاستعادة):", " ".join(target_output), "\n")


if __name__ == "__main__":
    bert_mlm_example(sentence)
    gpt_causal_lm_example(sentence)
    t5_span_corruption_example(sentence)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-12-chapter-12.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ مفاهيمي: BERT غير مُصمم للتوليد التسلسلي الحر
output = bert_model.generate(prompt)  # لا توجد دالة generate منطقية هنا أصلاً

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-12-chapter-12.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: استخدم GPT-style أو T5-style decoder للتوليد
output_ids = gpt_model.generate(input_ids, max_new_tokens=50)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-12-chapter-12.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ خطير جدًا وصامت: لا قناع إطلاقًا
attn_out = self.self_attn(x)  # بلا mask — تسريب كامل من المستقبل!

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-12-chapter-12.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح
causal_mask = make_causal_mask(seq_len, x.device)
attn_out = self.causal_self_attn(x, mask=causal_mask)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-12-chapter-12.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: استخدام الـ decoder نفسه كمصدر K/V بدلاً من مخرجات الـ encoder
cross_attn_out = self.cross_attn(query_input=x, kv_input=x)  # هذا Self-Attention وليس Cross!

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-12-chapter-12.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح
cross_attn_out = self.cross_attn(query_input=x, kv_input=encoder_output)

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-12-chapter-12.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# mini_project_toy_t5.py
from __future__ import annotations

import torch
import torch.nn as nn

from architecture_variants import (
    T5EncoderBlock,
    T5DecoderBlock,
    make_causal_mask,
)

VOCAB_SIZE = 12          # أرقام 0-9 + <BOS> + <EOS>
D_MODEL = 32
N_HEADS = 4
D_FF = 64
N_LAYERS = 2
MAX_LEN = 8


class ToyT5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(MAX_LEN, D_MODEL)

        self.encoder_blocks = nn.ModuleList(
            [T5EncoderBlock(D_MODEL, N_HEADS, D_FF) for _ in range(N_LAYERS)]
        )
        self.decoder_blocks = nn.ModuleList(
            [T5DecoderBlock(D_MODEL, N_HEADS, D_FF) for _ in range(N_LAYERS)]
        )
        self.output_head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(ids.size(1), device=ids.device)
        return self.token_emb(ids) + self.pos_emb(positions)

    def encode(self, src_ids: torch.Tensor) -> torch.Tensor:
        x = self._embed(src_ids)
        for block in self.encoder_blocks:
            x = block(x)
        return x

    def decode(self, tgt_ids: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        x = self._embed(tgt_ids)
        causal_mask = make_causal_mask(tgt_ids.size(1), tgt_ids.device)
        for block in self.decoder_blocks:
            x = block(x, encoder_output, causal_mask)
        return self.output_head(x)

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        encoder_output = self.encode(src_ids)
        return self.decode(tgt_ids, encoder_output)


# --- TODO لإكمال التمرين ---
# 1) اكتب دالة توليد بيانات: تسلسل عشوائي من الأرقام 0-9 وعكسه (أضف <BOS>=10, <EOS>=11)
# 2) اكتب حلقة تدريب باستخدام CrossEntropyLoss (تجاهل الحشو إن وُجد)
# 3) اكتب دالة استدلال Autoregressive: ابدأ بـ <BOS>، وَلِّد رمزًا رمزًا حتى <EOS>
# 4) قيّم الدقة: هل ينجح النموذج في عكس تسلسلات لم يرها أثناء التدريب؟
# 5) تجربة إضافية: احذف Cross-Attention بالكامل (استبدلها بتمرير صفري) وشاهد كيف ينهار الأداء تمامًا

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-12-chapter-12.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الثاني عشر: التنويعات المعمارية — Encoder-Only، Decoder-Only، Encoder-Decoder
Large Language Model Engineering — Encoder-Only

Book slug : llm-engineering-69-chapters
Chapter   : ch-12-chapter-12 (lesson 12)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_12.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 12: الفصل الثاني عشر: التنويعات المعمارية — Encoder-Only، Decoder-Only، Encoder-Decoder."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_masks() -> None:
    length = 5
    causal = [[int(key <= query) for key in range(length)] for query in range(length)]
    padding = [1, 1, 1, 0, 0]
    print('causal mask:')
    for row in causal:
        print(' ', row)
    print('padding-valid positions:', padding)
    print('encoder-decoder mask shape:', (length, 3))

if __name__ == "__main__":
    print("Chapter 12 lab: Encoder, decoder, and cross-attention masks")
    run_masks()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-03-transformers-core/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
# architecture_variants.py
"""
تنفيذ كامل للتنويعات المعمارية الثلاث على Transformer:
Encoder-Only (BERT-style), Decoder-Only (GPT-style), Encoder-Decoder (T5-style)
Python 3.13+, PyTorch 2.x
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


# ---------------------------------------------------------------------------
# 1) اللبنات المشتركة (معاد استخدامها من الفصل 10/11 مع توسيع للـ Cross-Attention)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    انتباه متعدد الرؤوس عام يدعم:
      - Self-Attention (عندما kv_input is None)
      - Cross-Attention (عندما kv_input يأتي من مصدر آخر، مثل مخرجات Encoder)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model يجب أن يقبل القسمة على n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, n_heads, seq_len, d_k)

    def forward(
        self,
        query_input: torch.Tensor,
        kv_input: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if kv_input is None:
            kv_input = query_input  # Self-Attention

        batch, q_len, _ = query_input.shape
        q = self._split_heads(self.w_q(query_input))
        k = self._split_heads(self.w_k(kv_input))
        v = self._split_heads(self.w_v(kv_input))

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        context = attn_weights @ v  # (batch, n_heads, q_len, d_k)

        context = context.transpose(1, 2).contiguous().view(batch, q_len, self.d_model)
        return self.w_o(context)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """قناع سببي: مثلث سفلي = 1 (مسموح)، مثلث علوي = 0 (ممنوع)."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len) للبث على batch/heads


def make_padding_mask(pad_mask_1d: torch.Tensor) -> torch.Tensor:
    """
    pad_mask_1d: (batch, seq_len) حيث 1 = رمز حقيقي، 0 = حشو (padding)
    نعيد (batch, 1, 1, seq_len) للبث على رؤوس الانتباه وكل استعلام.
    """
    return pad_mask_1d.unsqueeze(1).unsqueeze(2)


# ---------------------------------------------------------------------------
# 2) BERT-style Block (Encoder-Only): بلا قناع سببي إطلاقًا
# ---------------------------------------------------------------------------

class BERTBlock(nn.Module):
    """كتلة ثنائية الاتجاه بالكامل — Pre-LN كما اتفقنا في الفصل 11."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # ملاحظة جوهرية: لا يوجد أي causal mask هنا — فقط padding_mask إن وُجد
        attn_out = self.self_attn(self.ln1(x), mask=padding_mask)
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)
        return x


# ---------------------------------------------------------------------------
# 3) GPT-style Block (Decoder-Only): قناع سببي إلزامي
# ---------------------------------------------------------------------------

class GPTBlock(nn.Module):
    """كتلة أحادية الاتجاه — القناع السببي هو الفارق الوحيد بنيويًا عن BERTBlock."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.causal_self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.causal_self_attn(self.ln1(x), mask=causal_mask)
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)
        return x


# ---------------------------------------------------------------------------
# 4) T5-style Block (Encoder-Decoder): ثلاث طبقات انتباه فعليًا في الـ Decoder
# ---------------------------------------------------------------------------

class T5EncoderBlock(nn.Module):
    """مطابقة تمامًا لـ BERTBlock — الـ Encoder في T5 ثنائي الاتجاه كذلك."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = BERTBlock(d_model, n_heads, d_ff, dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.block(x, padding_mask)


class T5DecoderBlock(nn.Module):
    """
    ثلاث طبقات فرعية:
      1. Masked Self-Attention (سببي، على مخرجات الـ Decoder حتى الآن)
      2. Cross-Attention (Q من الـ Decoder، K/V من مخرجات الـ Encoder)
      3. Feed-Forward
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.masked_self_attn = MultiHeadAttention(d_model, n_heads, dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)

        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        causal_mask: torch.Tensor,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1) Masked self-attention داخل الـ decoder فقط
        self_attn_out = self.masked_self_attn(self.ln1(x), mask=causal_mask)
        x = x + self.dropout(self_attn_out)

        # 2) Cross-attention: الاستعلامات من x (decoder)، K/V من الـ encoder
        cross_attn_out = self.cross_attn(
            query_input=self.ln2(x), kv_input=encoder_output, mask=cross_mask
        )
        x = x + self.dropout(cross_attn_out)

        # 3) FFN كالمعتاد
        ffn_out = self.ffn(self.ln3(x))
        x = x + self.dropout(ffn_out)
        return x


# ---------------------------------------------------------------------------
# 5) تحقّق سريع: هل القناع فعليًا يمنع رؤية المستقبل؟
# ---------------------------------------------------------------------------

def sanity_check_masks() -> None:
    device = torch.device("cpu")
    seq_len = 5
    causal = make_causal_mask(seq_len, device).squeeze()
    print("قناع GPT السببي (1=مسموح، 0=ممنوع):")
    print(causal.int())
    # تحقق: العنصر [0, 4] يجب أن يكون 0 (الرمز الأول لا يرى الرمز الخامس)
    assert causal[0, 4].item() == 0, "خطأ: الرمز الأول يرى المستقبل!"
    assert causal[4, 0].item() == 1, "خطأ: الرمز الأخير يجب أن يرى الماضي كله"
    print("✅ القناع السببي يعمل بشكل صحيح: لا تسريب من المستقبل.\n")


if __name__ == "__main__":
    sanity_check_masks()

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-03-transformers-core/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
قناع GPT السببي (1=مسموح، 0=ممنوع):
tensor([[1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1]], dtype=torch.int32)
✅ القناع السببي يعمل بشكل صحيح: لا تسريب من المستقبل.

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-03-transformers-core/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
# training_objectives_demo.py
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(seed=7)

sentence = ["اذهب", "إلى", "المتجر", "واشترِ", "خمسة", "تفاحات"]


def bert_mlm_example(tokens: list[str]) -> None:
    """محاكاة MLM: اختر 15% من الرموز، طبّق قاعدة 80/10/10."""
    n_to_mask = max(1, round(0.15 * len(tokens)))
    mask_indices = rng.choice(len(tokens), size=n_to_mask, replace=False)

    masked_tokens = tokens.copy()
    for idx in mask_indices:
        roll = rng.random()
        if roll < 0.8:
            masked_tokens[idx] = "[MASK]"
        elif roll < 0.9:
            masked_tokens[idx] = rng.choice(tokens)  # استبدال عشوائي
        # وإلا: يبقى كما هو (10% المتبقية)

    print("BERT — مدخل MLM     :", " ".join(masked_tokens))
    print("BERT — الهدف (كامل) :", " ".join(tokens), "\n")


def gpt_causal_lm_example(tokens: list[str]) -> None:
    """محاكاة Next-Token Prediction: كل موضع يتنبأ بالتالي مباشرة."""
    print("GPT — تسلسل التدريب (مدخل → هدف لكل خطوة):")
    for t in range(len(tokens) - 1):
        context = " ".join(tokens[: t + 1])
        target = tokens[t + 1]
        print(f"   السياق: '{context}'  →  الهدف التالي: '{target}'")
    print()


def t5_span_corruption_example(tokens: list[str]) -> None:
    """محاكاة Span Corruption: إتلاف نطاق متجاور واحد أو أكثر."""
    span_start, span_len = 2, 2  # إتلاف "المتجر واشترِ"
    sentinel = "<X>"

    corrupted_input = (
        tokens[:span_start] + [sentinel] + tokens[span_start + span_len :]
    )
    target_output = [sentinel] + tokens[span_start : span_start + span_len] + ["<Z>"]

    print("T5 — مدخل Encoder (مُتلَف):", " ".join(corrupted_input))
    print("T5 — هدف Decoder (للاستعادة):", " ".join(target_output), "\n")


if __name__ == "__main__":
    bert_mlm_example(sentence)
    gpt_causal_lm_example(sentence)
    t5_span_corruption_example(sentence)

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-03-transformers-core/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ مفاهيمي: BERT غير مُصمم للتوليد التسلسلي الحر
output = bert_model.generate(prompt)  # لا توجد دالة generate منطقية هنا أصلاً

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-03-transformers-core/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: استخدم GPT-style أو T5-style decoder للتوليد
output_ids = gpt_model.generate(input_ids, max_new_tokens=50)

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-03-transformers-core/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ خطير جدًا وصامت: لا قناع إطلاقًا
attn_out = self.self_attn(x)  # بلا mask — تسريب كامل من المستقبل!

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-03-transformers-core/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح
causal_mask = make_causal_mask(seq_len, x.device)
attn_out = self.causal_self_attn(x, mask=causal_mask)

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-03-transformers-core/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: استخدام الـ decoder نفسه كمصدر K/V بدلاً من مخرجات الـ encoder
cross_attn_out = self.cross_attn(query_input=x, kv_input=x)  # هذا Self-Attention وليس Cross!

# ────────────────────────────────────────────────────────────────────────
# [010] book/chapters/part-03-transformers-core/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح
cross_attn_out = self.cross_attn(query_input=x, kv_input=encoder_output)

# ────────────────────────────────────────────────────────────────────────
# [011] book/chapters/part-03-transformers-core/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
# mini_project_toy_t5.py
from __future__ import annotations

import torch
import torch.nn as nn

from architecture_variants import (
    T5EncoderBlock,
    T5DecoderBlock,
    make_causal_mask,
)

VOCAB_SIZE = 12          # أرقام 0-9 + <BOS> + <EOS>
D_MODEL = 32
N_HEADS = 4
D_FF = 64
N_LAYERS = 2
MAX_LEN = 8


class ToyT5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(MAX_LEN, D_MODEL)

        self.encoder_blocks = nn.ModuleList(
            [T5EncoderBlock(D_MODEL, N_HEADS, D_FF) for _ in range(N_LAYERS)]
        )
        self.decoder_blocks = nn.ModuleList(
            [T5DecoderBlock(D_MODEL, N_HEADS, D_FF) for _ in range(N_LAYERS)]
        )
        self.output_head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(ids.size(1), device=ids.device)
        return self.token_emb(ids) + self.pos_emb(positions)

    def encode(self, src_ids: torch.Tensor) -> torch.Tensor:
        x = self._embed(src_ids)
        for block in self.encoder_blocks:
            x = block(x)
        return x

    def decode(self, tgt_ids: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        x = self._embed(tgt_ids)
        causal_mask = make_causal_mask(tgt_ids.size(1), tgt_ids.device)
        for block in self.decoder_blocks:
            x = block(x, encoder_output, causal_mask)
        return self.output_head(x)

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        encoder_output = self.encode(src_ids)
        return self.decode(tgt_ids, encoder_output)


# --- TODO لإكمال التمرين ---
# 1) اكتب دالة توليد بيانات: تسلسل عشوائي من الأرقام 0-9 وعكسه (أضف <BOS>=10, <EOS>=11)
# 2) اكتب حلقة تدريب باستخدام CrossEntropyLoss (تجاهل الحشو إن وُجد)
# 3) اكتب دالة استدلال Autoregressive: ابدأ بـ <BOS>، وَلِّد رمزًا رمزًا حتى <EOS>
# 4) قيّم الدقة: هل ينجح النموذج في عكس تسلسلات لم يرها أثناء التدريب؟
# 5) تجربة إضافية: احذف Cross-Attention بالكامل (استبدلها بتمرير صفري) وشاهد كيف ينهار الأداء تمامًا

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-12-chapter-12.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 12: الفصل الثاني عشر: التنويعات المعمارية — Encoder-Only، Decoder-Only، Encoder-Decoder."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_masks() -> None:
    length = 5
    causal = [[int(key <= query) for key in range(length)] for query in range(length)]
    padding = [1, 1, 1, 0, 0]
    print('causal mask:')
    for row in causal:
        print(' ', row)
    print('padding-valid positions:', padding)
    print('encoder-decoder mask shape:', (length, 3))

if __name__ == "__main__":
    print("Chapter 12 lab: Encoder, decoder, and cross-attention masks")
    run_masks()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-12-chapter-12.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# architecture_variants.py
"""
تنفيذ كامل للتنويعات المعمارية الثلاث على Transformer:
Encoder-Only (BERT-style), Decoder-Only (GPT-style), Encoder-Decoder (T5-style)
Python 3.13+, PyTorch 2.x
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


# ---------------------------------------------------------------------------
# 1) اللبنات المشتركة (معاد استخدامها من الفصل 10/11 مع توسيع للـ Cross-Attention)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    انتباه متعدد الرؤوس عام يدعم:
      - Self-Attention (عندما kv_input is None)
      - Cross-Attention (عندما kv_input يأتي من مصدر آخر، مثل مخرجات Encoder)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model يجب أن يقبل القسمة على n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, n_heads, seq_len, d_k)

    def forward(
        self,
        query_input: torch.Tensor,
        kv_input: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if kv_input is None:
            kv_input = query_input  # Self-Attention

        batch, q_len, _ = query_input.shape
        q = self._split_heads(self.w_q(query_input))
        k = self._split_heads(self.w_k(kv_input))
        v = self._split_heads(self.w_v(kv_input))

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        context = attn_weights @ v  # (batch, n_heads, q_len, d_k)

        context = context.transpose(1, 2).contiguous().view(batch, q_len, self.d_model)
        return self.w_o(context)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """قناع سببي: مثلث سفلي = 1 (مسموح)، مثلث علوي = 0 (ممنوع)."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len) للبث على batch/heads


def make_padding_mask(pad_mask_1d: torch.Tensor) -> torch.Tensor:
    """
    pad_mask_1d: (batch, seq_len) حيث 1 = رمز حقيقي، 0 = حشو (padding)
    نعيد (batch, 1, 1, seq_len) للبث على رؤوس الانتباه وكل استعلام.
    """
    return pad_mask_1d.unsqueeze(1).unsqueeze(2)


# ---------------------------------------------------------------------------
# 2) BERT-style Block (Encoder-Only): بلا قناع سببي إطلاقًا
# ---------------------------------------------------------------------------

class BERTBlock(nn.Module):
    """كتلة ثنائية الاتجاه بالكامل — Pre-LN كما اتفقنا في الفصل 11."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # ملاحظة جوهرية: لا يوجد أي causal mask هنا — فقط padding_mask إن وُجد
        attn_out = self.self_attn(self.ln1(x), mask=padding_mask)
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)
        return x


# ---------------------------------------------------------------------------
# 3) GPT-style Block (Decoder-Only): قناع سببي إلزامي
# ---------------------------------------------------------------------------

class GPTBlock(nn.Module):
    """كتلة أحادية الاتجاه — القناع السببي هو الفارق الوحيد بنيويًا عن BERTBlock."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.causal_self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.causal_self_attn(self.ln1(x), mask=causal_mask)
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)
        return x


# ---------------------------------------------------------------------------
# 4) T5-style Block (Encoder-Decoder): ثلاث طبقات انتباه فعليًا في الـ Decoder
# ---------------------------------------------------------------------------

class T5EncoderBlock(nn.Module):
    """مطابقة تمامًا لـ BERTBlock — الـ Encoder في T5 ثنائي الاتجاه كذلك."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = BERTBlock(d_model, n_heads, d_ff, dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.block(x, padding_mask)


class T5DecoderBlock(nn.Module):
    """
    ثلاث طبقات فرعية:
      1. Masked Self-Attention (سببي، على مخرجات الـ Decoder حتى الآن)
      2. Cross-Attention (Q من الـ Decoder، K/V من مخرجات الـ Encoder)
      3. Feed-Forward
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.masked_self_attn = MultiHeadAttention(d_model, n_heads, dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)

        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        causal_mask: torch.Tensor,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1) Masked self-attention داخل الـ decoder فقط
        self_attn_out = self.masked_self_attn(self.ln1(x), mask=causal_mask)
        x = x + self.dropout(self_attn_out)

        # 2) Cross-attention: الاستعلامات من x (decoder)، K/V من الـ encoder
        cross_attn_out = self.cross_attn(
            query_input=self.ln2(x), kv_input=encoder_output, mask=cross_mask
        )
        x = x + self.dropout(cross_attn_out)

        # 3) FFN كالمعتاد
        ffn_out = self.ffn(self.ln3(x))
        x = x + self.dropout(ffn_out)
        return x


# ---------------------------------------------------------------------------
# 5) تحقّق سريع: هل القناع فعليًا يمنع رؤية المستقبل؟
# ---------------------------------------------------------------------------

def sanity_check_masks() -> None:
    device = torch.device("cpu")
    seq_len = 5
    causal = make_causal_mask(seq_len, device).squeeze()
    print("قناع GPT السببي (1=مسموح، 0=ممنوع):")
    print(causal.int())
    # تحقق: العنصر [0, 4] يجب أن يكون 0 (الرمز الأول لا يرى الرمز الخامس)
    assert causal[0, 4].item() == 0, "خطأ: الرمز الأول يرى المستقبل!"
    assert causal[4, 0].item() == 1, "خطأ: الرمز الأخير يجب أن يرى الماضي كله"
    print("✅ القناع السببي يعمل بشكل صحيح: لا تسريب من المستقبل.\n")


if __name__ == "__main__":
    sanity_check_masks()

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-12-chapter-12.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
قناع GPT السببي (1=مسموح، 0=ممنوع):
tensor([[1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1]], dtype=torch.int32)
✅ القناع السببي يعمل بشكل صحيح: لا تسريب من المستقبل.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-12-chapter-12.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# training_objectives_demo.py
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(seed=7)

sentence = ["اذهب", "إلى", "المتجر", "واشترِ", "خمسة", "تفاحات"]


def bert_mlm_example(tokens: list[str]) -> None:
    """محاكاة MLM: اختر 15% من الرموز، طبّق قاعدة 80/10/10."""
    n_to_mask = max(1, round(0.15 * len(tokens)))
    mask_indices = rng.choice(len(tokens), size=n_to_mask, replace=False)

    masked_tokens = tokens.copy()
    for idx in mask_indices:
        roll = rng.random()
        if roll < 0.8:
            masked_tokens[idx] = "[MASK]"
        elif roll < 0.9:
            masked_tokens[idx] = rng.choice(tokens)  # استبدال عشوائي
        # وإلا: يبقى كما هو (10% المتبقية)

    print("BERT — مدخل MLM     :", " ".join(masked_tokens))
    print("BERT — الهدف (كامل) :", " ".join(tokens), "\n")


def gpt_causal_lm_example(tokens: list[str]) -> None:
    """محاكاة Next-Token Prediction: كل موضع يتنبأ بالتالي مباشرة."""
    print("GPT — تسلسل التدريب (مدخل → هدف لكل خطوة):")
    for t in range(len(tokens) - 1):
        context = " ".join(tokens[: t + 1])
        target = tokens[t + 1]
        print(f"   السياق: '{context}'  →  الهدف التالي: '{target}'")
    print()


def t5_span_corruption_example(tokens: list[str]) -> None:
    """محاكاة Span Corruption: إتلاف نطاق متجاور واحد أو أكثر."""
    span_start, span_len = 2, 2  # إتلاف "المتجر واشترِ"
    sentinel = "<X>"

    corrupted_input = (
        tokens[:span_start] + [sentinel] + tokens[span_start + span_len :]
    )
    target_output = [sentinel] + tokens[span_start : span_start + span_len] + ["<Z>"]

    print("T5 — مدخل Encoder (مُتلَف):", " ".join(corrupted_input))
    print("T5 — هدف Decoder (للاستعادة):", " ".join(target_output), "\n")


if __name__ == "__main__":
    bert_mlm_example(sentence)
    gpt_causal_lm_example(sentence)
    t5_span_corruption_example(sentence)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-12-chapter-12.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ مفاهيمي: BERT غير مُصمم للتوليد التسلسلي الحر
output = bert_model.generate(prompt)  # لا توجد دالة generate منطقية هنا أصلاً

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-12-chapter-12.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: استخدم GPT-style أو T5-style decoder للتوليد
output_ids = gpt_model.generate(input_ids, max_new_tokens=50)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-12-chapter-12.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ خطير جدًا وصامت: لا قناع إطلاقًا
attn_out = self.self_attn(x)  # بلا mask — تسريب كامل من المستقبل!

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-12-chapter-12.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح
causal_mask = make_causal_mask(seq_len, x.device)
attn_out = self.causal_self_attn(x, mask=causal_mask)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-12-chapter-12.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: استخدام الـ decoder نفسه كمصدر K/V بدلاً من مخرجات الـ encoder
cross_attn_out = self.cross_attn(query_input=x, kv_input=x)  # هذا Self-Attention وليس Cross!

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-12-chapter-12.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح
cross_attn_out = self.cross_attn(query_input=x, kv_input=encoder_output)

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-12-chapter-12.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# mini_project_toy_t5.py
from __future__ import annotations

import torch
import torch.nn as nn

from architecture_variants import (
    T5EncoderBlock,
    T5DecoderBlock,
    make_causal_mask,
)

VOCAB_SIZE = 12          # أرقام 0-9 + <BOS> + <EOS>
D_MODEL = 32
N_HEADS = 4
D_FF = 64
N_LAYERS = 2
MAX_LEN = 8


class ToyT5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(MAX_LEN, D_MODEL)

        self.encoder_blocks = nn.ModuleList(
            [T5EncoderBlock(D_MODEL, N_HEADS, D_FF) for _ in range(N_LAYERS)]
        )
        self.decoder_blocks = nn.ModuleList(
            [T5DecoderBlock(D_MODEL, N_HEADS, D_FF) for _ in range(N_LAYERS)]
        )
        self.output_head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(ids.size(1), device=ids.device)
        return self.token_emb(ids) + self.pos_emb(positions)

    def encode(self, src_ids: torch.Tensor) -> torch.Tensor:
        x = self._embed(src_ids)
        for block in self.encoder_blocks:
            x = block(x)
        return x

    def decode(self, tgt_ids: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        x = self._embed(tgt_ids)
        causal_mask = make_causal_mask(tgt_ids.size(1), tgt_ids.device)
        for block in self.decoder_blocks:
            x = block(x, encoder_output, causal_mask)
        return self.output_head(x)

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        encoder_output = self.encode(src_ids)
        return self.decode(tgt_ids, encoder_output)


# --- TODO لإكمال التمرين ---
# 1) اكتب دالة توليد بيانات: تسلسل عشوائي من الأرقام 0-9 وعكسه (أضف <BOS>=10, <EOS>=11)
# 2) اكتب حلقة تدريب باستخدام CrossEntropyLoss (تجاهل الحشو إن وُجد)
# 3) اكتب دالة استدلال Autoregressive: ابدأ بـ <BOS>، وَلِّد رمزًا رمزًا حتى <EOS>
# 4) قيّم الدقة: هل ينجح النموذج في عكس تسلسلات لم يرها أثناء التدريب؟
# 5) تجربة إضافية: احذف Cross-Attention بالكامل (استبدلها بتمرير صفري) وشاهد كيف ينهار الأداء تمامًا
