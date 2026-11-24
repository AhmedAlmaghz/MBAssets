#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — آلية الانتباه الذاتي — قلب الثورة النابض
Large Language Model Engineering — Chapter 10 — Transformers Core

Book slug : llm-engineering-69-chapters
Chapter   : ch-10-chapter-10 (lesson 10)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-10-chapter-10.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — آلية الانتباه الذاتي — قلب الثورة النابض
Large Language Model Engineering — Chapter 10 — Transformers Core

Book slug : llm-engineering-69-chapters
Chapter   : ch-10-chapter-10 (lesson 10)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-10-chapter-10.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — آلية الانتباه الذاتي — قلب الثورة النابض
Large Language Model Engineering — Chapter 10 — Transformers Core

Book slug : llm-engineering-69-chapters
Chapter   : ch-10-chapter-10 (lesson 10)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_10.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 10: الفصل العاشر: آلية الانتباه الذاتي — قلب الثورة النابض."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponents = [math.exp(value - maximum) for value in values]
    denominator = sum(exponents)
    return [value / denominator for value in exponents]

def run_attention() -> None:
    queries = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    keys, values = ([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    outputs = []
    for position, query in enumerate(queries):
        scores = [sum((a * b for a, b in zip(query, key))) / math.sqrt(len(query)) if index <= position else -math.inf for index, key in enumerate(keys)]
        weights = softmax(scores)
        outputs.append([sum((weight * value[axis] for weight, value in zip(weights, values))) for axis in range(2)])
        if position == 1:
            assert weights[2] == 0.0
    print('second-token causal weights:', [round(x, 3) for x in softmax([sum((a * b for a, b in zip(queries[1], key))) / math.sqrt(2) if index <= 1 else -math.inf for index, key in enumerate(keys)])])
    print('attention output shape:', (len(outputs), len(outputs[0])))
    print('future positions masked:', True)

if __name__ == "__main__":
    print("Chapter 10 lab: Scaled dot-product causal attention")
    run_attention()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-03-transformers-core/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
"""
scaled_dot_product_attention.py
تطبيق كامل لآلية الانتباه الذاتي من الصفر
يتطلب: Python 3.13+, PyTorch 2.3+
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor
import torch.nn.functional as F


class AttentionOutput(NamedTuple):
    """حاوية نتائج الانتباه - لإرجاع القيم والأوزان معًا بوضوح"""
    output: Tensor
    attention_weights: Tensor


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
    dropout_p: float = 0.0,
) -> AttentionOutput:
    """
    التطبيق اليدوي الكامل لمعادلة:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        query: (batch, ..., seq_len_q, d_k)
        key:   (batch, ..., seq_len_k, d_k)
        value: (batch, ..., seq_len_k, d_v)
        mask:  (batch, ..., seq_len_q, seq_len_k) — True/1 يعني "اسمح بالانتباه"
        dropout_p: احتمال إسقاط بعض أوزان الانتباه (Regularization)

    Returns:
        AttentionOutput(output, attention_weights)
    """
    d_k = query.size(-1)

    # الخطوة 1+2: QK^T ثم القسمة على sqrt(d_k)
    # transpose(-2, -1) يقلب آخر بعدين فقط، بغض النظر عن عدد الأبعاد الكلي
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # scores shape: (batch, ..., seq_len_q, seq_len_k)

    # الخطوة 3: تطبيق القناع (Mask) قبل Softmax - لا بعده!
    if mask is not None:
        # نملأ المواضع الممنوعة بقيمة سالبة ضخمة جدًا
        # بحيث يصبح exp(-inf) ≈ 0 بعد Softmax
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # الخطوة 4: Softmax على المحور الأخير (seq_len_k)
    attention_weights = F.softmax(scores, dim=-1)

    if dropout_p > 0.0:
        attention_weights = F.dropout(attention_weights, p=dropout_p)

    # الخطوة 5: المتوسط المرجّح للـ Values
    output = torch.matmul(attention_weights, value)

    return AttentionOutput(output=output, attention_weights=attention_weights)

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-03-transformers-core/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)

batch_size, seq_len, d_k, d_v = 1, 4, 8, 8

Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_v)

result = scaled_dot_product_attention(Q, K, V)

print(f"شكل المخرجات: {result.output.shape}")           # (1, 4, 8)
print(f"شكل أوزان الانتباه: {result.attention_weights.shape}")  # (1, 4, 4)
print(f"مجموع كل صف من أوزان الانتباه (يجب = 1.0):")
print(result.attention_weights.sum(dim=-1))              # tensor([[1., 1., 1., 1.]])

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-03-transformers-core/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention - التطبيق القياسي الكامل
    يُستخدم كمكوّن أساسي سنعيد استخدامه في الفصل 11 لبناء Transformer كامل
    """

    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model يجب أن يقبل القسمة على num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # بُعد كل رأس على حدة

        # مصفوفات إسقاط موحّدة لكل الرؤوس معًا (أكفأ حسابيًا من رؤوس منفصلة)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout_p = dropout_p

    def _split_heads(self, x: Tensor, batch_size: int) -> Tensor:
        """ (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k) """
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: Tensor, batch_size: int) -> Tensor:
        """ العملية العكسية: (batch, num_heads, seq_len, d_k) → (batch, seq_len, d_model) """
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, -1, self.d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        batch_size = query.size(0)

        # 1. الإسقاط الخطي ثم تقسيم إلى رؤوس
        Q = self._split_heads(self.W_q(query), batch_size)
        K = self._split_heads(self.W_k(key), batch_size)
        V = self._split_heads(self.W_v(value), batch_size)

        # 2. تطبيق Attention على كل الرؤوس بالتوازي (عملية مصفوفية واحدة!)
        if mask is not None:
            mask = mask.unsqueeze(1)  # توسيع البعد ليطابق num_heads

        attn_output, _ = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout_p=self.dropout_p if self.training else 0.0
        )

        # 3. دمج الرؤوس وإسقاط نهائي
        merged = self._merge_heads(attn_output, batch_size)
        return self.W_o(merged)

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-03-transformers-core/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 10, 512)  # (batch=2, seq_len=10, d_model=512)

output = mha(x, x, x)  # Self-Attention: Q=K=V=نفس المدخل
print(output.shape)  # torch.Size([2, 10, 512]) ← نفس شكل المدخل تمامًا!

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-03-transformers-core/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
torch.manual_seed(0)

# نفترض d_model = 4 فقط لتبسيط التتبع اليدوي
sentence_embeddings = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],   # كلمة 1: "القطة"
    [0.0, 1.0, 0.0, 1.0],   # كلمة 2: "جلست"
])  # shape: (2, 4) → (seq_len=2, d_model=4)

# أوزان مبسّطة (في الواقع تُتعلَّم، هنا نحددها يدويًا للتوضيح)
W_Q = torch.eye(4)  # مصفوفة هوية لتبسيط التتبع
W_K = torch.eye(4)
W_V = torch.tensor([
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 2., 0.],
    [0., 0., 0., 2.],
])

Q = sentence_embeddings @ W_Q
K = sentence_embeddings @ W_K
V = sentence_embeddings @ W_V

print("Q =\n", Q)
print("K =\n", K)

scores = Q @ K.T / math.sqrt(4)
print("\nScores (قبل Softmax) =\n", scores)

weights = F.softmax(scores, dim=-1)
print("\nAttention Weights (بعد Softmax) =\n", weights)

output = weights @ V
print("\nالمخرج النهائي (تمثيل واعٍ بالسياق) =\n", output)

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-03-transformers-core/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
   output = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, is_causal=False)
   ```
   هذه الدالة أسرع 2-4 مرات وتستهلك ذاكرة أقل بكثير — سنتعمّق في السبب الهندسي في الفصل 13 (Flash Attention).

2. **لا تنسَ `bias=False` في الإسقاطات الخطية لـ Q, K, V.** معظم التطبيقات الحديثة (LLaMA وغيرها) تتجاهل الـ bias في هذه الطبقات تحديدًا — يُقلّل عدد المعاملات قليلًا دون أثر يُذكر على الأداء.

3. **تحقّق دائمًا من مجموع صفوف `attention_weights` يساوي 1.0** كاختبار سلامة أولي عند بناء أي تطبيق جديد — إذا لم يكن كذلك، فهناك خطأ في محور الـ `softmax`.

4. **د_k الأصغر ليس بالضرورة أسوأ.** تقسيم `d_model=512` على `8` رؤوس يعطي `d_k=64` لكل رأس — هذا مقصود: رؤوس متعددة "أضيق" غالبًا أفضل من رأس واحد "عريض" لنفس الميزانية الحسابية.

---

## ⚠️ 10.5 أخطاء شائعة ومصائد (احذر منها!)

### ❌ الخطأ #1: الخلط بين Padding Mask و Causal Mask

هذا **أشهر خطأ** يرتكبه المهندسون عند بناء Transformer لأول مرة.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-10-chapter-10.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 10: الفصل العاشر: آلية الانتباه الذاتي — قلب الثورة النابض."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponents = [math.exp(value - maximum) for value in values]
    denominator = sum(exponents)
    return [value / denominator for value in exponents]

def run_attention() -> None:
    queries = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    keys, values = ([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    outputs = []
    for position, query in enumerate(queries):
        scores = [sum((a * b for a, b in zip(query, key))) / math.sqrt(len(query)) if index <= position else -math.inf for index, key in enumerate(keys)]
        weights = softmax(scores)
        outputs.append([sum((weight * value[axis] for weight, value in zip(weights, values))) for axis in range(2)])
        if position == 1:
            assert weights[2] == 0.0
    print('second-token causal weights:', [round(x, 3) for x in softmax([sum((a * b for a, b in zip(queries[1], key))) / math.sqrt(2) if index <= 1 else -math.inf for index, key in enumerate(keys)])])
    print('attention output shape:', (len(outputs), len(outputs[0])))
    print('future positions masked:', True)

if __name__ == "__main__":
    print("Chapter 10 lab: Scaled dot-product causal attention")
    run_attention()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-10-chapter-10.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
scaled_dot_product_attention.py
تطبيق كامل لآلية الانتباه الذاتي من الصفر
يتطلب: Python 3.13+, PyTorch 2.3+
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor
import torch.nn.functional as F


class AttentionOutput(NamedTuple):
    """حاوية نتائج الانتباه - لإرجاع القيم والأوزان معًا بوضوح"""
    output: Tensor
    attention_weights: Tensor


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
    dropout_p: float = 0.0,
) -> AttentionOutput:
    """
    التطبيق اليدوي الكامل لمعادلة:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        query: (batch, ..., seq_len_q, d_k)
        key:   (batch, ..., seq_len_k, d_k)
        value: (batch, ..., seq_len_k, d_v)
        mask:  (batch, ..., seq_len_q, seq_len_k) — True/1 يعني "اسمح بالانتباه"
        dropout_p: احتمال إسقاط بعض أوزان الانتباه (Regularization)

    Returns:
        AttentionOutput(output, attention_weights)
    """
    d_k = query.size(-1)

    # الخطوة 1+2: QK^T ثم القسمة على sqrt(d_k)
    # transpose(-2, -1) يقلب آخر بعدين فقط، بغض النظر عن عدد الأبعاد الكلي
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # scores shape: (batch, ..., seq_len_q, seq_len_k)

    # الخطوة 3: تطبيق القناع (Mask) قبل Softmax - لا بعده!
    if mask is not None:
        # نملأ المواضع الممنوعة بقيمة سالبة ضخمة جدًا
        # بحيث يصبح exp(-inf) ≈ 0 بعد Softmax
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # الخطوة 4: Softmax على المحور الأخير (seq_len_k)
    attention_weights = F.softmax(scores, dim=-1)

    if dropout_p > 0.0:
        attention_weights = F.dropout(attention_weights, p=dropout_p)

    # الخطوة 5: المتوسط المرجّح للـ Values
    output = torch.matmul(attention_weights, value)

    return AttentionOutput(output=output, attention_weights=attention_weights)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-10-chapter-10.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)

batch_size, seq_len, d_k, d_v = 1, 4, 8, 8

Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_v)

result = scaled_dot_product_attention(Q, K, V)

print(f"شكل المخرجات: {result.output.shape}")           # (1, 4, 8)
print(f"شكل أوزان الانتباه: {result.attention_weights.shape}")  # (1, 4, 4)
print(f"مجموع كل صف من أوزان الانتباه (يجب = 1.0):")
print(result.attention_weights.sum(dim=-1))              # tensor([[1., 1., 1., 1.]])

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-10-chapter-10.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention - التطبيق القياسي الكامل
    يُستخدم كمكوّن أساسي سنعيد استخدامه في الفصل 11 لبناء Transformer كامل
    """

    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model يجب أن يقبل القسمة على num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # بُعد كل رأس على حدة

        # مصفوفات إسقاط موحّدة لكل الرؤوس معًا (أكفأ حسابيًا من رؤوس منفصلة)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout_p = dropout_p

    def _split_heads(self, x: Tensor, batch_size: int) -> Tensor:
        """ (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k) """
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: Tensor, batch_size: int) -> Tensor:
        """ العملية العكسية: (batch, num_heads, seq_len, d_k) → (batch, seq_len, d_model) """
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, -1, self.d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        batch_size = query.size(0)

        # 1. الإسقاط الخطي ثم تقسيم إلى رؤوس
        Q = self._split_heads(self.W_q(query), batch_size)
        K = self._split_heads(self.W_k(key), batch_size)
        V = self._split_heads(self.W_v(value), batch_size)

        # 2. تطبيق Attention على كل الرؤوس بالتوازي (عملية مصفوفية واحدة!)
        if mask is not None:
            mask = mask.unsqueeze(1)  # توسيع البعد ليطابق num_heads

        attn_output, _ = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout_p=self.dropout_p if self.training else 0.0
        )

        # 3. دمج الرؤوس وإسقاط نهائي
        merged = self._merge_heads(attn_output, batch_size)
        return self.W_o(merged)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-10-chapter-10.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 10, 512)  # (batch=2, seq_len=10, d_model=512)

output = mha(x, x, x)  # Self-Attention: Q=K=V=نفس المدخل
print(output.shape)  # torch.Size([2, 10, 512]) ← نفس شكل المدخل تمامًا!

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-10-chapter-10.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
torch.manual_seed(0)

# نفترض d_model = 4 فقط لتبسيط التتبع اليدوي
sentence_embeddings = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],   # كلمة 1: "القطة"
    [0.0, 1.0, 0.0, 1.0],   # كلمة 2: "جلست"
])  # shape: (2, 4) → (seq_len=2, d_model=4)

# أوزان مبسّطة (في الواقع تُتعلَّم، هنا نحددها يدويًا للتوضيح)
W_Q = torch.eye(4)  # مصفوفة هوية لتبسيط التتبع
W_K = torch.eye(4)
W_V = torch.tensor([
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 2., 0.],
    [0., 0., 0., 2.],
])

Q = sentence_embeddings @ W_Q
K = sentence_embeddings @ W_K
V = sentence_embeddings @ W_V

print("Q =\n", Q)
print("K =\n", K)

scores = Q @ K.T / math.sqrt(4)
print("\nScores (قبل Softmax) =\n", scores)

weights = F.softmax(scores, dim=-1)
print("\nAttention Weights (بعد Softmax) =\n", weights)

output = weights @ V
print("\nالمخرج النهائي (تمثيل واعٍ بالسياق) =\n", output)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-10-chapter-10.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
   output = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, is_causal=False)
   ```
   هذه الدالة أسرع 2-4 مرات وتستهلك ذاكرة أقل بكثير — سنتعمّق في السبب الهندسي في الفصل 13 (Flash Attention).

2. **لا تنسَ `bias=False` في الإسقاطات الخطية لـ Q, K, V.** معظم التطبيقات الحديثة (LLaMA وغيرها) تتجاهل الـ bias في هذه الطبقات تحديدًا — يُقلّل عدد المعاملات قليلًا دون أثر يُذكر على الأداء.

3. **تحقّق دائمًا من مجموع صفوف `attention_weights` يساوي 1.0** كاختبار سلامة أولي عند بناء أي تطبيق جديد — إذا لم يكن كذلك، فهناك خطأ في محور الـ `softmax`.

4. **د_k الأصغر ليس بالضرورة أسوأ.** تقسيم `d_model=512` على `8` رؤوس يعطي `d_k=64` لكل رأس — هذا مقصود: رؤوس متعددة "أضيق" غالبًا أفضل من رأس واحد "عريض" لنفس الميزانية الحسابية.

---

## ⚠️ 10.5 أخطاء شائعة ومصائد (احذر منها!)

### ❌ الخطأ #1: الخلط بين Padding Mask و Causal Mask

هذا **أشهر خطأ** يرتكبه المهندسون عند بناء Transformer لأول مرة.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-10-chapter-10.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — آلية الانتباه الذاتي — قلب الثورة النابض
Large Language Model Engineering — Chapter 10 — Transformers Core

Book slug : llm-engineering-69-chapters
Chapter   : ch-10-chapter-10 (lesson 10)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_10.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 10: الفصل العاشر: آلية الانتباه الذاتي — قلب الثورة النابض."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponents = [math.exp(value - maximum) for value in values]
    denominator = sum(exponents)
    return [value / denominator for value in exponents]

def run_attention() -> None:
    queries = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    keys, values = ([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    outputs = []
    for position, query in enumerate(queries):
        scores = [sum((a * b for a, b in zip(query, key))) / math.sqrt(len(query)) if index <= position else -math.inf for index, key in enumerate(keys)]
        weights = softmax(scores)
        outputs.append([sum((weight * value[axis] for weight, value in zip(weights, values))) for axis in range(2)])
        if position == 1:
            assert weights[2] == 0.0
    print('second-token causal weights:', [round(x, 3) for x in softmax([sum((a * b for a, b in zip(queries[1], key))) / math.sqrt(2) if index <= 1 else -math.inf for index, key in enumerate(keys)])])
    print('attention output shape:', (len(outputs), len(outputs[0])))
    print('future positions masked:', True)

if __name__ == "__main__":
    print("Chapter 10 lab: Scaled dot-product causal attention")
    run_attention()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-03-transformers-core/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
"""
scaled_dot_product_attention.py
تطبيق كامل لآلية الانتباه الذاتي من الصفر
يتطلب: Python 3.13+, PyTorch 2.3+
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor
import torch.nn.functional as F


class AttentionOutput(NamedTuple):
    """حاوية نتائج الانتباه - لإرجاع القيم والأوزان معًا بوضوح"""
    output: Tensor
    attention_weights: Tensor


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
    dropout_p: float = 0.0,
) -> AttentionOutput:
    """
    التطبيق اليدوي الكامل لمعادلة:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        query: (batch, ..., seq_len_q, d_k)
        key:   (batch, ..., seq_len_k, d_k)
        value: (batch, ..., seq_len_k, d_v)
        mask:  (batch, ..., seq_len_q, seq_len_k) — True/1 يعني "اسمح بالانتباه"
        dropout_p: احتمال إسقاط بعض أوزان الانتباه (Regularization)

    Returns:
        AttentionOutput(output, attention_weights)
    """
    d_k = query.size(-1)

    # الخطوة 1+2: QK^T ثم القسمة على sqrt(d_k)
    # transpose(-2, -1) يقلب آخر بعدين فقط، بغض النظر عن عدد الأبعاد الكلي
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # scores shape: (batch, ..., seq_len_q, seq_len_k)

    # الخطوة 3: تطبيق القناع (Mask) قبل Softmax - لا بعده!
    if mask is not None:
        # نملأ المواضع الممنوعة بقيمة سالبة ضخمة جدًا
        # بحيث يصبح exp(-inf) ≈ 0 بعد Softmax
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # الخطوة 4: Softmax على المحور الأخير (seq_len_k)
    attention_weights = F.softmax(scores, dim=-1)

    if dropout_p > 0.0:
        attention_weights = F.dropout(attention_weights, p=dropout_p)

    # الخطوة 5: المتوسط المرجّح للـ Values
    output = torch.matmul(attention_weights, value)

    return AttentionOutput(output=output, attention_weights=attention_weights)

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-03-transformers-core/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)

batch_size, seq_len, d_k, d_v = 1, 4, 8, 8

Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_v)

result = scaled_dot_product_attention(Q, K, V)

print(f"شكل المخرجات: {result.output.shape}")           # (1, 4, 8)
print(f"شكل أوزان الانتباه: {result.attention_weights.shape}")  # (1, 4, 4)
print(f"مجموع كل صف من أوزان الانتباه (يجب = 1.0):")
print(result.attention_weights.sum(dim=-1))              # tensor([[1., 1., 1., 1.]])

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-03-transformers-core/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention - التطبيق القياسي الكامل
    يُستخدم كمكوّن أساسي سنعيد استخدامه في الفصل 11 لبناء Transformer كامل
    """

    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model يجب أن يقبل القسمة على num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # بُعد كل رأس على حدة

        # مصفوفات إسقاط موحّدة لكل الرؤوس معًا (أكفأ حسابيًا من رؤوس منفصلة)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout_p = dropout_p

    def _split_heads(self, x: Tensor, batch_size: int) -> Tensor:
        """ (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k) """
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: Tensor, batch_size: int) -> Tensor:
        """ العملية العكسية: (batch, num_heads, seq_len, d_k) → (batch, seq_len, d_model) """
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, -1, self.d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        batch_size = query.size(0)

        # 1. الإسقاط الخطي ثم تقسيم إلى رؤوس
        Q = self._split_heads(self.W_q(query), batch_size)
        K = self._split_heads(self.W_k(key), batch_size)
        V = self._split_heads(self.W_v(value), batch_size)

        # 2. تطبيق Attention على كل الرؤوس بالتوازي (عملية مصفوفية واحدة!)
        if mask is not None:
            mask = mask.unsqueeze(1)  # توسيع البعد ليطابق num_heads

        attn_output, _ = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout_p=self.dropout_p if self.training else 0.0
        )

        # 3. دمج الرؤوس وإسقاط نهائي
        merged = self._merge_heads(attn_output, batch_size)
        return self.W_o(merged)

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-03-transformers-core/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 10, 512)  # (batch=2, seq_len=10, d_model=512)

output = mha(x, x, x)  # Self-Attention: Q=K=V=نفس المدخل
print(output.shape)  # torch.Size([2, 10, 512]) ← نفس شكل المدخل تمامًا!

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-03-transformers-core/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
torch.manual_seed(0)

# نفترض d_model = 4 فقط لتبسيط التتبع اليدوي
sentence_embeddings = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],   # كلمة 1: "القطة"
    [0.0, 1.0, 0.0, 1.0],   # كلمة 2: "جلست"
])  # shape: (2, 4) → (seq_len=2, d_model=4)

# أوزان مبسّطة (في الواقع تُتعلَّم، هنا نحددها يدويًا للتوضيح)
W_Q = torch.eye(4)  # مصفوفة هوية لتبسيط التتبع
W_K = torch.eye(4)
W_V = torch.tensor([
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 2., 0.],
    [0., 0., 0., 2.],
])

Q = sentence_embeddings @ W_Q
K = sentence_embeddings @ W_K
V = sentence_embeddings @ W_V

print("Q =\n", Q)
print("K =\n", K)

scores = Q @ K.T / math.sqrt(4)
print("\nScores (قبل Softmax) =\n", scores)

weights = F.softmax(scores, dim=-1)
print("\nAttention Weights (بعد Softmax) =\n", weights)

output = weights @ V
print("\nالمخرج النهائي (تمثيل واعٍ بالسياق) =\n", output)

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-03-transformers-core/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
   output = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, is_causal=False)
   ```
   هذه الدالة أسرع 2-4 مرات وتستهلك ذاكرة أقل بكثير — سنتعمّق في السبب الهندسي في الفصل 13 (Flash Attention).

2. **لا تنسَ `bias=False` في الإسقاطات الخطية لـ Q, K, V.** معظم التطبيقات الحديثة (LLaMA وغيرها) تتجاهل الـ bias في هذه الطبقات تحديدًا — يُقلّل عدد المعاملات قليلًا دون أثر يُذكر على الأداء.

3. **تحقّق دائمًا من مجموع صفوف `attention_weights` يساوي 1.0** كاختبار سلامة أولي عند بناء أي تطبيق جديد — إذا لم يكن كذلك، فهناك خطأ في محور الـ `softmax`.

4. **د_k الأصغر ليس بالضرورة أسوأ.** تقسيم `d_model=512` على `8` رؤوس يعطي `d_k=64` لكل رأس — هذا مقصود: رؤوس متعددة "أضيق" غالبًا أفضل من رأس واحد "عريض" لنفس الميزانية الحسابية.

---

## ⚠️ 10.5 أخطاء شائعة ومصائد (احذر منها!)

### ❌ الخطأ #1: الخلط بين Padding Mask و Causal Mask

هذا **أشهر خطأ** يرتكبه المهندسون عند بناء Transformer لأول مرة.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-10-chapter-10.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 10: الفصل العاشر: آلية الانتباه الذاتي — قلب الثورة النابض."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponents = [math.exp(value - maximum) for value in values]
    denominator = sum(exponents)
    return [value / denominator for value in exponents]

def run_attention() -> None:
    queries = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    keys, values = ([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    outputs = []
    for position, query in enumerate(queries):
        scores = [sum((a * b for a, b in zip(query, key))) / math.sqrt(len(query)) if index <= position else -math.inf for index, key in enumerate(keys)]
        weights = softmax(scores)
        outputs.append([sum((weight * value[axis] for weight, value in zip(weights, values))) for axis in range(2)])
        if position == 1:
            assert weights[2] == 0.0
    print('second-token causal weights:', [round(x, 3) for x in softmax([sum((a * b for a, b in zip(queries[1], key))) / math.sqrt(2) if index <= 1 else -math.inf for index, key in enumerate(keys)])])
    print('attention output shape:', (len(outputs), len(outputs[0])))
    print('future positions masked:', True)

if __name__ == "__main__":
    print("Chapter 10 lab: Scaled dot-product causal attention")
    run_attention()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-10-chapter-10.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
scaled_dot_product_attention.py
تطبيق كامل لآلية الانتباه الذاتي من الصفر
يتطلب: Python 3.13+, PyTorch 2.3+
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor
import torch.nn.functional as F


class AttentionOutput(NamedTuple):
    """حاوية نتائج الانتباه - لإرجاع القيم والأوزان معًا بوضوح"""
    output: Tensor
    attention_weights: Tensor


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
    dropout_p: float = 0.0,
) -> AttentionOutput:
    """
    التطبيق اليدوي الكامل لمعادلة:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        query: (batch, ..., seq_len_q, d_k)
        key:   (batch, ..., seq_len_k, d_k)
        value: (batch, ..., seq_len_k, d_v)
        mask:  (batch, ..., seq_len_q, seq_len_k) — True/1 يعني "اسمح بالانتباه"
        dropout_p: احتمال إسقاط بعض أوزان الانتباه (Regularization)

    Returns:
        AttentionOutput(output, attention_weights)
    """
    d_k = query.size(-1)

    # الخطوة 1+2: QK^T ثم القسمة على sqrt(d_k)
    # transpose(-2, -1) يقلب آخر بعدين فقط، بغض النظر عن عدد الأبعاد الكلي
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # scores shape: (batch, ..., seq_len_q, seq_len_k)

    # الخطوة 3: تطبيق القناع (Mask) قبل Softmax - لا بعده!
    if mask is not None:
        # نملأ المواضع الممنوعة بقيمة سالبة ضخمة جدًا
        # بحيث يصبح exp(-inf) ≈ 0 بعد Softmax
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # الخطوة 4: Softmax على المحور الأخير (seq_len_k)
    attention_weights = F.softmax(scores, dim=-1)

    if dropout_p > 0.0:
        attention_weights = F.dropout(attention_weights, p=dropout_p)

    # الخطوة 5: المتوسط المرجّح للـ Values
    output = torch.matmul(attention_weights, value)

    return AttentionOutput(output=output, attention_weights=attention_weights)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-10-chapter-10.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)

batch_size, seq_len, d_k, d_v = 1, 4, 8, 8

Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_v)

result = scaled_dot_product_attention(Q, K, V)

print(f"شكل المخرجات: {result.output.shape}")           # (1, 4, 8)
print(f"شكل أوزان الانتباه: {result.attention_weights.shape}")  # (1, 4, 4)
print(f"مجموع كل صف من أوزان الانتباه (يجب = 1.0):")
print(result.attention_weights.sum(dim=-1))              # tensor([[1., 1., 1., 1.]])

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-10-chapter-10.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention - التطبيق القياسي الكامل
    يُستخدم كمكوّن أساسي سنعيد استخدامه في الفصل 11 لبناء Transformer كامل
    """

    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model يجب أن يقبل القسمة على num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # بُعد كل رأس على حدة

        # مصفوفات إسقاط موحّدة لكل الرؤوس معًا (أكفأ حسابيًا من رؤوس منفصلة)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout_p = dropout_p

    def _split_heads(self, x: Tensor, batch_size: int) -> Tensor:
        """ (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k) """
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: Tensor, batch_size: int) -> Tensor:
        """ العملية العكسية: (batch, num_heads, seq_len, d_k) → (batch, seq_len, d_model) """
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, -1, self.d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        batch_size = query.size(0)

        # 1. الإسقاط الخطي ثم تقسيم إلى رؤوس
        Q = self._split_heads(self.W_q(query), batch_size)
        K = self._split_heads(self.W_k(key), batch_size)
        V = self._split_heads(self.W_v(value), batch_size)

        # 2. تطبيق Attention على كل الرؤوس بالتوازي (عملية مصفوفية واحدة!)
        if mask is not None:
            mask = mask.unsqueeze(1)  # توسيع البعد ليطابق num_heads

        attn_output, _ = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout_p=self.dropout_p if self.training else 0.0
        )

        # 3. دمج الرؤوس وإسقاط نهائي
        merged = self._merge_heads(attn_output, batch_size)
        return self.W_o(merged)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-10-chapter-10.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 10, 512)  # (batch=2, seq_len=10, d_model=512)

output = mha(x, x, x)  # Self-Attention: Q=K=V=نفس المدخل
print(output.shape)  # torch.Size([2, 10, 512]) ← نفس شكل المدخل تمامًا!

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-10-chapter-10.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
torch.manual_seed(0)

# نفترض d_model = 4 فقط لتبسيط التتبع اليدوي
sentence_embeddings = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],   # كلمة 1: "القطة"
    [0.0, 1.0, 0.0, 1.0],   # كلمة 2: "جلست"
])  # shape: (2, 4) → (seq_len=2, d_model=4)

# أوزان مبسّطة (في الواقع تُتعلَّم، هنا نحددها يدويًا للتوضيح)
W_Q = torch.eye(4)  # مصفوفة هوية لتبسيط التتبع
W_K = torch.eye(4)
W_V = torch.tensor([
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 2., 0.],
    [0., 0., 0., 2.],
])

Q = sentence_embeddings @ W_Q
K = sentence_embeddings @ W_K
V = sentence_embeddings @ W_V

print("Q =\n", Q)
print("K =\n", K)

scores = Q @ K.T / math.sqrt(4)
print("\nScores (قبل Softmax) =\n", scores)

weights = F.softmax(scores, dim=-1)
print("\nAttention Weights (بعد Softmax) =\n", weights)

output = weights @ V
print("\nالمخرج النهائي (تمثيل واعٍ بالسياق) =\n", output)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-10-chapter-10.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
   output = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, is_causal=False)
   ```
   هذه الدالة أسرع 2-4 مرات وتستهلك ذاكرة أقل بكثير — سنتعمّق في السبب الهندسي في الفصل 13 (Flash Attention).

2. **لا تنسَ `bias=False` في الإسقاطات الخطية لـ Q, K, V.** معظم التطبيقات الحديثة (LLaMA وغيرها) تتجاهل الـ bias في هذه الطبقات تحديدًا — يُقلّل عدد المعاملات قليلًا دون أثر يُذكر على الأداء.

3. **تحقّق دائمًا من مجموع صفوف `attention_weights` يساوي 1.0** كاختبار سلامة أولي عند بناء أي تطبيق جديد — إذا لم يكن كذلك، فهناك خطأ في محور الـ `softmax`.

4. **د_k الأصغر ليس بالضرورة أسوأ.** تقسيم `d_model=512` على `8` رؤوس يعطي `d_k=64` لكل رأس — هذا مقصود: رؤوس متعددة "أضيق" غالبًا أفضل من رأس واحد "عريض" لنفس الميزانية الحسابية.

---

## ⚠️ 10.5 أخطاء شائعة ومصائد (احذر منها!)

### ❌ الخطأ #1: الخلط بين Padding Mask و Causal Mask

هذا **أشهر خطأ** يرتكبه المهندسون عند بناء Transformer لأول مرة.
