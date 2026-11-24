#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل السابع عشر: نماذج الفضاء الحالة (State Space Models) وMamba
Large Language Model Engineering — State

Book slug : llm-engineering-69-chapters
Chapter   : ch-17-chapter-17 (lesson 17)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-17-chapter-17.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل السابع عشر: نماذج الفضاء الحالة (State Space Models) وMamba
Large Language Model Engineering — State

Book slug : llm-engineering-69-chapters
Chapter   : ch-17-chapter-17 (lesson 17)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-17-chapter-17.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل السابع عشر: نماذج الفضاء الحالة (State Space Models) وMamba
Large Language Model Engineering — State

Book slug : llm-engineering-69-chapters
Chapter   : ch-17-chapter-17 (lesson 17)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_17.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 17: الفصل السابع عشر: نماذج الفضاء الحالة (State Space Models) وMamba."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_state_space() -> None:
    inputs = [1.0, 0.0, 0.5, -0.2, 0.8]
    state, decay, gain = (0.0, 0.8, 0.25)
    outputs = []
    for value in inputs:
        state = decay * state + gain * value
        outputs.append(round(state, 4))
    print('input:', inputs)
    print('linear recurrent states:', outputs)
    print('work grows linearly with sequence length:', len(inputs))

if __name__ == "__main__":
    print("Chapter 17 lab: A recurrent state-space sequence model")
    run_state_space()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-04-modern-architectures/chapter-17.md
# ────────────────────────────────────────────────────────────────────────
# mamba_block_simplified.py
"""
تنفيذ مبسّط (توضيحي، وليس محسَّنًا للعتاد) لكتلة Mamba
لفهم الآلية الرياضية الأساسية.
Python 3.13+, PyTorch 2.x
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


class SimplifiedSelectiveSSM(nn.Module):
    """
    تطبيق توضيحي مبسّط لـ Selective SSM.
    يستخدم حلقة Python تسلسلية بدلاً من Parallel Scan المحسَّن
    لغرض الوضوح التعليمي فقط (وليس للكفاءة الفعلية على GPU).
    """

    def __init__(self, d_inner: int, d_state: int = 16) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        # A: مصفوفة ثابتة (لكن سالبة لضمان الاستقرار)، مُهيَّأة بطريقة تُشبه HiPPO المبسَّطة
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))   # نُخزّن log(A) للاستقرار العددي

        # الإسقاطات الانتقائية: B_t, C_t, Δ_t جميعها دوال من x
        self.proj_B = nn.Linear(d_inner, d_state, bias=False)
        self.proj_C = nn.Linear(d_inner, d_state, bias=False)
        self.proj_delta = nn.Linear(d_inner, d_inner, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape
        A = -torch.exp(self.A_log)  # (d_inner, d_state)، سالبة لضمان الاستقرار (اضمحلال وليس انفجار)

        delta = F.softplus(self.proj_delta(x))     # (batch, seq_len, d_inner)
        B = self.proj_B(x)                          # (batch, seq_len, d_state)
        C = self.proj_C(x)                          # (batch, seq_len, d_state)

        # تقطيع زمني مبسّط (تقريب أولي من رتبة أولى بدلاً من exp الدقيق)
        # Ā_t = exp(Δ_t · A) تقريبًا، B̄_t = Δ_t · B_t
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device)
        outputs = []

        for t in range(seq_len):
            delta_t = delta[:, t, :].unsqueeze(-1)      # (batch, d_inner, 1)
            A_bar = torch.exp(delta_t * A.unsqueeze(0))   # (batch, d_inner, d_state)
            B_bar = delta_t * B[:, t, :].unsqueeze(1)     # (batch, d_inner, d_state)

            # معادلة التكرار الأساسية: h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_t
            h = A_bar * h + B_bar * x[:, t, :].unsqueeze(-1)

            # y_t = C_t · h_t  (ضرب نقطي عبر بُعد الحالة d_state)
            y_t = torch.einsum("bdn,bn->bd", h, C[:, t, :])
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)   # (batch, seq_len, d_inner)


class MambaBlock(nn.Module):
    """كتلة Mamba كاملة: توسيع + Conv1D محلي + SSM انتقائي + بوابة + ضغط."""

    def __init__(self, d_model: int, expand_factor: int = 2, d_state: int = 16, d_conv: int = 4) -> None:
        super().__init__()
        d_inner = expand_factor * d_model

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)   # لتوليد الفرعين (x وgate) معًا
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv, groups=d_inner, padding=d_conv - 1
        )
        self.ssm = SimplifiedSelectiveSSM(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.d_inner = d_inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        x_and_gate = self.in_proj(x)                       # (batch, seq_len, 2*d_inner)
        x_branch, gate = x_and_gate.chunk(2, dim=-1)        # كل فرع (batch, seq_len, d_inner)

        # Conv1D محلي قصير (يلتقط أنماطًا محلية صغيرة، مشابهًا لـ n-gram بسيط)
        x_branch = x_branch.transpose(1, 2)                  # (batch, d_inner, seq_len)
        x_branch = self.conv1d(x_branch)[:, :, :seq_len]     # قص الحشو الزائد من padding
        x_branch = x_branch.transpose(1, 2)                  # (batch, seq_len, d_inner)
        x_branch = F.silu(x_branch)

        ssm_output = self.ssm(x_branch)                      # (batch, seq_len, d_inner)

        gated_output = ssm_output * F.silu(gate)              # البوابة الانتقائية
        return self.out_proj(gated_output)


if __name__ == "__main__":
    d_model, seq_len, batch = 64, 20, 2
    mamba_block = MambaBlock(d_model)

    x = torch.randn(batch, seq_len, d_model)
    output = mamba_block(x)

    print(f"شكل الإدخال : {x.shape}")
    print(f"شكل الإخراج : {output.shape}")

    n_params = sum(p.numel() for p in mamba_block.parameters())
    print(f"عدد معاملات كتلة Mamba واحدة: {n_params:,}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-04-modern-architectures/chapter-17.md
# ────────────────────────────────────────────────────────────────────────
شكل الإدخال : torch.Size([2, 20, 64])
شكل الإخراج : torch.Size([2, 20, 64])
عدد معاملات كتلة Mamba واحدة: 25,536

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-04-modern-architectures/chapter-17.md
# ────────────────────────────────────────────────────────────────────────
# ❌ فهم خاطئ: معاملة delta كأنها hyperparameter تدريب عادي
learning_rate = delta  # خطأ مفاهيمي كامل!

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-04-modern-architectures/chapter-17.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: السماح لـ A أن تكون موجبة قد يُسبب انفجارًا أسيًا في الحالة الخفية
A = nn.Parameter(torch.randn(d_inner, d_state))  # بدون ضمان السالبية!

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-04-modern-architectures/chapter-17.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: نُخزّن log(-A) ونُطبّق -exp() لضمان البقاء سالبة دائمًا
self.A_log = nn.Parameter(torch.log(A_positive_init))
A = -torch.exp(self.A_log)   # دائمًا سالبة → اضمحلال مستقر، لا انفجار

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-04-modern-architectures/chapter-17.md
# ────────────────────────────────────────────────────────────────────────
# mini_project_mamba_vs_transformer_copying.py
from __future__ import annotations

import torch
import torch.nn as nn

from mamba_block_simplified import MambaBlock

# TODO:
# 1) بناء مولّد بيانات لمهمة النسخ: تسلسل عشوائي بطول L، ثم <COPY>، ثم نفس التسلسل كهدف
# 2) بناء نموذجين صغيرين متكافئين تقريبًا في عدد المعاملات:
#    (أ) GPT-style (من الفصل 12) بطبقات Self-Attention + FFN
#    (ب) Mamba-style بطبقات MambaBlock فقط
# 3) درّب كلا النموذجين على تسلسلات قصيرة (L=32) فقط
# 4) اختبر على L = [32, 128, 512, 2048] (معظمها أطول بكثير من التدريب!)
# 5) قِس دقة النسخ (نسبة الرموز الصحيحة) وزمن الاستدلال الفعلي لكل طول ولكل معمارية
# 6) لاحظ وناقش: كيف يتدهور أداء كل معمارية مع زيادة الطول؟ وكيف يتغير زمن الاستدلال نسبيًا؟

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-17-chapter-17.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 17: الفصل السابع عشر: نماذج الفضاء الحالة (State Space Models) وMamba."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_state_space() -> None:
    inputs = [1.0, 0.0, 0.5, -0.2, 0.8]
    state, decay, gain = (0.0, 0.8, 0.25)
    outputs = []
    for value in inputs:
        state = decay * state + gain * value
        outputs.append(round(state, 4))
    print('input:', inputs)
    print('linear recurrent states:', outputs)
    print('work grows linearly with sequence length:', len(inputs))

if __name__ == "__main__":
    print("Chapter 17 lab: A recurrent state-space sequence model")
    run_state_space()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-17-chapter-17.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# mamba_block_simplified.py
"""
تنفيذ مبسّط (توضيحي، وليس محسَّنًا للعتاد) لكتلة Mamba
لفهم الآلية الرياضية الأساسية.
Python 3.13+, PyTorch 2.x
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


class SimplifiedSelectiveSSM(nn.Module):
    """
    تطبيق توضيحي مبسّط لـ Selective SSM.
    يستخدم حلقة Python تسلسلية بدلاً من Parallel Scan المحسَّن
    لغرض الوضوح التعليمي فقط (وليس للكفاءة الفعلية على GPU).
    """

    def __init__(self, d_inner: int, d_state: int = 16) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        # A: مصفوفة ثابتة (لكن سالبة لضمان الاستقرار)، مُهيَّأة بطريقة تُشبه HiPPO المبسَّطة
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))   # نُخزّن log(A) للاستقرار العددي

        # الإسقاطات الانتقائية: B_t, C_t, Δ_t جميعها دوال من x
        self.proj_B = nn.Linear(d_inner, d_state, bias=False)
        self.proj_C = nn.Linear(d_inner, d_state, bias=False)
        self.proj_delta = nn.Linear(d_inner, d_inner, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape
        A = -torch.exp(self.A_log)  # (d_inner, d_state)، سالبة لضمان الاستقرار (اضمحلال وليس انفجار)

        delta = F.softplus(self.proj_delta(x))     # (batch, seq_len, d_inner)
        B = self.proj_B(x)                          # (batch, seq_len, d_state)
        C = self.proj_C(x)                          # (batch, seq_len, d_state)

        # تقطيع زمني مبسّط (تقريب أولي من رتبة أولى بدلاً من exp الدقيق)
        # Ā_t = exp(Δ_t · A) تقريبًا، B̄_t = Δ_t · B_t
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device)
        outputs = []

        for t in range(seq_len):
            delta_t = delta[:, t, :].unsqueeze(-1)      # (batch, d_inner, 1)
            A_bar = torch.exp(delta_t * A.unsqueeze(0))   # (batch, d_inner, d_state)
            B_bar = delta_t * B[:, t, :].unsqueeze(1)     # (batch, d_inner, d_state)

            # معادلة التكرار الأساسية: h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_t
            h = A_bar * h + B_bar * x[:, t, :].unsqueeze(-1)

            # y_t = C_t · h_t  (ضرب نقطي عبر بُعد الحالة d_state)
            y_t = torch.einsum("bdn,bn->bd", h, C[:, t, :])
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)   # (batch, seq_len, d_inner)


class MambaBlock(nn.Module):
    """كتلة Mamba كاملة: توسيع + Conv1D محلي + SSM انتقائي + بوابة + ضغط."""

    def __init__(self, d_model: int, expand_factor: int = 2, d_state: int = 16, d_conv: int = 4) -> None:
        super().__init__()
        d_inner = expand_factor * d_model

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)   # لتوليد الفرعين (x وgate) معًا
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv, groups=d_inner, padding=d_conv - 1
        )
        self.ssm = SimplifiedSelectiveSSM(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.d_inner = d_inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        x_and_gate = self.in_proj(x)                       # (batch, seq_len, 2*d_inner)
        x_branch, gate = x_and_gate.chunk(2, dim=-1)        # كل فرع (batch, seq_len, d_inner)

        # Conv1D محلي قصير (يلتقط أنماطًا محلية صغيرة، مشابهًا لـ n-gram بسيط)
        x_branch = x_branch.transpose(1, 2)                  # (batch, d_inner, seq_len)
        x_branch = self.conv1d(x_branch)[:, :, :seq_len]     # قص الحشو الزائد من padding
        x_branch = x_branch.transpose(1, 2)                  # (batch, seq_len, d_inner)
        x_branch = F.silu(x_branch)

        ssm_output = self.ssm(x_branch)                      # (batch, seq_len, d_inner)

        gated_output = ssm_output * F.silu(gate)              # البوابة الانتقائية
        return self.out_proj(gated_output)


if __name__ == "__main__":
    d_model, seq_len, batch = 64, 20, 2
    mamba_block = MambaBlock(d_model)

    x = torch.randn(batch, seq_len, d_model)
    output = mamba_block(x)

    print(f"شكل الإدخال : {x.shape}")
    print(f"شكل الإخراج : {output.shape}")

    n_params = sum(p.numel() for p in mamba_block.parameters())
    print(f"عدد معاملات كتلة Mamba واحدة: {n_params:,}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-17-chapter-17.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
شكل الإدخال : torch.Size([2, 20, 64])
شكل الإخراج : torch.Size([2, 20, 64])
عدد معاملات كتلة Mamba واحدة: 25,536

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-17-chapter-17.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ❌ فهم خاطئ: معاملة delta كأنها hyperparameter تدريب عادي
learning_rate = delta  # خطأ مفاهيمي كامل!

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-17-chapter-17.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: السماح لـ A أن تكون موجبة قد يُسبب انفجارًا أسيًا في الحالة الخفية
A = nn.Parameter(torch.randn(d_inner, d_state))  # بدون ضمان السالبية!

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-17-chapter-17.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: نُخزّن log(-A) ونُطبّق -exp() لضمان البقاء سالبة دائمًا
self.A_log = nn.Parameter(torch.log(A_positive_init))
A = -torch.exp(self.A_log)   # دائمًا سالبة → اضمحلال مستقر، لا انفجار

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-17-chapter-17.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# mini_project_mamba_vs_transformer_copying.py
from __future__ import annotations

import torch
import torch.nn as nn

from mamba_block_simplified import MambaBlock

# TODO:
# 1) بناء مولّد بيانات لمهمة النسخ: تسلسل عشوائي بطول L، ثم <COPY>، ثم نفس التسلسل كهدف
# 2) بناء نموذجين صغيرين متكافئين تقريبًا في عدد المعاملات:
#    (أ) GPT-style (من الفصل 12) بطبقات Self-Attention + FFN
#    (ب) Mamba-style بطبقات MambaBlock فقط
# 3) درّب كلا النموذجين على تسلسلات قصيرة (L=32) فقط
# 4) اختبر على L = [32, 128, 512, 2048] (معظمها أطول بكثير من التدريب!)
# 5) قِس دقة النسخ (نسبة الرموز الصحيحة) وزمن الاستدلال الفعلي لكل طول ولكل معمارية
# 6) لاحظ وناقش: كيف يتدهور أداء كل معمارية مع زيادة الطول؟ وكيف يتغير زمن الاستدلال نسبيًا؟

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-17-chapter-17.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل السابع عشر: نماذج الفضاء الحالة (State Space Models) وMamba
Large Language Model Engineering — State

Book slug : llm-engineering-69-chapters
Chapter   : ch-17-chapter-17 (lesson 17)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_17.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 17: الفصل السابع عشر: نماذج الفضاء الحالة (State Space Models) وMamba."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_state_space() -> None:
    inputs = [1.0, 0.0, 0.5, -0.2, 0.8]
    state, decay, gain = (0.0, 0.8, 0.25)
    outputs = []
    for value in inputs:
        state = decay * state + gain * value
        outputs.append(round(state, 4))
    print('input:', inputs)
    print('linear recurrent states:', outputs)
    print('work grows linearly with sequence length:', len(inputs))

if __name__ == "__main__":
    print("Chapter 17 lab: A recurrent state-space sequence model")
    run_state_space()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-04-modern-architectures/chapter-17.md
# ────────────────────────────────────────────────────────────────────────
# mamba_block_simplified.py
"""
تنفيذ مبسّط (توضيحي، وليس محسَّنًا للعتاد) لكتلة Mamba
لفهم الآلية الرياضية الأساسية.
Python 3.13+, PyTorch 2.x
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


class SimplifiedSelectiveSSM(nn.Module):
    """
    تطبيق توضيحي مبسّط لـ Selective SSM.
    يستخدم حلقة Python تسلسلية بدلاً من Parallel Scan المحسَّن
    لغرض الوضوح التعليمي فقط (وليس للكفاءة الفعلية على GPU).
    """

    def __init__(self, d_inner: int, d_state: int = 16) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        # A: مصفوفة ثابتة (لكن سالبة لضمان الاستقرار)، مُهيَّأة بطريقة تُشبه HiPPO المبسَّطة
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))   # نُخزّن log(A) للاستقرار العددي

        # الإسقاطات الانتقائية: B_t, C_t, Δ_t جميعها دوال من x
        self.proj_B = nn.Linear(d_inner, d_state, bias=False)
        self.proj_C = nn.Linear(d_inner, d_state, bias=False)
        self.proj_delta = nn.Linear(d_inner, d_inner, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape
        A = -torch.exp(self.A_log)  # (d_inner, d_state)، سالبة لضمان الاستقرار (اضمحلال وليس انفجار)

        delta = F.softplus(self.proj_delta(x))     # (batch, seq_len, d_inner)
        B = self.proj_B(x)                          # (batch, seq_len, d_state)
        C = self.proj_C(x)                          # (batch, seq_len, d_state)

        # تقطيع زمني مبسّط (تقريب أولي من رتبة أولى بدلاً من exp الدقيق)
        # Ā_t = exp(Δ_t · A) تقريبًا، B̄_t = Δ_t · B_t
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device)
        outputs = []

        for t in range(seq_len):
            delta_t = delta[:, t, :].unsqueeze(-1)      # (batch, d_inner, 1)
            A_bar = torch.exp(delta_t * A.unsqueeze(0))   # (batch, d_inner, d_state)
            B_bar = delta_t * B[:, t, :].unsqueeze(1)     # (batch, d_inner, d_state)

            # معادلة التكرار الأساسية: h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_t
            h = A_bar * h + B_bar * x[:, t, :].unsqueeze(-1)

            # y_t = C_t · h_t  (ضرب نقطي عبر بُعد الحالة d_state)
            y_t = torch.einsum("bdn,bn->bd", h, C[:, t, :])
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)   # (batch, seq_len, d_inner)


class MambaBlock(nn.Module):
    """كتلة Mamba كاملة: توسيع + Conv1D محلي + SSM انتقائي + بوابة + ضغط."""

    def __init__(self, d_model: int, expand_factor: int = 2, d_state: int = 16, d_conv: int = 4) -> None:
        super().__init__()
        d_inner = expand_factor * d_model

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)   # لتوليد الفرعين (x وgate) معًا
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv, groups=d_inner, padding=d_conv - 1
        )
        self.ssm = SimplifiedSelectiveSSM(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.d_inner = d_inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        x_and_gate = self.in_proj(x)                       # (batch, seq_len, 2*d_inner)
        x_branch, gate = x_and_gate.chunk(2, dim=-1)        # كل فرع (batch, seq_len, d_inner)

        # Conv1D محلي قصير (يلتقط أنماطًا محلية صغيرة، مشابهًا لـ n-gram بسيط)
        x_branch = x_branch.transpose(1, 2)                  # (batch, d_inner, seq_len)
        x_branch = self.conv1d(x_branch)[:, :, :seq_len]     # قص الحشو الزائد من padding
        x_branch = x_branch.transpose(1, 2)                  # (batch, seq_len, d_inner)
        x_branch = F.silu(x_branch)

        ssm_output = self.ssm(x_branch)                      # (batch, seq_len, d_inner)

        gated_output = ssm_output * F.silu(gate)              # البوابة الانتقائية
        return self.out_proj(gated_output)


if __name__ == "__main__":
    d_model, seq_len, batch = 64, 20, 2
    mamba_block = MambaBlock(d_model)

    x = torch.randn(batch, seq_len, d_model)
    output = mamba_block(x)

    print(f"شكل الإدخال : {x.shape}")
    print(f"شكل الإخراج : {output.shape}")

    n_params = sum(p.numel() for p in mamba_block.parameters())
    print(f"عدد معاملات كتلة Mamba واحدة: {n_params:,}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-04-modern-architectures/chapter-17.md
# ────────────────────────────────────────────────────────────────────────
شكل الإدخال : torch.Size([2, 20, 64])
شكل الإخراج : torch.Size([2, 20, 64])
عدد معاملات كتلة Mamba واحدة: 25,536

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-04-modern-architectures/chapter-17.md
# ────────────────────────────────────────────────────────────────────────
# ❌ فهم خاطئ: معاملة delta كأنها hyperparameter تدريب عادي
learning_rate = delta  # خطأ مفاهيمي كامل!

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-04-modern-architectures/chapter-17.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: السماح لـ A أن تكون موجبة قد يُسبب انفجارًا أسيًا في الحالة الخفية
A = nn.Parameter(torch.randn(d_inner, d_state))  # بدون ضمان السالبية!

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-04-modern-architectures/chapter-17.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: نُخزّن log(-A) ونُطبّق -exp() لضمان البقاء سالبة دائمًا
self.A_log = nn.Parameter(torch.log(A_positive_init))
A = -torch.exp(self.A_log)   # دائمًا سالبة → اضمحلال مستقر، لا انفجار

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-04-modern-architectures/chapter-17.md
# ────────────────────────────────────────────────────────────────────────
# mini_project_mamba_vs_transformer_copying.py
from __future__ import annotations

import torch
import torch.nn as nn

from mamba_block_simplified import MambaBlock

# TODO:
# 1) بناء مولّد بيانات لمهمة النسخ: تسلسل عشوائي بطول L، ثم <COPY>، ثم نفس التسلسل كهدف
# 2) بناء نموذجين صغيرين متكافئين تقريبًا في عدد المعاملات:
#    (أ) GPT-style (من الفصل 12) بطبقات Self-Attention + FFN
#    (ب) Mamba-style بطبقات MambaBlock فقط
# 3) درّب كلا النموذجين على تسلسلات قصيرة (L=32) فقط
# 4) اختبر على L = [32, 128, 512, 2048] (معظمها أطول بكثير من التدريب!)
# 5) قِس دقة النسخ (نسبة الرموز الصحيحة) وزمن الاستدلال الفعلي لكل طول ولكل معمارية
# 6) لاحظ وناقش: كيف يتدهور أداء كل معمارية مع زيادة الطول؟ وكيف يتغير زمن الاستدلال نسبيًا؟

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-17-chapter-17.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 17: الفصل السابع عشر: نماذج الفضاء الحالة (State Space Models) وMamba."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_state_space() -> None:
    inputs = [1.0, 0.0, 0.5, -0.2, 0.8]
    state, decay, gain = (0.0, 0.8, 0.25)
    outputs = []
    for value in inputs:
        state = decay * state + gain * value
        outputs.append(round(state, 4))
    print('input:', inputs)
    print('linear recurrent states:', outputs)
    print('work grows linearly with sequence length:', len(inputs))

if __name__ == "__main__":
    print("Chapter 17 lab: A recurrent state-space sequence model")
    run_state_space()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-17-chapter-17.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# mamba_block_simplified.py
"""
تنفيذ مبسّط (توضيحي، وليس محسَّنًا للعتاد) لكتلة Mamba
لفهم الآلية الرياضية الأساسية.
Python 3.13+, PyTorch 2.x
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


class SimplifiedSelectiveSSM(nn.Module):
    """
    تطبيق توضيحي مبسّط لـ Selective SSM.
    يستخدم حلقة Python تسلسلية بدلاً من Parallel Scan المحسَّن
    لغرض الوضوح التعليمي فقط (وليس للكفاءة الفعلية على GPU).
    """

    def __init__(self, d_inner: int, d_state: int = 16) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        # A: مصفوفة ثابتة (لكن سالبة لضمان الاستقرار)، مُهيَّأة بطريقة تُشبه HiPPO المبسَّطة
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))   # نُخزّن log(A) للاستقرار العددي

        # الإسقاطات الانتقائية: B_t, C_t, Δ_t جميعها دوال من x
        self.proj_B = nn.Linear(d_inner, d_state, bias=False)
        self.proj_C = nn.Linear(d_inner, d_state, bias=False)
        self.proj_delta = nn.Linear(d_inner, d_inner, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape
        A = -torch.exp(self.A_log)  # (d_inner, d_state)، سالبة لضمان الاستقرار (اضمحلال وليس انفجار)

        delta = F.softplus(self.proj_delta(x))     # (batch, seq_len, d_inner)
        B = self.proj_B(x)                          # (batch, seq_len, d_state)
        C = self.proj_C(x)                          # (batch, seq_len, d_state)

        # تقطيع زمني مبسّط (تقريب أولي من رتبة أولى بدلاً من exp الدقيق)
        # Ā_t = exp(Δ_t · A) تقريبًا، B̄_t = Δ_t · B_t
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device)
        outputs = []

        for t in range(seq_len):
            delta_t = delta[:, t, :].unsqueeze(-1)      # (batch, d_inner, 1)
            A_bar = torch.exp(delta_t * A.unsqueeze(0))   # (batch, d_inner, d_state)
            B_bar = delta_t * B[:, t, :].unsqueeze(1)     # (batch, d_inner, d_state)

            # معادلة التكرار الأساسية: h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_t
            h = A_bar * h + B_bar * x[:, t, :].unsqueeze(-1)

            # y_t = C_t · h_t  (ضرب نقطي عبر بُعد الحالة d_state)
            y_t = torch.einsum("bdn,bn->bd", h, C[:, t, :])
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)   # (batch, seq_len, d_inner)


class MambaBlock(nn.Module):
    """كتلة Mamba كاملة: توسيع + Conv1D محلي + SSM انتقائي + بوابة + ضغط."""

    def __init__(self, d_model: int, expand_factor: int = 2, d_state: int = 16, d_conv: int = 4) -> None:
        super().__init__()
        d_inner = expand_factor * d_model

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)   # لتوليد الفرعين (x وgate) معًا
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv, groups=d_inner, padding=d_conv - 1
        )
        self.ssm = SimplifiedSelectiveSSM(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.d_inner = d_inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        x_and_gate = self.in_proj(x)                       # (batch, seq_len, 2*d_inner)
        x_branch, gate = x_and_gate.chunk(2, dim=-1)        # كل فرع (batch, seq_len, d_inner)

        # Conv1D محلي قصير (يلتقط أنماطًا محلية صغيرة، مشابهًا لـ n-gram بسيط)
        x_branch = x_branch.transpose(1, 2)                  # (batch, d_inner, seq_len)
        x_branch = self.conv1d(x_branch)[:, :, :seq_len]     # قص الحشو الزائد من padding
        x_branch = x_branch.transpose(1, 2)                  # (batch, seq_len, d_inner)
        x_branch = F.silu(x_branch)

        ssm_output = self.ssm(x_branch)                      # (batch, seq_len, d_inner)

        gated_output = ssm_output * F.silu(gate)              # البوابة الانتقائية
        return self.out_proj(gated_output)


if __name__ == "__main__":
    d_model, seq_len, batch = 64, 20, 2
    mamba_block = MambaBlock(d_model)

    x = torch.randn(batch, seq_len, d_model)
    output = mamba_block(x)

    print(f"شكل الإدخال : {x.shape}")
    print(f"شكل الإخراج : {output.shape}")

    n_params = sum(p.numel() for p in mamba_block.parameters())
    print(f"عدد معاملات كتلة Mamba واحدة: {n_params:,}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-17-chapter-17.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
شكل الإدخال : torch.Size([2, 20, 64])
شكل الإخراج : torch.Size([2, 20, 64])
عدد معاملات كتلة Mamba واحدة: 25,536

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-17-chapter-17.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ❌ فهم خاطئ: معاملة delta كأنها hyperparameter تدريب عادي
learning_rate = delta  # خطأ مفاهيمي كامل!

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-17-chapter-17.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: السماح لـ A أن تكون موجبة قد يُسبب انفجارًا أسيًا في الحالة الخفية
A = nn.Parameter(torch.randn(d_inner, d_state))  # بدون ضمان السالبية!

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-17-chapter-17.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: نُخزّن log(-A) ونُطبّق -exp() لضمان البقاء سالبة دائمًا
self.A_log = nn.Parameter(torch.log(A_positive_init))
A = -torch.exp(self.A_log)   # دائمًا سالبة → اضمحلال مستقر، لا انفجار

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-17-chapter-17.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# mini_project_mamba_vs_transformer_copying.py
from __future__ import annotations

import torch
import torch.nn as nn

from mamba_block_simplified import MambaBlock

# TODO:
# 1) بناء مولّد بيانات لمهمة النسخ: تسلسل عشوائي بطول L، ثم <COPY>، ثم نفس التسلسل كهدف
# 2) بناء نموذجين صغيرين متكافئين تقريبًا في عدد المعاملات:
#    (أ) GPT-style (من الفصل 12) بطبقات Self-Attention + FFN
#    (ب) Mamba-style بطبقات MambaBlock فقط
# 3) درّب كلا النموذجين على تسلسلات قصيرة (L=32) فقط
# 4) اختبر على L = [32, 128, 512, 2048] (معظمها أطول بكثير من التدريب!)
# 5) قِس دقة النسخ (نسبة الرموز الصحيحة) وزمن الاستدلال الفعلي لكل طول ولكل معمارية
# 6) لاحظ وناقش: كيف يتدهور أداء كل معمارية مع زيادة الطول؟ وكيف يتغير زمن الاستدلال نسبيًا؟
