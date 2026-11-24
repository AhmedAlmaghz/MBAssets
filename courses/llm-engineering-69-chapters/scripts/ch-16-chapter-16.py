#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل السادس عشر: Mixture of Experts — فصل الحجم عن التكلفة
Large Language Model Engineering — Mixture

Book slug : llm-engineering-69-chapters
Chapter   : ch-16-chapter-16 (lesson 16)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-16-chapter-16.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل السادس عشر: Mixture of Experts — فصل الحجم عن التكلفة
Large Language Model Engineering — Mixture

Book slug : llm-engineering-69-chapters
Chapter   : ch-16-chapter-16 (lesson 16)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-16-chapter-16.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل السادس عشر: Mixture of Experts — فصل الحجم عن التكلفة
Large Language Model Engineering — Mixture

Book slug : llm-engineering-69-chapters
Chapter   : ch-16-chapter-16 (lesson 16)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_16.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 16: الفصل السادس عشر: Mixture of Experts — فصل الحجم عن التكلفة."""

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

def run_moe() -> None:
    scores = [0.2, 1.4, 0.8, -0.1]
    top_two = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:2]
    weights = softmax([scores[index] for index in top_two])
    expert_load = Counter(top_two)
    print('selected experts:', top_two)
    print('mixture weights:', [round(x, 3) for x in weights])
    print('active parameters:', '2 of', len(scores), 'experts')
    print('routing counts:', dict(expert_load))

if __name__ == "__main__":
    print("Chapter 16 lab: Top-k mixture-of-experts routing")
    run_moe()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-04-modern-architectures/chapter-16.md
# ────────────────────────────────────────────────────────────────────────
# mixture_of_experts.py
"""
تنفيذ كامل لطبقة Mixture of Experts مع Top-K Routing
وLoad Balancing Loss، بأسلوب مشابه لـ Switch Transformer / Mixtral.
Python 3.13+, PyTorch 2.x
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


class Expert(nn.Module):
    """خبير واحد: شبكة FFN قياسية بالضبط كما في الفصل 11."""

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


class MoELayer(nn.Module):
    """
    طبقة Mixture of Experts مع Top-K Routing.
    تُعيد: (الإخراج، خسارة موازنة الحمل)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff) for _ in range(n_experts)]
        )
        self.router = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)                     # (batch*seq_len, d_model)
        n_tokens = x_flat.size(0)

        # 1) حساب درجات الموجِّه واحتمالاته
        router_logits = self.router(x_flat)               # (n_tokens, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)    # (n_tokens, n_experts)

        # 2) اختيار Top-K خبراء لكل رمز
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # إعادة تطبيع الاحتمالات المختارة لتجمع إلى 1 (ممارسة شائعة)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # 3) حساب سعة كل خبير (لتجنب الفيضان على خبير واحد)
        capacity = int(self.capacity_factor * n_tokens * self.top_k / self.n_experts)

        output = torch.zeros_like(x_flat)
        expert_usage_count = torch.zeros(self.n_experts, device=x.device)

        for expert_idx in range(self.n_experts):
            # أي الرموز اختارت هذا الخبير ضمن Top-K؟
            mask = (topk_indices == expert_idx)             # (n_tokens, top_k) bool
            token_positions, k_positions = mask.nonzero(as_tuple=True)

            if token_positions.numel() == 0:
                continue

            # تطبيق سعة الخبير: أخذ فقط أول `capacity` رمز، إسقاط الباقي
            if token_positions.numel() > capacity:
                token_positions = token_positions[:capacity]
                k_positions = k_positions[:capacity]

            expert_usage_count[expert_idx] = token_positions.numel()

            selected_tokens = x_flat[token_positions]
            expert_output = self.experts[expert_idx](selected_tokens)

            gating_weights = topk_probs[token_positions, k_positions].unsqueeze(-1)
            output[token_positions] += gating_weights * expert_output

        output = output.view(batch, seq_len, d_model)

        # 4) خسارة موازنة الحمل
        balance_loss = self._compute_load_balancing_loss(router_probs, topk_indices, n_tokens)

        return output, balance_loss

    def _compute_load_balancing_loss(
        self, router_probs: torch.Tensor, topk_indices: torch.Tensor, n_tokens: int
    ) -> torch.Tensor:
        # f_i: نسبة الرموز التي اختارت الخبير i ضمن Top-K (تجريبيًا)
        one_hot = F.one_hot(topk_indices, num_classes=self.n_experts).float()  # (n_tokens, top_k, n_experts)
        f_i = one_hot.sum(dim=(0, 1)) / (n_tokens * self.top_k)

        # P_i: متوسط احتمال الموجِّه لكل خبير عبر كل الرموز
        p_i = router_probs.mean(dim=0)

        return self.n_experts * torch.sum(f_i * p_i)


if __name__ == "__main__":
    d_model, d_ff, n_experts, top_k = 64, 256, 8, 2
    moe_layer = MoELayer(d_model, d_ff, n_experts, top_k)

    batch, seq_len = 4, 16
    x = torch.randn(batch, seq_len, d_model)

    output, balance_loss = moe_layer(x)

    print(f"شكل الإدخال  : {x.shape}")
    print(f"شكل الإخراج  : {output.shape}")
    print(f"خسارة التوازن: {balance_loss.item():.4f}")

    # حساب عدد المعاملات: إجمالية مقابل فعّالة لكل رمز
    total_params = sum(p.numel() for p in moe_layer.experts.parameters())
    active_params_per_token = total_params * top_k / n_experts
    print(f"\nإجمالي معاملات الخبراء      : {total_params:,}")
    print(f"المعاملات الفعّالة لكل رمز   : {active_params_per_token:,.0f}")
    print(f"نسبة التفعيل                : {100*top_k/n_experts:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-04-modern-architectures/chapter-16.md
# ────────────────────────────────────────────────────────────────────────
شكل الإدخال  : torch.Size([4, 16, 64])
شكل الإخراج  : torch.Size([4, 16, 64])
خسارة التوازن: 1.0234

إجمالي معاملات الخبراء      : 266,240
المعاملات الفعّالة لكل رمز   : 66,560
نسبة التفعيل                : 25.0%

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-04-modern-architectures/chapter-16.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: معامل موازنة كبير جدًا يُهيمن على هدف التعلّم الأساسي
total_loss = task_loss + 1.0 * balance_loss  # α=1.0 كبير جدًا، سيُشوّه التعلّم!

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-04-modern-architectures/chapter-16.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: معامل صغير يُحافظ على توازن خفيف دون التأثير على المهمة الأساسية
total_loss = task_loss + 0.01 * balance_loss

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-04-modern-architectures/chapter-16.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: استخدام الاحتمالات الخام من softmax الكامل دون تطبيع بعد اختيار Top-K
# قد يجمع إلى قيمة أقل من 1 بكثير، مما يُصغّر إخراج الطبقة بشكل غير مقصود
y = topk_probs_raw[0] * E_a(x) + topk_probs_raw[1] * E_b(x)  # قد يكون المجموع 0.3 فقط!

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-04-modern-architectures/chapter-16.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: إعادة التطبيع لضمان أن أوزان الدمج تجمع إلى 1
topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-04-modern-architectures/chapter-16.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: capacity_factor = 1.0 مع عدم مراقبة نسبة الرموز المُسقَطة
moe = MoELayer(d_model, d_ff, n_experts=8, top_k=2, capacity_factor=1.0)
# إذا كان التوزيع غير متوازن حتى قليلاً، سيُسقط رموزًا كثيرة دون أن تلاحظ!

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-04-modern-architectures/chapter-16.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: راقب فعليًا نسبة الإسقاط، واضبط capacity_factor بناءً عليها
dropped_ratio = 1 - (total_processed_tokens / total_tokens)
assert dropped_ratio < 0.05, "نسبة إسقاط مرتفعة جدًا، فكّر في زيادة capacity_factor"

# ────────────────────────────────────────────────────────────────────────
# [010] book/chapters/part-04-modern-architectures/chapter-16.md
# ────────────────────────────────────────────────────────────────────────
# mini_project_moe_gpt_block.py
from __future__ import annotations

import torch
import torch.nn as nn

from mixture_of_experts import MoELayer

# TODO:
# 1) بناء GPTMoEBlock: نفس GPTBlock من الفصل 12، لكن استبدل self.ffn بـ MoELayer
#    تذكّر: forward يجب أن يُعيد أيضًا balance_loss لتجميعها عبر كل الطبقات
# 2) بناء GPTMoEModel كامل بعدة طبقات، يُجمّع balance_loss من كل الطبقات
# 3) درّب نموذجين (Dense وMoE) على نفس بيانات نصية بسيطة لعدد خطوات متساوٍ
# 4) قارن: (أ) منحنى الخسارة، (ب) الوقت الفعلي للتدريب لكل خطوة،
#    (ج) إجمالي المعاملات مقابل المعاملات الفعّالة لكل نموذج
# 5) ارسم توزيع استخدام الخبراء عبر التدريب (هل تقارب التوزيع المتساوي مع التقدم؟)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-16-chapter-16.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 16: الفصل السادس عشر: Mixture of Experts — فصل الحجم عن التكلفة."""

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

def run_moe() -> None:
    scores = [0.2, 1.4, 0.8, -0.1]
    top_two = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:2]
    weights = softmax([scores[index] for index in top_two])
    expert_load = Counter(top_two)
    print('selected experts:', top_two)
    print('mixture weights:', [round(x, 3) for x in weights])
    print('active parameters:', '2 of', len(scores), 'experts')
    print('routing counts:', dict(expert_load))

if __name__ == "__main__":
    print("Chapter 16 lab: Top-k mixture-of-experts routing")
    run_moe()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-16-chapter-16.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# mixture_of_experts.py
"""
تنفيذ كامل لطبقة Mixture of Experts مع Top-K Routing
وLoad Balancing Loss، بأسلوب مشابه لـ Switch Transformer / Mixtral.
Python 3.13+, PyTorch 2.x
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


class Expert(nn.Module):
    """خبير واحد: شبكة FFN قياسية بالضبط كما في الفصل 11."""

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


class MoELayer(nn.Module):
    """
    طبقة Mixture of Experts مع Top-K Routing.
    تُعيد: (الإخراج، خسارة موازنة الحمل)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff) for _ in range(n_experts)]
        )
        self.router = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)                     # (batch*seq_len, d_model)
        n_tokens = x_flat.size(0)

        # 1) حساب درجات الموجِّه واحتمالاته
        router_logits = self.router(x_flat)               # (n_tokens, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)    # (n_tokens, n_experts)

        # 2) اختيار Top-K خبراء لكل رمز
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # إعادة تطبيع الاحتمالات المختارة لتجمع إلى 1 (ممارسة شائعة)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # 3) حساب سعة كل خبير (لتجنب الفيضان على خبير واحد)
        capacity = int(self.capacity_factor * n_tokens * self.top_k / self.n_experts)

        output = torch.zeros_like(x_flat)
        expert_usage_count = torch.zeros(self.n_experts, device=x.device)

        for expert_idx in range(self.n_experts):
            # أي الرموز اختارت هذا الخبير ضمن Top-K؟
            mask = (topk_indices == expert_idx)             # (n_tokens, top_k) bool
            token_positions, k_positions = mask.nonzero(as_tuple=True)

            if token_positions.numel() == 0:
                continue

            # تطبيق سعة الخبير: أخذ فقط أول `capacity` رمز، إسقاط الباقي
            if token_positions.numel() > capacity:
                token_positions = token_positions[:capacity]
                k_positions = k_positions[:capacity]

            expert_usage_count[expert_idx] = token_positions.numel()

            selected_tokens = x_flat[token_positions]
            expert_output = self.experts[expert_idx](selected_tokens)

            gating_weights = topk_probs[token_positions, k_positions].unsqueeze(-1)
            output[token_positions] += gating_weights * expert_output

        output = output.view(batch, seq_len, d_model)

        # 4) خسارة موازنة الحمل
        balance_loss = self._compute_load_balancing_loss(router_probs, topk_indices, n_tokens)

        return output, balance_loss

    def _compute_load_balancing_loss(
        self, router_probs: torch.Tensor, topk_indices: torch.Tensor, n_tokens: int
    ) -> torch.Tensor:
        # f_i: نسبة الرموز التي اختارت الخبير i ضمن Top-K (تجريبيًا)
        one_hot = F.one_hot(topk_indices, num_classes=self.n_experts).float()  # (n_tokens, top_k, n_experts)
        f_i = one_hot.sum(dim=(0, 1)) / (n_tokens * self.top_k)

        # P_i: متوسط احتمال الموجِّه لكل خبير عبر كل الرموز
        p_i = router_probs.mean(dim=0)

        return self.n_experts * torch.sum(f_i * p_i)


if __name__ == "__main__":
    d_model, d_ff, n_experts, top_k = 64, 256, 8, 2
    moe_layer = MoELayer(d_model, d_ff, n_experts, top_k)

    batch, seq_len = 4, 16
    x = torch.randn(batch, seq_len, d_model)

    output, balance_loss = moe_layer(x)

    print(f"شكل الإدخال  : {x.shape}")
    print(f"شكل الإخراج  : {output.shape}")
    print(f"خسارة التوازن: {balance_loss.item():.4f}")

    # حساب عدد المعاملات: إجمالية مقابل فعّالة لكل رمز
    total_params = sum(p.numel() for p in moe_layer.experts.parameters())
    active_params_per_token = total_params * top_k / n_experts
    print(f"\nإجمالي معاملات الخبراء      : {total_params:,}")
    print(f"المعاملات الفعّالة لكل رمز   : {active_params_per_token:,.0f}")
    print(f"نسبة التفعيل                : {100*top_k/n_experts:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-16-chapter-16.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
شكل الإدخال  : torch.Size([4, 16, 64])
شكل الإخراج  : torch.Size([4, 16, 64])
خسارة التوازن: 1.0234

إجمالي معاملات الخبراء      : 266,240
المعاملات الفعّالة لكل رمز   : 66,560
نسبة التفعيل                : 25.0%

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-16-chapter-16.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: معامل موازنة كبير جدًا يُهيمن على هدف التعلّم الأساسي
total_loss = task_loss + 1.0 * balance_loss  # α=1.0 كبير جدًا، سيُشوّه التعلّم!

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-16-chapter-16.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: معامل صغير يُحافظ على توازن خفيف دون التأثير على المهمة الأساسية
total_loss = task_loss + 0.01 * balance_loss

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-16-chapter-16.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: استخدام الاحتمالات الخام من softmax الكامل دون تطبيع بعد اختيار Top-K
# قد يجمع إلى قيمة أقل من 1 بكثير، مما يُصغّر إخراج الطبقة بشكل غير مقصود
y = topk_probs_raw[0] * E_a(x) + topk_probs_raw[1] * E_b(x)  # قد يكون المجموع 0.3 فقط!

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-16-chapter-16.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: إعادة التطبيع لضمان أن أوزان الدمج تجمع إلى 1
topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-16-chapter-16.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: capacity_factor = 1.0 مع عدم مراقبة نسبة الرموز المُسقَطة
moe = MoELayer(d_model, d_ff, n_experts=8, top_k=2, capacity_factor=1.0)
# إذا كان التوزيع غير متوازن حتى قليلاً، سيُسقط رموزًا كثيرة دون أن تلاحظ!

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-16-chapter-16.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: راقب فعليًا نسبة الإسقاط، واضبط capacity_factor بناءً عليها
dropped_ratio = 1 - (total_processed_tokens / total_tokens)
assert dropped_ratio < 0.05, "نسبة إسقاط مرتفعة جدًا، فكّر في زيادة capacity_factor"

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-16-chapter-16.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# mini_project_moe_gpt_block.py
from __future__ import annotations

import torch
import torch.nn as nn

from mixture_of_experts import MoELayer

# TODO:
# 1) بناء GPTMoEBlock: نفس GPTBlock من الفصل 12، لكن استبدل self.ffn بـ MoELayer
#    تذكّر: forward يجب أن يُعيد أيضًا balance_loss لتجميعها عبر كل الطبقات
# 2) بناء GPTMoEModel كامل بعدة طبقات، يُجمّع balance_loss من كل الطبقات
# 3) درّب نموذجين (Dense وMoE) على نفس بيانات نصية بسيطة لعدد خطوات متساوٍ
# 4) قارن: (أ) منحنى الخسارة، (ب) الوقت الفعلي للتدريب لكل خطوة،
#    (ج) إجمالي المعاملات مقابل المعاملات الفعّالة لكل نموذج
# 5) ارسم توزيع استخدام الخبراء عبر التدريب (هل تقارب التوزيع المتساوي مع التقدم؟)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-16-chapter-16.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل السادس عشر: Mixture of Experts — فصل الحجم عن التكلفة
Large Language Model Engineering — Mixture

Book slug : llm-engineering-69-chapters
Chapter   : ch-16-chapter-16 (lesson 16)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_16.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 16: الفصل السادس عشر: Mixture of Experts — فصل الحجم عن التكلفة."""

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

def run_moe() -> None:
    scores = [0.2, 1.4, 0.8, -0.1]
    top_two = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:2]
    weights = softmax([scores[index] for index in top_two])
    expert_load = Counter(top_two)
    print('selected experts:', top_two)
    print('mixture weights:', [round(x, 3) for x in weights])
    print('active parameters:', '2 of', len(scores), 'experts')
    print('routing counts:', dict(expert_load))

if __name__ == "__main__":
    print("Chapter 16 lab: Top-k mixture-of-experts routing")
    run_moe()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-04-modern-architectures/chapter-16.md
# ────────────────────────────────────────────────────────────────────────
# mixture_of_experts.py
"""
تنفيذ كامل لطبقة Mixture of Experts مع Top-K Routing
وLoad Balancing Loss، بأسلوب مشابه لـ Switch Transformer / Mixtral.
Python 3.13+, PyTorch 2.x
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


class Expert(nn.Module):
    """خبير واحد: شبكة FFN قياسية بالضبط كما في الفصل 11."""

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


class MoELayer(nn.Module):
    """
    طبقة Mixture of Experts مع Top-K Routing.
    تُعيد: (الإخراج، خسارة موازنة الحمل)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff) for _ in range(n_experts)]
        )
        self.router = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)                     # (batch*seq_len, d_model)
        n_tokens = x_flat.size(0)

        # 1) حساب درجات الموجِّه واحتمالاته
        router_logits = self.router(x_flat)               # (n_tokens, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)    # (n_tokens, n_experts)

        # 2) اختيار Top-K خبراء لكل رمز
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # إعادة تطبيع الاحتمالات المختارة لتجمع إلى 1 (ممارسة شائعة)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # 3) حساب سعة كل خبير (لتجنب الفيضان على خبير واحد)
        capacity = int(self.capacity_factor * n_tokens * self.top_k / self.n_experts)

        output = torch.zeros_like(x_flat)
        expert_usage_count = torch.zeros(self.n_experts, device=x.device)

        for expert_idx in range(self.n_experts):
            # أي الرموز اختارت هذا الخبير ضمن Top-K؟
            mask = (topk_indices == expert_idx)             # (n_tokens, top_k) bool
            token_positions, k_positions = mask.nonzero(as_tuple=True)

            if token_positions.numel() == 0:
                continue

            # تطبيق سعة الخبير: أخذ فقط أول `capacity` رمز، إسقاط الباقي
            if token_positions.numel() > capacity:
                token_positions = token_positions[:capacity]
                k_positions = k_positions[:capacity]

            expert_usage_count[expert_idx] = token_positions.numel()

            selected_tokens = x_flat[token_positions]
            expert_output = self.experts[expert_idx](selected_tokens)

            gating_weights = topk_probs[token_positions, k_positions].unsqueeze(-1)
            output[token_positions] += gating_weights * expert_output

        output = output.view(batch, seq_len, d_model)

        # 4) خسارة موازنة الحمل
        balance_loss = self._compute_load_balancing_loss(router_probs, topk_indices, n_tokens)

        return output, balance_loss

    def _compute_load_balancing_loss(
        self, router_probs: torch.Tensor, topk_indices: torch.Tensor, n_tokens: int
    ) -> torch.Tensor:
        # f_i: نسبة الرموز التي اختارت الخبير i ضمن Top-K (تجريبيًا)
        one_hot = F.one_hot(topk_indices, num_classes=self.n_experts).float()  # (n_tokens, top_k, n_experts)
        f_i = one_hot.sum(dim=(0, 1)) / (n_tokens * self.top_k)

        # P_i: متوسط احتمال الموجِّه لكل خبير عبر كل الرموز
        p_i = router_probs.mean(dim=0)

        return self.n_experts * torch.sum(f_i * p_i)


if __name__ == "__main__":
    d_model, d_ff, n_experts, top_k = 64, 256, 8, 2
    moe_layer = MoELayer(d_model, d_ff, n_experts, top_k)

    batch, seq_len = 4, 16
    x = torch.randn(batch, seq_len, d_model)

    output, balance_loss = moe_layer(x)

    print(f"شكل الإدخال  : {x.shape}")
    print(f"شكل الإخراج  : {output.shape}")
    print(f"خسارة التوازن: {balance_loss.item():.4f}")

    # حساب عدد المعاملات: إجمالية مقابل فعّالة لكل رمز
    total_params = sum(p.numel() for p in moe_layer.experts.parameters())
    active_params_per_token = total_params * top_k / n_experts
    print(f"\nإجمالي معاملات الخبراء      : {total_params:,}")
    print(f"المعاملات الفعّالة لكل رمز   : {active_params_per_token:,.0f}")
    print(f"نسبة التفعيل                : {100*top_k/n_experts:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-04-modern-architectures/chapter-16.md
# ────────────────────────────────────────────────────────────────────────
شكل الإدخال  : torch.Size([4, 16, 64])
شكل الإخراج  : torch.Size([4, 16, 64])
خسارة التوازن: 1.0234

إجمالي معاملات الخبراء      : 266,240
المعاملات الفعّالة لكل رمز   : 66,560
نسبة التفعيل                : 25.0%

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-04-modern-architectures/chapter-16.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: معامل موازنة كبير جدًا يُهيمن على هدف التعلّم الأساسي
total_loss = task_loss + 1.0 * balance_loss  # α=1.0 كبير جدًا، سيُشوّه التعلّم!

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-04-modern-architectures/chapter-16.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: معامل صغير يُحافظ على توازن خفيف دون التأثير على المهمة الأساسية
total_loss = task_loss + 0.01 * balance_loss

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-04-modern-architectures/chapter-16.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: استخدام الاحتمالات الخام من softmax الكامل دون تطبيع بعد اختيار Top-K
# قد يجمع إلى قيمة أقل من 1 بكثير، مما يُصغّر إخراج الطبقة بشكل غير مقصود
y = topk_probs_raw[0] * E_a(x) + topk_probs_raw[1] * E_b(x)  # قد يكون المجموع 0.3 فقط!

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-04-modern-architectures/chapter-16.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: إعادة التطبيع لضمان أن أوزان الدمج تجمع إلى 1
topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-04-modern-architectures/chapter-16.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: capacity_factor = 1.0 مع عدم مراقبة نسبة الرموز المُسقَطة
moe = MoELayer(d_model, d_ff, n_experts=8, top_k=2, capacity_factor=1.0)
# إذا كان التوزيع غير متوازن حتى قليلاً، سيُسقط رموزًا كثيرة دون أن تلاحظ!

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-04-modern-architectures/chapter-16.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: راقب فعليًا نسبة الإسقاط، واضبط capacity_factor بناءً عليها
dropped_ratio = 1 - (total_processed_tokens / total_tokens)
assert dropped_ratio < 0.05, "نسبة إسقاط مرتفعة جدًا، فكّر في زيادة capacity_factor"

# ────────────────────────────────────────────────────────────────────────
# [010] book/chapters/part-04-modern-architectures/chapter-16.md
# ────────────────────────────────────────────────────────────────────────
# mini_project_moe_gpt_block.py
from __future__ import annotations

import torch
import torch.nn as nn

from mixture_of_experts import MoELayer

# TODO:
# 1) بناء GPTMoEBlock: نفس GPTBlock من الفصل 12، لكن استبدل self.ffn بـ MoELayer
#    تذكّر: forward يجب أن يُعيد أيضًا balance_loss لتجميعها عبر كل الطبقات
# 2) بناء GPTMoEModel كامل بعدة طبقات، يُجمّع balance_loss من كل الطبقات
# 3) درّب نموذجين (Dense وMoE) على نفس بيانات نصية بسيطة لعدد خطوات متساوٍ
# 4) قارن: (أ) منحنى الخسارة، (ب) الوقت الفعلي للتدريب لكل خطوة،
#    (ج) إجمالي المعاملات مقابل المعاملات الفعّالة لكل نموذج
# 5) ارسم توزيع استخدام الخبراء عبر التدريب (هل تقارب التوزيع المتساوي مع التقدم؟)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-16-chapter-16.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 16: الفصل السادس عشر: Mixture of Experts — فصل الحجم عن التكلفة."""

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

def run_moe() -> None:
    scores = [0.2, 1.4, 0.8, -0.1]
    top_two = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:2]
    weights = softmax([scores[index] for index in top_two])
    expert_load = Counter(top_two)
    print('selected experts:', top_two)
    print('mixture weights:', [round(x, 3) for x in weights])
    print('active parameters:', '2 of', len(scores), 'experts')
    print('routing counts:', dict(expert_load))

if __name__ == "__main__":
    print("Chapter 16 lab: Top-k mixture-of-experts routing")
    run_moe()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-16-chapter-16.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# mixture_of_experts.py
"""
تنفيذ كامل لطبقة Mixture of Experts مع Top-K Routing
وLoad Balancing Loss، بأسلوب مشابه لـ Switch Transformer / Mixtral.
Python 3.13+, PyTorch 2.x
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


class Expert(nn.Module):
    """خبير واحد: شبكة FFN قياسية بالضبط كما في الفصل 11."""

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


class MoELayer(nn.Module):
    """
    طبقة Mixture of Experts مع Top-K Routing.
    تُعيد: (الإخراج، خسارة موازنة الحمل)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff) for _ in range(n_experts)]
        )
        self.router = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)                     # (batch*seq_len, d_model)
        n_tokens = x_flat.size(0)

        # 1) حساب درجات الموجِّه واحتمالاته
        router_logits = self.router(x_flat)               # (n_tokens, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)    # (n_tokens, n_experts)

        # 2) اختيار Top-K خبراء لكل رمز
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # إعادة تطبيع الاحتمالات المختارة لتجمع إلى 1 (ممارسة شائعة)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # 3) حساب سعة كل خبير (لتجنب الفيضان على خبير واحد)
        capacity = int(self.capacity_factor * n_tokens * self.top_k / self.n_experts)

        output = torch.zeros_like(x_flat)
        expert_usage_count = torch.zeros(self.n_experts, device=x.device)

        for expert_idx in range(self.n_experts):
            # أي الرموز اختارت هذا الخبير ضمن Top-K؟
            mask = (topk_indices == expert_idx)             # (n_tokens, top_k) bool
            token_positions, k_positions = mask.nonzero(as_tuple=True)

            if token_positions.numel() == 0:
                continue

            # تطبيق سعة الخبير: أخذ فقط أول `capacity` رمز، إسقاط الباقي
            if token_positions.numel() > capacity:
                token_positions = token_positions[:capacity]
                k_positions = k_positions[:capacity]

            expert_usage_count[expert_idx] = token_positions.numel()

            selected_tokens = x_flat[token_positions]
            expert_output = self.experts[expert_idx](selected_tokens)

            gating_weights = topk_probs[token_positions, k_positions].unsqueeze(-1)
            output[token_positions] += gating_weights * expert_output

        output = output.view(batch, seq_len, d_model)

        # 4) خسارة موازنة الحمل
        balance_loss = self._compute_load_balancing_loss(router_probs, topk_indices, n_tokens)

        return output, balance_loss

    def _compute_load_balancing_loss(
        self, router_probs: torch.Tensor, topk_indices: torch.Tensor, n_tokens: int
    ) -> torch.Tensor:
        # f_i: نسبة الرموز التي اختارت الخبير i ضمن Top-K (تجريبيًا)
        one_hot = F.one_hot(topk_indices, num_classes=self.n_experts).float()  # (n_tokens, top_k, n_experts)
        f_i = one_hot.sum(dim=(0, 1)) / (n_tokens * self.top_k)

        # P_i: متوسط احتمال الموجِّه لكل خبير عبر كل الرموز
        p_i = router_probs.mean(dim=0)

        return self.n_experts * torch.sum(f_i * p_i)


if __name__ == "__main__":
    d_model, d_ff, n_experts, top_k = 64, 256, 8, 2
    moe_layer = MoELayer(d_model, d_ff, n_experts, top_k)

    batch, seq_len = 4, 16
    x = torch.randn(batch, seq_len, d_model)

    output, balance_loss = moe_layer(x)

    print(f"شكل الإدخال  : {x.shape}")
    print(f"شكل الإخراج  : {output.shape}")
    print(f"خسارة التوازن: {balance_loss.item():.4f}")

    # حساب عدد المعاملات: إجمالية مقابل فعّالة لكل رمز
    total_params = sum(p.numel() for p in moe_layer.experts.parameters())
    active_params_per_token = total_params * top_k / n_experts
    print(f"\nإجمالي معاملات الخبراء      : {total_params:,}")
    print(f"المعاملات الفعّالة لكل رمز   : {active_params_per_token:,.0f}")
    print(f"نسبة التفعيل                : {100*top_k/n_experts:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-16-chapter-16.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
شكل الإدخال  : torch.Size([4, 16, 64])
شكل الإخراج  : torch.Size([4, 16, 64])
خسارة التوازن: 1.0234

إجمالي معاملات الخبراء      : 266,240
المعاملات الفعّالة لكل رمز   : 66,560
نسبة التفعيل                : 25.0%

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-16-chapter-16.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: معامل موازنة كبير جدًا يُهيمن على هدف التعلّم الأساسي
total_loss = task_loss + 1.0 * balance_loss  # α=1.0 كبير جدًا، سيُشوّه التعلّم!

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-16-chapter-16.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: معامل صغير يُحافظ على توازن خفيف دون التأثير على المهمة الأساسية
total_loss = task_loss + 0.01 * balance_loss

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-16-chapter-16.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: استخدام الاحتمالات الخام من softmax الكامل دون تطبيع بعد اختيار Top-K
# قد يجمع إلى قيمة أقل من 1 بكثير، مما يُصغّر إخراج الطبقة بشكل غير مقصود
y = topk_probs_raw[0] * E_a(x) + topk_probs_raw[1] * E_b(x)  # قد يكون المجموع 0.3 فقط!

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-16-chapter-16.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: إعادة التطبيع لضمان أن أوزان الدمج تجمع إلى 1
topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-16-chapter-16.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: capacity_factor = 1.0 مع عدم مراقبة نسبة الرموز المُسقَطة
moe = MoELayer(d_model, d_ff, n_experts=8, top_k=2, capacity_factor=1.0)
# إذا كان التوزيع غير متوازن حتى قليلاً، سيُسقط رموزًا كثيرة دون أن تلاحظ!

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-16-chapter-16.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: راقب فعليًا نسبة الإسقاط، واضبط capacity_factor بناءً عليها
dropped_ratio = 1 - (total_processed_tokens / total_tokens)
assert dropped_ratio < 0.05, "نسبة إسقاط مرتفعة جدًا، فكّر في زيادة capacity_factor"

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-16-chapter-16.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# mini_project_moe_gpt_block.py
from __future__ import annotations

import torch
import torch.nn as nn

from mixture_of_experts import MoELayer

# TODO:
# 1) بناء GPTMoEBlock: نفس GPTBlock من الفصل 12، لكن استبدل self.ffn بـ MoELayer
#    تذكّر: forward يجب أن يُعيد أيضًا balance_loss لتجميعها عبر كل الطبقات
# 2) بناء GPTMoEModel كامل بعدة طبقات، يُجمّع balance_loss من كل الطبقات
# 3) درّب نموذجين (Dense وMoE) على نفس بيانات نصية بسيطة لعدد خطوات متساوٍ
# 4) قارن: (أ) منحنى الخسارة، (ب) الوقت الفعلي للتدريب لكل خطوة،
#    (ج) إجمالي المعاملات مقابل المعاملات الفعّالة لكل نموذج
# 5) ارسم توزيع استخدام الخبراء عبر التدريب (هل تقارب التوزيع المتساوي مع التقدم؟)
