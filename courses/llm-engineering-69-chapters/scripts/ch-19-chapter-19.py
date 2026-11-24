#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل التاسع عشر: دراسة حالة تطبيقية — تشريح معماري لـ LLaMA 3، Mistral، وDeepSeek-V3
Large Language Model Engineering — LLaMA

Book slug : llm-engineering-69-chapters
Chapter   : ch-19-chapter-19 (lesson 19)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-19-chapter-19.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل التاسع عشر: دراسة حالة تطبيقية — تشريح معماري لـ LLaMA 3، Mistral، وDeepSeek-V3
Large Language Model Engineering — LLaMA

Book slug : llm-engineering-69-chapters
Chapter   : ch-19-chapter-19 (lesson 19)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-19-chapter-19.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل التاسع عشر: دراسة حالة تطبيقية — تشريح معماري لـ LLaMA 3، Mistral، وDeepSeek-V3
Large Language Model Engineering — LLaMA

Book slug : llm-engineering-69-chapters
Chapter   : ch-19-chapter-19 (lesson 19)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_19.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 19: الفصل التاسع عشر: دراسة حالة تطبيقية — تشريح معماري لـ LLaMA 3، Mistral، وDeepSeek-V3."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_architecture() -> None:
    models = [{'name': 'encoder', 'bidirectional': True, 'generation': False}, {'name': 'decoder', 'bidirectional': False, 'generation': True}, {'name': 'encoder-decoder', 'bidirectional': True, 'generation': True}]
    for model in models:
        print(model['name'], 'context:', 'both' if model['bidirectional'] else 'causal', '| generation:', model['generation'])

if __name__ == "__main__":
    print("Chapter 19 lab: Compare model architecture configurations")
    run_architecture()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-04-modern-architectures/chapter-19.md
# ────────────────────────────────────────────────────────────────────────
"""
chapter19_architecture_comparison.py
مقارنة عملية بين MHA، GQA، وMLA من حيث حجم KV Cache،
مع تطبيق مصغّر قابل للتشغيل لكل آلية.
Python 3.13+, PyTorch 2.x
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


@dataclass
class AttentionMemoryReport:
    name: str
    bytes_per_layer: float

    @property
    def mb_per_layer(self) -> float:
        return self.bytes_per_layer / (1024 ** 2)


def compute_kv_cache_sizes(
    seq_len: int,
    n_heads: int,
    d_head: int,
    n_kv_groups: int,
    d_latent: int,
    dtype_bytes: int = 2,
) -> list[AttentionMemoryReport]:
    """حساب حجم ذاكرة KV Cache لكل آلية انتباه (لكل طبقة واحدة)."""
    reports = []

    # 1) MHA القياسية: كل رأس له K,V منفصلان بالكامل
    mha_bytes = 2 * seq_len * n_heads * d_head * dtype_bytes
    reports.append(AttentionMemoryReport("MHA القياسية", mha_bytes))

    # 2) GQA: عدد مجموعات KV أقل من عدد رؤوس Q
    gqa_bytes = 2 * seq_len * n_kv_groups * d_head * dtype_bytes
    reports.append(AttentionMemoryReport(f"GQA ({n_kv_groups} مجموعات)", gqa_bytes))

    # 3) MQA: حالة خاصة من GQA بمجموعة واحدة فقط
    mqa_bytes = 2 * seq_len * 1 * d_head * dtype_bytes
    reports.append(AttentionMemoryReport("MQA (مجموعة واحدة)", mqa_bytes))

    # 4) MLA: متّجه كامن واحد مضغوط فقط (بلا تكرار عبر الرؤوس)
    mla_bytes = seq_len * d_latent * dtype_bytes
    reports.append(AttentionMemoryReport(f"MLA (بُعد كامن={d_latent})", mla_bytes))

    return reports


class SimplifiedMLA(nn.Module):
    """
    تطبيق مبسّط تعليمي لـ Multi-Head Latent Attention (DeepSeek-V2/V3).

    الفكرة: ضغط K,V إلى متّجه كامن مشترك c_kv، ثم بسطه (up-project)
    إلى K,V الكاملين عند الحاجة الفعلية لحساب الانتباه.
    الذي يُخزَّن في الكاش وقت التوليد هو c_kv فقط (صغير جدًا).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_latent: int = 128,
    ) -> None:
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.d_latent = d_latent

        # إسقاط الاستعلام (Query) كالمعتاد — كل رأس له Q خاص به
        self.w_q = nn.Linear(dim, dim, bias=False)

        # مرحلة الضغط: من البُعد الكامل إلى البُعد الكامن الصغير
        self.w_down_kv = nn.Linear(dim, d_latent, bias=False)

        # مرحلة البسط: من البُعد الكامن إلى K و V الكاملين
        self.w_up_k = nn.Linear(d_latent, dim, bias=False)
        self.w_up_v = nn.Linear(d_latent, dim, bias=False)

        self.w_out = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        latent_cache: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape

        # --- مرحلة الضغط: نحسب المتّجه الكامن ونُخزّنه فقط في الكاش ---
        c_kv = self.w_down_kv(x)  # (B, T, d_latent) — هذا ما يُخزَّن فعليًا!

        if latent_cache is not None:
            c_kv_full = torch.cat([latent_cache, c_kv], dim=1)
        else:
            c_kv_full = c_kv

        # --- مرحلة البسط: نُعيد بناء K و V الكاملين عند الحاجة فقط ---
        k_full = self.w_up_k(c_kv_full)  # (B, T_total, D)
        v_full = self.w_up_v(c_kv_full)  # (B, T_total, D)
        q = self.w_q(x)                   # (B, T, D)

        # إعادة تشكيل لرؤوس متعددة
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            B_, T_, _ = t.shape
            return t.view(B_, T_, self.n_heads, self.d_head).transpose(1, 2)

        q, k_full, v_full = split_heads(q), split_heads(k_full), split_heads(v_full)

        attn_out = F.scaled_dot_product_attention(
            q, k_full, v_full, attn_mask=causal_mask, is_causal=(causal_mask is None)
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)

        return self.w_out(attn_out), c_kv_full  # نُعيد الكاش الكامن المُحدَّث


if __name__ == "__main__":
    print("=== مقارنة حجم KV Cache لكل طبقة واحدة ===\n")
    reports = compute_kv_cache_sizes(
        seq_len=2048, n_heads=16, d_head=64, n_kv_groups=4, d_latent=128
    )
    baseline = reports[0].bytes_per_layer
    for r in reports:
        ratio = baseline / r.bytes_per_layer
        print(f"{r.name:30s}: {r.mb_per_layer:7.3f} MB   (تقليص {ratio:.1f}x عن MHA)")

    print("\n=== اختبار SimplifiedMLA عمليًا ===")
    mla = SimplifiedMLA(dim=1024, n_heads=16, d_latent=128)
    x = torch.randn(1, 10, 1024)
    causal = torch.triu(torch.full((10, 10), float("-inf")), diagonal=1)
    out, cache = mla(x, latent_cache=None, causal_mask=causal)
    print(f"شكل المُخرج: {tuple(out.shape)}")
    print(f"شكل الكاش الكامن المُخزَّن: {tuple(cache.shape)}  <- هذا فقط ما يُحفظ!")

    # محاكاة خطوة توليد تالية: رمز واحد جديد، يستخدم الكاش السابق
    new_token = torch.randn(1, 1, 1024)
    out2, cache2 = mla(new_token, latent_cache=cache, causal_mask=None)
    print(f"بعد خطوة توليد إضافية، شكل الكاش: {tuple(cache2.shape)}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-04-modern-architectures/chapter-19.md
# ────────────────────────────────────────────────────────────────────────
# ❌ تفكير ساذج: "لنُقلّص دائمًا إلى أقصى حد ممكن (مثل MQA أو بُعد كامن صغير جدًا)"
mla_too_aggressive = SimplifiedMLA(dim=1024, n_heads=16, d_latent=16)  # صغير جدًا!
# النتيجة: فقدان حاد في جودة النموذج لأن المتّجه الكامن لا يحمل معلومة كافية

# ✅ صحيح: اختر d_latent بعد تجربة تجريبية موازنة بين الذاكرة والجودة
# (في DeepSeek-V2 الفعلي، d_latent تم اختياره بعناية بعد تجارب Ablation واسعة)
mla_balanced = SimplifiedMLA(dim=1024, n_heads=16, d_latent=128)

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-04-modern-architectures/chapter-19.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع: "Mixtral لديه 8 خبراء إذن قدرته تعادل 8 نماذج منفصلة"
# هذا غير دقيق - الخبراء يتشاركون نفس طبقات الانتباه، والتوجيه ديناميكي لكل رمز

# ✅ الفهم الصحيح: السعة الإجمالية (knowledge capacity) أكبر،
# لكن الحساب الفعلي لكل رمز يعادل فقط Top-K من الخبراء (عادة 2)
# أي: فكّر في MoE كـ "مكتبة ضخمة من الخبرات المتخصصة" يُستدعى منها القليل فقط حسب الحاجة

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-04-modern-architectures/chapter-19.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: افتراض أن نافذة 4096 تعني أن النموذج "لا يرى إطلاقًا" أبعد من 4096 رمز
# حتى في الطبقة الأولى - هذا صحيح لطبقة واحدة، لكن خاطئ للنموذج ككل

# ✅ الفهم الصحيح: احسب الحقل الاستقبالي الفعّال عبر كل الطبقات
def effective_receptive_field(window_size: int, n_layers: int) -> int:
    """تقدير تقريبي للحقل الاستقبالي الفعّال عبر طبقات متراصّة من SWA."""
    return window_size * n_layers

print(effective_receptive_field(4096, 32))  # 131,072 - أكبر بكثير من النافذة المفردة!

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-04-modern-architectures/chapter-19.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: مقارنة "سرعة استدلال LLaMA 70B مقابل DeepSeek-V3 671B"
# دون توضيح أن DeepSeek تستخدم فقط 37B معامل نشط لكل رمز

# ✅ المقارنة العادلة يجب أن توضّح الأبعاد الثلاثة دائمًا:
comparison = {
    "LLaMA 3 70B":  {"total_params": "70B",  "active_params": "70B",  "type": "Dense"},
    "DeepSeek-V3":  {"total_params": "671B", "active_params": "37B",  "type": "MoE"},
}
# المقارنة العادلة لسرعة الاستدلال: قارن active_params وليس total_params

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-04-modern-architectures/chapter-19.md
# ────────────────────────────────────────────────────────────────────────
"""
mini_project_architecture_dashboard.py
لوحة مقارنة معمارية: تحسب ذاكرة KV Cache، عدد المعاملات،
و FLOPs التقريبية لثلاثة تصاميم مستوحاة من LLaMA/Mistral/DeepSeek-V3.

المطلوب إكماله:
1. استخدم الأكواد من الفصول 11، 15، 16، 18، 19 كمكوّنات بناء.
2. ابنِ ثلاث دوال build_llama_style_block / build_mistral_style_block /
   build_deepseek_style_block، كل واحدة تُعيد نموذج nn.Module لكتلة واحدة.
3. لكل كتلة، احسب:
   - إجمالي عدد المعاملات (sum(p.numel())).
   - حجم KV Cache لتسلسل بطول 4096 (استخدم compute_kv_cache_sizes من هذا الفصل).
   - FLOPs تقريبية باستخدام القاعدة 6ND من الفصل 14 (للتدريب) وتقدير الاستدلال.
4. اطبع جدولًا نهائيًا يقارن الثلاثة جنبًا إلى جنب.
5. (تحدٍّ إضافي) ارسم رسمًا بيانيًا بـ matplotlib يُظهر نمو حجم KV Cache
   مع ازدياد طول التسلسل (512, 1024, 2048, 4096, 8192, 16384) لكل من الثلاثة.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BlockSpec:
    name: str
    dim: int = 1024
    n_heads: int = 16
    n_kv_groups: int | None = None   # None يعني MHA كاملة
    d_latent: int | None = None       # None يعني بدون MLA
    use_moe: bool = False
    n_experts: int = 8
    top_k: int = 2


def build_and_report(spec: BlockSpec, seq_len: int = 4096) -> dict:
    # TODO 1: بناء الكتلة الفعلية (استخدم المكوّنات من الفصول السابقة)
    # TODO 2: حساب إجمالي المعاملات
    # TODO 3: حساب حجم KV Cache المناسب (MHA/GQA/MLA حسب spec)
    # TODO 4: حساب FLOPs تقريبية بالقاعدة 6ND (افترض N=عدد المعاملات، وD=عدد رموز افتراضي)
    # TODO 5: إعادة قاموس بكل هذه الإحصائيات
    raise NotImplementedError


def main() -> None:
    specs = [
        BlockSpec(name="LLaMA-style", n_kv_groups=4),
        BlockSpec(name="Mistral-style", n_kv_groups=4),  # + سيحتاج Sliding Window منفصلًا
        BlockSpec(name="DeepSeek-style", d_latent=128, use_moe=True),
    ]
    # TODO: استدعِ build_and_report لكل spec واطبع جدول مقارنة منسّق


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-19-chapter-19.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 19: الفصل التاسع عشر: دراسة حالة تطبيقية — تشريح معماري لـ LLaMA 3، Mistral، وDeepSeek-V3."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_architecture() -> None:
    models = [{'name': 'encoder', 'bidirectional': True, 'generation': False}, {'name': 'decoder', 'bidirectional': False, 'generation': True}, {'name': 'encoder-decoder', 'bidirectional': True, 'generation': True}]
    for model in models:
        print(model['name'], 'context:', 'both' if model['bidirectional'] else 'causal', '| generation:', model['generation'])

if __name__ == "__main__":
    print("Chapter 19 lab: Compare model architecture configurations")
    run_architecture()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-19-chapter-19.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
chapter19_architecture_comparison.py
مقارنة عملية بين MHA، GQA، وMLA من حيث حجم KV Cache،
مع تطبيق مصغّر قابل للتشغيل لكل آلية.
Python 3.13+, PyTorch 2.x
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


@dataclass
class AttentionMemoryReport:
    name: str
    bytes_per_layer: float

    @property
    def mb_per_layer(self) -> float:
        return self.bytes_per_layer / (1024 ** 2)


def compute_kv_cache_sizes(
    seq_len: int,
    n_heads: int,
    d_head: int,
    n_kv_groups: int,
    d_latent: int,
    dtype_bytes: int = 2,
) -> list[AttentionMemoryReport]:
    """حساب حجم ذاكرة KV Cache لكل آلية انتباه (لكل طبقة واحدة)."""
    reports = []

    # 1) MHA القياسية: كل رأس له K,V منفصلان بالكامل
    mha_bytes = 2 * seq_len * n_heads * d_head * dtype_bytes
    reports.append(AttentionMemoryReport("MHA القياسية", mha_bytes))

    # 2) GQA: عدد مجموعات KV أقل من عدد رؤوس Q
    gqa_bytes = 2 * seq_len * n_kv_groups * d_head * dtype_bytes
    reports.append(AttentionMemoryReport(f"GQA ({n_kv_groups} مجموعات)", gqa_bytes))

    # 3) MQA: حالة خاصة من GQA بمجموعة واحدة فقط
    mqa_bytes = 2 * seq_len * 1 * d_head * dtype_bytes
    reports.append(AttentionMemoryReport("MQA (مجموعة واحدة)", mqa_bytes))

    # 4) MLA: متّجه كامن واحد مضغوط فقط (بلا تكرار عبر الرؤوس)
    mla_bytes = seq_len * d_latent * dtype_bytes
    reports.append(AttentionMemoryReport(f"MLA (بُعد كامن={d_latent})", mla_bytes))

    return reports


class SimplifiedMLA(nn.Module):
    """
    تطبيق مبسّط تعليمي لـ Multi-Head Latent Attention (DeepSeek-V2/V3).

    الفكرة: ضغط K,V إلى متّجه كامن مشترك c_kv، ثم بسطه (up-project)
    إلى K,V الكاملين عند الحاجة الفعلية لحساب الانتباه.
    الذي يُخزَّن في الكاش وقت التوليد هو c_kv فقط (صغير جدًا).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_latent: int = 128,
    ) -> None:
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.d_latent = d_latent

        # إسقاط الاستعلام (Query) كالمعتاد — كل رأس له Q خاص به
        self.w_q = nn.Linear(dim, dim, bias=False)

        # مرحلة الضغط: من البُعد الكامل إلى البُعد الكامن الصغير
        self.w_down_kv = nn.Linear(dim, d_latent, bias=False)

        # مرحلة البسط: من البُعد الكامن إلى K و V الكاملين
        self.w_up_k = nn.Linear(d_latent, dim, bias=False)
        self.w_up_v = nn.Linear(d_latent, dim, bias=False)

        self.w_out = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        latent_cache: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape

        # --- مرحلة الضغط: نحسب المتّجه الكامن ونُخزّنه فقط في الكاش ---
        c_kv = self.w_down_kv(x)  # (B, T, d_latent) — هذا ما يُخزَّن فعليًا!

        if latent_cache is not None:
            c_kv_full = torch.cat([latent_cache, c_kv], dim=1)
        else:
            c_kv_full = c_kv

        # --- مرحلة البسط: نُعيد بناء K و V الكاملين عند الحاجة فقط ---
        k_full = self.w_up_k(c_kv_full)  # (B, T_total, D)
        v_full = self.w_up_v(c_kv_full)  # (B, T_total, D)
        q = self.w_q(x)                   # (B, T, D)

        # إعادة تشكيل لرؤوس متعددة
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            B_, T_, _ = t.shape
            return t.view(B_, T_, self.n_heads, self.d_head).transpose(1, 2)

        q, k_full, v_full = split_heads(q), split_heads(k_full), split_heads(v_full)

        attn_out = F.scaled_dot_product_attention(
            q, k_full, v_full, attn_mask=causal_mask, is_causal=(causal_mask is None)
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)

        return self.w_out(attn_out), c_kv_full  # نُعيد الكاش الكامن المُحدَّث


if __name__ == "__main__":
    print("=== مقارنة حجم KV Cache لكل طبقة واحدة ===\n")
    reports = compute_kv_cache_sizes(
        seq_len=2048, n_heads=16, d_head=64, n_kv_groups=4, d_latent=128
    )
    baseline = reports[0].bytes_per_layer
    for r in reports:
        ratio = baseline / r.bytes_per_layer
        print(f"{r.name:30s}: {r.mb_per_layer:7.3f} MB   (تقليص {ratio:.1f}x عن MHA)")

    print("\n=== اختبار SimplifiedMLA عمليًا ===")
    mla = SimplifiedMLA(dim=1024, n_heads=16, d_latent=128)
    x = torch.randn(1, 10, 1024)
    causal = torch.triu(torch.full((10, 10), float("-inf")), diagonal=1)
    out, cache = mla(x, latent_cache=None, causal_mask=causal)
    print(f"شكل المُخرج: {tuple(out.shape)}")
    print(f"شكل الكاش الكامن المُخزَّن: {tuple(cache.shape)}  <- هذا فقط ما يُحفظ!")

    # محاكاة خطوة توليد تالية: رمز واحد جديد، يستخدم الكاش السابق
    new_token = torch.randn(1, 1, 1024)
    out2, cache2 = mla(new_token, latent_cache=cache, causal_mask=None)
    print(f"بعد خطوة توليد إضافية، شكل الكاش: {tuple(cache2.shape)}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-19-chapter-19.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ❌ تفكير ساذج: "لنُقلّص دائمًا إلى أقصى حد ممكن (مثل MQA أو بُعد كامن صغير جدًا)"
mla_too_aggressive = SimplifiedMLA(dim=1024, n_heads=16, d_latent=16)  # صغير جدًا!
# النتيجة: فقدان حاد في جودة النموذج لأن المتّجه الكامن لا يحمل معلومة كافية

# ✅ صحيح: اختر d_latent بعد تجربة تجريبية موازنة بين الذاكرة والجودة
# (في DeepSeek-V2 الفعلي، d_latent تم اختياره بعناية بعد تجارب Ablation واسعة)
mla_balanced = SimplifiedMLA(dim=1024, n_heads=16, d_latent=128)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-19-chapter-19.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع: "Mixtral لديه 8 خبراء إذن قدرته تعادل 8 نماذج منفصلة"
# هذا غير دقيق - الخبراء يتشاركون نفس طبقات الانتباه، والتوجيه ديناميكي لكل رمز

# ✅ الفهم الصحيح: السعة الإجمالية (knowledge capacity) أكبر،
# لكن الحساب الفعلي لكل رمز يعادل فقط Top-K من الخبراء (عادة 2)
# أي: فكّر في MoE كـ "مكتبة ضخمة من الخبرات المتخصصة" يُستدعى منها القليل فقط حسب الحاجة

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-19-chapter-19.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: افتراض أن نافذة 4096 تعني أن النموذج "لا يرى إطلاقًا" أبعد من 4096 رمز
# حتى في الطبقة الأولى - هذا صحيح لطبقة واحدة، لكن خاطئ للنموذج ككل

# ✅ الفهم الصحيح: احسب الحقل الاستقبالي الفعّال عبر كل الطبقات
def effective_receptive_field(window_size: int, n_layers: int) -> int:
    """تقدير تقريبي للحقل الاستقبالي الفعّال عبر طبقات متراصّة من SWA."""
    return window_size * n_layers

print(effective_receptive_field(4096, 32))  # 131,072 - أكبر بكثير من النافذة المفردة!

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-19-chapter-19.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: مقارنة "سرعة استدلال LLaMA 70B مقابل DeepSeek-V3 671B"
# دون توضيح أن DeepSeek تستخدم فقط 37B معامل نشط لكل رمز

# ✅ المقارنة العادلة يجب أن توضّح الأبعاد الثلاثة دائمًا:
comparison = {
    "LLaMA 3 70B":  {"total_params": "70B",  "active_params": "70B",  "type": "Dense"},
    "DeepSeek-V3":  {"total_params": "671B", "active_params": "37B",  "type": "MoE"},
}
# المقارنة العادلة لسرعة الاستدلال: قارن active_params وليس total_params

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-19-chapter-19.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
"""
mini_project_architecture_dashboard.py
لوحة مقارنة معمارية: تحسب ذاكرة KV Cache، عدد المعاملات،
و FLOPs التقريبية لثلاثة تصاميم مستوحاة من LLaMA/Mistral/DeepSeek-V3.

المطلوب إكماله:
1. استخدم الأكواد من الفصول 11، 15، 16، 18، 19 كمكوّنات بناء.
2. ابنِ ثلاث دوال build_llama_style_block / build_mistral_style_block /
   build_deepseek_style_block، كل واحدة تُعيد نموذج nn.Module لكتلة واحدة.
3. لكل كتلة، احسب:
   - إجمالي عدد المعاملات (sum(p.numel())).
   - حجم KV Cache لتسلسل بطول 4096 (استخدم compute_kv_cache_sizes من هذا الفصل).
   - FLOPs تقريبية باستخدام القاعدة 6ND من الفصل 14 (للتدريب) وتقدير الاستدلال.
4. اطبع جدولًا نهائيًا يقارن الثلاثة جنبًا إلى جنب.
5. (تحدٍّ إضافي) ارسم رسمًا بيانيًا بـ matplotlib يُظهر نمو حجم KV Cache
   مع ازدياد طول التسلسل (512, 1024, 2048, 4096, 8192, 16384) لكل من الثلاثة.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BlockSpec:
    name: str
    dim: int = 1024
    n_heads: int = 16
    n_kv_groups: int | None = None   # None يعني MHA كاملة
    d_latent: int | None = None       # None يعني بدون MLA
    use_moe: bool = False
    n_experts: int = 8
    top_k: int = 2


def build_and_report(spec: BlockSpec, seq_len: int = 4096) -> dict:
    # TODO 1: بناء الكتلة الفعلية (استخدم المكوّنات من الفصول السابقة)
    # TODO 2: حساب إجمالي المعاملات
    # TODO 3: حساب حجم KV Cache المناسب (MHA/GQA/MLA حسب spec)
    # TODO 4: حساب FLOPs تقريبية بالقاعدة 6ND (افترض N=عدد المعاملات، وD=عدد رموز افتراضي)
    # TODO 5: إعادة قاموس بكل هذه الإحصائيات
    raise NotImplementedError


def main() -> None:
    specs = [
        BlockSpec(name="LLaMA-style", n_kv_groups=4),
        BlockSpec(name="Mistral-style", n_kv_groups=4),  # + سيحتاج Sliding Window منفصلًا
        BlockSpec(name="DeepSeek-style", d_latent=128, use_moe=True),
    ]
    # TODO: استدعِ build_and_report لكل spec واطبع جدول مقارنة منسّق


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-19-chapter-19.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل التاسع عشر: دراسة حالة تطبيقية — تشريح معماري لـ LLaMA 3، Mistral، وDeepSeek-V3
Large Language Model Engineering — LLaMA

Book slug : llm-engineering-69-chapters
Chapter   : ch-19-chapter-19 (lesson 19)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_19.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 19: الفصل التاسع عشر: دراسة حالة تطبيقية — تشريح معماري لـ LLaMA 3، Mistral، وDeepSeek-V3."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_architecture() -> None:
    models = [{'name': 'encoder', 'bidirectional': True, 'generation': False}, {'name': 'decoder', 'bidirectional': False, 'generation': True}, {'name': 'encoder-decoder', 'bidirectional': True, 'generation': True}]
    for model in models:
        print(model['name'], 'context:', 'both' if model['bidirectional'] else 'causal', '| generation:', model['generation'])

if __name__ == "__main__":
    print("Chapter 19 lab: Compare model architecture configurations")
    run_architecture()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-04-modern-architectures/chapter-19.md
# ────────────────────────────────────────────────────────────────────────
"""
chapter19_architecture_comparison.py
مقارنة عملية بين MHA، GQA، وMLA من حيث حجم KV Cache،
مع تطبيق مصغّر قابل للتشغيل لكل آلية.
Python 3.13+, PyTorch 2.x
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


@dataclass
class AttentionMemoryReport:
    name: str
    bytes_per_layer: float

    @property
    def mb_per_layer(self) -> float:
        return self.bytes_per_layer / (1024 ** 2)


def compute_kv_cache_sizes(
    seq_len: int,
    n_heads: int,
    d_head: int,
    n_kv_groups: int,
    d_latent: int,
    dtype_bytes: int = 2,
) -> list[AttentionMemoryReport]:
    """حساب حجم ذاكرة KV Cache لكل آلية انتباه (لكل طبقة واحدة)."""
    reports = []

    # 1) MHA القياسية: كل رأس له K,V منفصلان بالكامل
    mha_bytes = 2 * seq_len * n_heads * d_head * dtype_bytes
    reports.append(AttentionMemoryReport("MHA القياسية", mha_bytes))

    # 2) GQA: عدد مجموعات KV أقل من عدد رؤوس Q
    gqa_bytes = 2 * seq_len * n_kv_groups * d_head * dtype_bytes
    reports.append(AttentionMemoryReport(f"GQA ({n_kv_groups} مجموعات)", gqa_bytes))

    # 3) MQA: حالة خاصة من GQA بمجموعة واحدة فقط
    mqa_bytes = 2 * seq_len * 1 * d_head * dtype_bytes
    reports.append(AttentionMemoryReport("MQA (مجموعة واحدة)", mqa_bytes))

    # 4) MLA: متّجه كامن واحد مضغوط فقط (بلا تكرار عبر الرؤوس)
    mla_bytes = seq_len * d_latent * dtype_bytes
    reports.append(AttentionMemoryReport(f"MLA (بُعد كامن={d_latent})", mla_bytes))

    return reports


class SimplifiedMLA(nn.Module):
    """
    تطبيق مبسّط تعليمي لـ Multi-Head Latent Attention (DeepSeek-V2/V3).

    الفكرة: ضغط K,V إلى متّجه كامن مشترك c_kv، ثم بسطه (up-project)
    إلى K,V الكاملين عند الحاجة الفعلية لحساب الانتباه.
    الذي يُخزَّن في الكاش وقت التوليد هو c_kv فقط (صغير جدًا).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_latent: int = 128,
    ) -> None:
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.d_latent = d_latent

        # إسقاط الاستعلام (Query) كالمعتاد — كل رأس له Q خاص به
        self.w_q = nn.Linear(dim, dim, bias=False)

        # مرحلة الضغط: من البُعد الكامل إلى البُعد الكامن الصغير
        self.w_down_kv = nn.Linear(dim, d_latent, bias=False)

        # مرحلة البسط: من البُعد الكامن إلى K و V الكاملين
        self.w_up_k = nn.Linear(d_latent, dim, bias=False)
        self.w_up_v = nn.Linear(d_latent, dim, bias=False)

        self.w_out = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        latent_cache: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape

        # --- مرحلة الضغط: نحسب المتّجه الكامن ونُخزّنه فقط في الكاش ---
        c_kv = self.w_down_kv(x)  # (B, T, d_latent) — هذا ما يُخزَّن فعليًا!

        if latent_cache is not None:
            c_kv_full = torch.cat([latent_cache, c_kv], dim=1)
        else:
            c_kv_full = c_kv

        # --- مرحلة البسط: نُعيد بناء K و V الكاملين عند الحاجة فقط ---
        k_full = self.w_up_k(c_kv_full)  # (B, T_total, D)
        v_full = self.w_up_v(c_kv_full)  # (B, T_total, D)
        q = self.w_q(x)                   # (B, T, D)

        # إعادة تشكيل لرؤوس متعددة
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            B_, T_, _ = t.shape
            return t.view(B_, T_, self.n_heads, self.d_head).transpose(1, 2)

        q, k_full, v_full = split_heads(q), split_heads(k_full), split_heads(v_full)

        attn_out = F.scaled_dot_product_attention(
            q, k_full, v_full, attn_mask=causal_mask, is_causal=(causal_mask is None)
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)

        return self.w_out(attn_out), c_kv_full  # نُعيد الكاش الكامن المُحدَّث


if __name__ == "__main__":
    print("=== مقارنة حجم KV Cache لكل طبقة واحدة ===\n")
    reports = compute_kv_cache_sizes(
        seq_len=2048, n_heads=16, d_head=64, n_kv_groups=4, d_latent=128
    )
    baseline = reports[0].bytes_per_layer
    for r in reports:
        ratio = baseline / r.bytes_per_layer
        print(f"{r.name:30s}: {r.mb_per_layer:7.3f} MB   (تقليص {ratio:.1f}x عن MHA)")

    print("\n=== اختبار SimplifiedMLA عمليًا ===")
    mla = SimplifiedMLA(dim=1024, n_heads=16, d_latent=128)
    x = torch.randn(1, 10, 1024)
    causal = torch.triu(torch.full((10, 10), float("-inf")), diagonal=1)
    out, cache = mla(x, latent_cache=None, causal_mask=causal)
    print(f"شكل المُخرج: {tuple(out.shape)}")
    print(f"شكل الكاش الكامن المُخزَّن: {tuple(cache.shape)}  <- هذا فقط ما يُحفظ!")

    # محاكاة خطوة توليد تالية: رمز واحد جديد، يستخدم الكاش السابق
    new_token = torch.randn(1, 1, 1024)
    out2, cache2 = mla(new_token, latent_cache=cache, causal_mask=None)
    print(f"بعد خطوة توليد إضافية، شكل الكاش: {tuple(cache2.shape)}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-04-modern-architectures/chapter-19.md
# ────────────────────────────────────────────────────────────────────────
# ❌ تفكير ساذج: "لنُقلّص دائمًا إلى أقصى حد ممكن (مثل MQA أو بُعد كامن صغير جدًا)"
mla_too_aggressive = SimplifiedMLA(dim=1024, n_heads=16, d_latent=16)  # صغير جدًا!
# النتيجة: فقدان حاد في جودة النموذج لأن المتّجه الكامن لا يحمل معلومة كافية

# ✅ صحيح: اختر d_latent بعد تجربة تجريبية موازنة بين الذاكرة والجودة
# (في DeepSeek-V2 الفعلي، d_latent تم اختياره بعناية بعد تجارب Ablation واسعة)
mla_balanced = SimplifiedMLA(dim=1024, n_heads=16, d_latent=128)

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-04-modern-architectures/chapter-19.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع: "Mixtral لديه 8 خبراء إذن قدرته تعادل 8 نماذج منفصلة"
# هذا غير دقيق - الخبراء يتشاركون نفس طبقات الانتباه، والتوجيه ديناميكي لكل رمز

# ✅ الفهم الصحيح: السعة الإجمالية (knowledge capacity) أكبر،
# لكن الحساب الفعلي لكل رمز يعادل فقط Top-K من الخبراء (عادة 2)
# أي: فكّر في MoE كـ "مكتبة ضخمة من الخبرات المتخصصة" يُستدعى منها القليل فقط حسب الحاجة

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-04-modern-architectures/chapter-19.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: افتراض أن نافذة 4096 تعني أن النموذج "لا يرى إطلاقًا" أبعد من 4096 رمز
# حتى في الطبقة الأولى - هذا صحيح لطبقة واحدة، لكن خاطئ للنموذج ككل

# ✅ الفهم الصحيح: احسب الحقل الاستقبالي الفعّال عبر كل الطبقات
def effective_receptive_field(window_size: int, n_layers: int) -> int:
    """تقدير تقريبي للحقل الاستقبالي الفعّال عبر طبقات متراصّة من SWA."""
    return window_size * n_layers

print(effective_receptive_field(4096, 32))  # 131,072 - أكبر بكثير من النافذة المفردة!

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-04-modern-architectures/chapter-19.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: مقارنة "سرعة استدلال LLaMA 70B مقابل DeepSeek-V3 671B"
# دون توضيح أن DeepSeek تستخدم فقط 37B معامل نشط لكل رمز

# ✅ المقارنة العادلة يجب أن توضّح الأبعاد الثلاثة دائمًا:
comparison = {
    "LLaMA 3 70B":  {"total_params": "70B",  "active_params": "70B",  "type": "Dense"},
    "DeepSeek-V3":  {"total_params": "671B", "active_params": "37B",  "type": "MoE"},
}
# المقارنة العادلة لسرعة الاستدلال: قارن active_params وليس total_params

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-04-modern-architectures/chapter-19.md
# ────────────────────────────────────────────────────────────────────────
"""
mini_project_architecture_dashboard.py
لوحة مقارنة معمارية: تحسب ذاكرة KV Cache، عدد المعاملات،
و FLOPs التقريبية لثلاثة تصاميم مستوحاة من LLaMA/Mistral/DeepSeek-V3.

المطلوب إكماله:
1. استخدم الأكواد من الفصول 11، 15، 16، 18، 19 كمكوّنات بناء.
2. ابنِ ثلاث دوال build_llama_style_block / build_mistral_style_block /
   build_deepseek_style_block، كل واحدة تُعيد نموذج nn.Module لكتلة واحدة.
3. لكل كتلة، احسب:
   - إجمالي عدد المعاملات (sum(p.numel())).
   - حجم KV Cache لتسلسل بطول 4096 (استخدم compute_kv_cache_sizes من هذا الفصل).
   - FLOPs تقريبية باستخدام القاعدة 6ND من الفصل 14 (للتدريب) وتقدير الاستدلال.
4. اطبع جدولًا نهائيًا يقارن الثلاثة جنبًا إلى جنب.
5. (تحدٍّ إضافي) ارسم رسمًا بيانيًا بـ matplotlib يُظهر نمو حجم KV Cache
   مع ازدياد طول التسلسل (512, 1024, 2048, 4096, 8192, 16384) لكل من الثلاثة.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BlockSpec:
    name: str
    dim: int = 1024
    n_heads: int = 16
    n_kv_groups: int | None = None   # None يعني MHA كاملة
    d_latent: int | None = None       # None يعني بدون MLA
    use_moe: bool = False
    n_experts: int = 8
    top_k: int = 2


def build_and_report(spec: BlockSpec, seq_len: int = 4096) -> dict:
    # TODO 1: بناء الكتلة الفعلية (استخدم المكوّنات من الفصول السابقة)
    # TODO 2: حساب إجمالي المعاملات
    # TODO 3: حساب حجم KV Cache المناسب (MHA/GQA/MLA حسب spec)
    # TODO 4: حساب FLOPs تقريبية بالقاعدة 6ND (افترض N=عدد المعاملات، وD=عدد رموز افتراضي)
    # TODO 5: إعادة قاموس بكل هذه الإحصائيات
    raise NotImplementedError


def main() -> None:
    specs = [
        BlockSpec(name="LLaMA-style", n_kv_groups=4),
        BlockSpec(name="Mistral-style", n_kv_groups=4),  # + سيحتاج Sliding Window منفصلًا
        BlockSpec(name="DeepSeek-style", d_latent=128, use_moe=True),
    ]
    # TODO: استدعِ build_and_report لكل spec واطبع جدول مقارنة منسّق


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-19-chapter-19.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 19: الفصل التاسع عشر: دراسة حالة تطبيقية — تشريح معماري لـ LLaMA 3، Mistral، وDeepSeek-V3."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_architecture() -> None:
    models = [{'name': 'encoder', 'bidirectional': True, 'generation': False}, {'name': 'decoder', 'bidirectional': False, 'generation': True}, {'name': 'encoder-decoder', 'bidirectional': True, 'generation': True}]
    for model in models:
        print(model['name'], 'context:', 'both' if model['bidirectional'] else 'causal', '| generation:', model['generation'])

if __name__ == "__main__":
    print("Chapter 19 lab: Compare model architecture configurations")
    run_architecture()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-19-chapter-19.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
chapter19_architecture_comparison.py
مقارنة عملية بين MHA، GQA، وMLA من حيث حجم KV Cache،
مع تطبيق مصغّر قابل للتشغيل لكل آلية.
Python 3.13+, PyTorch 2.x
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


@dataclass
class AttentionMemoryReport:
    name: str
    bytes_per_layer: float

    @property
    def mb_per_layer(self) -> float:
        return self.bytes_per_layer / (1024 ** 2)


def compute_kv_cache_sizes(
    seq_len: int,
    n_heads: int,
    d_head: int,
    n_kv_groups: int,
    d_latent: int,
    dtype_bytes: int = 2,
) -> list[AttentionMemoryReport]:
    """حساب حجم ذاكرة KV Cache لكل آلية انتباه (لكل طبقة واحدة)."""
    reports = []

    # 1) MHA القياسية: كل رأس له K,V منفصلان بالكامل
    mha_bytes = 2 * seq_len * n_heads * d_head * dtype_bytes
    reports.append(AttentionMemoryReport("MHA القياسية", mha_bytes))

    # 2) GQA: عدد مجموعات KV أقل من عدد رؤوس Q
    gqa_bytes = 2 * seq_len * n_kv_groups * d_head * dtype_bytes
    reports.append(AttentionMemoryReport(f"GQA ({n_kv_groups} مجموعات)", gqa_bytes))

    # 3) MQA: حالة خاصة من GQA بمجموعة واحدة فقط
    mqa_bytes = 2 * seq_len * 1 * d_head * dtype_bytes
    reports.append(AttentionMemoryReport("MQA (مجموعة واحدة)", mqa_bytes))

    # 4) MLA: متّجه كامن واحد مضغوط فقط (بلا تكرار عبر الرؤوس)
    mla_bytes = seq_len * d_latent * dtype_bytes
    reports.append(AttentionMemoryReport(f"MLA (بُعد كامن={d_latent})", mla_bytes))

    return reports


class SimplifiedMLA(nn.Module):
    """
    تطبيق مبسّط تعليمي لـ Multi-Head Latent Attention (DeepSeek-V2/V3).

    الفكرة: ضغط K,V إلى متّجه كامن مشترك c_kv، ثم بسطه (up-project)
    إلى K,V الكاملين عند الحاجة الفعلية لحساب الانتباه.
    الذي يُخزَّن في الكاش وقت التوليد هو c_kv فقط (صغير جدًا).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_latent: int = 128,
    ) -> None:
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.d_latent = d_latent

        # إسقاط الاستعلام (Query) كالمعتاد — كل رأس له Q خاص به
        self.w_q = nn.Linear(dim, dim, bias=False)

        # مرحلة الضغط: من البُعد الكامل إلى البُعد الكامن الصغير
        self.w_down_kv = nn.Linear(dim, d_latent, bias=False)

        # مرحلة البسط: من البُعد الكامن إلى K و V الكاملين
        self.w_up_k = nn.Linear(d_latent, dim, bias=False)
        self.w_up_v = nn.Linear(d_latent, dim, bias=False)

        self.w_out = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        latent_cache: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape

        # --- مرحلة الضغط: نحسب المتّجه الكامن ونُخزّنه فقط في الكاش ---
        c_kv = self.w_down_kv(x)  # (B, T, d_latent) — هذا ما يُخزَّن فعليًا!

        if latent_cache is not None:
            c_kv_full = torch.cat([latent_cache, c_kv], dim=1)
        else:
            c_kv_full = c_kv

        # --- مرحلة البسط: نُعيد بناء K و V الكاملين عند الحاجة فقط ---
        k_full = self.w_up_k(c_kv_full)  # (B, T_total, D)
        v_full = self.w_up_v(c_kv_full)  # (B, T_total, D)
        q = self.w_q(x)                   # (B, T, D)

        # إعادة تشكيل لرؤوس متعددة
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            B_, T_, _ = t.shape
            return t.view(B_, T_, self.n_heads, self.d_head).transpose(1, 2)

        q, k_full, v_full = split_heads(q), split_heads(k_full), split_heads(v_full)

        attn_out = F.scaled_dot_product_attention(
            q, k_full, v_full, attn_mask=causal_mask, is_causal=(causal_mask is None)
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)

        return self.w_out(attn_out), c_kv_full  # نُعيد الكاش الكامن المُحدَّث


if __name__ == "__main__":
    print("=== مقارنة حجم KV Cache لكل طبقة واحدة ===\n")
    reports = compute_kv_cache_sizes(
        seq_len=2048, n_heads=16, d_head=64, n_kv_groups=4, d_latent=128
    )
    baseline = reports[0].bytes_per_layer
    for r in reports:
        ratio = baseline / r.bytes_per_layer
        print(f"{r.name:30s}: {r.mb_per_layer:7.3f} MB   (تقليص {ratio:.1f}x عن MHA)")

    print("\n=== اختبار SimplifiedMLA عمليًا ===")
    mla = SimplifiedMLA(dim=1024, n_heads=16, d_latent=128)
    x = torch.randn(1, 10, 1024)
    causal = torch.triu(torch.full((10, 10), float("-inf")), diagonal=1)
    out, cache = mla(x, latent_cache=None, causal_mask=causal)
    print(f"شكل المُخرج: {tuple(out.shape)}")
    print(f"شكل الكاش الكامن المُخزَّن: {tuple(cache.shape)}  <- هذا فقط ما يُحفظ!")

    # محاكاة خطوة توليد تالية: رمز واحد جديد، يستخدم الكاش السابق
    new_token = torch.randn(1, 1, 1024)
    out2, cache2 = mla(new_token, latent_cache=cache, causal_mask=None)
    print(f"بعد خطوة توليد إضافية، شكل الكاش: {tuple(cache2.shape)}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-19-chapter-19.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ❌ تفكير ساذج: "لنُقلّص دائمًا إلى أقصى حد ممكن (مثل MQA أو بُعد كامن صغير جدًا)"
mla_too_aggressive = SimplifiedMLA(dim=1024, n_heads=16, d_latent=16)  # صغير جدًا!
# النتيجة: فقدان حاد في جودة النموذج لأن المتّجه الكامن لا يحمل معلومة كافية

# ✅ صحيح: اختر d_latent بعد تجربة تجريبية موازنة بين الذاكرة والجودة
# (في DeepSeek-V2 الفعلي، d_latent تم اختياره بعناية بعد تجارب Ablation واسعة)
mla_balanced = SimplifiedMLA(dim=1024, n_heads=16, d_latent=128)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-19-chapter-19.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع: "Mixtral لديه 8 خبراء إذن قدرته تعادل 8 نماذج منفصلة"
# هذا غير دقيق - الخبراء يتشاركون نفس طبقات الانتباه، والتوجيه ديناميكي لكل رمز

# ✅ الفهم الصحيح: السعة الإجمالية (knowledge capacity) أكبر،
# لكن الحساب الفعلي لكل رمز يعادل فقط Top-K من الخبراء (عادة 2)
# أي: فكّر في MoE كـ "مكتبة ضخمة من الخبرات المتخصصة" يُستدعى منها القليل فقط حسب الحاجة

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-19-chapter-19.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: افتراض أن نافذة 4096 تعني أن النموذج "لا يرى إطلاقًا" أبعد من 4096 رمز
# حتى في الطبقة الأولى - هذا صحيح لطبقة واحدة، لكن خاطئ للنموذج ككل

# ✅ الفهم الصحيح: احسب الحقل الاستقبالي الفعّال عبر كل الطبقات
def effective_receptive_field(window_size: int, n_layers: int) -> int:
    """تقدير تقريبي للحقل الاستقبالي الفعّال عبر طبقات متراصّة من SWA."""
    return window_size * n_layers

print(effective_receptive_field(4096, 32))  # 131,072 - أكبر بكثير من النافذة المفردة!

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-19-chapter-19.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: مقارنة "سرعة استدلال LLaMA 70B مقابل DeepSeek-V3 671B"
# دون توضيح أن DeepSeek تستخدم فقط 37B معامل نشط لكل رمز

# ✅ المقارنة العادلة يجب أن توضّح الأبعاد الثلاثة دائمًا:
comparison = {
    "LLaMA 3 70B":  {"total_params": "70B",  "active_params": "70B",  "type": "Dense"},
    "DeepSeek-V3":  {"total_params": "671B", "active_params": "37B",  "type": "MoE"},
}
# المقارنة العادلة لسرعة الاستدلال: قارن active_params وليس total_params

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-19-chapter-19.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
"""
mini_project_architecture_dashboard.py
لوحة مقارنة معمارية: تحسب ذاكرة KV Cache، عدد المعاملات،
و FLOPs التقريبية لثلاثة تصاميم مستوحاة من LLaMA/Mistral/DeepSeek-V3.

المطلوب إكماله:
1. استخدم الأكواد من الفصول 11، 15، 16، 18، 19 كمكوّنات بناء.
2. ابنِ ثلاث دوال build_llama_style_block / build_mistral_style_block /
   build_deepseek_style_block، كل واحدة تُعيد نموذج nn.Module لكتلة واحدة.
3. لكل كتلة، احسب:
   - إجمالي عدد المعاملات (sum(p.numel())).
   - حجم KV Cache لتسلسل بطول 4096 (استخدم compute_kv_cache_sizes من هذا الفصل).
   - FLOPs تقريبية باستخدام القاعدة 6ND من الفصل 14 (للتدريب) وتقدير الاستدلال.
4. اطبع جدولًا نهائيًا يقارن الثلاثة جنبًا إلى جنب.
5. (تحدٍّ إضافي) ارسم رسمًا بيانيًا بـ matplotlib يُظهر نمو حجم KV Cache
   مع ازدياد طول التسلسل (512, 1024, 2048, 4096, 8192, 16384) لكل من الثلاثة.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BlockSpec:
    name: str
    dim: int = 1024
    n_heads: int = 16
    n_kv_groups: int | None = None   # None يعني MHA كاملة
    d_latent: int | None = None       # None يعني بدون MLA
    use_moe: bool = False
    n_experts: int = 8
    top_k: int = 2


def build_and_report(spec: BlockSpec, seq_len: int = 4096) -> dict:
    # TODO 1: بناء الكتلة الفعلية (استخدم المكوّنات من الفصول السابقة)
    # TODO 2: حساب إجمالي المعاملات
    # TODO 3: حساب حجم KV Cache المناسب (MHA/GQA/MLA حسب spec)
    # TODO 4: حساب FLOPs تقريبية بالقاعدة 6ND (افترض N=عدد المعاملات، وD=عدد رموز افتراضي)
    # TODO 5: إعادة قاموس بكل هذه الإحصائيات
    raise NotImplementedError


def main() -> None:
    specs = [
        BlockSpec(name="LLaMA-style", n_kv_groups=4),
        BlockSpec(name="Mistral-style", n_kv_groups=4),  # + سيحتاج Sliding Window منفصلًا
        BlockSpec(name="DeepSeek-style", d_latent=128, use_moe=True),
    ]
    # TODO: استدعِ build_and_report لكل spec واطبع جدول مقارنة منسّق


if __name__ == "__main__":
    main()
