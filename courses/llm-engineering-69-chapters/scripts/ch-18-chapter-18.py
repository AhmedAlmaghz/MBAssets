#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الثامن عشر: تقنيات معمارية حديثة — RMSNorm، SwiGLU، Multi-Token Prediction، وYaRN/NTK-aware Scaling
Large Language Model Engineering — RMSNorm

Book slug : llm-engineering-69-chapters
Chapter   : ch-18-chapter-18 (lesson 18)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-18-chapter-18.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الثامن عشر: تقنيات معمارية حديثة — RMSNorm، SwiGLU، Multi-Token Prediction، وYaRN/NTK-aware Scaling
Large Language Model Engineering — RMSNorm

Book slug : llm-engineering-69-chapters
Chapter   : ch-18-chapter-18 (lesson 18)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-18-chapter-18.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الثامن عشر: تقنيات معمارية حديثة — RMSNorm، SwiGLU، Multi-Token Prediction، وYaRN/NTK-aware Scaling
Large Language Model Engineering — RMSNorm

Book slug : llm-engineering-69-chapters
Chapter   : ch-18-chapter-18 (lesson 18)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_18.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 18: الفصل الثامن عشر: تقنيات معمارية حديثة — RMSNorm، SwiGLU، Multi-Token Prediction، وYaRN/NTK-aware Scaling."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_modern_blocks() -> None:
    values = [1.0, -2.0, 0.5, 3.0]
    rms = math.sqrt(sum((x * x for x in values)) / len(values) + 1e-06)
    normalized = [x / rms for x in values]
    silu = lambda x: x / (1 + math.exp(-x))
    swiglu = [silu(x) * math.tanh(0.8 * x) for x in normalized]
    print('RMSNorm:', [round(x, 3) for x in normalized])
    print('SwiGLU values:', [round(x, 3) for x in swiglu])

if __name__ == "__main__":
    print("Chapter 18 lab: RMSNorm and SwiGLU")
    run_modern_blocks()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-04-modern-architectures/chapter-18.md
# ────────────────────────────────────────────────────────────────────────
"""
chapter18_modern_architecture.py
تطبيق كامل لتقنيات معمارية حديثة: RMSNorm، SwiGLU، Multi-Token Prediction، وYaRN.
Python 3.13+, PyTorch 2.x
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


# =========================================================================
# 1) RMSNorm
# =========================================================================
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    على عكس LayerNorm: لا يطرح المتوسط، ولا يملك معامل انحياز (bias).
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # معامل القياس γ فقط

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        # نحسب RMS على آخر محور (بُعد التضمين)
        # نستخدم float32 داخليًا للاستقرار العددي حتى لو كان x بدقة أقل (FP16/BF16)
        in_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_normed = x_f32 / rms
        return (self.weight * x_normed.to(in_dtype))


def compare_layernorm_vs_rmsnorm() -> None:
    """مقارنة عملية بين LayerNorm وRMSNorm من حيث السلوك العددي."""
    dim = 512
    x = torch.randn(4, 10, dim) * 3.0 + 1.5  # بيانات بها إزاحة (bias) ملحوظة

    ln = nn.LayerNorm(dim)
    rn = RMSNorm(dim)

    y_ln = ln(x)
    y_rn = rn(x)

    print("=== مقارنة LayerNorm و RMSNorm ===")
    print(f"متوسط مُدخل x (يجب أن يكون ~1.5): {x.mean().item():.4f}")
    print(f"متوسط مخرج LayerNorm (يجب أن يكون ~0): {y_ln.mean().item():.4f}")
    print(f"متوسط مخرج RMSNorm (لن يكون صفرًا بالضرورة): {y_rn.mean().item():.4f}")
    print(f"عدد معاملات LayerNorm: {sum(p.numel() for p in ln.parameters())}")
    print(f"عدد معاملات RMSNorm:   {sum(p.numel() for p in rn.parameters())}")


# =========================================================================
# 2) SwiGLU FFN
# =========================================================================
class SwiGLUFeedForward(nn.Module):
    """
    شبكة تغذية أمامية بآلية بوابة SwiGLU (Shazeer, 2020).

    FFN_SwiGLU(x) = (SiLU(xW1) ⊗ xW3) W2

    يُعاد ضبط d_ff تلقائيًا إلى ~2/3 من القيمة الكلاسيكية
    للحفاظ على نفس ميزانية المعاملات التقريبية، مع التقريب
    لأقرب مضاعف multiple_of لتحسين كفاءة الـ GPU.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        multiple_of: int = 256,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim               # القيمة الكلاسيكية
            hidden_dim = int(2 * hidden_dim / 3)  # إعادة الموازنة لـ SwiGLU
            # التقريب لأعلى مضاعف لـ multiple_of
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)   # مسار البوابة (gate)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)   # المسار الخطي (value)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)   # إسقاط العودة

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(x))   # SiLU(xW1)
        value = self.w3(x)          # xW3
        return self.w2(gate * value)  # (SiLU(xW1) ⊗ xW3) W2


class ClassicFeedForward(nn.Module):
    """FFN كلاسيكية بـ GELU للمقارنة."""

    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


def compare_ffn_parameter_budgets(dim: int = 512) -> None:
    classic = ClassicFeedForward(dim)
    swiglu = SwiGLUFeedForward(dim)

    n_classic = sum(p.numel() for p in classic.parameters())
    n_swiglu = sum(p.numel() for p in swiglu.parameters())

    print("\n=== مقارنة ميزانية المعاملات: FFN كلاسيكية مقابل SwiGLU ===")
    print(f"FFN كلاسيكية (GELU): {n_classic:,} معامل")
    print(f"FFN حديثة   (SwiGLU): {n_swiglu:,} معامل")
    print(f"نسبة الزيادة: {(n_swiglu / n_classic - 1) * 100:.2f}%")


# =========================================================================
# 3) كتلة Transformer حديثة كاملة (Pre-Norm + RoPE + RMSNorm + SwiGLU)
# =========================================================================
class ModernTransformerBlock(nn.Module):
    """
    كتلة Transformer بمواصفات حديثة (على غرار LLaMA):
    - Pre-Norm باستخدام RMSNorm
    - Self-Attention (بدون تفاصيل RoPE هنا لتبسيط المثال، انظر الفصل 15)
    - SwiGLU FFN
    """

    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLUFeedForward(dim)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-Norm + Residual حول الانتباه
        normed = self.attn_norm(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out

        # Pre-Norm + Residual حول FFN
        normed2 = self.ffn_norm(x)
        x = x + self.ffn(normed2)
        return x


if __name__ == "__main__":
    compare_layernorm_vs_rmsnorm()
    compare_ffn_parameter_budgets(dim=512)

    print("\n=== اختبار كتلة Transformer حديثة كاملة ===")
    block = ModernTransformerBlock(dim=512, n_heads=8)
    dummy_input = torch.randn(2, 16, 512)
    seq_len = dummy_input.size(1)
    causal = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    output = block(dummy_input, causal_mask=causal)
    print(f"شكل المُدخل:  {tuple(dummy_input.shape)}")
    print(f"شكل المُخرج: {tuple(output.shape)}")
    total_params = sum(p.numel() for p in block.parameters())
    print(f"إجمالي معاملات الكتلة: {total_params:,}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-04-modern-architectures/chapter-18.md
# ────────────────────────────────────────────────────────────────────────
"""
yarn_ntk_scaling.py
تطبيق مبسّط لمفهوم NTK-aware scaling و YaRN لتمديد سياق RoPE.
"""
from __future__ import annotations

import torch


def compute_rope_base_frequencies(
    dim: int, base: float = 10000.0
) -> torch.Tensor:
    """حساب الترددات الأصلية لـ RoPE: θ_i = base^(-2i/dim)."""
    i = torch.arange(0, dim, 2, dtype=torch.float32)
    return base ** (-i / dim)


def ntk_aware_base(
    original_base: float, scale_factor: float, dim: int
) -> float:
    """
    حساب القاعدة الجديدة (base') وفق NTK-aware interpolation.

    base' = base * scale_factor^(dim / (dim - 2))
    """
    exponent = dim / (dim - 2)
    return original_base * (scale_factor ** exponent)


def yarn_ramp_mask(
    dim: int,
    original_max_pos: int,
    extended_max_pos: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> torch.Tensor:
    """
    حساب دالة Ramp لـ YaRN: تُحدّد لكل بُعد (تردد) مدى تطبيق
    التدخل التفاعلي (0 = بدون تدخل/extrapolation طبيعي،
    1 = تدخل كامل/interpolation خطي).

    التبسيط هنا تعليمي، التفاصيل الكاملة في ورقة YaRN الأصلية.
    """
    scale = extended_max_pos / original_max_pos
    dims = torch.arange(0, dim, 2, dtype=torch.float32)

    # عدد "الدورات" الكاملة لكل تردد ضمن طول السياق الأصلي
    # الترددات ذات دورات قليلة جدًا (منخفضة التردد) تحتاج تدخلًا كاملًا
    # الترددات ذات دورات كثيرة (عالية التردد) لا تحتاج تدخلًا يُذكر
    low = dim * torch.log(torch.tensor(original_max_pos / (beta_fast * 2 * torch.pi))) / (
        2 * torch.log(torch.tensor(10000.0))
    )
    high = dim * torch.log(torch.tensor(original_max_pos / (beta_slow * 2 * torch.pi))) / (
        2 * torch.log(torch.tensor(10000.0))
    )
    low, high = max(low.item(), 0), min(high.item(), dim - 1)

    ramp = torch.clamp((dims - low) / max(high - low, 1e-3), 0, 1)
    return ramp  # قيمة بين 0 و 1 لكل بُعد تردد


def yarn_attention_temperature(scale_factor: float) -> float:
    """معامل تصحيح درجة حرارة الانتباه في YaRN: t = 0.1*ln(s) + 1."""
    return 0.1 * math.log(scale_factor) + 1.0


if __name__ == "__main__":
    import math

    dim = 128
    original_len = 4096
    target_len = 32768
    scale = target_len / original_len

    base_freqs = compute_rope_base_frequencies(dim)
    new_base = ntk_aware_base(10000.0, scale, dim)
    new_freqs = compute_rope_base_frequencies(dim, base=new_base)
    ramp = yarn_ramp_mask(dim, original_len, target_len)
    temp = yarn_attention_temperature(scale)

    print(f"نسبة التمديد المطلوبة (s): {scale:.2f}x")
    print(f"القاعدة الأصلية: 10000.0  |  القاعدة الجديدة NTK-aware: {new_base:.1f}")
    print(f"تردد أول بُعد (الأعلى) — أصلي: {base_freqs[0]:.6f}  جديد: {new_freqs[0]:.6f}")
    print(f"تردد آخر بُعد (الأدنى) — أصلي: {base_freqs[-1]:.8f}  جديد: {new_freqs[-1]:.8f}")
    print(f"معامل تصحيح درجة الحرارة (attention temperature): {temp:.4f}")
    print(f"\nعيّنة من دالة Ramp (أول 5 وآخر 5 قيم):")
    print(f"  بداية (ترددات عالية، تدخل ضئيل): {ramp[:5].tolist()}")
    print(f"  نهاية (ترددات منخفضة، تدخل كامل): {ramp[-5:].tolist()}")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-04-modern-architectures/chapter-18.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: eps الافتراضي قد يكون صغيرًا جدًا لـ FP16
norm = RMSNorm(dim=4096, eps=1e-8)  # قد يُسبب تقسيمًا على رقم قريب من صفر في FP16

# ✅ صحيح: استخدم eps أكبر قليلًا (1e-5 أو 1e-6) وتأكد من الحساب بـ float32
norm = RMSNorm(dim=4096, eps=1e-6)

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-04-modern-architectures/chapter-18.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع: عكس أي من المسارين يُمرَّر عبر التفعيل
def wrong_swiglu(x, w1, w2, w3):
    gate = w3(x)              # الخطأ: هذا يجب أن يكون w1
    value = F.silu(w1(x))     # الخطأ: SiLU يجب أن تُطبَّق على w1، لا w3
    return w2(gate * value)

# ✅ صحيح: SiLU تُطبَّق دائمًا على w1 (البوابة)، وw3 يبقى خطيًا بحتًا (القيمة)
def correct_swiglu(x, w1, w2, w3):
    gate = F.silu(w1(x))
    value = w3(x)
    return w2(gate * value)

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-04-modern-architectures/chapter-18.md
# ────────────────────────────────────────────────────────────────────────
# ❌ تبسيط ساذج يُضحّي بالدقة المحلية دون داعٍ
scaled_positions = positions * (original_len / target_len)  # PI خطي بسيط

# ✅ استخدام مكتبة جاهزة تُطبّق YaRN الكامل بكل تفاصيله
# (مثلًا عبر إعداد rope_scaling في Hugging Face، أو تطبيق كامل الخوارزمية)
config.rope_scaling = {"type": "yarn", "factor": 8.0, "original_max_position_embeddings": 4096}

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-04-modern-architectures/chapter-18.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: كل الرؤوس بنفس الأهمية يُشتت تركيز النموذج عن الهدف الأساسي (t+1)
total_loss = sum(losses)  # λ_k = 1 للجميع

# ✅ صحيح: الرأس الأقرب (t+1) يجب أن يحمل الوزن الأكبر دائمًا
weights = [1.0, 0.5, 0.3, 0.2]  # تتناقص مع ازدياد k
total_loss = sum(w * l for w, l in zip(weights, losses))

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-04-modern-architectures/chapter-18.md
# ────────────────────────────────────────────────────────────────────────
"""
mini_project_modern_gpt.py
مشروع مصغّر: مقارنة GPT كلاسيكي (LayerNorm+GELU) مقابل GPT حديث (RMSNorm+SwiGLU)
على مهمة توليد نص بسيطة (character-level على نص عربي قصير).

المطلوب إكماله:
1. بناء TransformerBlock كلاسيكي وآخر حديث (استخدم المكوّنات من هذا الفصل).
2. بناء GPTModel يستخدم أيًّا من الكتلتين حسب إعداد config.
3. تدريب كلا النموذجين لنفس عدد الخطوات على نفس البيانات بنفس seed.
4. رسم منحنى الخسارة (Loss Curve) لكليهما على نفس الرسم البياني.
5. قياس الزمن الفعلي (wall-clock time) لكل 100 خطوة تدريب لكل نموذج.
6. الإجابة كتابيًا: أي نموذج تقارب أسرع؟ أيهما أسرع زمنيًا لكل خطوة؟
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MiniGPTConfig:
    vocab_size: int
    dim: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_seq_len: int = 64
    use_modern_arch: bool = True  # True = RMSNorm+SwiGLU, False = LayerNorm+GELU


class MiniGPT(nn.Module):
    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        # TODO 1: بناء طبقة تضمين الرموز (token embedding)
        # TODO 2: بناء طبقة تضمين موضعي (يمكن استخدام تضمين مُتعلَّم بسيط هنا)
        # TODO 3: بناء قائمة من TransformerBlock حسب config.use_modern_arch
        #         (استخدم ModernTransformerBlock أو كتلة كلاسيكية مكافئة)
        # TODO 4: بناء طبقة تطبيع نهائية (RMSNorm أو LayerNorm حسب الإعداد)
        # TODO 5: بناء رأس الإخراج (Linear إلى vocab_size)
        raise NotImplementedError("أكمل التصميم هنا")

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # TODO: نفّذ التمرير الأمامي الكامل، أعد logits بشكل (B, T, vocab_size)
        raise NotImplementedError("أكمل التمرير الأمامي هنا")


def train_and_compare() -> None:
    # TODO: حمّل نصًا عربيًا قصيرًا (مثلًا فقرة من كتاب عام)
    # TODO: ابنِ character-level tokenizer بسيطًا
    # TODO: درّب نموذجين (modern=True/False) بنفس البيانات ونفس seed
    # TODO: سجّل loss كل خطوة والزمن الكلي لكل 100 خطوة
    # TODO: ارسم بـ matplotlib منحنيين على نفس الرسم البياني للمقارنة
    pass


if __name__ == "__main__":
    train_and_compare()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-18-chapter-18.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 18: الفصل الثامن عشر: تقنيات معمارية حديثة — RMSNorm، SwiGLU، Multi-Token Prediction، وYaRN/NTK-aware Scaling."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_modern_blocks() -> None:
    values = [1.0, -2.0, 0.5, 3.0]
    rms = math.sqrt(sum((x * x for x in values)) / len(values) + 1e-06)
    normalized = [x / rms for x in values]
    silu = lambda x: x / (1 + math.exp(-x))
    swiglu = [silu(x) * math.tanh(0.8 * x) for x in normalized]
    print('RMSNorm:', [round(x, 3) for x in normalized])
    print('SwiGLU values:', [round(x, 3) for x in swiglu])

if __name__ == "__main__":
    print("Chapter 18 lab: RMSNorm and SwiGLU")
    run_modern_blocks()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-18-chapter-18.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
chapter18_modern_architecture.py
تطبيق كامل لتقنيات معمارية حديثة: RMSNorm، SwiGLU، Multi-Token Prediction، وYaRN.
Python 3.13+, PyTorch 2.x
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


# =========================================================================
# 1) RMSNorm
# =========================================================================
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    على عكس LayerNorm: لا يطرح المتوسط، ولا يملك معامل انحياز (bias).
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # معامل القياس γ فقط

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        # نحسب RMS على آخر محور (بُعد التضمين)
        # نستخدم float32 داخليًا للاستقرار العددي حتى لو كان x بدقة أقل (FP16/BF16)
        in_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_normed = x_f32 / rms
        return (self.weight * x_normed.to(in_dtype))


def compare_layernorm_vs_rmsnorm() -> None:
    """مقارنة عملية بين LayerNorm وRMSNorm من حيث السلوك العددي."""
    dim = 512
    x = torch.randn(4, 10, dim) * 3.0 + 1.5  # بيانات بها إزاحة (bias) ملحوظة

    ln = nn.LayerNorm(dim)
    rn = RMSNorm(dim)

    y_ln = ln(x)
    y_rn = rn(x)

    print("=== مقارنة LayerNorm و RMSNorm ===")
    print(f"متوسط مُدخل x (يجب أن يكون ~1.5): {x.mean().item():.4f}")
    print(f"متوسط مخرج LayerNorm (يجب أن يكون ~0): {y_ln.mean().item():.4f}")
    print(f"متوسط مخرج RMSNorm (لن يكون صفرًا بالضرورة): {y_rn.mean().item():.4f}")
    print(f"عدد معاملات LayerNorm: {sum(p.numel() for p in ln.parameters())}")
    print(f"عدد معاملات RMSNorm:   {sum(p.numel() for p in rn.parameters())}")


# =========================================================================
# 2) SwiGLU FFN
# =========================================================================
class SwiGLUFeedForward(nn.Module):
    """
    شبكة تغذية أمامية بآلية بوابة SwiGLU (Shazeer, 2020).

    FFN_SwiGLU(x) = (SiLU(xW1) ⊗ xW3) W2

    يُعاد ضبط d_ff تلقائيًا إلى ~2/3 من القيمة الكلاسيكية
    للحفاظ على نفس ميزانية المعاملات التقريبية، مع التقريب
    لأقرب مضاعف multiple_of لتحسين كفاءة الـ GPU.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        multiple_of: int = 256,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim               # القيمة الكلاسيكية
            hidden_dim = int(2 * hidden_dim / 3)  # إعادة الموازنة لـ SwiGLU
            # التقريب لأعلى مضاعف لـ multiple_of
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)   # مسار البوابة (gate)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)   # المسار الخطي (value)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)   # إسقاط العودة

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(x))   # SiLU(xW1)
        value = self.w3(x)          # xW3
        return self.w2(gate * value)  # (SiLU(xW1) ⊗ xW3) W2


class ClassicFeedForward(nn.Module):
    """FFN كلاسيكية بـ GELU للمقارنة."""

    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


def compare_ffn_parameter_budgets(dim: int = 512) -> None:
    classic = ClassicFeedForward(dim)
    swiglu = SwiGLUFeedForward(dim)

    n_classic = sum(p.numel() for p in classic.parameters())
    n_swiglu = sum(p.numel() for p in swiglu.parameters())

    print("\n=== مقارنة ميزانية المعاملات: FFN كلاسيكية مقابل SwiGLU ===")
    print(f"FFN كلاسيكية (GELU): {n_classic:,} معامل")
    print(f"FFN حديثة   (SwiGLU): {n_swiglu:,} معامل")
    print(f"نسبة الزيادة: {(n_swiglu / n_classic - 1) * 100:.2f}%")


# =========================================================================
# 3) كتلة Transformer حديثة كاملة (Pre-Norm + RoPE + RMSNorm + SwiGLU)
# =========================================================================
class ModernTransformerBlock(nn.Module):
    """
    كتلة Transformer بمواصفات حديثة (على غرار LLaMA):
    - Pre-Norm باستخدام RMSNorm
    - Self-Attention (بدون تفاصيل RoPE هنا لتبسيط المثال، انظر الفصل 15)
    - SwiGLU FFN
    """

    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLUFeedForward(dim)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-Norm + Residual حول الانتباه
        normed = self.attn_norm(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out

        # Pre-Norm + Residual حول FFN
        normed2 = self.ffn_norm(x)
        x = x + self.ffn(normed2)
        return x


if __name__ == "__main__":
    compare_layernorm_vs_rmsnorm()
    compare_ffn_parameter_budgets(dim=512)

    print("\n=== اختبار كتلة Transformer حديثة كاملة ===")
    block = ModernTransformerBlock(dim=512, n_heads=8)
    dummy_input = torch.randn(2, 16, 512)
    seq_len = dummy_input.size(1)
    causal = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    output = block(dummy_input, causal_mask=causal)
    print(f"شكل المُدخل:  {tuple(dummy_input.shape)}")
    print(f"شكل المُخرج: {tuple(output.shape)}")
    total_params = sum(p.numel() for p in block.parameters())
    print(f"إجمالي معاملات الكتلة: {total_params:,}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-18-chapter-18.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
yarn_ntk_scaling.py
تطبيق مبسّط لمفهوم NTK-aware scaling و YaRN لتمديد سياق RoPE.
"""
from __future__ import annotations

import torch


def compute_rope_base_frequencies(
    dim: int, base: float = 10000.0
) -> torch.Tensor:
    """حساب الترددات الأصلية لـ RoPE: θ_i = base^(-2i/dim)."""
    i = torch.arange(0, dim, 2, dtype=torch.float32)
    return base ** (-i / dim)


def ntk_aware_base(
    original_base: float, scale_factor: float, dim: int
) -> float:
    """
    حساب القاعدة الجديدة (base') وفق NTK-aware interpolation.

    base' = base * scale_factor^(dim / (dim - 2))
    """
    exponent = dim / (dim - 2)
    return original_base * (scale_factor ** exponent)


def yarn_ramp_mask(
    dim: int,
    original_max_pos: int,
    extended_max_pos: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> torch.Tensor:
    """
    حساب دالة Ramp لـ YaRN: تُحدّد لكل بُعد (تردد) مدى تطبيق
    التدخل التفاعلي (0 = بدون تدخل/extrapolation طبيعي،
    1 = تدخل كامل/interpolation خطي).

    التبسيط هنا تعليمي، التفاصيل الكاملة في ورقة YaRN الأصلية.
    """
    scale = extended_max_pos / original_max_pos
    dims = torch.arange(0, dim, 2, dtype=torch.float32)

    # عدد "الدورات" الكاملة لكل تردد ضمن طول السياق الأصلي
    # الترددات ذات دورات قليلة جدًا (منخفضة التردد) تحتاج تدخلًا كاملًا
    # الترددات ذات دورات كثيرة (عالية التردد) لا تحتاج تدخلًا يُذكر
    low = dim * torch.log(torch.tensor(original_max_pos / (beta_fast * 2 * torch.pi))) / (
        2 * torch.log(torch.tensor(10000.0))
    )
    high = dim * torch.log(torch.tensor(original_max_pos / (beta_slow * 2 * torch.pi))) / (
        2 * torch.log(torch.tensor(10000.0))
    )
    low, high = max(low.item(), 0), min(high.item(), dim - 1)

    ramp = torch.clamp((dims - low) / max(high - low, 1e-3), 0, 1)
    return ramp  # قيمة بين 0 و 1 لكل بُعد تردد


def yarn_attention_temperature(scale_factor: float) -> float:
    """معامل تصحيح درجة حرارة الانتباه في YaRN: t = 0.1*ln(s) + 1."""
    return 0.1 * math.log(scale_factor) + 1.0


if __name__ == "__main__":
    import math

    dim = 128
    original_len = 4096
    target_len = 32768
    scale = target_len / original_len

    base_freqs = compute_rope_base_frequencies(dim)
    new_base = ntk_aware_base(10000.0, scale, dim)
    new_freqs = compute_rope_base_frequencies(dim, base=new_base)
    ramp = yarn_ramp_mask(dim, original_len, target_len)
    temp = yarn_attention_temperature(scale)

    print(f"نسبة التمديد المطلوبة (s): {scale:.2f}x")
    print(f"القاعدة الأصلية: 10000.0  |  القاعدة الجديدة NTK-aware: {new_base:.1f}")
    print(f"تردد أول بُعد (الأعلى) — أصلي: {base_freqs[0]:.6f}  جديد: {new_freqs[0]:.6f}")
    print(f"تردد آخر بُعد (الأدنى) — أصلي: {base_freqs[-1]:.8f}  جديد: {new_freqs[-1]:.8f}")
    print(f"معامل تصحيح درجة الحرارة (attention temperature): {temp:.4f}")
    print(f"\nعيّنة من دالة Ramp (أول 5 وآخر 5 قيم):")
    print(f"  بداية (ترددات عالية، تدخل ضئيل): {ramp[:5].tolist()}")
    print(f"  نهاية (ترددات منخفضة، تدخل كامل): {ramp[-5:].tolist()}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-18-chapter-18.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: eps الافتراضي قد يكون صغيرًا جدًا لـ FP16
norm = RMSNorm(dim=4096, eps=1e-8)  # قد يُسبب تقسيمًا على رقم قريب من صفر في FP16

# ✅ صحيح: استخدم eps أكبر قليلًا (1e-5 أو 1e-6) وتأكد من الحساب بـ float32
norm = RMSNorm(dim=4096, eps=1e-6)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-18-chapter-18.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع: عكس أي من المسارين يُمرَّر عبر التفعيل
def wrong_swiglu(x, w1, w2, w3):
    gate = w3(x)              # الخطأ: هذا يجب أن يكون w1
    value = F.silu(w1(x))     # الخطأ: SiLU يجب أن تُطبَّق على w1، لا w3
    return w2(gate * value)

# ✅ صحيح: SiLU تُطبَّق دائمًا على w1 (البوابة)، وw3 يبقى خطيًا بحتًا (القيمة)
def correct_swiglu(x, w1, w2, w3):
    gate = F.silu(w1(x))
    value = w3(x)
    return w2(gate * value)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-18-chapter-18.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ❌ تبسيط ساذج يُضحّي بالدقة المحلية دون داعٍ
scaled_positions = positions * (original_len / target_len)  # PI خطي بسيط

# ✅ استخدام مكتبة جاهزة تُطبّق YaRN الكامل بكل تفاصيله
# (مثلًا عبر إعداد rope_scaling في Hugging Face، أو تطبيق كامل الخوارزمية)
config.rope_scaling = {"type": "yarn", "factor": 8.0, "original_max_position_embeddings": 4096}

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-18-chapter-18.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: كل الرؤوس بنفس الأهمية يُشتت تركيز النموذج عن الهدف الأساسي (t+1)
total_loss = sum(losses)  # λ_k = 1 للجميع

# ✅ صحيح: الرأس الأقرب (t+1) يجب أن يحمل الوزن الأكبر دائمًا
weights = [1.0, 0.5, 0.3, 0.2]  # تتناقص مع ازدياد k
total_loss = sum(w * l for w, l in zip(weights, losses))

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-18-chapter-18.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
"""
mini_project_modern_gpt.py
مشروع مصغّر: مقارنة GPT كلاسيكي (LayerNorm+GELU) مقابل GPT حديث (RMSNorm+SwiGLU)
على مهمة توليد نص بسيطة (character-level على نص عربي قصير).

المطلوب إكماله:
1. بناء TransformerBlock كلاسيكي وآخر حديث (استخدم المكوّنات من هذا الفصل).
2. بناء GPTModel يستخدم أيًّا من الكتلتين حسب إعداد config.
3. تدريب كلا النموذجين لنفس عدد الخطوات على نفس البيانات بنفس seed.
4. رسم منحنى الخسارة (Loss Curve) لكليهما على نفس الرسم البياني.
5. قياس الزمن الفعلي (wall-clock time) لكل 100 خطوة تدريب لكل نموذج.
6. الإجابة كتابيًا: أي نموذج تقارب أسرع؟ أيهما أسرع زمنيًا لكل خطوة؟
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MiniGPTConfig:
    vocab_size: int
    dim: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_seq_len: int = 64
    use_modern_arch: bool = True  # True = RMSNorm+SwiGLU, False = LayerNorm+GELU


class MiniGPT(nn.Module):
    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        # TODO 1: بناء طبقة تضمين الرموز (token embedding)
        # TODO 2: بناء طبقة تضمين موضعي (يمكن استخدام تضمين مُتعلَّم بسيط هنا)
        # TODO 3: بناء قائمة من TransformerBlock حسب config.use_modern_arch
        #         (استخدم ModernTransformerBlock أو كتلة كلاسيكية مكافئة)
        # TODO 4: بناء طبقة تطبيع نهائية (RMSNorm أو LayerNorm حسب الإعداد)
        # TODO 5: بناء رأس الإخراج (Linear إلى vocab_size)
        raise NotImplementedError("أكمل التصميم هنا")

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # TODO: نفّذ التمرير الأمامي الكامل، أعد logits بشكل (B, T, vocab_size)
        raise NotImplementedError("أكمل التمرير الأمامي هنا")


def train_and_compare() -> None:
    # TODO: حمّل نصًا عربيًا قصيرًا (مثلًا فقرة من كتاب عام)
    # TODO: ابنِ character-level tokenizer بسيطًا
    # TODO: درّب نموذجين (modern=True/False) بنفس البيانات ونفس seed
    # TODO: سجّل loss كل خطوة والزمن الكلي لكل 100 خطوة
    # TODO: ارسم بـ matplotlib منحنيين على نفس الرسم البياني للمقارنة
    pass


if __name__ == "__main__":
    train_and_compare()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-18-chapter-18.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الثامن عشر: تقنيات معمارية حديثة — RMSNorm، SwiGLU، Multi-Token Prediction، وYaRN/NTK-aware Scaling
Large Language Model Engineering — RMSNorm

Book slug : llm-engineering-69-chapters
Chapter   : ch-18-chapter-18 (lesson 18)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_18.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 18: الفصل الثامن عشر: تقنيات معمارية حديثة — RMSNorm، SwiGLU، Multi-Token Prediction، وYaRN/NTK-aware Scaling."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_modern_blocks() -> None:
    values = [1.0, -2.0, 0.5, 3.0]
    rms = math.sqrt(sum((x * x for x in values)) / len(values) + 1e-06)
    normalized = [x / rms for x in values]
    silu = lambda x: x / (1 + math.exp(-x))
    swiglu = [silu(x) * math.tanh(0.8 * x) for x in normalized]
    print('RMSNorm:', [round(x, 3) for x in normalized])
    print('SwiGLU values:', [round(x, 3) for x in swiglu])

if __name__ == "__main__":
    print("Chapter 18 lab: RMSNorm and SwiGLU")
    run_modern_blocks()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-04-modern-architectures/chapter-18.md
# ────────────────────────────────────────────────────────────────────────
"""
chapter18_modern_architecture.py
تطبيق كامل لتقنيات معمارية حديثة: RMSNorm، SwiGLU، Multi-Token Prediction، وYaRN.
Python 3.13+, PyTorch 2.x
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


# =========================================================================
# 1) RMSNorm
# =========================================================================
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    على عكس LayerNorm: لا يطرح المتوسط، ولا يملك معامل انحياز (bias).
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # معامل القياس γ فقط

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        # نحسب RMS على آخر محور (بُعد التضمين)
        # نستخدم float32 داخليًا للاستقرار العددي حتى لو كان x بدقة أقل (FP16/BF16)
        in_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_normed = x_f32 / rms
        return (self.weight * x_normed.to(in_dtype))


def compare_layernorm_vs_rmsnorm() -> None:
    """مقارنة عملية بين LayerNorm وRMSNorm من حيث السلوك العددي."""
    dim = 512
    x = torch.randn(4, 10, dim) * 3.0 + 1.5  # بيانات بها إزاحة (bias) ملحوظة

    ln = nn.LayerNorm(dim)
    rn = RMSNorm(dim)

    y_ln = ln(x)
    y_rn = rn(x)

    print("=== مقارنة LayerNorm و RMSNorm ===")
    print(f"متوسط مُدخل x (يجب أن يكون ~1.5): {x.mean().item():.4f}")
    print(f"متوسط مخرج LayerNorm (يجب أن يكون ~0): {y_ln.mean().item():.4f}")
    print(f"متوسط مخرج RMSNorm (لن يكون صفرًا بالضرورة): {y_rn.mean().item():.4f}")
    print(f"عدد معاملات LayerNorm: {sum(p.numel() for p in ln.parameters())}")
    print(f"عدد معاملات RMSNorm:   {sum(p.numel() for p in rn.parameters())}")


# =========================================================================
# 2) SwiGLU FFN
# =========================================================================
class SwiGLUFeedForward(nn.Module):
    """
    شبكة تغذية أمامية بآلية بوابة SwiGLU (Shazeer, 2020).

    FFN_SwiGLU(x) = (SiLU(xW1) ⊗ xW3) W2

    يُعاد ضبط d_ff تلقائيًا إلى ~2/3 من القيمة الكلاسيكية
    للحفاظ على نفس ميزانية المعاملات التقريبية، مع التقريب
    لأقرب مضاعف multiple_of لتحسين كفاءة الـ GPU.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        multiple_of: int = 256,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim               # القيمة الكلاسيكية
            hidden_dim = int(2 * hidden_dim / 3)  # إعادة الموازنة لـ SwiGLU
            # التقريب لأعلى مضاعف لـ multiple_of
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)   # مسار البوابة (gate)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)   # المسار الخطي (value)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)   # إسقاط العودة

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(x))   # SiLU(xW1)
        value = self.w3(x)          # xW3
        return self.w2(gate * value)  # (SiLU(xW1) ⊗ xW3) W2


class ClassicFeedForward(nn.Module):
    """FFN كلاسيكية بـ GELU للمقارنة."""

    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


def compare_ffn_parameter_budgets(dim: int = 512) -> None:
    classic = ClassicFeedForward(dim)
    swiglu = SwiGLUFeedForward(dim)

    n_classic = sum(p.numel() for p in classic.parameters())
    n_swiglu = sum(p.numel() for p in swiglu.parameters())

    print("\n=== مقارنة ميزانية المعاملات: FFN كلاسيكية مقابل SwiGLU ===")
    print(f"FFN كلاسيكية (GELU): {n_classic:,} معامل")
    print(f"FFN حديثة   (SwiGLU): {n_swiglu:,} معامل")
    print(f"نسبة الزيادة: {(n_swiglu / n_classic - 1) * 100:.2f}%")


# =========================================================================
# 3) كتلة Transformer حديثة كاملة (Pre-Norm + RoPE + RMSNorm + SwiGLU)
# =========================================================================
class ModernTransformerBlock(nn.Module):
    """
    كتلة Transformer بمواصفات حديثة (على غرار LLaMA):
    - Pre-Norm باستخدام RMSNorm
    - Self-Attention (بدون تفاصيل RoPE هنا لتبسيط المثال، انظر الفصل 15)
    - SwiGLU FFN
    """

    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLUFeedForward(dim)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-Norm + Residual حول الانتباه
        normed = self.attn_norm(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out

        # Pre-Norm + Residual حول FFN
        normed2 = self.ffn_norm(x)
        x = x + self.ffn(normed2)
        return x


if __name__ == "__main__":
    compare_layernorm_vs_rmsnorm()
    compare_ffn_parameter_budgets(dim=512)

    print("\n=== اختبار كتلة Transformer حديثة كاملة ===")
    block = ModernTransformerBlock(dim=512, n_heads=8)
    dummy_input = torch.randn(2, 16, 512)
    seq_len = dummy_input.size(1)
    causal = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    output = block(dummy_input, causal_mask=causal)
    print(f"شكل المُدخل:  {tuple(dummy_input.shape)}")
    print(f"شكل المُخرج: {tuple(output.shape)}")
    total_params = sum(p.numel() for p in block.parameters())
    print(f"إجمالي معاملات الكتلة: {total_params:,}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-04-modern-architectures/chapter-18.md
# ────────────────────────────────────────────────────────────────────────
"""
yarn_ntk_scaling.py
تطبيق مبسّط لمفهوم NTK-aware scaling و YaRN لتمديد سياق RoPE.
"""
from __future__ import annotations

import torch


def compute_rope_base_frequencies(
    dim: int, base: float = 10000.0
) -> torch.Tensor:
    """حساب الترددات الأصلية لـ RoPE: θ_i = base^(-2i/dim)."""
    i = torch.arange(0, dim, 2, dtype=torch.float32)
    return base ** (-i / dim)


def ntk_aware_base(
    original_base: float, scale_factor: float, dim: int
) -> float:
    """
    حساب القاعدة الجديدة (base') وفق NTK-aware interpolation.

    base' = base * scale_factor^(dim / (dim - 2))
    """
    exponent = dim / (dim - 2)
    return original_base * (scale_factor ** exponent)


def yarn_ramp_mask(
    dim: int,
    original_max_pos: int,
    extended_max_pos: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> torch.Tensor:
    """
    حساب دالة Ramp لـ YaRN: تُحدّد لكل بُعد (تردد) مدى تطبيق
    التدخل التفاعلي (0 = بدون تدخل/extrapolation طبيعي،
    1 = تدخل كامل/interpolation خطي).

    التبسيط هنا تعليمي، التفاصيل الكاملة في ورقة YaRN الأصلية.
    """
    scale = extended_max_pos / original_max_pos
    dims = torch.arange(0, dim, 2, dtype=torch.float32)

    # عدد "الدورات" الكاملة لكل تردد ضمن طول السياق الأصلي
    # الترددات ذات دورات قليلة جدًا (منخفضة التردد) تحتاج تدخلًا كاملًا
    # الترددات ذات دورات كثيرة (عالية التردد) لا تحتاج تدخلًا يُذكر
    low = dim * torch.log(torch.tensor(original_max_pos / (beta_fast * 2 * torch.pi))) / (
        2 * torch.log(torch.tensor(10000.0))
    )
    high = dim * torch.log(torch.tensor(original_max_pos / (beta_slow * 2 * torch.pi))) / (
        2 * torch.log(torch.tensor(10000.0))
    )
    low, high = max(low.item(), 0), min(high.item(), dim - 1)

    ramp = torch.clamp((dims - low) / max(high - low, 1e-3), 0, 1)
    return ramp  # قيمة بين 0 و 1 لكل بُعد تردد


def yarn_attention_temperature(scale_factor: float) -> float:
    """معامل تصحيح درجة حرارة الانتباه في YaRN: t = 0.1*ln(s) + 1."""
    return 0.1 * math.log(scale_factor) + 1.0


if __name__ == "__main__":
    import math

    dim = 128
    original_len = 4096
    target_len = 32768
    scale = target_len / original_len

    base_freqs = compute_rope_base_frequencies(dim)
    new_base = ntk_aware_base(10000.0, scale, dim)
    new_freqs = compute_rope_base_frequencies(dim, base=new_base)
    ramp = yarn_ramp_mask(dim, original_len, target_len)
    temp = yarn_attention_temperature(scale)

    print(f"نسبة التمديد المطلوبة (s): {scale:.2f}x")
    print(f"القاعدة الأصلية: 10000.0  |  القاعدة الجديدة NTK-aware: {new_base:.1f}")
    print(f"تردد أول بُعد (الأعلى) — أصلي: {base_freqs[0]:.6f}  جديد: {new_freqs[0]:.6f}")
    print(f"تردد آخر بُعد (الأدنى) — أصلي: {base_freqs[-1]:.8f}  جديد: {new_freqs[-1]:.8f}")
    print(f"معامل تصحيح درجة الحرارة (attention temperature): {temp:.4f}")
    print(f"\nعيّنة من دالة Ramp (أول 5 وآخر 5 قيم):")
    print(f"  بداية (ترددات عالية، تدخل ضئيل): {ramp[:5].tolist()}")
    print(f"  نهاية (ترددات منخفضة، تدخل كامل): {ramp[-5:].tolist()}")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-04-modern-architectures/chapter-18.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: eps الافتراضي قد يكون صغيرًا جدًا لـ FP16
norm = RMSNorm(dim=4096, eps=1e-8)  # قد يُسبب تقسيمًا على رقم قريب من صفر في FP16

# ✅ صحيح: استخدم eps أكبر قليلًا (1e-5 أو 1e-6) وتأكد من الحساب بـ float32
norm = RMSNorm(dim=4096, eps=1e-6)

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-04-modern-architectures/chapter-18.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع: عكس أي من المسارين يُمرَّر عبر التفعيل
def wrong_swiglu(x, w1, w2, w3):
    gate = w3(x)              # الخطأ: هذا يجب أن يكون w1
    value = F.silu(w1(x))     # الخطأ: SiLU يجب أن تُطبَّق على w1، لا w3
    return w2(gate * value)

# ✅ صحيح: SiLU تُطبَّق دائمًا على w1 (البوابة)، وw3 يبقى خطيًا بحتًا (القيمة)
def correct_swiglu(x, w1, w2, w3):
    gate = F.silu(w1(x))
    value = w3(x)
    return w2(gate * value)

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-04-modern-architectures/chapter-18.md
# ────────────────────────────────────────────────────────────────────────
# ❌ تبسيط ساذج يُضحّي بالدقة المحلية دون داعٍ
scaled_positions = positions * (original_len / target_len)  # PI خطي بسيط

# ✅ استخدام مكتبة جاهزة تُطبّق YaRN الكامل بكل تفاصيله
# (مثلًا عبر إعداد rope_scaling في Hugging Face، أو تطبيق كامل الخوارزمية)
config.rope_scaling = {"type": "yarn", "factor": 8.0, "original_max_position_embeddings": 4096}

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-04-modern-architectures/chapter-18.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: كل الرؤوس بنفس الأهمية يُشتت تركيز النموذج عن الهدف الأساسي (t+1)
total_loss = sum(losses)  # λ_k = 1 للجميع

# ✅ صحيح: الرأس الأقرب (t+1) يجب أن يحمل الوزن الأكبر دائمًا
weights = [1.0, 0.5, 0.3, 0.2]  # تتناقص مع ازدياد k
total_loss = sum(w * l for w, l in zip(weights, losses))

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-04-modern-architectures/chapter-18.md
# ────────────────────────────────────────────────────────────────────────
"""
mini_project_modern_gpt.py
مشروع مصغّر: مقارنة GPT كلاسيكي (LayerNorm+GELU) مقابل GPT حديث (RMSNorm+SwiGLU)
على مهمة توليد نص بسيطة (character-level على نص عربي قصير).

المطلوب إكماله:
1. بناء TransformerBlock كلاسيكي وآخر حديث (استخدم المكوّنات من هذا الفصل).
2. بناء GPTModel يستخدم أيًّا من الكتلتين حسب إعداد config.
3. تدريب كلا النموذجين لنفس عدد الخطوات على نفس البيانات بنفس seed.
4. رسم منحنى الخسارة (Loss Curve) لكليهما على نفس الرسم البياني.
5. قياس الزمن الفعلي (wall-clock time) لكل 100 خطوة تدريب لكل نموذج.
6. الإجابة كتابيًا: أي نموذج تقارب أسرع؟ أيهما أسرع زمنيًا لكل خطوة؟
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MiniGPTConfig:
    vocab_size: int
    dim: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_seq_len: int = 64
    use_modern_arch: bool = True  # True = RMSNorm+SwiGLU, False = LayerNorm+GELU


class MiniGPT(nn.Module):
    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        # TODO 1: بناء طبقة تضمين الرموز (token embedding)
        # TODO 2: بناء طبقة تضمين موضعي (يمكن استخدام تضمين مُتعلَّم بسيط هنا)
        # TODO 3: بناء قائمة من TransformerBlock حسب config.use_modern_arch
        #         (استخدم ModernTransformerBlock أو كتلة كلاسيكية مكافئة)
        # TODO 4: بناء طبقة تطبيع نهائية (RMSNorm أو LayerNorm حسب الإعداد)
        # TODO 5: بناء رأس الإخراج (Linear إلى vocab_size)
        raise NotImplementedError("أكمل التصميم هنا")

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # TODO: نفّذ التمرير الأمامي الكامل، أعد logits بشكل (B, T, vocab_size)
        raise NotImplementedError("أكمل التمرير الأمامي هنا")


def train_and_compare() -> None:
    # TODO: حمّل نصًا عربيًا قصيرًا (مثلًا فقرة من كتاب عام)
    # TODO: ابنِ character-level tokenizer بسيطًا
    # TODO: درّب نموذجين (modern=True/False) بنفس البيانات ونفس seed
    # TODO: سجّل loss كل خطوة والزمن الكلي لكل 100 خطوة
    # TODO: ارسم بـ matplotlib منحنيين على نفس الرسم البياني للمقارنة
    pass


if __name__ == "__main__":
    train_and_compare()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-18-chapter-18.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 18: الفصل الثامن عشر: تقنيات معمارية حديثة — RMSNorm، SwiGLU، Multi-Token Prediction، وYaRN/NTK-aware Scaling."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_modern_blocks() -> None:
    values = [1.0, -2.0, 0.5, 3.0]
    rms = math.sqrt(sum((x * x for x in values)) / len(values) + 1e-06)
    normalized = [x / rms for x in values]
    silu = lambda x: x / (1 + math.exp(-x))
    swiglu = [silu(x) * math.tanh(0.8 * x) for x in normalized]
    print('RMSNorm:', [round(x, 3) for x in normalized])
    print('SwiGLU values:', [round(x, 3) for x in swiglu])

if __name__ == "__main__":
    print("Chapter 18 lab: RMSNorm and SwiGLU")
    run_modern_blocks()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-18-chapter-18.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
chapter18_modern_architecture.py
تطبيق كامل لتقنيات معمارية حديثة: RMSNorm، SwiGLU، Multi-Token Prediction، وYaRN.
Python 3.13+, PyTorch 2.x
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


# =========================================================================
# 1) RMSNorm
# =========================================================================
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    على عكس LayerNorm: لا يطرح المتوسط، ولا يملك معامل انحياز (bias).
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # معامل القياس γ فقط

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        # نحسب RMS على آخر محور (بُعد التضمين)
        # نستخدم float32 داخليًا للاستقرار العددي حتى لو كان x بدقة أقل (FP16/BF16)
        in_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_normed = x_f32 / rms
        return (self.weight * x_normed.to(in_dtype))


def compare_layernorm_vs_rmsnorm() -> None:
    """مقارنة عملية بين LayerNorm وRMSNorm من حيث السلوك العددي."""
    dim = 512
    x = torch.randn(4, 10, dim) * 3.0 + 1.5  # بيانات بها إزاحة (bias) ملحوظة

    ln = nn.LayerNorm(dim)
    rn = RMSNorm(dim)

    y_ln = ln(x)
    y_rn = rn(x)

    print("=== مقارنة LayerNorm و RMSNorm ===")
    print(f"متوسط مُدخل x (يجب أن يكون ~1.5): {x.mean().item():.4f}")
    print(f"متوسط مخرج LayerNorm (يجب أن يكون ~0): {y_ln.mean().item():.4f}")
    print(f"متوسط مخرج RMSNorm (لن يكون صفرًا بالضرورة): {y_rn.mean().item():.4f}")
    print(f"عدد معاملات LayerNorm: {sum(p.numel() for p in ln.parameters())}")
    print(f"عدد معاملات RMSNorm:   {sum(p.numel() for p in rn.parameters())}")


# =========================================================================
# 2) SwiGLU FFN
# =========================================================================
class SwiGLUFeedForward(nn.Module):
    """
    شبكة تغذية أمامية بآلية بوابة SwiGLU (Shazeer, 2020).

    FFN_SwiGLU(x) = (SiLU(xW1) ⊗ xW3) W2

    يُعاد ضبط d_ff تلقائيًا إلى ~2/3 من القيمة الكلاسيكية
    للحفاظ على نفس ميزانية المعاملات التقريبية، مع التقريب
    لأقرب مضاعف multiple_of لتحسين كفاءة الـ GPU.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        multiple_of: int = 256,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim               # القيمة الكلاسيكية
            hidden_dim = int(2 * hidden_dim / 3)  # إعادة الموازنة لـ SwiGLU
            # التقريب لأعلى مضاعف لـ multiple_of
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)   # مسار البوابة (gate)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)   # المسار الخطي (value)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)   # إسقاط العودة

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(x))   # SiLU(xW1)
        value = self.w3(x)          # xW3
        return self.w2(gate * value)  # (SiLU(xW1) ⊗ xW3) W2


class ClassicFeedForward(nn.Module):
    """FFN كلاسيكية بـ GELU للمقارنة."""

    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


def compare_ffn_parameter_budgets(dim: int = 512) -> None:
    classic = ClassicFeedForward(dim)
    swiglu = SwiGLUFeedForward(dim)

    n_classic = sum(p.numel() for p in classic.parameters())
    n_swiglu = sum(p.numel() for p in swiglu.parameters())

    print("\n=== مقارنة ميزانية المعاملات: FFN كلاسيكية مقابل SwiGLU ===")
    print(f"FFN كلاسيكية (GELU): {n_classic:,} معامل")
    print(f"FFN حديثة   (SwiGLU): {n_swiglu:,} معامل")
    print(f"نسبة الزيادة: {(n_swiglu / n_classic - 1) * 100:.2f}%")


# =========================================================================
# 3) كتلة Transformer حديثة كاملة (Pre-Norm + RoPE + RMSNorm + SwiGLU)
# =========================================================================
class ModernTransformerBlock(nn.Module):
    """
    كتلة Transformer بمواصفات حديثة (على غرار LLaMA):
    - Pre-Norm باستخدام RMSNorm
    - Self-Attention (بدون تفاصيل RoPE هنا لتبسيط المثال، انظر الفصل 15)
    - SwiGLU FFN
    """

    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLUFeedForward(dim)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-Norm + Residual حول الانتباه
        normed = self.attn_norm(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out

        # Pre-Norm + Residual حول FFN
        normed2 = self.ffn_norm(x)
        x = x + self.ffn(normed2)
        return x


if __name__ == "__main__":
    compare_layernorm_vs_rmsnorm()
    compare_ffn_parameter_budgets(dim=512)

    print("\n=== اختبار كتلة Transformer حديثة كاملة ===")
    block = ModernTransformerBlock(dim=512, n_heads=8)
    dummy_input = torch.randn(2, 16, 512)
    seq_len = dummy_input.size(1)
    causal = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    output = block(dummy_input, causal_mask=causal)
    print(f"شكل المُدخل:  {tuple(dummy_input.shape)}")
    print(f"شكل المُخرج: {tuple(output.shape)}")
    total_params = sum(p.numel() for p in block.parameters())
    print(f"إجمالي معاملات الكتلة: {total_params:,}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-18-chapter-18.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
yarn_ntk_scaling.py
تطبيق مبسّط لمفهوم NTK-aware scaling و YaRN لتمديد سياق RoPE.
"""
from __future__ import annotations

import torch


def compute_rope_base_frequencies(
    dim: int, base: float = 10000.0
) -> torch.Tensor:
    """حساب الترددات الأصلية لـ RoPE: θ_i = base^(-2i/dim)."""
    i = torch.arange(0, dim, 2, dtype=torch.float32)
    return base ** (-i / dim)


def ntk_aware_base(
    original_base: float, scale_factor: float, dim: int
) -> float:
    """
    حساب القاعدة الجديدة (base') وفق NTK-aware interpolation.

    base' = base * scale_factor^(dim / (dim - 2))
    """
    exponent = dim / (dim - 2)
    return original_base * (scale_factor ** exponent)


def yarn_ramp_mask(
    dim: int,
    original_max_pos: int,
    extended_max_pos: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> torch.Tensor:
    """
    حساب دالة Ramp لـ YaRN: تُحدّد لكل بُعد (تردد) مدى تطبيق
    التدخل التفاعلي (0 = بدون تدخل/extrapolation طبيعي،
    1 = تدخل كامل/interpolation خطي).

    التبسيط هنا تعليمي، التفاصيل الكاملة في ورقة YaRN الأصلية.
    """
    scale = extended_max_pos / original_max_pos
    dims = torch.arange(0, dim, 2, dtype=torch.float32)

    # عدد "الدورات" الكاملة لكل تردد ضمن طول السياق الأصلي
    # الترددات ذات دورات قليلة جدًا (منخفضة التردد) تحتاج تدخلًا كاملًا
    # الترددات ذات دورات كثيرة (عالية التردد) لا تحتاج تدخلًا يُذكر
    low = dim * torch.log(torch.tensor(original_max_pos / (beta_fast * 2 * torch.pi))) / (
        2 * torch.log(torch.tensor(10000.0))
    )
    high = dim * torch.log(torch.tensor(original_max_pos / (beta_slow * 2 * torch.pi))) / (
        2 * torch.log(torch.tensor(10000.0))
    )
    low, high = max(low.item(), 0), min(high.item(), dim - 1)

    ramp = torch.clamp((dims - low) / max(high - low, 1e-3), 0, 1)
    return ramp  # قيمة بين 0 و 1 لكل بُعد تردد


def yarn_attention_temperature(scale_factor: float) -> float:
    """معامل تصحيح درجة حرارة الانتباه في YaRN: t = 0.1*ln(s) + 1."""
    return 0.1 * math.log(scale_factor) + 1.0


if __name__ == "__main__":
    import math

    dim = 128
    original_len = 4096
    target_len = 32768
    scale = target_len / original_len

    base_freqs = compute_rope_base_frequencies(dim)
    new_base = ntk_aware_base(10000.0, scale, dim)
    new_freqs = compute_rope_base_frequencies(dim, base=new_base)
    ramp = yarn_ramp_mask(dim, original_len, target_len)
    temp = yarn_attention_temperature(scale)

    print(f"نسبة التمديد المطلوبة (s): {scale:.2f}x")
    print(f"القاعدة الأصلية: 10000.0  |  القاعدة الجديدة NTK-aware: {new_base:.1f}")
    print(f"تردد أول بُعد (الأعلى) — أصلي: {base_freqs[0]:.6f}  جديد: {new_freqs[0]:.6f}")
    print(f"تردد آخر بُعد (الأدنى) — أصلي: {base_freqs[-1]:.8f}  جديد: {new_freqs[-1]:.8f}")
    print(f"معامل تصحيح درجة الحرارة (attention temperature): {temp:.4f}")
    print(f"\nعيّنة من دالة Ramp (أول 5 وآخر 5 قيم):")
    print(f"  بداية (ترددات عالية، تدخل ضئيل): {ramp[:5].tolist()}")
    print(f"  نهاية (ترددات منخفضة، تدخل كامل): {ramp[-5:].tolist()}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-18-chapter-18.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: eps الافتراضي قد يكون صغيرًا جدًا لـ FP16
norm = RMSNorm(dim=4096, eps=1e-8)  # قد يُسبب تقسيمًا على رقم قريب من صفر في FP16

# ✅ صحيح: استخدم eps أكبر قليلًا (1e-5 أو 1e-6) وتأكد من الحساب بـ float32
norm = RMSNorm(dim=4096, eps=1e-6)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-18-chapter-18.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع: عكس أي من المسارين يُمرَّر عبر التفعيل
def wrong_swiglu(x, w1, w2, w3):
    gate = w3(x)              # الخطأ: هذا يجب أن يكون w1
    value = F.silu(w1(x))     # الخطأ: SiLU يجب أن تُطبَّق على w1، لا w3
    return w2(gate * value)

# ✅ صحيح: SiLU تُطبَّق دائمًا على w1 (البوابة)، وw3 يبقى خطيًا بحتًا (القيمة)
def correct_swiglu(x, w1, w2, w3):
    gate = F.silu(w1(x))
    value = w3(x)
    return w2(gate * value)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-18-chapter-18.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ❌ تبسيط ساذج يُضحّي بالدقة المحلية دون داعٍ
scaled_positions = positions * (original_len / target_len)  # PI خطي بسيط

# ✅ استخدام مكتبة جاهزة تُطبّق YaRN الكامل بكل تفاصيله
# (مثلًا عبر إعداد rope_scaling في Hugging Face، أو تطبيق كامل الخوارزمية)
config.rope_scaling = {"type": "yarn", "factor": 8.0, "original_max_position_embeddings": 4096}

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-18-chapter-18.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: كل الرؤوس بنفس الأهمية يُشتت تركيز النموذج عن الهدف الأساسي (t+1)
total_loss = sum(losses)  # λ_k = 1 للجميع

# ✅ صحيح: الرأس الأقرب (t+1) يجب أن يحمل الوزن الأكبر دائمًا
weights = [1.0, 0.5, 0.3, 0.2]  # تتناقص مع ازدياد k
total_loss = sum(w * l for w, l in zip(weights, losses))

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-18-chapter-18.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
"""
mini_project_modern_gpt.py
مشروع مصغّر: مقارنة GPT كلاسيكي (LayerNorm+GELU) مقابل GPT حديث (RMSNorm+SwiGLU)
على مهمة توليد نص بسيطة (character-level على نص عربي قصير).

المطلوب إكماله:
1. بناء TransformerBlock كلاسيكي وآخر حديث (استخدم المكوّنات من هذا الفصل).
2. بناء GPTModel يستخدم أيًّا من الكتلتين حسب إعداد config.
3. تدريب كلا النموذجين لنفس عدد الخطوات على نفس البيانات بنفس seed.
4. رسم منحنى الخسارة (Loss Curve) لكليهما على نفس الرسم البياني.
5. قياس الزمن الفعلي (wall-clock time) لكل 100 خطوة تدريب لكل نموذج.
6. الإجابة كتابيًا: أي نموذج تقارب أسرع؟ أيهما أسرع زمنيًا لكل خطوة؟
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MiniGPTConfig:
    vocab_size: int
    dim: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_seq_len: int = 64
    use_modern_arch: bool = True  # True = RMSNorm+SwiGLU, False = LayerNorm+GELU


class MiniGPT(nn.Module):
    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        # TODO 1: بناء طبقة تضمين الرموز (token embedding)
        # TODO 2: بناء طبقة تضمين موضعي (يمكن استخدام تضمين مُتعلَّم بسيط هنا)
        # TODO 3: بناء قائمة من TransformerBlock حسب config.use_modern_arch
        #         (استخدم ModernTransformerBlock أو كتلة كلاسيكية مكافئة)
        # TODO 4: بناء طبقة تطبيع نهائية (RMSNorm أو LayerNorm حسب الإعداد)
        # TODO 5: بناء رأس الإخراج (Linear إلى vocab_size)
        raise NotImplementedError("أكمل التصميم هنا")

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # TODO: نفّذ التمرير الأمامي الكامل، أعد logits بشكل (B, T, vocab_size)
        raise NotImplementedError("أكمل التمرير الأمامي هنا")


def train_and_compare() -> None:
    # TODO: حمّل نصًا عربيًا قصيرًا (مثلًا فقرة من كتاب عام)
    # TODO: ابنِ character-level tokenizer بسيطًا
    # TODO: درّب نموذجين (modern=True/False) بنفس البيانات ونفس seed
    # TODO: سجّل loss كل خطوة والزمن الكلي لكل 100 خطوة
    # TODO: ارسم بـ matplotlib منحنيين على نفس الرسم البياني للمقارنة
    pass


if __name__ == "__main__":
    train_and_compare()
