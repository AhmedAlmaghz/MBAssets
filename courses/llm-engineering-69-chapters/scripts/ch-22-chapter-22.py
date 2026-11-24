#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الثاني والعشرون: تصميم المعمارية وScaling Laws — فن اختيار الحجم الأمثل لنموذجك
Large Language Model Engineering — Scaling

Book slug : llm-engineering-69-chapters
Chapter   : ch-22-chapter-22 (lesson 22)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-22-chapter-22.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الثاني والعشرون: تصميم المعمارية وScaling Laws — فن اختيار الحجم الأمثل لنموذجك
Large Language Model Engineering — Scaling

Book slug : llm-engineering-69-chapters
Chapter   : ch-22-chapter-22 (lesson 22)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-22-chapter-22.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الثاني والعشرون: تصميم المعمارية وScaling Laws — فن اختيار الحجم الأمثل لنموذجك
Large Language Model Engineering — Scaling

Book slug : llm-engineering-69-chapters
Chapter   : ch-22-chapter-22 (lesson 22)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_22.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 22: الفصل الثاني والعشرون: تصميم المعمارية وScaling Laws — فن اختيار الحجم الأمثل لنموذجك."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_scaling() -> None:
    parameters, tokens, coefficient, exponent = (1000000000.0, 20000000000.0, 1.0, 0.34)
    compute = 6 * parameters * tokens
    proxy = coefficient * (parameters ** (-exponent) + tokens ** (-exponent))
    print('training FLOPs proxy:', f'{compute:.3e}')
    print('loss scaling proxy:', f'{proxy:.5f}')
    print('double parameters only compute multiplier:', 2.0)

if __name__ == "__main__":
    print("Chapter 22 lab: Estimate compute with a scaling-law proxy")
    run_scaling()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-05-build-an-llm/chapter-22.md
# ────────────────────────────────────────────────────────────────────────
"""
chapter22_scaling_laws_calculator.py
حاسبة Scaling Laws كاملة: من ميزانية حوسبة إلى حجم نموذج،
كمية بيانات، ومعمارية مقترحة، بأسلوب Chinchilla-Optimal.

Python 3.13+
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# =========================================================================
# 1) معاملات قانون Chinchilla المُقدَّرة تجريبيًا (Hoffmann et al. 2022)
# =========================================================================
@dataclass(frozen=True)
class ChinchillaConstants:
    E: float = 1.69     # الإنتروبيا المثالية (أدنى حد نظري للخسارة)
    A: float = 406.4    # ثابت حد حجم النموذج
    B: float = 410.7    # ثابت حد حجم البيانات
    alpha: float = 0.34
    beta: float = 0.28


def predicted_loss(n_params: float, n_tokens: float, c: ChinchillaConstants) -> float:
    """
    التنبؤ بدالة الخسارة النهائية وفق صيغة Chinchilla:
    L(N, D) = E + A/N^alpha + B/D^beta
    """
    return c.E + c.A / (n_params ** c.alpha) + c.B / (n_tokens ** c.beta)


def compute_optimal_allocation(
    compute_budget_flops: float, tokens_per_param: float = 20.0
) -> tuple[float, float]:
    """
    حساب N* و D* الأمثل لميزانية حوسبة معيّنة (بالـ FLOPs)،
    باستخدام القاعدة التقريبية D ≈ tokens_per_param × N
    مُشتقّة من C ≈ 6ND.
    """
    # C = 6 * N * (tokens_per_param * N) = 6 * tokens_per_param * N^2
    n_optimal = math.sqrt(compute_budget_flops / (6 * tokens_per_param))
    d_optimal = tokens_per_param * n_optimal
    return n_optimal, d_optimal


def flops_from_gpu_hours(
    n_gpus: int, hours: float, flops_per_gpu_per_sec: float, mfu: float = 0.4
) -> float:
    """
    تحويل عدد ساعات GPU إلى ميزانية حوسبة فعلية بالـ FLOPs،
    مع الأخذ في الحسبان كفاءة استغلال الأجهزة (MFU - Model FLOPs Utilization).
    """
    total_seconds = n_gpus * hours * 3600
    return total_seconds * flops_per_gpu_per_sec * mfu


def suggest_architecture(n_params: float) -> dict[str, int]:
    """
    اقتراح أبعاد معمارية تقريبية بناءً على قواعد تجريبية شائعة
    (جدول القيم المُستخلص من ملاحظة نماذج ناجحة فعليًا).
    """
    # جدول مرجعي مبني على ملاحظات عملية (مبسّط لأغراض تعليمية)
    reference_points = [
        (125e6, 12, 768, 12),
        (1.3e9, 24, 2048, 16),
        (7e9, 32, 4096, 32),
        (13e9, 40, 5120, 40),
        (70e9, 80, 8192, 64),
    ]
    # اختيار أقرب نقطة مرجعية (في تطبيق حقيقي: استيفاء (interpolation) أدق)
    closest = min(reference_points, key=lambda p: abs(math.log(p[0]) - math.log(n_params)))
    _, n_layers, d_model, n_heads = closest
    return {"n_layers": n_layers, "d_model": d_model, "n_heads": n_heads}


if __name__ == "__main__":
    constants = ChinchillaConstants()

    print("=" * 70)
    print("سيناريو: لديك 512 وحدة A100، لمدة 30 يومًا متواصلة")
    print("=" * 70)

    # A100: ~312 TFLOPS (BF16) نظريًا، نفترض MFU=40% (واقعي جدًا للتدريب الموزّع)
    a100_peak_flops = 312e12
    budget = flops_from_gpu_hours(
        n_gpus=512, hours=30 * 24, flops_per_gpu_per_sec=a100_peak_flops, mfu=0.4
    )
    print(f"ميزانية الحوسبة الكلية: {budget:.3e} FLOPs\n")

    n_star, d_star = compute_optimal_allocation(budget, tokens_per_param=20.0)
    print(f"[Chinchilla-Optimal] حجم النموذج الأمثل N*: {n_star / 1e9:.2f} مليار معامل")
    print(f"[Chinchilla-Optimal] عدد رموز التدريب الأمثل D*: {d_star / 1e9:.1f} مليار رمز")
    print(f"نسبة D*/N*: {d_star / n_star:.1f}  (يجب أن تقارب 20)\n")

    loss_at_optimal = predicted_loss(n_star, d_star, constants)
    print(f"الخسارة المتوقعة عند التركيبة المثلى: {loss_at_optimal:.4f}\n")

    # --- مقارنة: ماذا لو اخترنا نموذجًا أكبر بكثير لكن بنفس الميزانية؟ ---
    print("=" * 70)
    print("تجربة مقارنة: نموذج أكبر 2x مع بيانات أقل (نفس الميزانية تقريبًا)")
    print("=" * 70)
    n_bigger = n_star * 2
    d_smaller = budget / (6 * n_bigger)  # نفس الميزانية C تقريبًا
    loss_bigger = predicted_loss(n_bigger, d_smaller, constants)
    print(f"N أكبر: {n_bigger / 1e9:.2f}B   D أصغر: {d_smaller / 1e9:.1f}B رمز")
    print(f"الخسارة المتوقعة: {loss_bigger:.4f}  "
          f"({'أسوأ' if loss_bigger > loss_at_optimal else 'أفضل'} من المثالية)\n")

    # --- اقتراح معمارية فعلية للنموذج الأمثل ---
    print("=" * 70)
    print("اقتراح أبعاد معمارية تقريبية للنموذج الأمثل")
    print("=" * 70)
    arch = suggest_architecture(n_star)
    for k, v in arch.items():
        print(f"  {k}: {v}")

    # --- محاكاة استراتيجية LLaMA 3: تدريب "فوق-الأمثل" لصالح استدلال أرخص ---
    print("\n" + "=" * 70)
    print("مقارنة إضافية: استراتيجية LLaMA-3-style (نموذج أصغر، بيانات أكثر بكثير)")
    print("=" * 70)
    n_llama_style = n_star / 3           # نموذج أصغر بكثير من Chinchilla-Optimal
    d_llama_style = budget / (6 * n_llama_style)  # نفس ميزانية التدريب تقريبًا
    loss_llama_style = predicted_loss(n_llama_style, d_llama_style, constants)
    print(f"N: {n_llama_style/1e9:.2f}B   D: {d_llama_style/1e9:.1f}B رمز "
          f"(نسبة D/N={d_llama_style/n_llama_style:.0f})")
    print(f"الخسارة المتوقعة: {loss_llama_style:.4f} "
          f"(أعلى قليلًا من المثالية، لكن الاستدلال أرخص بـ 3x لاحقًا)")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-05-build-an-llm/chapter-22.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: تطبيق Chinchilla حرفيًا لمشروع يستهدف نشرًا تجاريًا ضخمًا
n_params = 70e9
d_tokens_naive = 20 * n_params  # 1.4 تريليون رمز - الأمثل "حسابيًا" فقط

# ✅ صحيح: فكّر في عدد مرات الاستدلال المتوقعة طوال عمر المنتج
# إذا كان الاستدلال سيحدث مليارات المرات، فكّر بنموذج أصغر وبيانات أكثر
# (مقايضة قابلة للحساب: وفر في الاستدلال طويل المدى مقابل خسارة طفيفة في Loss)

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-05-build-an-llm/chapter-22.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: تطبيق صيغة Chinchilla القياسية على نموذج MoE دون تمييز
# N الإجمالي في MoE لا يُكافئ N في نموذج Dense من حيث العلاقة بـ Loss
# (لأن المعاملات النشطة فقط هي التي تُحدد الحوسبة الفعلية لكل رمز)

# ✅ صحيح: استخدم أبحاث Scaling Laws المخصصة لـ MoE
# (مثل Clark et al. 2022 "Unified Scaling Laws for Routed Language Models")
# التي تُميّز بوضوح بين N الإجمالي و N النشط في المعادلات

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-05-build-an-llm/chapter-22.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: مقارنة "D* المطلوب" بين مشروعك ومشروع آخر دون مراعاة جودة البيانات
# نموذج مُدرَّب على بيانات مُصفّاة بعناية (FineWeb-Edu) قد يحتاج D أقل بكثير
# لتحقيق نفس Loss مقارنة ببيانات خام غير مُصفّاة

# ✅ صحيح: أعد تقدير ثوابت Scaling الخاصة ببياناتك دائمًا، ولا تفترض
# أن أرقام الأوراق البحثية الأخرى تنطبق مباشرة على مجموعة بياناتك

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-05-build-an-llm/chapter-22.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: افتراض أن النموذج ذا الخسارة الأقل (Loss) هو بالضرورة الأفضل
# لمهمة محددة (مثل الدردشة أو البرمجة) دون اختبار فعلي

# ✅ صحيح: استخدم Scaling Laws لتحديد ميزانية التدريب الأولية بذكاء،
# لكن اعتمد دائمًا تقييمًا شاملًا على معايير حقيقية (الفصل 46) للقرار النهائي

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-05-build-an-llm/chapter-22.md
# ────────────────────────────────────────────────────────────────────────
"""
mini_project_isoflop_experiment.py
تجربة Scaling Laws مصغّرة حقيقية وقابلة للتشغيل على جهاز واحد:
درّب عدة نماذج GPT مصغّرة جدًا بأحجام مختلفة على نفس البيانات،
بميزانيات حوسبة متساوية تقريبًا (IsoFLOP)، وارسم منحنى U لإيجاد
الحجم الأمثل تجريبيًا بنفسك.

المطلوب إكماله:
1. استخدم نموذج GPT المصغّر من الفصل 18/19 (ModernTransformerBlock).
2. اختر 5-6 أحجام نماذج مختلفة (من ~100K إلى ~5M معامل، صغيرة جدًا
   عمدًا لتكون التجربة قابلة للتشغيل على جهاز واحد خلال دقائق).
3. لكل حجم نموذج N، احسب D بحيث يبقى حاصل الضرب 6*N*D ثابتًا تقريبًا
   (نفس ميزانية الحوسبة لكل التجارب - هذا هو جوهر IsoFLOP).
4. درّب كل نموذج على جزء من نص عربي (مثل عيّنة من ويكيبيديا) بالعدد
   المحسوب من الخطوات، وسجّل الخسارة النهائية.
5. ارسم منحنى (N على المحور الأفقي، Loss النهائية على العمودي) -
   يجب أن يُظهر شكل U تقريبي.
6. حدد النموذج صاحب أقل Loss من بين التجارب - هذا هو N* التجريبي
   لميزانية حوسبتك المحددة.
7. قارن النتيجة بالتنبؤ النظري من حاسبة Chinchilla في هذا الفصل.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class IsoFLOPExperiment:
    n_params_target: int
    compute_budget: float  # 6*N*D ثابت لكل التجارب

    @property
    def d_tokens(self) -> int:
        return int(self.compute_budget / (6 * self.n_params_target))


def run_single_experiment(exp: IsoFLOPExperiment) -> float:
    # TODO 1: ابنِ نموذج GPT مصغّر بعدد معاملات قريب من exp.n_params_target
    #         (اضبط n_layers/d_model يدويًا للوصول لعدد المعاملات المطلوب تقريبًا)
    # TODO 2: درّب النموذج على exp.d_tokens رمزًا بالضبط من نص عربي
    # TODO 3: أعد الخسارة النهائية (على عيّنة تحقق منفصلة held-out)
    raise NotImplementedError


def main() -> None:
    compute_budget = 1e15  # ميزانية ثابتة صغيرة لكل التجارب (قابلة للتشغيل محليًا)
    param_sizes = [100_000, 300_000, 1_000_000, 3_000_000, 5_000_000]

    results = []
    for n in param_sizes:
        exp = IsoFLOPExperiment(n_params_target=n, compute_budget=compute_budget)
        loss = run_single_experiment(exp)
        results.append((n, exp.d_tokens, loss))
        print(f"N={n:>10,}  D={exp.d_tokens:>12,}  Loss={loss:.4f}")

    # TODO: ارسم بـ matplotlib منحنى N مقابل Loss، وحدد أدنى نقطة (N* التجريبي)


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-22-chapter-22.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 22: الفصل الثاني والعشرون: تصميم المعمارية وScaling Laws — فن اختيار الحجم الأمثل لنموذجك."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_scaling() -> None:
    parameters, tokens, coefficient, exponent = (1000000000.0, 20000000000.0, 1.0, 0.34)
    compute = 6 * parameters * tokens
    proxy = coefficient * (parameters ** (-exponent) + tokens ** (-exponent))
    print('training FLOPs proxy:', f'{compute:.3e}')
    print('loss scaling proxy:', f'{proxy:.5f}')
    print('double parameters only compute multiplier:', 2.0)

if __name__ == "__main__":
    print("Chapter 22 lab: Estimate compute with a scaling-law proxy")
    run_scaling()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-22-chapter-22.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
chapter22_scaling_laws_calculator.py
حاسبة Scaling Laws كاملة: من ميزانية حوسبة إلى حجم نموذج،
كمية بيانات، ومعمارية مقترحة، بأسلوب Chinchilla-Optimal.

Python 3.13+
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# =========================================================================
# 1) معاملات قانون Chinchilla المُقدَّرة تجريبيًا (Hoffmann et al. 2022)
# =========================================================================
@dataclass(frozen=True)
class ChinchillaConstants:
    E: float = 1.69     # الإنتروبيا المثالية (أدنى حد نظري للخسارة)
    A: float = 406.4    # ثابت حد حجم النموذج
    B: float = 410.7    # ثابت حد حجم البيانات
    alpha: float = 0.34
    beta: float = 0.28


def predicted_loss(n_params: float, n_tokens: float, c: ChinchillaConstants) -> float:
    """
    التنبؤ بدالة الخسارة النهائية وفق صيغة Chinchilla:
    L(N, D) = E + A/N^alpha + B/D^beta
    """
    return c.E + c.A / (n_params ** c.alpha) + c.B / (n_tokens ** c.beta)


def compute_optimal_allocation(
    compute_budget_flops: float, tokens_per_param: float = 20.0
) -> tuple[float, float]:
    """
    حساب N* و D* الأمثل لميزانية حوسبة معيّنة (بالـ FLOPs)،
    باستخدام القاعدة التقريبية D ≈ tokens_per_param × N
    مُشتقّة من C ≈ 6ND.
    """
    # C = 6 * N * (tokens_per_param * N) = 6 * tokens_per_param * N^2
    n_optimal = math.sqrt(compute_budget_flops / (6 * tokens_per_param))
    d_optimal = tokens_per_param * n_optimal
    return n_optimal, d_optimal


def flops_from_gpu_hours(
    n_gpus: int, hours: float, flops_per_gpu_per_sec: float, mfu: float = 0.4
) -> float:
    """
    تحويل عدد ساعات GPU إلى ميزانية حوسبة فعلية بالـ FLOPs،
    مع الأخذ في الحسبان كفاءة استغلال الأجهزة (MFU - Model FLOPs Utilization).
    """
    total_seconds = n_gpus * hours * 3600
    return total_seconds * flops_per_gpu_per_sec * mfu


def suggest_architecture(n_params: float) -> dict[str, int]:
    """
    اقتراح أبعاد معمارية تقريبية بناءً على قواعد تجريبية شائعة
    (جدول القيم المُستخلص من ملاحظة نماذج ناجحة فعليًا).
    """
    # جدول مرجعي مبني على ملاحظات عملية (مبسّط لأغراض تعليمية)
    reference_points = [
        (125e6, 12, 768, 12),
        (1.3e9, 24, 2048, 16),
        (7e9, 32, 4096, 32),
        (13e9, 40, 5120, 40),
        (70e9, 80, 8192, 64),
    ]
    # اختيار أقرب نقطة مرجعية (في تطبيق حقيقي: استيفاء (interpolation) أدق)
    closest = min(reference_points, key=lambda p: abs(math.log(p[0]) - math.log(n_params)))
    _, n_layers, d_model, n_heads = closest
    return {"n_layers": n_layers, "d_model": d_model, "n_heads": n_heads}


if __name__ == "__main__":
    constants = ChinchillaConstants()

    print("=" * 70)
    print("سيناريو: لديك 512 وحدة A100، لمدة 30 يومًا متواصلة")
    print("=" * 70)

    # A100: ~312 TFLOPS (BF16) نظريًا، نفترض MFU=40% (واقعي جدًا للتدريب الموزّع)
    a100_peak_flops = 312e12
    budget = flops_from_gpu_hours(
        n_gpus=512, hours=30 * 24, flops_per_gpu_per_sec=a100_peak_flops, mfu=0.4
    )
    print(f"ميزانية الحوسبة الكلية: {budget:.3e} FLOPs\n")

    n_star, d_star = compute_optimal_allocation(budget, tokens_per_param=20.0)
    print(f"[Chinchilla-Optimal] حجم النموذج الأمثل N*: {n_star / 1e9:.2f} مليار معامل")
    print(f"[Chinchilla-Optimal] عدد رموز التدريب الأمثل D*: {d_star / 1e9:.1f} مليار رمز")
    print(f"نسبة D*/N*: {d_star / n_star:.1f}  (يجب أن تقارب 20)\n")

    loss_at_optimal = predicted_loss(n_star, d_star, constants)
    print(f"الخسارة المتوقعة عند التركيبة المثلى: {loss_at_optimal:.4f}\n")

    # --- مقارنة: ماذا لو اخترنا نموذجًا أكبر بكثير لكن بنفس الميزانية؟ ---
    print("=" * 70)
    print("تجربة مقارنة: نموذج أكبر 2x مع بيانات أقل (نفس الميزانية تقريبًا)")
    print("=" * 70)
    n_bigger = n_star * 2
    d_smaller = budget / (6 * n_bigger)  # نفس الميزانية C تقريبًا
    loss_bigger = predicted_loss(n_bigger, d_smaller, constants)
    print(f"N أكبر: {n_bigger / 1e9:.2f}B   D أصغر: {d_smaller / 1e9:.1f}B رمز")
    print(f"الخسارة المتوقعة: {loss_bigger:.4f}  "
          f"({'أسوأ' if loss_bigger > loss_at_optimal else 'أفضل'} من المثالية)\n")

    # --- اقتراح معمارية فعلية للنموذج الأمثل ---
    print("=" * 70)
    print("اقتراح أبعاد معمارية تقريبية للنموذج الأمثل")
    print("=" * 70)
    arch = suggest_architecture(n_star)
    for k, v in arch.items():
        print(f"  {k}: {v}")

    # --- محاكاة استراتيجية LLaMA 3: تدريب "فوق-الأمثل" لصالح استدلال أرخص ---
    print("\n" + "=" * 70)
    print("مقارنة إضافية: استراتيجية LLaMA-3-style (نموذج أصغر، بيانات أكثر بكثير)")
    print("=" * 70)
    n_llama_style = n_star / 3           # نموذج أصغر بكثير من Chinchilla-Optimal
    d_llama_style = budget / (6 * n_llama_style)  # نفس ميزانية التدريب تقريبًا
    loss_llama_style = predicted_loss(n_llama_style, d_llama_style, constants)
    print(f"N: {n_llama_style/1e9:.2f}B   D: {d_llama_style/1e9:.1f}B رمز "
          f"(نسبة D/N={d_llama_style/n_llama_style:.0f})")
    print(f"الخسارة المتوقعة: {loss_llama_style:.4f} "
          f"(أعلى قليلًا من المثالية، لكن الاستدلال أرخص بـ 3x لاحقًا)")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-22-chapter-22.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: تطبيق Chinchilla حرفيًا لمشروع يستهدف نشرًا تجاريًا ضخمًا
n_params = 70e9
d_tokens_naive = 20 * n_params  # 1.4 تريليون رمز - الأمثل "حسابيًا" فقط

# ✅ صحيح: فكّر في عدد مرات الاستدلال المتوقعة طوال عمر المنتج
# إذا كان الاستدلال سيحدث مليارات المرات، فكّر بنموذج أصغر وبيانات أكثر
# (مقايضة قابلة للحساب: وفر في الاستدلال طويل المدى مقابل خسارة طفيفة في Loss)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-22-chapter-22.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: تطبيق صيغة Chinchilla القياسية على نموذج MoE دون تمييز
# N الإجمالي في MoE لا يُكافئ N في نموذج Dense من حيث العلاقة بـ Loss
# (لأن المعاملات النشطة فقط هي التي تُحدد الحوسبة الفعلية لكل رمز)

# ✅ صحيح: استخدم أبحاث Scaling Laws المخصصة لـ MoE
# (مثل Clark et al. 2022 "Unified Scaling Laws for Routed Language Models")
# التي تُميّز بوضوح بين N الإجمالي و N النشط في المعادلات

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-22-chapter-22.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: مقارنة "D* المطلوب" بين مشروعك ومشروع آخر دون مراعاة جودة البيانات
# نموذج مُدرَّب على بيانات مُصفّاة بعناية (FineWeb-Edu) قد يحتاج D أقل بكثير
# لتحقيق نفس Loss مقارنة ببيانات خام غير مُصفّاة

# ✅ صحيح: أعد تقدير ثوابت Scaling الخاصة ببياناتك دائمًا، ولا تفترض
# أن أرقام الأوراق البحثية الأخرى تنطبق مباشرة على مجموعة بياناتك

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-22-chapter-22.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: افتراض أن النموذج ذا الخسارة الأقل (Loss) هو بالضرورة الأفضل
# لمهمة محددة (مثل الدردشة أو البرمجة) دون اختبار فعلي

# ✅ صحيح: استخدم Scaling Laws لتحديد ميزانية التدريب الأولية بذكاء،
# لكن اعتمد دائمًا تقييمًا شاملًا على معايير حقيقية (الفصل 46) للقرار النهائي

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-22-chapter-22.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
"""
mini_project_isoflop_experiment.py
تجربة Scaling Laws مصغّرة حقيقية وقابلة للتشغيل على جهاز واحد:
درّب عدة نماذج GPT مصغّرة جدًا بأحجام مختلفة على نفس البيانات،
بميزانيات حوسبة متساوية تقريبًا (IsoFLOP)، وارسم منحنى U لإيجاد
الحجم الأمثل تجريبيًا بنفسك.

المطلوب إكماله:
1. استخدم نموذج GPT المصغّر من الفصل 18/19 (ModernTransformerBlock).
2. اختر 5-6 أحجام نماذج مختلفة (من ~100K إلى ~5M معامل، صغيرة جدًا
   عمدًا لتكون التجربة قابلة للتشغيل على جهاز واحد خلال دقائق).
3. لكل حجم نموذج N، احسب D بحيث يبقى حاصل الضرب 6*N*D ثابتًا تقريبًا
   (نفس ميزانية الحوسبة لكل التجارب - هذا هو جوهر IsoFLOP).
4. درّب كل نموذج على جزء من نص عربي (مثل عيّنة من ويكيبيديا) بالعدد
   المحسوب من الخطوات، وسجّل الخسارة النهائية.
5. ارسم منحنى (N على المحور الأفقي، Loss النهائية على العمودي) -
   يجب أن يُظهر شكل U تقريبي.
6. حدد النموذج صاحب أقل Loss من بين التجارب - هذا هو N* التجريبي
   لميزانية حوسبتك المحددة.
7. قارن النتيجة بالتنبؤ النظري من حاسبة Chinchilla في هذا الفصل.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class IsoFLOPExperiment:
    n_params_target: int
    compute_budget: float  # 6*N*D ثابت لكل التجارب

    @property
    def d_tokens(self) -> int:
        return int(self.compute_budget / (6 * self.n_params_target))


def run_single_experiment(exp: IsoFLOPExperiment) -> float:
    # TODO 1: ابنِ نموذج GPT مصغّر بعدد معاملات قريب من exp.n_params_target
    #         (اضبط n_layers/d_model يدويًا للوصول لعدد المعاملات المطلوب تقريبًا)
    # TODO 2: درّب النموذج على exp.d_tokens رمزًا بالضبط من نص عربي
    # TODO 3: أعد الخسارة النهائية (على عيّنة تحقق منفصلة held-out)
    raise NotImplementedError


def main() -> None:
    compute_budget = 1e15  # ميزانية ثابتة صغيرة لكل التجارب (قابلة للتشغيل محليًا)
    param_sizes = [100_000, 300_000, 1_000_000, 3_000_000, 5_000_000]

    results = []
    for n in param_sizes:
        exp = IsoFLOPExperiment(n_params_target=n, compute_budget=compute_budget)
        loss = run_single_experiment(exp)
        results.append((n, exp.d_tokens, loss))
        print(f"N={n:>10,}  D={exp.d_tokens:>12,}  Loss={loss:.4f}")

    # TODO: ارسم بـ matplotlib منحنى N مقابل Loss، وحدد أدنى نقطة (N* التجريبي)


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-22-chapter-22.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الثاني والعشرون: تصميم المعمارية وScaling Laws — فن اختيار الحجم الأمثل لنموذجك
Large Language Model Engineering — Scaling

Book slug : llm-engineering-69-chapters
Chapter   : ch-22-chapter-22 (lesson 22)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_22.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 22: الفصل الثاني والعشرون: تصميم المعمارية وScaling Laws — فن اختيار الحجم الأمثل لنموذجك."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_scaling() -> None:
    parameters, tokens, coefficient, exponent = (1000000000.0, 20000000000.0, 1.0, 0.34)
    compute = 6 * parameters * tokens
    proxy = coefficient * (parameters ** (-exponent) + tokens ** (-exponent))
    print('training FLOPs proxy:', f'{compute:.3e}')
    print('loss scaling proxy:', f'{proxy:.5f}')
    print('double parameters only compute multiplier:', 2.0)

if __name__ == "__main__":
    print("Chapter 22 lab: Estimate compute with a scaling-law proxy")
    run_scaling()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-05-build-an-llm/chapter-22.md
# ────────────────────────────────────────────────────────────────────────
"""
chapter22_scaling_laws_calculator.py
حاسبة Scaling Laws كاملة: من ميزانية حوسبة إلى حجم نموذج،
كمية بيانات، ومعمارية مقترحة، بأسلوب Chinchilla-Optimal.

Python 3.13+
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# =========================================================================
# 1) معاملات قانون Chinchilla المُقدَّرة تجريبيًا (Hoffmann et al. 2022)
# =========================================================================
@dataclass(frozen=True)
class ChinchillaConstants:
    E: float = 1.69     # الإنتروبيا المثالية (أدنى حد نظري للخسارة)
    A: float = 406.4    # ثابت حد حجم النموذج
    B: float = 410.7    # ثابت حد حجم البيانات
    alpha: float = 0.34
    beta: float = 0.28


def predicted_loss(n_params: float, n_tokens: float, c: ChinchillaConstants) -> float:
    """
    التنبؤ بدالة الخسارة النهائية وفق صيغة Chinchilla:
    L(N, D) = E + A/N^alpha + B/D^beta
    """
    return c.E + c.A / (n_params ** c.alpha) + c.B / (n_tokens ** c.beta)


def compute_optimal_allocation(
    compute_budget_flops: float, tokens_per_param: float = 20.0
) -> tuple[float, float]:
    """
    حساب N* و D* الأمثل لميزانية حوسبة معيّنة (بالـ FLOPs)،
    باستخدام القاعدة التقريبية D ≈ tokens_per_param × N
    مُشتقّة من C ≈ 6ND.
    """
    # C = 6 * N * (tokens_per_param * N) = 6 * tokens_per_param * N^2
    n_optimal = math.sqrt(compute_budget_flops / (6 * tokens_per_param))
    d_optimal = tokens_per_param * n_optimal
    return n_optimal, d_optimal


def flops_from_gpu_hours(
    n_gpus: int, hours: float, flops_per_gpu_per_sec: float, mfu: float = 0.4
) -> float:
    """
    تحويل عدد ساعات GPU إلى ميزانية حوسبة فعلية بالـ FLOPs،
    مع الأخذ في الحسبان كفاءة استغلال الأجهزة (MFU - Model FLOPs Utilization).
    """
    total_seconds = n_gpus * hours * 3600
    return total_seconds * flops_per_gpu_per_sec * mfu


def suggest_architecture(n_params: float) -> dict[str, int]:
    """
    اقتراح أبعاد معمارية تقريبية بناءً على قواعد تجريبية شائعة
    (جدول القيم المُستخلص من ملاحظة نماذج ناجحة فعليًا).
    """
    # جدول مرجعي مبني على ملاحظات عملية (مبسّط لأغراض تعليمية)
    reference_points = [
        (125e6, 12, 768, 12),
        (1.3e9, 24, 2048, 16),
        (7e9, 32, 4096, 32),
        (13e9, 40, 5120, 40),
        (70e9, 80, 8192, 64),
    ]
    # اختيار أقرب نقطة مرجعية (في تطبيق حقيقي: استيفاء (interpolation) أدق)
    closest = min(reference_points, key=lambda p: abs(math.log(p[0]) - math.log(n_params)))
    _, n_layers, d_model, n_heads = closest
    return {"n_layers": n_layers, "d_model": d_model, "n_heads": n_heads}


if __name__ == "__main__":
    constants = ChinchillaConstants()

    print("=" * 70)
    print("سيناريو: لديك 512 وحدة A100، لمدة 30 يومًا متواصلة")
    print("=" * 70)

    # A100: ~312 TFLOPS (BF16) نظريًا، نفترض MFU=40% (واقعي جدًا للتدريب الموزّع)
    a100_peak_flops = 312e12
    budget = flops_from_gpu_hours(
        n_gpus=512, hours=30 * 24, flops_per_gpu_per_sec=a100_peak_flops, mfu=0.4
    )
    print(f"ميزانية الحوسبة الكلية: {budget:.3e} FLOPs\n")

    n_star, d_star = compute_optimal_allocation(budget, tokens_per_param=20.0)
    print(f"[Chinchilla-Optimal] حجم النموذج الأمثل N*: {n_star / 1e9:.2f} مليار معامل")
    print(f"[Chinchilla-Optimal] عدد رموز التدريب الأمثل D*: {d_star / 1e9:.1f} مليار رمز")
    print(f"نسبة D*/N*: {d_star / n_star:.1f}  (يجب أن تقارب 20)\n")

    loss_at_optimal = predicted_loss(n_star, d_star, constants)
    print(f"الخسارة المتوقعة عند التركيبة المثلى: {loss_at_optimal:.4f}\n")

    # --- مقارنة: ماذا لو اخترنا نموذجًا أكبر بكثير لكن بنفس الميزانية؟ ---
    print("=" * 70)
    print("تجربة مقارنة: نموذج أكبر 2x مع بيانات أقل (نفس الميزانية تقريبًا)")
    print("=" * 70)
    n_bigger = n_star * 2
    d_smaller = budget / (6 * n_bigger)  # نفس الميزانية C تقريبًا
    loss_bigger = predicted_loss(n_bigger, d_smaller, constants)
    print(f"N أكبر: {n_bigger / 1e9:.2f}B   D أصغر: {d_smaller / 1e9:.1f}B رمز")
    print(f"الخسارة المتوقعة: {loss_bigger:.4f}  "
          f"({'أسوأ' if loss_bigger > loss_at_optimal else 'أفضل'} من المثالية)\n")

    # --- اقتراح معمارية فعلية للنموذج الأمثل ---
    print("=" * 70)
    print("اقتراح أبعاد معمارية تقريبية للنموذج الأمثل")
    print("=" * 70)
    arch = suggest_architecture(n_star)
    for k, v in arch.items():
        print(f"  {k}: {v}")

    # --- محاكاة استراتيجية LLaMA 3: تدريب "فوق-الأمثل" لصالح استدلال أرخص ---
    print("\n" + "=" * 70)
    print("مقارنة إضافية: استراتيجية LLaMA-3-style (نموذج أصغر، بيانات أكثر بكثير)")
    print("=" * 70)
    n_llama_style = n_star / 3           # نموذج أصغر بكثير من Chinchilla-Optimal
    d_llama_style = budget / (6 * n_llama_style)  # نفس ميزانية التدريب تقريبًا
    loss_llama_style = predicted_loss(n_llama_style, d_llama_style, constants)
    print(f"N: {n_llama_style/1e9:.2f}B   D: {d_llama_style/1e9:.1f}B رمز "
          f"(نسبة D/N={d_llama_style/n_llama_style:.0f})")
    print(f"الخسارة المتوقعة: {loss_llama_style:.4f} "
          f"(أعلى قليلًا من المثالية، لكن الاستدلال أرخص بـ 3x لاحقًا)")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-05-build-an-llm/chapter-22.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: تطبيق Chinchilla حرفيًا لمشروع يستهدف نشرًا تجاريًا ضخمًا
n_params = 70e9
d_tokens_naive = 20 * n_params  # 1.4 تريليون رمز - الأمثل "حسابيًا" فقط

# ✅ صحيح: فكّر في عدد مرات الاستدلال المتوقعة طوال عمر المنتج
# إذا كان الاستدلال سيحدث مليارات المرات، فكّر بنموذج أصغر وبيانات أكثر
# (مقايضة قابلة للحساب: وفر في الاستدلال طويل المدى مقابل خسارة طفيفة في Loss)

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-05-build-an-llm/chapter-22.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: تطبيق صيغة Chinchilla القياسية على نموذج MoE دون تمييز
# N الإجمالي في MoE لا يُكافئ N في نموذج Dense من حيث العلاقة بـ Loss
# (لأن المعاملات النشطة فقط هي التي تُحدد الحوسبة الفعلية لكل رمز)

# ✅ صحيح: استخدم أبحاث Scaling Laws المخصصة لـ MoE
# (مثل Clark et al. 2022 "Unified Scaling Laws for Routed Language Models")
# التي تُميّز بوضوح بين N الإجمالي و N النشط في المعادلات

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-05-build-an-llm/chapter-22.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: مقارنة "D* المطلوب" بين مشروعك ومشروع آخر دون مراعاة جودة البيانات
# نموذج مُدرَّب على بيانات مُصفّاة بعناية (FineWeb-Edu) قد يحتاج D أقل بكثير
# لتحقيق نفس Loss مقارنة ببيانات خام غير مُصفّاة

# ✅ صحيح: أعد تقدير ثوابت Scaling الخاصة ببياناتك دائمًا، ولا تفترض
# أن أرقام الأوراق البحثية الأخرى تنطبق مباشرة على مجموعة بياناتك

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-05-build-an-llm/chapter-22.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: افتراض أن النموذج ذا الخسارة الأقل (Loss) هو بالضرورة الأفضل
# لمهمة محددة (مثل الدردشة أو البرمجة) دون اختبار فعلي

# ✅ صحيح: استخدم Scaling Laws لتحديد ميزانية التدريب الأولية بذكاء،
# لكن اعتمد دائمًا تقييمًا شاملًا على معايير حقيقية (الفصل 46) للقرار النهائي

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-05-build-an-llm/chapter-22.md
# ────────────────────────────────────────────────────────────────────────
"""
mini_project_isoflop_experiment.py
تجربة Scaling Laws مصغّرة حقيقية وقابلة للتشغيل على جهاز واحد:
درّب عدة نماذج GPT مصغّرة جدًا بأحجام مختلفة على نفس البيانات،
بميزانيات حوسبة متساوية تقريبًا (IsoFLOP)، وارسم منحنى U لإيجاد
الحجم الأمثل تجريبيًا بنفسك.

المطلوب إكماله:
1. استخدم نموذج GPT المصغّر من الفصل 18/19 (ModernTransformerBlock).
2. اختر 5-6 أحجام نماذج مختلفة (من ~100K إلى ~5M معامل، صغيرة جدًا
   عمدًا لتكون التجربة قابلة للتشغيل على جهاز واحد خلال دقائق).
3. لكل حجم نموذج N، احسب D بحيث يبقى حاصل الضرب 6*N*D ثابتًا تقريبًا
   (نفس ميزانية الحوسبة لكل التجارب - هذا هو جوهر IsoFLOP).
4. درّب كل نموذج على جزء من نص عربي (مثل عيّنة من ويكيبيديا) بالعدد
   المحسوب من الخطوات، وسجّل الخسارة النهائية.
5. ارسم منحنى (N على المحور الأفقي، Loss النهائية على العمودي) -
   يجب أن يُظهر شكل U تقريبي.
6. حدد النموذج صاحب أقل Loss من بين التجارب - هذا هو N* التجريبي
   لميزانية حوسبتك المحددة.
7. قارن النتيجة بالتنبؤ النظري من حاسبة Chinchilla في هذا الفصل.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class IsoFLOPExperiment:
    n_params_target: int
    compute_budget: float  # 6*N*D ثابت لكل التجارب

    @property
    def d_tokens(self) -> int:
        return int(self.compute_budget / (6 * self.n_params_target))


def run_single_experiment(exp: IsoFLOPExperiment) -> float:
    # TODO 1: ابنِ نموذج GPT مصغّر بعدد معاملات قريب من exp.n_params_target
    #         (اضبط n_layers/d_model يدويًا للوصول لعدد المعاملات المطلوب تقريبًا)
    # TODO 2: درّب النموذج على exp.d_tokens رمزًا بالضبط من نص عربي
    # TODO 3: أعد الخسارة النهائية (على عيّنة تحقق منفصلة held-out)
    raise NotImplementedError


def main() -> None:
    compute_budget = 1e15  # ميزانية ثابتة صغيرة لكل التجارب (قابلة للتشغيل محليًا)
    param_sizes = [100_000, 300_000, 1_000_000, 3_000_000, 5_000_000]

    results = []
    for n in param_sizes:
        exp = IsoFLOPExperiment(n_params_target=n, compute_budget=compute_budget)
        loss = run_single_experiment(exp)
        results.append((n, exp.d_tokens, loss))
        print(f"N={n:>10,}  D={exp.d_tokens:>12,}  Loss={loss:.4f}")

    # TODO: ارسم بـ matplotlib منحنى N مقابل Loss، وحدد أدنى نقطة (N* التجريبي)


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-22-chapter-22.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 22: الفصل الثاني والعشرون: تصميم المعمارية وScaling Laws — فن اختيار الحجم الأمثل لنموذجك."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_scaling() -> None:
    parameters, tokens, coefficient, exponent = (1000000000.0, 20000000000.0, 1.0, 0.34)
    compute = 6 * parameters * tokens
    proxy = coefficient * (parameters ** (-exponent) + tokens ** (-exponent))
    print('training FLOPs proxy:', f'{compute:.3e}')
    print('loss scaling proxy:', f'{proxy:.5f}')
    print('double parameters only compute multiplier:', 2.0)

if __name__ == "__main__":
    print("Chapter 22 lab: Estimate compute with a scaling-law proxy")
    run_scaling()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-22-chapter-22.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
chapter22_scaling_laws_calculator.py
حاسبة Scaling Laws كاملة: من ميزانية حوسبة إلى حجم نموذج،
كمية بيانات، ومعمارية مقترحة، بأسلوب Chinchilla-Optimal.

Python 3.13+
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# =========================================================================
# 1) معاملات قانون Chinchilla المُقدَّرة تجريبيًا (Hoffmann et al. 2022)
# =========================================================================
@dataclass(frozen=True)
class ChinchillaConstants:
    E: float = 1.69     # الإنتروبيا المثالية (أدنى حد نظري للخسارة)
    A: float = 406.4    # ثابت حد حجم النموذج
    B: float = 410.7    # ثابت حد حجم البيانات
    alpha: float = 0.34
    beta: float = 0.28


def predicted_loss(n_params: float, n_tokens: float, c: ChinchillaConstants) -> float:
    """
    التنبؤ بدالة الخسارة النهائية وفق صيغة Chinchilla:
    L(N, D) = E + A/N^alpha + B/D^beta
    """
    return c.E + c.A / (n_params ** c.alpha) + c.B / (n_tokens ** c.beta)


def compute_optimal_allocation(
    compute_budget_flops: float, tokens_per_param: float = 20.0
) -> tuple[float, float]:
    """
    حساب N* و D* الأمثل لميزانية حوسبة معيّنة (بالـ FLOPs)،
    باستخدام القاعدة التقريبية D ≈ tokens_per_param × N
    مُشتقّة من C ≈ 6ND.
    """
    # C = 6 * N * (tokens_per_param * N) = 6 * tokens_per_param * N^2
    n_optimal = math.sqrt(compute_budget_flops / (6 * tokens_per_param))
    d_optimal = tokens_per_param * n_optimal
    return n_optimal, d_optimal


def flops_from_gpu_hours(
    n_gpus: int, hours: float, flops_per_gpu_per_sec: float, mfu: float = 0.4
) -> float:
    """
    تحويل عدد ساعات GPU إلى ميزانية حوسبة فعلية بالـ FLOPs،
    مع الأخذ في الحسبان كفاءة استغلال الأجهزة (MFU - Model FLOPs Utilization).
    """
    total_seconds = n_gpus * hours * 3600
    return total_seconds * flops_per_gpu_per_sec * mfu


def suggest_architecture(n_params: float) -> dict[str, int]:
    """
    اقتراح أبعاد معمارية تقريبية بناءً على قواعد تجريبية شائعة
    (جدول القيم المُستخلص من ملاحظة نماذج ناجحة فعليًا).
    """
    # جدول مرجعي مبني على ملاحظات عملية (مبسّط لأغراض تعليمية)
    reference_points = [
        (125e6, 12, 768, 12),
        (1.3e9, 24, 2048, 16),
        (7e9, 32, 4096, 32),
        (13e9, 40, 5120, 40),
        (70e9, 80, 8192, 64),
    ]
    # اختيار أقرب نقطة مرجعية (في تطبيق حقيقي: استيفاء (interpolation) أدق)
    closest = min(reference_points, key=lambda p: abs(math.log(p[0]) - math.log(n_params)))
    _, n_layers, d_model, n_heads = closest
    return {"n_layers": n_layers, "d_model": d_model, "n_heads": n_heads}


if __name__ == "__main__":
    constants = ChinchillaConstants()

    print("=" * 70)
    print("سيناريو: لديك 512 وحدة A100، لمدة 30 يومًا متواصلة")
    print("=" * 70)

    # A100: ~312 TFLOPS (BF16) نظريًا، نفترض MFU=40% (واقعي جدًا للتدريب الموزّع)
    a100_peak_flops = 312e12
    budget = flops_from_gpu_hours(
        n_gpus=512, hours=30 * 24, flops_per_gpu_per_sec=a100_peak_flops, mfu=0.4
    )
    print(f"ميزانية الحوسبة الكلية: {budget:.3e} FLOPs\n")

    n_star, d_star = compute_optimal_allocation(budget, tokens_per_param=20.0)
    print(f"[Chinchilla-Optimal] حجم النموذج الأمثل N*: {n_star / 1e9:.2f} مليار معامل")
    print(f"[Chinchilla-Optimal] عدد رموز التدريب الأمثل D*: {d_star / 1e9:.1f} مليار رمز")
    print(f"نسبة D*/N*: {d_star / n_star:.1f}  (يجب أن تقارب 20)\n")

    loss_at_optimal = predicted_loss(n_star, d_star, constants)
    print(f"الخسارة المتوقعة عند التركيبة المثلى: {loss_at_optimal:.4f}\n")

    # --- مقارنة: ماذا لو اخترنا نموذجًا أكبر بكثير لكن بنفس الميزانية؟ ---
    print("=" * 70)
    print("تجربة مقارنة: نموذج أكبر 2x مع بيانات أقل (نفس الميزانية تقريبًا)")
    print("=" * 70)
    n_bigger = n_star * 2
    d_smaller = budget / (6 * n_bigger)  # نفس الميزانية C تقريبًا
    loss_bigger = predicted_loss(n_bigger, d_smaller, constants)
    print(f"N أكبر: {n_bigger / 1e9:.2f}B   D أصغر: {d_smaller / 1e9:.1f}B رمز")
    print(f"الخسارة المتوقعة: {loss_bigger:.4f}  "
          f"({'أسوأ' if loss_bigger > loss_at_optimal else 'أفضل'} من المثالية)\n")

    # --- اقتراح معمارية فعلية للنموذج الأمثل ---
    print("=" * 70)
    print("اقتراح أبعاد معمارية تقريبية للنموذج الأمثل")
    print("=" * 70)
    arch = suggest_architecture(n_star)
    for k, v in arch.items():
        print(f"  {k}: {v}")

    # --- محاكاة استراتيجية LLaMA 3: تدريب "فوق-الأمثل" لصالح استدلال أرخص ---
    print("\n" + "=" * 70)
    print("مقارنة إضافية: استراتيجية LLaMA-3-style (نموذج أصغر، بيانات أكثر بكثير)")
    print("=" * 70)
    n_llama_style = n_star / 3           # نموذج أصغر بكثير من Chinchilla-Optimal
    d_llama_style = budget / (6 * n_llama_style)  # نفس ميزانية التدريب تقريبًا
    loss_llama_style = predicted_loss(n_llama_style, d_llama_style, constants)
    print(f"N: {n_llama_style/1e9:.2f}B   D: {d_llama_style/1e9:.1f}B رمز "
          f"(نسبة D/N={d_llama_style/n_llama_style:.0f})")
    print(f"الخسارة المتوقعة: {loss_llama_style:.4f} "
          f"(أعلى قليلًا من المثالية، لكن الاستدلال أرخص بـ 3x لاحقًا)")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-22-chapter-22.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: تطبيق Chinchilla حرفيًا لمشروع يستهدف نشرًا تجاريًا ضخمًا
n_params = 70e9
d_tokens_naive = 20 * n_params  # 1.4 تريليون رمز - الأمثل "حسابيًا" فقط

# ✅ صحيح: فكّر في عدد مرات الاستدلال المتوقعة طوال عمر المنتج
# إذا كان الاستدلال سيحدث مليارات المرات، فكّر بنموذج أصغر وبيانات أكثر
# (مقايضة قابلة للحساب: وفر في الاستدلال طويل المدى مقابل خسارة طفيفة في Loss)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-22-chapter-22.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: تطبيق صيغة Chinchilla القياسية على نموذج MoE دون تمييز
# N الإجمالي في MoE لا يُكافئ N في نموذج Dense من حيث العلاقة بـ Loss
# (لأن المعاملات النشطة فقط هي التي تُحدد الحوسبة الفعلية لكل رمز)

# ✅ صحيح: استخدم أبحاث Scaling Laws المخصصة لـ MoE
# (مثل Clark et al. 2022 "Unified Scaling Laws for Routed Language Models")
# التي تُميّز بوضوح بين N الإجمالي و N النشط في المعادلات

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-22-chapter-22.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: مقارنة "D* المطلوب" بين مشروعك ومشروع آخر دون مراعاة جودة البيانات
# نموذج مُدرَّب على بيانات مُصفّاة بعناية (FineWeb-Edu) قد يحتاج D أقل بكثير
# لتحقيق نفس Loss مقارنة ببيانات خام غير مُصفّاة

# ✅ صحيح: أعد تقدير ثوابت Scaling الخاصة ببياناتك دائمًا، ولا تفترض
# أن أرقام الأوراق البحثية الأخرى تنطبق مباشرة على مجموعة بياناتك

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-22-chapter-22.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: افتراض أن النموذج ذا الخسارة الأقل (Loss) هو بالضرورة الأفضل
# لمهمة محددة (مثل الدردشة أو البرمجة) دون اختبار فعلي

# ✅ صحيح: استخدم Scaling Laws لتحديد ميزانية التدريب الأولية بذكاء،
# لكن اعتمد دائمًا تقييمًا شاملًا على معايير حقيقية (الفصل 46) للقرار النهائي

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-22-chapter-22.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
"""
mini_project_isoflop_experiment.py
تجربة Scaling Laws مصغّرة حقيقية وقابلة للتشغيل على جهاز واحد:
درّب عدة نماذج GPT مصغّرة جدًا بأحجام مختلفة على نفس البيانات،
بميزانيات حوسبة متساوية تقريبًا (IsoFLOP)، وارسم منحنى U لإيجاد
الحجم الأمثل تجريبيًا بنفسك.

المطلوب إكماله:
1. استخدم نموذج GPT المصغّر من الفصل 18/19 (ModernTransformerBlock).
2. اختر 5-6 أحجام نماذج مختلفة (من ~100K إلى ~5M معامل، صغيرة جدًا
   عمدًا لتكون التجربة قابلة للتشغيل على جهاز واحد خلال دقائق).
3. لكل حجم نموذج N، احسب D بحيث يبقى حاصل الضرب 6*N*D ثابتًا تقريبًا
   (نفس ميزانية الحوسبة لكل التجارب - هذا هو جوهر IsoFLOP).
4. درّب كل نموذج على جزء من نص عربي (مثل عيّنة من ويكيبيديا) بالعدد
   المحسوب من الخطوات، وسجّل الخسارة النهائية.
5. ارسم منحنى (N على المحور الأفقي، Loss النهائية على العمودي) -
   يجب أن يُظهر شكل U تقريبي.
6. حدد النموذج صاحب أقل Loss من بين التجارب - هذا هو N* التجريبي
   لميزانية حوسبتك المحددة.
7. قارن النتيجة بالتنبؤ النظري من حاسبة Chinchilla في هذا الفصل.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class IsoFLOPExperiment:
    n_params_target: int
    compute_budget: float  # 6*N*D ثابت لكل التجارب

    @property
    def d_tokens(self) -> int:
        return int(self.compute_budget / (6 * self.n_params_target))


def run_single_experiment(exp: IsoFLOPExperiment) -> float:
    # TODO 1: ابنِ نموذج GPT مصغّر بعدد معاملات قريب من exp.n_params_target
    #         (اضبط n_layers/d_model يدويًا للوصول لعدد المعاملات المطلوب تقريبًا)
    # TODO 2: درّب النموذج على exp.d_tokens رمزًا بالضبط من نص عربي
    # TODO 3: أعد الخسارة النهائية (على عيّنة تحقق منفصلة held-out)
    raise NotImplementedError


def main() -> None:
    compute_budget = 1e15  # ميزانية ثابتة صغيرة لكل التجارب (قابلة للتشغيل محليًا)
    param_sizes = [100_000, 300_000, 1_000_000, 3_000_000, 5_000_000]

    results = []
    for n in param_sizes:
        exp = IsoFLOPExperiment(n_params_target=n, compute_budget=compute_budget)
        loss = run_single_experiment(exp)
        results.append((n, exp.d_tokens, loss))
        print(f"N={n:>10,}  D={exp.d_tokens:>12,}  Loss={loss:.4f}")

    # TODO: ارسم بـ matplotlib منحنى N مقابل Loss، وحدد أدنى نقطة (N* التجريبي)


if __name__ == "__main__":
    main()
