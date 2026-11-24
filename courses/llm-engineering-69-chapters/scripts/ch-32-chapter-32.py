#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — مصفوفة القرار الاستراتيجي — متى تختار ماذا؟
Large Language Model Engineering — Chapter 32 — Model Reuse Adaptation

Book slug : llm-engineering-69-chapters
Chapter   : ch-32-chapter-32 (lesson 32)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-32-chapter-32.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — مصفوفة القرار الاستراتيجي — متى تختار ماذا؟
Large Language Model Engineering — Chapter 32 — Model Reuse Adaptation

Book slug : llm-engineering-69-chapters
Chapter   : ch-32-chapter-32 (lesson 32)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-32-chapter-32.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — مصفوفة القرار الاستراتيجي — متى تختار ماذا؟
Large Language Model Engineering — Chapter 32 — Model Reuse Adaptation

Book slug : llm-engineering-69-chapters
Chapter   : ch-32-chapter-32 (lesson 32)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_32.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 32: الفصل 32: مصفوفة القرار الاستراتيجي — متى تختار ماذا؟."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_decision() -> None:
    candidates = {'small-local': {'quality': 0.72, 'latency': 0.08, 'cost': 0.01}, 'large-api': {'quality': 0.91, 'latency': 0.8, 'cost': 0.2}}
    weights = {'quality': 0.6, 'latency': 0.25, 'cost': 0.15}
    for name, metrics in candidates.items():
        score = weights['quality'] * metrics['quality'] - weights['latency'] * metrics['latency'] - weights['cost'] * metrics['cost']
        print(name, 'decision score:', round(score, 3))

if __name__ == "__main__":
    print("Chapter 32 lab: Score model choices against project constraints")
    run_decision()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-06-model-reuse-adaptation/chapter-32.md
# ────────────────────────────────────────────────────────────────────────
"""
محرك القرار الاستراتيجي لاختيار تقنية تطوير LLM
Python 3.13+ | لا يحتاج مكتبات تدريب فعلية، هذا محرك تقييم منطقي-كمّي
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable


# ------------------------------------------------------------------
# 1) تعريف محاور المشكلة التي يصفها المستخدم
# ------------------------------------------------------------------

class KnowledgeType(Enum):
    """طبيعة الفجوة التي نريد سدّها."""
    VOLATILE_FACTS = auto()      # حقائق تتغير باستمرار (أسعار، أخبار)
    STABLE_DOMAIN = auto()       # معرفة مجال مستقرة لكن غير مُمثَّلة (طب نادر، لغة نادرة)
    BEHAVIOR_STYLE = auto()      # أسلوب/تنسيق/نبرة الاستجابة
    REASONING_SKILL = auto()     # مهارة استدلال أو حل مسائل جديدة
    COMBINED_CAPABILITY = auto() # دمج قدرات نماذج متخصصة متعددة موجودة فعلاً


@dataclass
class ProblemProfile:
    """وصف كمّي كامل للمشكلة التي يواجهها المهندس."""
    knowledge_type: KnowledgeType
    data_volume_examples: int          # عدد الأمثلة المتاحة للتدريب (0 إذا RAG فقط)
    update_frequency_days: float       # كل كم يوم تتغيّر المعرفة؟ (inf إذا ثابتة)
    traceability_required: bool        # هل تحتاج تتبّع مصدر الإجابة؟ (قانوني/طبي)
    latency_budget_ms: float           # الميزانية الزمنية لكل استعلام
    deployment_memory_gb: float        # الذاكرة المتاحة للنشر
    compute_budget_gpu_hours: float    # ميزانية التدريب المتاحة
    has_pretrained_specialist_models: bool = False  # هل توجد نماذج متخصصة جاهزة؟
    team_ml_expertise_level: int = 3   # 1 (مبتدئ) إلى 5 (خبير أبحاث)


class Technique(Enum):
    RAG = "Retrieval-Augmented Generation"
    LORA = "LoRA / PEFT Fine-tuning"
    CPT = "Continued Pretraining"
    MERGING = "Model Merging"
    DISTILLATION = "Knowledge Distillation"


# ------------------------------------------------------------------
# 2) دوال التقييم: كل تقنية لها دالة تُرجع درجة ملاءمة 0-100 + تبرير
# ------------------------------------------------------------------

@dataclass
class ScoreResult:
    technique: Technique
    score: float
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def score_rag(p: ProblemProfile) -> ScoreResult:
    score, reasons, warnings = 0.0, [], []

    if p.knowledge_type == KnowledgeType.VOLATILE_FACTS:
        score += 40
        reasons.append("المعرفة متغيرة زمنيًا؛ RAG يحدّثها دون إعادة تدريب")
    if p.update_frequency_days < 30:
        score += 25
        reasons.append(f"التحديث كل {p.update_frequency_days:.0f} يوم — تدريب متكرر غير عملي")
    if p.traceability_required:
        score += 20
        reasons.append("تتطلب تتبّع المصدر؛ RAG يوفر استشهادًا مباشرًا بالمستندات")
    if p.data_volume_examples < 50:
        score += 10
        reasons.append("بيانات تدريب قليلة جدًا لأي Fine-tuning فعّال")

    if p.latency_budget_ms < 200:
        score -= 15
        warnings.append("زمن استجابة ضيق جدًا؛ خطوة الاسترجاع (Retrieval) تضيف تأخيرًا")
    if p.knowledge_type == KnowledgeType.BEHAVIOR_STYLE:
        score -= 20
        warnings.append("RAG لا يُصلح مشاكل الأسلوب/التنسيق، فقط يضيف معلومات")

    return ScoreResult(Technique.RAG, max(0, min(100, score)), reasons, warnings)


def score_lora(p: ProblemProfile) -> ScoreResult:
    score, reasons, warnings = 0.0, [], []

    if p.knowledge_type in (KnowledgeType.BEHAVIOR_STYLE, KnowledgeType.REASONING_SKILL):
        score += 40
        reasons.append("هذه مشكلة سلوك/مهارة — المجال الأمثل لـ LoRA")
    if 100 <= p.data_volume_examples <= 50_000:
        score += 25
        reasons.append(f"{p.data_volume_examples} مثال — نطاق مثالي لـ PEFT")
    if p.compute_budget_gpu_hours < 50:
        score += 15
        reasons.append("ميزانية حوسبة محدودة؛ LoRA يحتاج جزءًا صغيرًا من تكلفة Full FT")
    if p.deployment_memory_gb < 24:
        score += 10
        reasons.append("ذاكرة نشر محدودة؛ يمكن دمج Adapters متعددة بكفاءة")

    if p.knowledge_type == KnowledgeType.VOLATILE_FACTS:
        score -= 30
        warnings.append("LoRA غير مناسب لمعرفة سريعة التغيّر — سيحتاج إعادة تدريب متكرر")
    if p.data_volume_examples < 50:
        score -= 20
        warnings.append("بيانات قليلة جدًا حتى لـ LoRA؛ خطر Overfitting مرتفع")

    return ScoreResult(Technique.LORA, max(0, min(100, score)), reasons, warnings)


def score_cpt(p: ProblemProfile) -> ScoreResult:
    score, reasons, warnings = 0.0, [], []

    if p.knowledge_type == KnowledgeType.STABLE_DOMAIN:
        score += 45
        reasons.append("فجوة معرفية بنيوية مستقرة — تحتاج تعديل التمثيلات الداخلية عميقًا")
    if p.data_volume_examples > 1_000_000:
        score += 25
        reasons.append("حجم بيانات ضخم يبرر تكلفة Continued Pretraining")
    if p.update_frequency_days > 180:
        score += 15
        reasons.append("معرفة مستقرة طويلة الأمد؛ لا حاجة لتحديث متكرر")
    if p.team_ml_expertise_level >= 4:
        score += 10
        reasons.append("الفريق يملك الخبرة لإدارة Catastrophic Forgetting والـ LR Warmup")

    if p.compute_budget_gpu_hours < 500:
        score -= 35
        warnings.append("ميزانية الحوسبة غير كافية لـ Continued Pretraining الجاد")
    if p.update_frequency_days < 30:
        score -= 30
        warnings.append("معرفة سريعة التغيّر؛ CPT مكلف جدًا للتحديث المتكرر")

    return ScoreResult(Technique.CPT, max(0, min(100, score)), reasons, warnings)


def score_merging(p: ProblemProfile) -> ScoreResult:
    score, reasons, warnings = 0.0, [], []

    if p.knowledge_type == KnowledgeType.COMBINED_CAPABILITY:
        score += 45
        reasons.append("الهدف دمج قدرات موجودة — هذا بالضبط ما صُمم له Merging")
    if p.has_pretrained_specialist_models:
        score += 30
        reasons.append("تتوفر نماذج متخصصة جاهزة — Merging يستغلها دون تدريب إضافي")
    if p.compute_budget_gpu_hours < 10:
        score += 15
        reasons.append("ميزانية حوسبة شبه معدومة؛ Merging يحتاج CPU غالبًا فقط")

    if not p.has_pretrained_specialist_models:
        score -= 40
        warnings.append("لا توجد نماذج متخصصة جاهزة للدمج — Merging غير قابل للتطبيق حاليًا")

    return ScoreResult(Technique.MERGING, max(0, min(100, score)), reasons, warnings)


def score_distillation(p: ProblemProfile) -> ScoreResult:
    score, reasons, warnings = 0.0, [], []

    if p.deployment_memory_gb < 8:
        score += 35
        reasons.append("قيد ذاكرة نشر صارم — Distillation يُنتج نموذجًا مضغوطًا")
    if p.latency_budget_ms < 100:
        score += 30
        reasons.append("ميزانية زمن استجابة ضيقة جدًا — نموذج Student أسرع بشكل جوهري")
    if p.knowledge_type in (KnowledgeType.BEHAVIOR_STYLE, KnowledgeType.REASONING_SKILL):
        score += 15
        reasons.append("يمكن نقل المهارة من Teacher كبير إلى Student صغير فعّال")

    if p.compute_budget_gpu_hours < 100:
        score -= 20
        warnings.append("Distillation الجيد يحتاج تدريبًا على مخرجات Teacher — تكلفة ليست صفرية")
    if p.deployment_memory_gb > 40:
        score -= 15
        warnings.append("لا قيد حقيقي على الذاكرة — فائدة Distillation أقل إلحاحًا")

    return ScoreResult(Technique.DISTILLATION, max(0, min(100, score)), reasons, warnings)


SCORERS: list[Callable[[ProblemProfile], ScoreResult]] = [
    score_rag, score_lora, score_cpt, score_merging, score_distillation
]


# ------------------------------------------------------------------
# 3) محرك القرار الرئيسي
# ------------------------------------------------------------------

def recommend(profile: ProblemProfile, top_k: int = 3) -> list[ScoreResult]:
    """يُرجع أفضل k تقنيات مرتبة تنازليًا بالدرجة."""
    results = [scorer(profile) for scorer in SCORERS]
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def print_recommendation(profile: ProblemProfile) -> None:
    print("=" * 70)
    print("تحليل المشكلة وتوصية الحل الاستراتيجي")
    print("=" * 70)
    results = recommend(profile, top_k=5)
    for i, r in enumerate(results, 1):
        bar = "█" * int(r.score / 5)
        print(f"\n{i}. {r.technique.value}  [{r.score:.0f}/100]  {bar}")
        for reason in r.reasons:
            print(f"   ✓ {reason}")
        for warning in r.warnings:
            print(f"   ⚠ {warning}")
    print("\n" + "=" * 70)

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-06-model-reuse-adaptation/chapter-32.md
# ────────────────────────────────────────────────────────────────────────
# ------------------------------------------------------------------
# السيناريو 1: شات بوت دعم فني يجيب عن سياسات شركة تتغيّر أسبوعيًا
# ------------------------------------------------------------------
scenario_1 = ProblemProfile(
    knowledge_type=KnowledgeType.VOLATILE_FACTS,
    data_volume_examples=0,
    update_frequency_days=7,
    traceability_required=True,
    latency_budget_ms=800,
    deployment_memory_gb=16,
    compute_budget_gpu_hours=20,
)
print_recommendation(scenario_1)
# التوقع: RAG يتصدّر بفارق كبير

# ------------------------------------------------------------------
# السيناريو 2: نموذج يجب أن يجيب دائمًا بصيغة JSON منظّمة
# بأسلوب علامة تجارية محدّدة (توجد 3000 مثال موسوم)
# ------------------------------------------------------------------
scenario_2 = ProblemProfile(
    knowledge_type=KnowledgeType.BEHAVIOR_STYLE,
    data_volume_examples=3000,
    update_frequency_days=365,
    traceability_required=False,
    latency_budget_ms=500,
    deployment_memory_gb=24,
    compute_budget_gpu_hours=30,
)
print_recommendation(scenario_2)
# التوقع: LoRA يتصدّر

# ------------------------------------------------------------------
# السيناريو 3: بناء نموذج يفهم لهجة طبية عربية متخصصة نادرة
# (2 مليار توكن من نصوص طبية عربية، ميزانية حوسبة كبيرة)
# ------------------------------------------------------------------
scenario_3 = ProblemProfile(
    knowledge_type=KnowledgeType.STABLE_DOMAIN,
    data_volume_examples=2_000_000_000,
    update_frequency_days=730,
    traceability_required=False,
    latency_budget_ms=1000,
    deployment_memory_gb=40,
    compute_budget_gpu_hours=2000,
    team_ml_expertise_level=5,
)
print_recommendation(scenario_3)
# التوقع: Continued Pretraining يتصدّر

# ------------------------------------------------------------------
# السيناريو 4: لدينا 3 نماذج متخصصة (كود، رياضيات، عربي) جاهزة
# نريد نموذجًا واحدًا يجمع الثلاث قدرات دون تدريب
# ------------------------------------------------------------------
scenario_4 = ProblemProfile(
    knowledge_type=KnowledgeType.COMBINED_CAPABILITY,
    data_volume_examples=0,
    update_frequency_days=float("inf"),
    traceability_required=False,
    latency_budget_ms=500,
    deployment_memory_gb=24,
    compute_budget_gpu_hours=5,
    has_pretrained_specialist_models=True,
)
print_recommendation(scenario_4)
# التوقع: Model Merging يتصدّر

# ------------------------------------------------------------------
# السيناريو 5: نموذج ممتاز الأداء لكن يجب أن يعمل على هاتف محمول
# ------------------------------------------------------------------
scenario_5 = ProblemProfile(
    knowledge_type=KnowledgeType.REASONING_SKILL,
    data_volume_examples=5000,
    update_frequency_days=365,
    traceability_required=False,
    latency_budget_ms=50,
    deployment_memory_gb=2,
    compute_budget_gpu_hours=150,
)
print_recommendation(scenario_5)
# التوقع: Distillation يتصدّر

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-06-model-reuse-adaptation/chapter-32.md
# ────────────────────────────────────────────────────────────────────────
@dataclass
class HybridRecommendation:
    primary: Technique
    complementary: list[Technique]
    rationale: str


HYBRID_PATTERNS: dict[tuple[KnowledgeType, bool], HybridRecommendation] = {
    (KnowledgeType.VOLATILE_FACTS, True): HybridRecommendation(
        primary=Technique.RAG,
        complementary=[Technique.LORA],
        rationale=(
            "RAG يجلب الحقائق المحدّثة، بينما LoRA خفيف يُدرَّب على كيفية "
            "استخدام هذا السياق بأسلوب الإجابة المطلوب (تنسيق، نبرة، لغة الاستشهاد)"
        ),
    ),
    (KnowledgeType.COMBINED_CAPABILITY, True): HybridRecommendation(
        primary=Technique.MERGING,
        complementary=[Technique.DISTILLATION],
        rationale=(
            "ادمج النماذج المتخصصة أولًا بـ TIES/DARE، ثم قطّر الناتج "
            "إلى نموذج أصغر إذا كانت الميزانية التشغيلية (Inference) هي القيد النهائي"
        ),
    ),
    (KnowledgeType.STABLE_DOMAIN, True): HybridRecommendation(
        primary=Technique.CPT,
        complementary=[Technique.LORA, Technique.RAG],
        rationale=(
            "CPT يبني الأساس المعرفي العميق للمجال، ثم LoRA يضبط المهام "
            "النهائية (Instruction Following)، ثم RAG يغطي أي تحديثات لاحقة"
        ),
    ),
}

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-32-chapter-32.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 32: الفصل 32: مصفوفة القرار الاستراتيجي — متى تختار ماذا؟."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_decision() -> None:
    candidates = {'small-local': {'quality': 0.72, 'latency': 0.08, 'cost': 0.01}, 'large-api': {'quality': 0.91, 'latency': 0.8, 'cost': 0.2}}
    weights = {'quality': 0.6, 'latency': 0.25, 'cost': 0.15}
    for name, metrics in candidates.items():
        score = weights['quality'] * metrics['quality'] - weights['latency'] * metrics['latency'] - weights['cost'] * metrics['cost']
        print(name, 'decision score:', round(score, 3))

if __name__ == "__main__":
    print("Chapter 32 lab: Score model choices against project constraints")
    run_decision()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-32-chapter-32.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
محرك القرار الاستراتيجي لاختيار تقنية تطوير LLM
Python 3.13+ | لا يحتاج مكتبات تدريب فعلية، هذا محرك تقييم منطقي-كمّي
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable


# ------------------------------------------------------------------
# 1) تعريف محاور المشكلة التي يصفها المستخدم
# ------------------------------------------------------------------

class KnowledgeType(Enum):
    """طبيعة الفجوة التي نريد سدّها."""
    VOLATILE_FACTS = auto()      # حقائق تتغير باستمرار (أسعار، أخبار)
    STABLE_DOMAIN = auto()       # معرفة مجال مستقرة لكن غير مُمثَّلة (طب نادر، لغة نادرة)
    BEHAVIOR_STYLE = auto()      # أسلوب/تنسيق/نبرة الاستجابة
    REASONING_SKILL = auto()     # مهارة استدلال أو حل مسائل جديدة
    COMBINED_CAPABILITY = auto() # دمج قدرات نماذج متخصصة متعددة موجودة فعلاً


@dataclass
class ProblemProfile:
    """وصف كمّي كامل للمشكلة التي يواجهها المهندس."""
    knowledge_type: KnowledgeType
    data_volume_examples: int          # عدد الأمثلة المتاحة للتدريب (0 إذا RAG فقط)
    update_frequency_days: float       # كل كم يوم تتغيّر المعرفة؟ (inf إذا ثابتة)
    traceability_required: bool        # هل تحتاج تتبّع مصدر الإجابة؟ (قانوني/طبي)
    latency_budget_ms: float           # الميزانية الزمنية لكل استعلام
    deployment_memory_gb: float        # الذاكرة المتاحة للنشر
    compute_budget_gpu_hours: float    # ميزانية التدريب المتاحة
    has_pretrained_specialist_models: bool = False  # هل توجد نماذج متخصصة جاهزة؟
    team_ml_expertise_level: int = 3   # 1 (مبتدئ) إلى 5 (خبير أبحاث)


class Technique(Enum):
    RAG = "Retrieval-Augmented Generation"
    LORA = "LoRA / PEFT Fine-tuning"
    CPT = "Continued Pretraining"
    MERGING = "Model Merging"
    DISTILLATION = "Knowledge Distillation"


# ------------------------------------------------------------------
# 2) دوال التقييم: كل تقنية لها دالة تُرجع درجة ملاءمة 0-100 + تبرير
# ------------------------------------------------------------------

@dataclass
class ScoreResult:
    technique: Technique
    score: float
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def score_rag(p: ProblemProfile) -> ScoreResult:
    score, reasons, warnings = 0.0, [], []

    if p.knowledge_type == KnowledgeType.VOLATILE_FACTS:
        score += 40
        reasons.append("المعرفة متغيرة زمنيًا؛ RAG يحدّثها دون إعادة تدريب")
    if p.update_frequency_days < 30:
        score += 25
        reasons.append(f"التحديث كل {p.update_frequency_days:.0f} يوم — تدريب متكرر غير عملي")
    if p.traceability_required:
        score += 20
        reasons.append("تتطلب تتبّع المصدر؛ RAG يوفر استشهادًا مباشرًا بالمستندات")
    if p.data_volume_examples < 50:
        score += 10
        reasons.append("بيانات تدريب قليلة جدًا لأي Fine-tuning فعّال")

    if p.latency_budget_ms < 200:
        score -= 15
        warnings.append("زمن استجابة ضيق جدًا؛ خطوة الاسترجاع (Retrieval) تضيف تأخيرًا")
    if p.knowledge_type == KnowledgeType.BEHAVIOR_STYLE:
        score -= 20
        warnings.append("RAG لا يُصلح مشاكل الأسلوب/التنسيق، فقط يضيف معلومات")

    return ScoreResult(Technique.RAG, max(0, min(100, score)), reasons, warnings)


def score_lora(p: ProblemProfile) -> ScoreResult:
    score, reasons, warnings = 0.0, [], []

    if p.knowledge_type in (KnowledgeType.BEHAVIOR_STYLE, KnowledgeType.REASONING_SKILL):
        score += 40
        reasons.append("هذه مشكلة سلوك/مهارة — المجال الأمثل لـ LoRA")
    if 100 <= p.data_volume_examples <= 50_000:
        score += 25
        reasons.append(f"{p.data_volume_examples} مثال — نطاق مثالي لـ PEFT")
    if p.compute_budget_gpu_hours < 50:
        score += 15
        reasons.append("ميزانية حوسبة محدودة؛ LoRA يحتاج جزءًا صغيرًا من تكلفة Full FT")
    if p.deployment_memory_gb < 24:
        score += 10
        reasons.append("ذاكرة نشر محدودة؛ يمكن دمج Adapters متعددة بكفاءة")

    if p.knowledge_type == KnowledgeType.VOLATILE_FACTS:
        score -= 30
        warnings.append("LoRA غير مناسب لمعرفة سريعة التغيّر — سيحتاج إعادة تدريب متكرر")
    if p.data_volume_examples < 50:
        score -= 20
        warnings.append("بيانات قليلة جدًا حتى لـ LoRA؛ خطر Overfitting مرتفع")

    return ScoreResult(Technique.LORA, max(0, min(100, score)), reasons, warnings)


def score_cpt(p: ProblemProfile) -> ScoreResult:
    score, reasons, warnings = 0.0, [], []

    if p.knowledge_type == KnowledgeType.STABLE_DOMAIN:
        score += 45
        reasons.append("فجوة معرفية بنيوية مستقرة — تحتاج تعديل التمثيلات الداخلية عميقًا")
    if p.data_volume_examples > 1_000_000:
        score += 25
        reasons.append("حجم بيانات ضخم يبرر تكلفة Continued Pretraining")
    if p.update_frequency_days > 180:
        score += 15
        reasons.append("معرفة مستقرة طويلة الأمد؛ لا حاجة لتحديث متكرر")
    if p.team_ml_expertise_level >= 4:
        score += 10
        reasons.append("الفريق يملك الخبرة لإدارة Catastrophic Forgetting والـ LR Warmup")

    if p.compute_budget_gpu_hours < 500:
        score -= 35
        warnings.append("ميزانية الحوسبة غير كافية لـ Continued Pretraining الجاد")
    if p.update_frequency_days < 30:
        score -= 30
        warnings.append("معرفة سريعة التغيّر؛ CPT مكلف جدًا للتحديث المتكرر")

    return ScoreResult(Technique.CPT, max(0, min(100, score)), reasons, warnings)


def score_merging(p: ProblemProfile) -> ScoreResult:
    score, reasons, warnings = 0.0, [], []

    if p.knowledge_type == KnowledgeType.COMBINED_CAPABILITY:
        score += 45
        reasons.append("الهدف دمج قدرات موجودة — هذا بالضبط ما صُمم له Merging")
    if p.has_pretrained_specialist_models:
        score += 30
        reasons.append("تتوفر نماذج متخصصة جاهزة — Merging يستغلها دون تدريب إضافي")
    if p.compute_budget_gpu_hours < 10:
        score += 15
        reasons.append("ميزانية حوسبة شبه معدومة؛ Merging يحتاج CPU غالبًا فقط")

    if not p.has_pretrained_specialist_models:
        score -= 40
        warnings.append("لا توجد نماذج متخصصة جاهزة للدمج — Merging غير قابل للتطبيق حاليًا")

    return ScoreResult(Technique.MERGING, max(0, min(100, score)), reasons, warnings)


def score_distillation(p: ProblemProfile) -> ScoreResult:
    score, reasons, warnings = 0.0, [], []

    if p.deployment_memory_gb < 8:
        score += 35
        reasons.append("قيد ذاكرة نشر صارم — Distillation يُنتج نموذجًا مضغوطًا")
    if p.latency_budget_ms < 100:
        score += 30
        reasons.append("ميزانية زمن استجابة ضيقة جدًا — نموذج Student أسرع بشكل جوهري")
    if p.knowledge_type in (KnowledgeType.BEHAVIOR_STYLE, KnowledgeType.REASONING_SKILL):
        score += 15
        reasons.append("يمكن نقل المهارة من Teacher كبير إلى Student صغير فعّال")

    if p.compute_budget_gpu_hours < 100:
        score -= 20
        warnings.append("Distillation الجيد يحتاج تدريبًا على مخرجات Teacher — تكلفة ليست صفرية")
    if p.deployment_memory_gb > 40:
        score -= 15
        warnings.append("لا قيد حقيقي على الذاكرة — فائدة Distillation أقل إلحاحًا")

    return ScoreResult(Technique.DISTILLATION, max(0, min(100, score)), reasons, warnings)


SCORERS: list[Callable[[ProblemProfile], ScoreResult]] = [
    score_rag, score_lora, score_cpt, score_merging, score_distillation
]


# ------------------------------------------------------------------
# 3) محرك القرار الرئيسي
# ------------------------------------------------------------------

def recommend(profile: ProblemProfile, top_k: int = 3) -> list[ScoreResult]:
    """يُرجع أفضل k تقنيات مرتبة تنازليًا بالدرجة."""
    results = [scorer(profile) for scorer in SCORERS]
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def print_recommendation(profile: ProblemProfile) -> None:
    print("=" * 70)
    print("تحليل المشكلة وتوصية الحل الاستراتيجي")
    print("=" * 70)
    results = recommend(profile, top_k=5)
    for i, r in enumerate(results, 1):
        bar = "█" * int(r.score / 5)
        print(f"\n{i}. {r.technique.value}  [{r.score:.0f}/100]  {bar}")
        for reason in r.reasons:
            print(f"   ✓ {reason}")
        for warning in r.warnings:
            print(f"   ⚠ {warning}")
    print("\n" + "=" * 70)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-32-chapter-32.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ------------------------------------------------------------------
# السيناريو 1: شات بوت دعم فني يجيب عن سياسات شركة تتغيّر أسبوعيًا
# ------------------------------------------------------------------
scenario_1 = ProblemProfile(
    knowledge_type=KnowledgeType.VOLATILE_FACTS,
    data_volume_examples=0,
    update_frequency_days=7,
    traceability_required=True,
    latency_budget_ms=800,
    deployment_memory_gb=16,
    compute_budget_gpu_hours=20,
)
print_recommendation(scenario_1)
# التوقع: RAG يتصدّر بفارق كبير

# ------------------------------------------------------------------
# السيناريو 2: نموذج يجب أن يجيب دائمًا بصيغة JSON منظّمة
# بأسلوب علامة تجارية محدّدة (توجد 3000 مثال موسوم)
# ------------------------------------------------------------------
scenario_2 = ProblemProfile(
    knowledge_type=KnowledgeType.BEHAVIOR_STYLE,
    data_volume_examples=3000,
    update_frequency_days=365,
    traceability_required=False,
    latency_budget_ms=500,
    deployment_memory_gb=24,
    compute_budget_gpu_hours=30,
)
print_recommendation(scenario_2)
# التوقع: LoRA يتصدّر

# ------------------------------------------------------------------
# السيناريو 3: بناء نموذج يفهم لهجة طبية عربية متخصصة نادرة
# (2 مليار توكن من نصوص طبية عربية، ميزانية حوسبة كبيرة)
# ------------------------------------------------------------------
scenario_3 = ProblemProfile(
    knowledge_type=KnowledgeType.STABLE_DOMAIN,
    data_volume_examples=2_000_000_000,
    update_frequency_days=730,
    traceability_required=False,
    latency_budget_ms=1000,
    deployment_memory_gb=40,
    compute_budget_gpu_hours=2000,
    team_ml_expertise_level=5,
)
print_recommendation(scenario_3)
# التوقع: Continued Pretraining يتصدّر

# ------------------------------------------------------------------
# السيناريو 4: لدينا 3 نماذج متخصصة (كود، رياضيات، عربي) جاهزة
# نريد نموذجًا واحدًا يجمع الثلاث قدرات دون تدريب
# ------------------------------------------------------------------
scenario_4 = ProblemProfile(
    knowledge_type=KnowledgeType.COMBINED_CAPABILITY,
    data_volume_examples=0,
    update_frequency_days=float("inf"),
    traceability_required=False,
    latency_budget_ms=500,
    deployment_memory_gb=24,
    compute_budget_gpu_hours=5,
    has_pretrained_specialist_models=True,
)
print_recommendation(scenario_4)
# التوقع: Model Merging يتصدّر

# ------------------------------------------------------------------
# السيناريو 5: نموذج ممتاز الأداء لكن يجب أن يعمل على هاتف محمول
# ------------------------------------------------------------------
scenario_5 = ProblemProfile(
    knowledge_type=KnowledgeType.REASONING_SKILL,
    data_volume_examples=5000,
    update_frequency_days=365,
    traceability_required=False,
    latency_budget_ms=50,
    deployment_memory_gb=2,
    compute_budget_gpu_hours=150,
)
print_recommendation(scenario_5)
# التوقع: Distillation يتصدّر

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-32-chapter-32.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
@dataclass
class HybridRecommendation:
    primary: Technique
    complementary: list[Technique]
    rationale: str


HYBRID_PATTERNS: dict[tuple[KnowledgeType, bool], HybridRecommendation] = {
    (KnowledgeType.VOLATILE_FACTS, True): HybridRecommendation(
        primary=Technique.RAG,
        complementary=[Technique.LORA],
        rationale=(
            "RAG يجلب الحقائق المحدّثة، بينما LoRA خفيف يُدرَّب على كيفية "
            "استخدام هذا السياق بأسلوب الإجابة المطلوب (تنسيق، نبرة، لغة الاستشهاد)"
        ),
    ),
    (KnowledgeType.COMBINED_CAPABILITY, True): HybridRecommendation(
        primary=Technique.MERGING,
        complementary=[Technique.DISTILLATION],
        rationale=(
            "ادمج النماذج المتخصصة أولًا بـ TIES/DARE، ثم قطّر الناتج "
            "إلى نموذج أصغر إذا كانت الميزانية التشغيلية (Inference) هي القيد النهائي"
        ),
    ),
    (KnowledgeType.STABLE_DOMAIN, True): HybridRecommendation(
        primary=Technique.CPT,
        complementary=[Technique.LORA, Technique.RAG],
        rationale=(
            "CPT يبني الأساس المعرفي العميق للمجال، ثم LoRA يضبط المهام "
            "النهائية (Instruction Following)، ثم RAG يغطي أي تحديثات لاحقة"
        ),
    ),
}

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-32-chapter-32.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — مصفوفة القرار الاستراتيجي — متى تختار ماذا؟
Large Language Model Engineering — Chapter 32 — Model Reuse Adaptation

Book slug : llm-engineering-69-chapters
Chapter   : ch-32-chapter-32 (lesson 32)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_32.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 32: الفصل 32: مصفوفة القرار الاستراتيجي — متى تختار ماذا؟."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_decision() -> None:
    candidates = {'small-local': {'quality': 0.72, 'latency': 0.08, 'cost': 0.01}, 'large-api': {'quality': 0.91, 'latency': 0.8, 'cost': 0.2}}
    weights = {'quality': 0.6, 'latency': 0.25, 'cost': 0.15}
    for name, metrics in candidates.items():
        score = weights['quality'] * metrics['quality'] - weights['latency'] * metrics['latency'] - weights['cost'] * metrics['cost']
        print(name, 'decision score:', round(score, 3))

if __name__ == "__main__":
    print("Chapter 32 lab: Score model choices against project constraints")
    run_decision()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-06-model-reuse-adaptation/chapter-32.md
# ────────────────────────────────────────────────────────────────────────
"""
محرك القرار الاستراتيجي لاختيار تقنية تطوير LLM
Python 3.13+ | لا يحتاج مكتبات تدريب فعلية، هذا محرك تقييم منطقي-كمّي
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable


# ------------------------------------------------------------------
# 1) تعريف محاور المشكلة التي يصفها المستخدم
# ------------------------------------------------------------------

class KnowledgeType(Enum):
    """طبيعة الفجوة التي نريد سدّها."""
    VOLATILE_FACTS = auto()      # حقائق تتغير باستمرار (أسعار، أخبار)
    STABLE_DOMAIN = auto()       # معرفة مجال مستقرة لكن غير مُمثَّلة (طب نادر، لغة نادرة)
    BEHAVIOR_STYLE = auto()      # أسلوب/تنسيق/نبرة الاستجابة
    REASONING_SKILL = auto()     # مهارة استدلال أو حل مسائل جديدة
    COMBINED_CAPABILITY = auto() # دمج قدرات نماذج متخصصة متعددة موجودة فعلاً


@dataclass
class ProblemProfile:
    """وصف كمّي كامل للمشكلة التي يواجهها المهندس."""
    knowledge_type: KnowledgeType
    data_volume_examples: int          # عدد الأمثلة المتاحة للتدريب (0 إذا RAG فقط)
    update_frequency_days: float       # كل كم يوم تتغيّر المعرفة؟ (inf إذا ثابتة)
    traceability_required: bool        # هل تحتاج تتبّع مصدر الإجابة؟ (قانوني/طبي)
    latency_budget_ms: float           # الميزانية الزمنية لكل استعلام
    deployment_memory_gb: float        # الذاكرة المتاحة للنشر
    compute_budget_gpu_hours: float    # ميزانية التدريب المتاحة
    has_pretrained_specialist_models: bool = False  # هل توجد نماذج متخصصة جاهزة؟
    team_ml_expertise_level: int = 3   # 1 (مبتدئ) إلى 5 (خبير أبحاث)


class Technique(Enum):
    RAG = "Retrieval-Augmented Generation"
    LORA = "LoRA / PEFT Fine-tuning"
    CPT = "Continued Pretraining"
    MERGING = "Model Merging"
    DISTILLATION = "Knowledge Distillation"


# ------------------------------------------------------------------
# 2) دوال التقييم: كل تقنية لها دالة تُرجع درجة ملاءمة 0-100 + تبرير
# ------------------------------------------------------------------

@dataclass
class ScoreResult:
    technique: Technique
    score: float
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def score_rag(p: ProblemProfile) -> ScoreResult:
    score, reasons, warnings = 0.0, [], []

    if p.knowledge_type == KnowledgeType.VOLATILE_FACTS:
        score += 40
        reasons.append("المعرفة متغيرة زمنيًا؛ RAG يحدّثها دون إعادة تدريب")
    if p.update_frequency_days < 30:
        score += 25
        reasons.append(f"التحديث كل {p.update_frequency_days:.0f} يوم — تدريب متكرر غير عملي")
    if p.traceability_required:
        score += 20
        reasons.append("تتطلب تتبّع المصدر؛ RAG يوفر استشهادًا مباشرًا بالمستندات")
    if p.data_volume_examples < 50:
        score += 10
        reasons.append("بيانات تدريب قليلة جدًا لأي Fine-tuning فعّال")

    if p.latency_budget_ms < 200:
        score -= 15
        warnings.append("زمن استجابة ضيق جدًا؛ خطوة الاسترجاع (Retrieval) تضيف تأخيرًا")
    if p.knowledge_type == KnowledgeType.BEHAVIOR_STYLE:
        score -= 20
        warnings.append("RAG لا يُصلح مشاكل الأسلوب/التنسيق، فقط يضيف معلومات")

    return ScoreResult(Technique.RAG, max(0, min(100, score)), reasons, warnings)


def score_lora(p: ProblemProfile) -> ScoreResult:
    score, reasons, warnings = 0.0, [], []

    if p.knowledge_type in (KnowledgeType.BEHAVIOR_STYLE, KnowledgeType.REASONING_SKILL):
        score += 40
        reasons.append("هذه مشكلة سلوك/مهارة — المجال الأمثل لـ LoRA")
    if 100 <= p.data_volume_examples <= 50_000:
        score += 25
        reasons.append(f"{p.data_volume_examples} مثال — نطاق مثالي لـ PEFT")
    if p.compute_budget_gpu_hours < 50:
        score += 15
        reasons.append("ميزانية حوسبة محدودة؛ LoRA يحتاج جزءًا صغيرًا من تكلفة Full FT")
    if p.deployment_memory_gb < 24:
        score += 10
        reasons.append("ذاكرة نشر محدودة؛ يمكن دمج Adapters متعددة بكفاءة")

    if p.knowledge_type == KnowledgeType.VOLATILE_FACTS:
        score -= 30
        warnings.append("LoRA غير مناسب لمعرفة سريعة التغيّر — سيحتاج إعادة تدريب متكرر")
    if p.data_volume_examples < 50:
        score -= 20
        warnings.append("بيانات قليلة جدًا حتى لـ LoRA؛ خطر Overfitting مرتفع")

    return ScoreResult(Technique.LORA, max(0, min(100, score)), reasons, warnings)


def score_cpt(p: ProblemProfile) -> ScoreResult:
    score, reasons, warnings = 0.0, [], []

    if p.knowledge_type == KnowledgeType.STABLE_DOMAIN:
        score += 45
        reasons.append("فجوة معرفية بنيوية مستقرة — تحتاج تعديل التمثيلات الداخلية عميقًا")
    if p.data_volume_examples > 1_000_000:
        score += 25
        reasons.append("حجم بيانات ضخم يبرر تكلفة Continued Pretraining")
    if p.update_frequency_days > 180:
        score += 15
        reasons.append("معرفة مستقرة طويلة الأمد؛ لا حاجة لتحديث متكرر")
    if p.team_ml_expertise_level >= 4:
        score += 10
        reasons.append("الفريق يملك الخبرة لإدارة Catastrophic Forgetting والـ LR Warmup")

    if p.compute_budget_gpu_hours < 500:
        score -= 35
        warnings.append("ميزانية الحوسبة غير كافية لـ Continued Pretraining الجاد")
    if p.update_frequency_days < 30:
        score -= 30
        warnings.append("معرفة سريعة التغيّر؛ CPT مكلف جدًا للتحديث المتكرر")

    return ScoreResult(Technique.CPT, max(0, min(100, score)), reasons, warnings)


def score_merging(p: ProblemProfile) -> ScoreResult:
    score, reasons, warnings = 0.0, [], []

    if p.knowledge_type == KnowledgeType.COMBINED_CAPABILITY:
        score += 45
        reasons.append("الهدف دمج قدرات موجودة — هذا بالضبط ما صُمم له Merging")
    if p.has_pretrained_specialist_models:
        score += 30
        reasons.append("تتوفر نماذج متخصصة جاهزة — Merging يستغلها دون تدريب إضافي")
    if p.compute_budget_gpu_hours < 10:
        score += 15
        reasons.append("ميزانية حوسبة شبه معدومة؛ Merging يحتاج CPU غالبًا فقط")

    if not p.has_pretrained_specialist_models:
        score -= 40
        warnings.append("لا توجد نماذج متخصصة جاهزة للدمج — Merging غير قابل للتطبيق حاليًا")

    return ScoreResult(Technique.MERGING, max(0, min(100, score)), reasons, warnings)


def score_distillation(p: ProblemProfile) -> ScoreResult:
    score, reasons, warnings = 0.0, [], []

    if p.deployment_memory_gb < 8:
        score += 35
        reasons.append("قيد ذاكرة نشر صارم — Distillation يُنتج نموذجًا مضغوطًا")
    if p.latency_budget_ms < 100:
        score += 30
        reasons.append("ميزانية زمن استجابة ضيقة جدًا — نموذج Student أسرع بشكل جوهري")
    if p.knowledge_type in (KnowledgeType.BEHAVIOR_STYLE, KnowledgeType.REASONING_SKILL):
        score += 15
        reasons.append("يمكن نقل المهارة من Teacher كبير إلى Student صغير فعّال")

    if p.compute_budget_gpu_hours < 100:
        score -= 20
        warnings.append("Distillation الجيد يحتاج تدريبًا على مخرجات Teacher — تكلفة ليست صفرية")
    if p.deployment_memory_gb > 40:
        score -= 15
        warnings.append("لا قيد حقيقي على الذاكرة — فائدة Distillation أقل إلحاحًا")

    return ScoreResult(Technique.DISTILLATION, max(0, min(100, score)), reasons, warnings)


SCORERS: list[Callable[[ProblemProfile], ScoreResult]] = [
    score_rag, score_lora, score_cpt, score_merging, score_distillation
]


# ------------------------------------------------------------------
# 3) محرك القرار الرئيسي
# ------------------------------------------------------------------

def recommend(profile: ProblemProfile, top_k: int = 3) -> list[ScoreResult]:
    """يُرجع أفضل k تقنيات مرتبة تنازليًا بالدرجة."""
    results = [scorer(profile) for scorer in SCORERS]
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def print_recommendation(profile: ProblemProfile) -> None:
    print("=" * 70)
    print("تحليل المشكلة وتوصية الحل الاستراتيجي")
    print("=" * 70)
    results = recommend(profile, top_k=5)
    for i, r in enumerate(results, 1):
        bar = "█" * int(r.score / 5)
        print(f"\n{i}. {r.technique.value}  [{r.score:.0f}/100]  {bar}")
        for reason in r.reasons:
            print(f"   ✓ {reason}")
        for warning in r.warnings:
            print(f"   ⚠ {warning}")
    print("\n" + "=" * 70)

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-06-model-reuse-adaptation/chapter-32.md
# ────────────────────────────────────────────────────────────────────────
# ------------------------------------------------------------------
# السيناريو 1: شات بوت دعم فني يجيب عن سياسات شركة تتغيّر أسبوعيًا
# ------------------------------------------------------------------
scenario_1 = ProblemProfile(
    knowledge_type=KnowledgeType.VOLATILE_FACTS,
    data_volume_examples=0,
    update_frequency_days=7,
    traceability_required=True,
    latency_budget_ms=800,
    deployment_memory_gb=16,
    compute_budget_gpu_hours=20,
)
print_recommendation(scenario_1)
# التوقع: RAG يتصدّر بفارق كبير

# ------------------------------------------------------------------
# السيناريو 2: نموذج يجب أن يجيب دائمًا بصيغة JSON منظّمة
# بأسلوب علامة تجارية محدّدة (توجد 3000 مثال موسوم)
# ------------------------------------------------------------------
scenario_2 = ProblemProfile(
    knowledge_type=KnowledgeType.BEHAVIOR_STYLE,
    data_volume_examples=3000,
    update_frequency_days=365,
    traceability_required=False,
    latency_budget_ms=500,
    deployment_memory_gb=24,
    compute_budget_gpu_hours=30,
)
print_recommendation(scenario_2)
# التوقع: LoRA يتصدّر

# ------------------------------------------------------------------
# السيناريو 3: بناء نموذج يفهم لهجة طبية عربية متخصصة نادرة
# (2 مليار توكن من نصوص طبية عربية، ميزانية حوسبة كبيرة)
# ------------------------------------------------------------------
scenario_3 = ProblemProfile(
    knowledge_type=KnowledgeType.STABLE_DOMAIN,
    data_volume_examples=2_000_000_000,
    update_frequency_days=730,
    traceability_required=False,
    latency_budget_ms=1000,
    deployment_memory_gb=40,
    compute_budget_gpu_hours=2000,
    team_ml_expertise_level=5,
)
print_recommendation(scenario_3)
# التوقع: Continued Pretraining يتصدّر

# ------------------------------------------------------------------
# السيناريو 4: لدينا 3 نماذج متخصصة (كود، رياضيات، عربي) جاهزة
# نريد نموذجًا واحدًا يجمع الثلاث قدرات دون تدريب
# ------------------------------------------------------------------
scenario_4 = ProblemProfile(
    knowledge_type=KnowledgeType.COMBINED_CAPABILITY,
    data_volume_examples=0,
    update_frequency_days=float("inf"),
    traceability_required=False,
    latency_budget_ms=500,
    deployment_memory_gb=24,
    compute_budget_gpu_hours=5,
    has_pretrained_specialist_models=True,
)
print_recommendation(scenario_4)
# التوقع: Model Merging يتصدّر

# ------------------------------------------------------------------
# السيناريو 5: نموذج ممتاز الأداء لكن يجب أن يعمل على هاتف محمول
# ------------------------------------------------------------------
scenario_5 = ProblemProfile(
    knowledge_type=KnowledgeType.REASONING_SKILL,
    data_volume_examples=5000,
    update_frequency_days=365,
    traceability_required=False,
    latency_budget_ms=50,
    deployment_memory_gb=2,
    compute_budget_gpu_hours=150,
)
print_recommendation(scenario_5)
# التوقع: Distillation يتصدّر

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-06-model-reuse-adaptation/chapter-32.md
# ────────────────────────────────────────────────────────────────────────
@dataclass
class HybridRecommendation:
    primary: Technique
    complementary: list[Technique]
    rationale: str


HYBRID_PATTERNS: dict[tuple[KnowledgeType, bool], HybridRecommendation] = {
    (KnowledgeType.VOLATILE_FACTS, True): HybridRecommendation(
        primary=Technique.RAG,
        complementary=[Technique.LORA],
        rationale=(
            "RAG يجلب الحقائق المحدّثة، بينما LoRA خفيف يُدرَّب على كيفية "
            "استخدام هذا السياق بأسلوب الإجابة المطلوب (تنسيق، نبرة، لغة الاستشهاد)"
        ),
    ),
    (KnowledgeType.COMBINED_CAPABILITY, True): HybridRecommendation(
        primary=Technique.MERGING,
        complementary=[Technique.DISTILLATION],
        rationale=(
            "ادمج النماذج المتخصصة أولًا بـ TIES/DARE، ثم قطّر الناتج "
            "إلى نموذج أصغر إذا كانت الميزانية التشغيلية (Inference) هي القيد النهائي"
        ),
    ),
    (KnowledgeType.STABLE_DOMAIN, True): HybridRecommendation(
        primary=Technique.CPT,
        complementary=[Technique.LORA, Technique.RAG],
        rationale=(
            "CPT يبني الأساس المعرفي العميق للمجال، ثم LoRA يضبط المهام "
            "النهائية (Instruction Following)، ثم RAG يغطي أي تحديثات لاحقة"
        ),
    ),
}

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-32-chapter-32.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 32: الفصل 32: مصفوفة القرار الاستراتيجي — متى تختار ماذا؟."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_decision() -> None:
    candidates = {'small-local': {'quality': 0.72, 'latency': 0.08, 'cost': 0.01}, 'large-api': {'quality': 0.91, 'latency': 0.8, 'cost': 0.2}}
    weights = {'quality': 0.6, 'latency': 0.25, 'cost': 0.15}
    for name, metrics in candidates.items():
        score = weights['quality'] * metrics['quality'] - weights['latency'] * metrics['latency'] - weights['cost'] * metrics['cost']
        print(name, 'decision score:', round(score, 3))

if __name__ == "__main__":
    print("Chapter 32 lab: Score model choices against project constraints")
    run_decision()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-32-chapter-32.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
محرك القرار الاستراتيجي لاختيار تقنية تطوير LLM
Python 3.13+ | لا يحتاج مكتبات تدريب فعلية، هذا محرك تقييم منطقي-كمّي
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable


# ------------------------------------------------------------------
# 1) تعريف محاور المشكلة التي يصفها المستخدم
# ------------------------------------------------------------------

class KnowledgeType(Enum):
    """طبيعة الفجوة التي نريد سدّها."""
    VOLATILE_FACTS = auto()      # حقائق تتغير باستمرار (أسعار، أخبار)
    STABLE_DOMAIN = auto()       # معرفة مجال مستقرة لكن غير مُمثَّلة (طب نادر، لغة نادرة)
    BEHAVIOR_STYLE = auto()      # أسلوب/تنسيق/نبرة الاستجابة
    REASONING_SKILL = auto()     # مهارة استدلال أو حل مسائل جديدة
    COMBINED_CAPABILITY = auto() # دمج قدرات نماذج متخصصة متعددة موجودة فعلاً


@dataclass
class ProblemProfile:
    """وصف كمّي كامل للمشكلة التي يواجهها المهندس."""
    knowledge_type: KnowledgeType
    data_volume_examples: int          # عدد الأمثلة المتاحة للتدريب (0 إذا RAG فقط)
    update_frequency_days: float       # كل كم يوم تتغيّر المعرفة؟ (inf إذا ثابتة)
    traceability_required: bool        # هل تحتاج تتبّع مصدر الإجابة؟ (قانوني/طبي)
    latency_budget_ms: float           # الميزانية الزمنية لكل استعلام
    deployment_memory_gb: float        # الذاكرة المتاحة للنشر
    compute_budget_gpu_hours: float    # ميزانية التدريب المتاحة
    has_pretrained_specialist_models: bool = False  # هل توجد نماذج متخصصة جاهزة؟
    team_ml_expertise_level: int = 3   # 1 (مبتدئ) إلى 5 (خبير أبحاث)


class Technique(Enum):
    RAG = "Retrieval-Augmented Generation"
    LORA = "LoRA / PEFT Fine-tuning"
    CPT = "Continued Pretraining"
    MERGING = "Model Merging"
    DISTILLATION = "Knowledge Distillation"


# ------------------------------------------------------------------
# 2) دوال التقييم: كل تقنية لها دالة تُرجع درجة ملاءمة 0-100 + تبرير
# ------------------------------------------------------------------

@dataclass
class ScoreResult:
    technique: Technique
    score: float
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def score_rag(p: ProblemProfile) -> ScoreResult:
    score, reasons, warnings = 0.0, [], []

    if p.knowledge_type == KnowledgeType.VOLATILE_FACTS:
        score += 40
        reasons.append("المعرفة متغيرة زمنيًا؛ RAG يحدّثها دون إعادة تدريب")
    if p.update_frequency_days < 30:
        score += 25
        reasons.append(f"التحديث كل {p.update_frequency_days:.0f} يوم — تدريب متكرر غير عملي")
    if p.traceability_required:
        score += 20
        reasons.append("تتطلب تتبّع المصدر؛ RAG يوفر استشهادًا مباشرًا بالمستندات")
    if p.data_volume_examples < 50:
        score += 10
        reasons.append("بيانات تدريب قليلة جدًا لأي Fine-tuning فعّال")

    if p.latency_budget_ms < 200:
        score -= 15
        warnings.append("زمن استجابة ضيق جدًا؛ خطوة الاسترجاع (Retrieval) تضيف تأخيرًا")
    if p.knowledge_type == KnowledgeType.BEHAVIOR_STYLE:
        score -= 20
        warnings.append("RAG لا يُصلح مشاكل الأسلوب/التنسيق، فقط يضيف معلومات")

    return ScoreResult(Technique.RAG, max(0, min(100, score)), reasons, warnings)


def score_lora(p: ProblemProfile) -> ScoreResult:
    score, reasons, warnings = 0.0, [], []

    if p.knowledge_type in (KnowledgeType.BEHAVIOR_STYLE, KnowledgeType.REASONING_SKILL):
        score += 40
        reasons.append("هذه مشكلة سلوك/مهارة — المجال الأمثل لـ LoRA")
    if 100 <= p.data_volume_examples <= 50_000:
        score += 25
        reasons.append(f"{p.data_volume_examples} مثال — نطاق مثالي لـ PEFT")
    if p.compute_budget_gpu_hours < 50:
        score += 15
        reasons.append("ميزانية حوسبة محدودة؛ LoRA يحتاج جزءًا صغيرًا من تكلفة Full FT")
    if p.deployment_memory_gb < 24:
        score += 10
        reasons.append("ذاكرة نشر محدودة؛ يمكن دمج Adapters متعددة بكفاءة")

    if p.knowledge_type == KnowledgeType.VOLATILE_FACTS:
        score -= 30
        warnings.append("LoRA غير مناسب لمعرفة سريعة التغيّر — سيحتاج إعادة تدريب متكرر")
    if p.data_volume_examples < 50:
        score -= 20
        warnings.append("بيانات قليلة جدًا حتى لـ LoRA؛ خطر Overfitting مرتفع")

    return ScoreResult(Technique.LORA, max(0, min(100, score)), reasons, warnings)


def score_cpt(p: ProblemProfile) -> ScoreResult:
    score, reasons, warnings = 0.0, [], []

    if p.knowledge_type == KnowledgeType.STABLE_DOMAIN:
        score += 45
        reasons.append("فجوة معرفية بنيوية مستقرة — تحتاج تعديل التمثيلات الداخلية عميقًا")
    if p.data_volume_examples > 1_000_000:
        score += 25
        reasons.append("حجم بيانات ضخم يبرر تكلفة Continued Pretraining")
    if p.update_frequency_days > 180:
        score += 15
        reasons.append("معرفة مستقرة طويلة الأمد؛ لا حاجة لتحديث متكرر")
    if p.team_ml_expertise_level >= 4:
        score += 10
        reasons.append("الفريق يملك الخبرة لإدارة Catastrophic Forgetting والـ LR Warmup")

    if p.compute_budget_gpu_hours < 500:
        score -= 35
        warnings.append("ميزانية الحوسبة غير كافية لـ Continued Pretraining الجاد")
    if p.update_frequency_days < 30:
        score -= 30
        warnings.append("معرفة سريعة التغيّر؛ CPT مكلف جدًا للتحديث المتكرر")

    return ScoreResult(Technique.CPT, max(0, min(100, score)), reasons, warnings)


def score_merging(p: ProblemProfile) -> ScoreResult:
    score, reasons, warnings = 0.0, [], []

    if p.knowledge_type == KnowledgeType.COMBINED_CAPABILITY:
        score += 45
        reasons.append("الهدف دمج قدرات موجودة — هذا بالضبط ما صُمم له Merging")
    if p.has_pretrained_specialist_models:
        score += 30
        reasons.append("تتوفر نماذج متخصصة جاهزة — Merging يستغلها دون تدريب إضافي")
    if p.compute_budget_gpu_hours < 10:
        score += 15
        reasons.append("ميزانية حوسبة شبه معدومة؛ Merging يحتاج CPU غالبًا فقط")

    if not p.has_pretrained_specialist_models:
        score -= 40
        warnings.append("لا توجد نماذج متخصصة جاهزة للدمج — Merging غير قابل للتطبيق حاليًا")

    return ScoreResult(Technique.MERGING, max(0, min(100, score)), reasons, warnings)


def score_distillation(p: ProblemProfile) -> ScoreResult:
    score, reasons, warnings = 0.0, [], []

    if p.deployment_memory_gb < 8:
        score += 35
        reasons.append("قيد ذاكرة نشر صارم — Distillation يُنتج نموذجًا مضغوطًا")
    if p.latency_budget_ms < 100:
        score += 30
        reasons.append("ميزانية زمن استجابة ضيقة جدًا — نموذج Student أسرع بشكل جوهري")
    if p.knowledge_type in (KnowledgeType.BEHAVIOR_STYLE, KnowledgeType.REASONING_SKILL):
        score += 15
        reasons.append("يمكن نقل المهارة من Teacher كبير إلى Student صغير فعّال")

    if p.compute_budget_gpu_hours < 100:
        score -= 20
        warnings.append("Distillation الجيد يحتاج تدريبًا على مخرجات Teacher — تكلفة ليست صفرية")
    if p.deployment_memory_gb > 40:
        score -= 15
        warnings.append("لا قيد حقيقي على الذاكرة — فائدة Distillation أقل إلحاحًا")

    return ScoreResult(Technique.DISTILLATION, max(0, min(100, score)), reasons, warnings)


SCORERS: list[Callable[[ProblemProfile], ScoreResult]] = [
    score_rag, score_lora, score_cpt, score_merging, score_distillation
]


# ------------------------------------------------------------------
# 3) محرك القرار الرئيسي
# ------------------------------------------------------------------

def recommend(profile: ProblemProfile, top_k: int = 3) -> list[ScoreResult]:
    """يُرجع أفضل k تقنيات مرتبة تنازليًا بالدرجة."""
    results = [scorer(profile) for scorer in SCORERS]
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def print_recommendation(profile: ProblemProfile) -> None:
    print("=" * 70)
    print("تحليل المشكلة وتوصية الحل الاستراتيجي")
    print("=" * 70)
    results = recommend(profile, top_k=5)
    for i, r in enumerate(results, 1):
        bar = "█" * int(r.score / 5)
        print(f"\n{i}. {r.technique.value}  [{r.score:.0f}/100]  {bar}")
        for reason in r.reasons:
            print(f"   ✓ {reason}")
        for warning in r.warnings:
            print(f"   ⚠ {warning}")
    print("\n" + "=" * 70)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-32-chapter-32.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ------------------------------------------------------------------
# السيناريو 1: شات بوت دعم فني يجيب عن سياسات شركة تتغيّر أسبوعيًا
# ------------------------------------------------------------------
scenario_1 = ProblemProfile(
    knowledge_type=KnowledgeType.VOLATILE_FACTS,
    data_volume_examples=0,
    update_frequency_days=7,
    traceability_required=True,
    latency_budget_ms=800,
    deployment_memory_gb=16,
    compute_budget_gpu_hours=20,
)
print_recommendation(scenario_1)
# التوقع: RAG يتصدّر بفارق كبير

# ------------------------------------------------------------------
# السيناريو 2: نموذج يجب أن يجيب دائمًا بصيغة JSON منظّمة
# بأسلوب علامة تجارية محدّدة (توجد 3000 مثال موسوم)
# ------------------------------------------------------------------
scenario_2 = ProblemProfile(
    knowledge_type=KnowledgeType.BEHAVIOR_STYLE,
    data_volume_examples=3000,
    update_frequency_days=365,
    traceability_required=False,
    latency_budget_ms=500,
    deployment_memory_gb=24,
    compute_budget_gpu_hours=30,
)
print_recommendation(scenario_2)
# التوقع: LoRA يتصدّر

# ------------------------------------------------------------------
# السيناريو 3: بناء نموذج يفهم لهجة طبية عربية متخصصة نادرة
# (2 مليار توكن من نصوص طبية عربية، ميزانية حوسبة كبيرة)
# ------------------------------------------------------------------
scenario_3 = ProblemProfile(
    knowledge_type=KnowledgeType.STABLE_DOMAIN,
    data_volume_examples=2_000_000_000,
    update_frequency_days=730,
    traceability_required=False,
    latency_budget_ms=1000,
    deployment_memory_gb=40,
    compute_budget_gpu_hours=2000,
    team_ml_expertise_level=5,
)
print_recommendation(scenario_3)
# التوقع: Continued Pretraining يتصدّر

# ------------------------------------------------------------------
# السيناريو 4: لدينا 3 نماذج متخصصة (كود، رياضيات، عربي) جاهزة
# نريد نموذجًا واحدًا يجمع الثلاث قدرات دون تدريب
# ------------------------------------------------------------------
scenario_4 = ProblemProfile(
    knowledge_type=KnowledgeType.COMBINED_CAPABILITY,
    data_volume_examples=0,
    update_frequency_days=float("inf"),
    traceability_required=False,
    latency_budget_ms=500,
    deployment_memory_gb=24,
    compute_budget_gpu_hours=5,
    has_pretrained_specialist_models=True,
)
print_recommendation(scenario_4)
# التوقع: Model Merging يتصدّر

# ------------------------------------------------------------------
# السيناريو 5: نموذج ممتاز الأداء لكن يجب أن يعمل على هاتف محمول
# ------------------------------------------------------------------
scenario_5 = ProblemProfile(
    knowledge_type=KnowledgeType.REASONING_SKILL,
    data_volume_examples=5000,
    update_frequency_days=365,
    traceability_required=False,
    latency_budget_ms=50,
    deployment_memory_gb=2,
    compute_budget_gpu_hours=150,
)
print_recommendation(scenario_5)
# التوقع: Distillation يتصدّر

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-32-chapter-32.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
@dataclass
class HybridRecommendation:
    primary: Technique
    complementary: list[Technique]
    rationale: str


HYBRID_PATTERNS: dict[tuple[KnowledgeType, bool], HybridRecommendation] = {
    (KnowledgeType.VOLATILE_FACTS, True): HybridRecommendation(
        primary=Technique.RAG,
        complementary=[Technique.LORA],
        rationale=(
            "RAG يجلب الحقائق المحدّثة، بينما LoRA خفيف يُدرَّب على كيفية "
            "استخدام هذا السياق بأسلوب الإجابة المطلوب (تنسيق، نبرة، لغة الاستشهاد)"
        ),
    ),
    (KnowledgeType.COMBINED_CAPABILITY, True): HybridRecommendation(
        primary=Technique.MERGING,
        complementary=[Technique.DISTILLATION],
        rationale=(
            "ادمج النماذج المتخصصة أولًا بـ TIES/DARE، ثم قطّر الناتج "
            "إلى نموذج أصغر إذا كانت الميزانية التشغيلية (Inference) هي القيد النهائي"
        ),
    ),
    (KnowledgeType.STABLE_DOMAIN, True): HybridRecommendation(
        primary=Technique.CPT,
        complementary=[Technique.LORA, Technique.RAG],
        rationale=(
            "CPT يبني الأساس المعرفي العميق للمجال، ثم LoRA يضبط المهام "
            "النهائية (Instruction Following)، ثم RAG يغطي أي تحديثات لاحقة"
        ),
    ),
}
