#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — هندسة التكلفة — Self-Hosting مقابل API، والتوجيه الذكي للنماذج
Large Language Model Engineering — Self-Hosting

Book slug : llm-engineering-69-chapters
Chapter   : ch-64-chapter-64 (lesson 64)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-64-chapter-64.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — هندسة التكلفة — Self-Hosting مقابل API، والتوجيه الذكي للنماذج
Large Language Model Engineering — Self-Hosting

Book slug : llm-engineering-69-chapters
Chapter   : ch-64-chapter-64 (lesson 64)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-64-chapter-64.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — هندسة التكلفة — Self-Hosting مقابل API، والتوجيه الذكي للنماذج
Large Language Model Engineering — Self-Hosting

Book slug : llm-engineering-69-chapters
Chapter   : ch-64-chapter-64 (lesson 64)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_64.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 64: الفصل 64: هندسة التكلفة — Self-Hosting مقابل API، والتوجيه الذكي للنماذج."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_cost() -> None:
    options = [{'name': 'local-small', 'input_per_m': 0.2, 'output_per_m': 0.4, 'latency': 0.12}, {'name': 'hosted-large', 'input_per_m': 2.0, 'output_per_m': 8.0, 'latency': 0.8}]
    tokens_in, tokens_out = (1000, 300)
    budget, latency_sla = (0.01, 0.5)
    for option in options:
        cost = (tokens_in * option['input_per_m'] + tokens_out * option['output_per_m']) / 1000000.0
        eligible = cost <= budget and option['latency'] <= latency_sla
        print(option['name'], 'cost:', round(cost, 6), '| eligible:', eligible)

if __name__ == "__main__":
    print("Chapter 64 lab: Route requests using latency and cost budgets")
    run_cost()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-11-deployment-llmops/chapter-64.md
# ────────────────────────────────────────────────────────────────────────
"""
cost_calculator.py
حاسبة تكلفة شاملة تقارن Self-Hosting مقابل API الخارجي
بناءً على معاملات حقيقية قابلة للتخصيص حسب سيناريو كل شركة.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass
class SelfHostingConfig:
    """معاملات تكلفة الاستضافة الذاتية"""
    num_gpus: int
    gpu_hourly_cost_usd: float          # مثال: H100 على AWS ≈ $4-8/ساعة حسب النوع والمنطقة
    engineering_fte_monthly_cost: float  # تكلفة مهندس LLMOps جزئي/كامل شهريًا
    engineering_fte_fraction: float      # أي جزء من وقت المهندس مخصص لهذا (0.0-1.0)
    overhead_monthly_usd: float = 2000.0  # شبكة، تخزين، مراقبة، نسخ احتياطي
    gpu_utilization_rate: float = 1.0     # نسبة الوقت الفعلي للاستخدام (أهم متغير!)

    @property
    def monthly_hours(self) -> float:
        return 730.0  # متوسط ساعات الشهر

    @property
    def total_monthly_cost(self) -> float:
        gpu_cost = (
            self.num_gpus * self.gpu_hourly_cost_usd *
            self.monthly_hours * self.gpu_utilization_rate
        )
        eng_cost = self.engineering_fte_monthly_cost * self.engineering_fte_fraction
        return gpu_cost + eng_cost + self.overhead_monthly_usd


@dataclass
class APIConfig:
    """معاملات تكلفة استخدام API خارجي (أسعار تقريبية توضيحية — تحقق من المزوّد الفعلي)"""
    price_per_million_input_tokens: float
    price_per_million_output_tokens: float
    avg_input_tokens_per_request: int
    avg_output_tokens_per_request: int

    def monthly_cost(self, monthly_requests: int) -> float:
        input_cost = (
            monthly_requests * self.avg_input_tokens_per_request / 1_000_000
            * self.price_per_million_input_tokens
        )
        output_cost = (
            monthly_requests * self.avg_output_tokens_per_request / 1_000_000
            * self.price_per_million_output_tokens
        )
        return input_cost + output_cost


class CostComparisonEngine:
    """
    محرك مقارنة شامل يحسب نقطة التعادل (Break-even Point)
    ويُولّد توصية واضحة بناءً على الحجم المتوقع للطلبات.
    """

    def __init__(self, self_hosting: SelfHostingConfig, api: APIConfig) -> None:
        self.self_hosting = self_hosting
        self.api = api

    def break_even_requests(self) -> float:
        """عند كم طلب شهري تتساوى التكلفتان؟"""
        cost_per_request = (
            self.api.avg_input_tokens_per_request / 1_000_000 * self.api.price_per_million_input_tokens
            + self.api.avg_output_tokens_per_request / 1_000_000 * self.api.price_per_million_output_tokens
        )
        if cost_per_request == 0:
            return float("inf")
        return self.self_hosting.total_monthly_cost / cost_per_request

    def compare_at_volume(self, monthly_requests: int) -> dict:
        self_cost = self.self_hosting.total_monthly_cost
        api_cost = self.api.monthly_cost(monthly_requests)
        cheaper = "self_hosting" if self_cost < api_cost else "api"
        savings = abs(self_cost - api_cost)

        return {
            "monthly_requests": monthly_requests,
            "self_hosting_cost_usd": round(self_cost, 2),
            "api_cost_usd": round(api_cost, 2),
            "cheaper_option": cheaper,
            "monthly_savings_usd": round(savings, 2),
            "break_even_requests": round(self.break_even_requests()),
        }

    def generate_recommendation(self, projected_monthly_requests: int, growth_rate: float = 0.1) -> str:
        """
        توصية استراتيجية تأخذ بالحسبان ليس فقط التكلفة الحالية
        بل اتجاه النمو المتوقع — قرار ديناميكي وليس لحظيًا فقط.
        """
        current = self.compare_at_volume(projected_monthly_requests)
        future_volume = int(projected_monthly_requests * (1 + growth_rate) ** 6)  # توقع 6 أشهر
        future = self.compare_at_volume(future_volume)

        lines = [
            f"📊 عند الحجم الحالي ({projected_monthly_requests:,} طلب/شهر):",
            f"   Self-Hosting: ${current['self_hosting_cost_usd']:,.2f} | "
            f"API: ${current['api_cost_usd']:,.2f}",
            f"   الأرخص: {current['cheaper_option']} "
            f"(توفير ${current['monthly_savings_usd']:,.2f})",
            "",
            f"📈 عند الحجم المتوقع بعد 6 أشهر ({future_volume:,} طلب/شهر "
            f"بافتراض نمو {growth_rate:.0%} شهريًا):",
            f"   Self-Hosting: ${future['self_hosting_cost_usd']:,.2f} | "
            f"API: ${future['api_cost_usd']:,.2f}",
            f"   الأرخص: {future['cheaper_option']}",
            "",
            f"⚖️ نقطة التعادل: {current['break_even_requests']:,} طلب/شهر",
        ]

        if current['cheaper_option'] != future['cheaper_option']:
            lines.append(
                "\n🔄 تنبيه استراتيجي: التوصية المثلى تتغيّر مع النمو! "
                "خطّط لهجرة تدريجية بدل قرار نهائي الآن."
            )

        return "\n".join(lines)


# ───────────────────────── مثال تطبيقي واقعي ─────────────────────────
if __name__ == "__main__":
    # سيناريو: شركة تفكر باستضافة LLaMA 3 70B على 4x H100
    self_hosting_config = SelfHostingConfig(
        num_gpus=4,
        gpu_hourly_cost_usd=5.5,              # تقريبي لـ H100 على سحابة تجارية
        engineering_fte_monthly_cost=12_000,   # راتب مهندس LLMOps متوسط
        engineering_fte_fraction=0.3,          # 30% من وقته فقط مخصص لهذا النظام
        overhead_monthly_usd=1500,
        gpu_utilization_rate=0.6,              # ⚠️ الأجهزة ليست مُستغَلّة 100% من الوقت!
    )

    api_config = APIConfig(
        price_per_million_input_tokens=3.0,    # مثال توضيحي قريب من أسعار GPT-4-class
        price_per_million_output_tokens=15.0,
        avg_input_tokens_per_request=800,
        avg_output_tokens_per_request=300,
    )

    engine = CostComparisonEngine(self_hosting_config, api_config)

    print(engine.generate_recommendation(projected_monthly_requests=500_000, growth_rate=0.15))
    print("\n" + "=" * 60)
    print("📋 مقارنات عند أحجام مختلفة:")
    for volume in [50_000, 200_000, 500_000, 2_000_000]:
        result = engine.compare_at_volume(volume)
        print(f"  {volume:>10,} طلب/شهر → الأرخص: {result['cheaper_option']:12} "
              f"(توفير ${result['monthly_savings_usd']:>10,.2f})")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-11-deployment-llmops/chapter-64.md
# ────────────────────────────────────────────────────────────────────────
"""
model_cascade_router.py
نظام توجيه هرمي (Cascading) يوجّه كل طلب للنموذج الأرخص الكافي،
ويُصعّد فقط عند عدم الثقة الكافية في الإجابة.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol


class ModelTier(Enum):
    TIER_1_SMALL = "small"      # مثال: نموذج محلي صغير (Phi-3, Gemma-2B) — شبه مجاني
    TIER_2_MEDIUM = "medium"    # مثال: GPT-4o-mini / Claude Haiku — رخيص
    TIER_3_LARGE = "large"       # مثال: GPT-4o / Claude Opus — الأغلى والأقوى


@dataclass
class TierConfig:
    tier: ModelTier
    cost_per_1k_tokens_usd: float
    confidence_threshold: float   # الحد الأدنى للثقة قبل القبول دون تصعيد
    call_fn: Callable[[str], "ModelResponse"]


@dataclass
class ModelResponse:
    text: str
    confidence_score: float       # 0.0-1.0، كيف نحصل عليها؟ (راجع القسم التالي)
    tokens_used: int
    tier_used: ModelTier


@dataclass
class CascadeResult:
    final_response: ModelResponse
    escalation_path: list[ModelTier] = field(default_factory=list)
    total_cost_usd: float = 0.0


class ModelCascadeRouter:
    """
    يوجّه الطلبات عبر سلسلة نماذج متصاعدة التكلفة والقوة.
    يتوقف عند أول Tier يُحقق ثقة كافية، موفرًا التكلفة في الحالات السهلة
    (وهي الأغلبية الساحقة من الطلبات الواقعية في معظم التطبيقات).
    """

    def __init__(self, tiers: list[TierConfig]) -> None:
        # التأكد أن الـ Tiers مرتبة من الأرخص للأغلى
        self.tiers = sorted(tiers, key=lambda t: t.cost_per_1k_tokens_usd)
        self.stats = {tier.tier: 0 for tier in self.tiers}
        self.total_cost_saved_estimate = 0.0

    def route(self, prompt: str) -> CascadeResult:
        escalation_path: list[ModelTier] = []
        total_cost = 0.0

        for i, tier_config in enumerate(self.tiers):
            response = tier_config.call_fn(prompt)
            escalation_path.append(tier_config.tier)

            cost = (response.tokens_used / 1000) * tier_config.cost_per_1k_tokens_usd
            total_cost += cost
            self.stats[tier_config.tier] += 1

            # هل هذا الـ Tier واثق بما يكفي من إجابته؟
            if response.confidence_score >= tier_config.confidence_threshold:
                # حساب التوفير التقديري: ماذا لو ذهبنا مباشرة للـ Tier الأعلى؟
                if i < len(self.tiers) - 1:
                    hypothetical_largest_cost = (
                        response.tokens_used / 1000 * self.tiers[-1].cost_per_1k_tokens_usd
                    )
                    self.total_cost_saved_estimate += (hypothetical_largest_cost - cost)

                return CascadeResult(
                    final_response=response,
                    escalation_path=escalation_path,
                    total_cost_usd=round(total_cost, 6),
                )

            # غير واثق بما يكفي → صعّد للـ Tier التالي (إن وُجد)

        # وصلنا لآخر Tier دون ثقة كافية — نُعيد أفضل ما لدينا على أي حال
        return CascadeResult(
            final_response=response,
            escalation_path=escalation_path,
            total_cost_usd=round(total_cost, 6),
        )

    def report(self) -> str:
        total_requests = sum(self.stats.values())
        lines = [f"📊 توزيع الطلبات عبر {total_requests} استدعاء:"]
        for tier, count in self.stats.items():
            pct = count / total_requests * 100 if total_requests else 0
            lines.append(f"   {tier.value:10}: {count:5} طلب ({pct:.1f}%)")
        lines.append(f"💰 التوفير التقديري الإجمالي: ${self.total_cost_saved_estimate:.4f}")
        return "\n".join(lines)


# ───────────────── محاكاة دوال استدعاء كل Tier ─────────────────
def simulate_small_model_call(prompt: str) -> ModelResponse:
    """نموذج صغير محلي — سريع ورخيص جدًا، لكن ثقته منخفضة للمهام المعقدة"""
    is_simple = len(prompt.split()) < 15  # قاعدة تبسيطية؛ في الواقع نستخدم heuristics أذكى
    confidence = 0.9 if is_simple else 0.4
    return ModelResponse(
        text=f"[Tier1] إجابة سريعة لـ: {prompt[:30]}",
        confidence_score=confidence,
        tokens_used=50,
        tier_used=ModelTier.TIER_1_SMALL,
    )


def simulate_medium_model_call(prompt: str) -> ModelResponse:
    confidence = 0.85 if "معقد جدًا" not in prompt else 0.5
    return ModelResponse(
        text=f"[Tier2] إجابة متوسطة لـ: {prompt[:30]}",
        confidence_score=confidence,
        tokens_used=150,
        tier_used=ModelTier.TIER_2_MEDIUM,
    )


def simulate_large_model_call(prompt: str) -> ModelResponse:
    return ModelResponse(
        text=f"[Tier3] إجابة شاملة وقوية لـ: {prompt[:30]}",
        confidence_score=0.98,  # النموذج الأقوى، ثقة عالية دائمًا تقريبًا
        tokens_used=300,
        tier_used=ModelTier.TIER_3_LARGE,
    )


# ───────────────────────── مثال تطبيقي ─────────────────────────
if __name__ == "__main__":
    router = ModelCascadeRouter(tiers=[
        TierConfig(ModelTier.TIER_1_SMALL, cost_per_1k_tokens_usd=0.0001,
                   confidence_threshold=0.8, call_fn=simulate_small_model_call),
        TierConfig(ModelTier.TIER_2_MEDIUM, cost_per_1k_tokens_usd=0.002,
                   confidence_threshold=0.75, call_fn=simulate_medium_model_call),
        TierConfig(ModelTier.TIER_3_LARGE, cost_per_1k_tokens_usd=0.03,
                   confidence_threshold=0.0, call_fn=simulate_large_model_call),  # آخر محطة، يقبل دائمًا
    ])

    test_prompts = [
        "ما هي ساعات العمل؟",                                    # بسيط → يُحل في Tier 1
        "كيف ألغي اشتراكي في الخطة السنوية؟",                      # متوسط → قد يحتاج Tier 2
        "حلّل لي استراتيجية تسعير معقدة جدًا لمنتج SaaS متعدد الطبقات مع مراعاة المنافسين",  # معقد → Tier 3
    ]

    for prompt in test_prompts:
        result = router.route(prompt)
        print(f"\nالطلب: {prompt[:50]}...")
        print(f"  مسار التصعيد: {[t.value for t in result.escalation_path]}")
        print(f"  التكلفة: ${result.total_cost_usd}")
        print(f"  الإجابة النهائية: {result.final_response.text}")

    print("\n" + router.report())

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-11-deployment-llmops/chapter-64.md
# ────────────────────────────────────────────────────────────────────────
"""
cost_governance_service.py
خدمة حوكمة تكلفة متكاملة: توجيه Cascading + تتبع لحظي + تنبيهات تجاوز الميزانية
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_cascade_router import (
    ModelCascadeRouter, TierConfig, ModelTier,
    simulate_small_model_call, simulate_medium_model_call, simulate_large_model_call,
)


@dataclass
class BudgetTracker:
    """يُراقب الإنفاق التراكمي مقابل ميزانية شهرية محددة، ويُنذر عند الاقتراب من الحد"""
    monthly_budget_usd: float
    current_spend_usd: float = 0.0
    alert_threshold_pct: float = 0.8  # نبّه عند 80% من الميزانية

    def record_spend(self, amount: float) -> str | None:
        self.current_spend_usd += amount
        usage_pct = self.current_spend_usd / self.monthly_budget_usd

        if usage_pct >= 1.0:
            return "🔴 تجاوزت الميزانية الشهرية بالكامل!"
        elif usage_pct >= self.alert_threshold_pct:
            return f"🟡 تحذير: استُهلك {usage_pct:.0%} من الميزانية الشهرية"
        return None

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.monthly_budget_usd - self.current_spend_usd)


class ChatRequest(BaseModel):
    prompt: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.router = ModelCascadeRouter(tiers=[
        TierConfig(ModelTier.TIER_1_SMALL, 0.0001, 0.8, simulate_small_model_call),
        TierConfig(ModelTier.TIER_2_MEDIUM, 0.002, 0.75, simulate_medium_model_call),
        TierConfig(ModelTier.TIER_3_LARGE, 0.03, 0.0, simulate_large_model_call),
    ])
    app.state.budget = BudgetTracker(monthly_budget_usd=5000.0)
    app.state.alerts_log: list[str] = []
    yield


app = FastAPI(title="LLM Cost Governance Service", lifespan=lifespan)


@app.post("/chat")
async def chat(req: ChatRequest) -> dict:
    budget: BudgetTracker = app.state.budget

    # 🚨 حارس صارم: ارفض الطلب تمامًا إذا تجاوزت الميزانية بالكامل
    if budget.remaining_budget <= 0:
        raise HTTPException(
            status_code=429,
            detail="تم تجاوز الميزانية الشهرية المحددة. يرجى مراجعة الفريق المالي.",
        )

    result = app.state.router.route(req.prompt)
    alert = budget.record_spend(result.total_cost_usd)

    if alert:
        app.state.alerts_log.append(f"{datetime.now(timezone.utc).isoformat()} | {alert}")
        print(f"[BUDGET ALERT] {alert}")

    return {
        "response": result.final_response.text,
        "cost_usd": result.total_cost_usd,
        "escalation_path": [t.value for t in result.escalation_path],
        "remaining_budget_usd": round(budget.remaining_budget, 2),
    }


@app.get("/cost-report")
async def cost_report() -> dict:
    """تقرير تنفيذي جاهز للعرض على الإدارة — إجابة مباشرة لسؤال 'أين يذهب المال؟'"""
    router: ModelCascadeRouter = app.state.router
    budget: BudgetTracker = app.state.budget

    total_requests = sum(router.stats.values())
    tier_breakdown = {
        tier.value: {
            "requests": count,
            "pct": round(count / total_requests * 100, 1) if total_requests else 0,
        }
        for tier, count in router.stats.items()
    }

    return {
        "budget_summary": {
            "monthly_budget_usd": budget.monthly_budget_usd,
            "spent_usd": round(budget.current_spend_usd, 2),
            "remaining_usd": round(budget.remaining_budget, 2),
            "usage_pct": round(budget.current_spend_usd / budget.monthly_budget_usd * 100, 1),
        },
        "tier_distribution": tier_breakdown,
        "estimated_cascade_savings_usd": round(router.total_cost_saved_estimate, 2),
        "recent_alerts": app.state.alerts_log[-5:],
    }

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-64-chapter-64.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 64: الفصل 64: هندسة التكلفة — Self-Hosting مقابل API، والتوجيه الذكي للنماذج."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_cost() -> None:
    options = [{'name': 'local-small', 'input_per_m': 0.2, 'output_per_m': 0.4, 'latency': 0.12}, {'name': 'hosted-large', 'input_per_m': 2.0, 'output_per_m': 8.0, 'latency': 0.8}]
    tokens_in, tokens_out = (1000, 300)
    budget, latency_sla = (0.01, 0.5)
    for option in options:
        cost = (tokens_in * option['input_per_m'] + tokens_out * option['output_per_m']) / 1000000.0
        eligible = cost <= budget and option['latency'] <= latency_sla
        print(option['name'], 'cost:', round(cost, 6), '| eligible:', eligible)

if __name__ == "__main__":
    print("Chapter 64 lab: Route requests using latency and cost budgets")
    run_cost()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-64-chapter-64.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
cost_calculator.py
حاسبة تكلفة شاملة تقارن Self-Hosting مقابل API الخارجي
بناءً على معاملات حقيقية قابلة للتخصيص حسب سيناريو كل شركة.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass
class SelfHostingConfig:
    """معاملات تكلفة الاستضافة الذاتية"""
    num_gpus: int
    gpu_hourly_cost_usd: float          # مثال: H100 على AWS ≈ $4-8/ساعة حسب النوع والمنطقة
    engineering_fte_monthly_cost: float  # تكلفة مهندس LLMOps جزئي/كامل شهريًا
    engineering_fte_fraction: float      # أي جزء من وقت المهندس مخصص لهذا (0.0-1.0)
    overhead_monthly_usd: float = 2000.0  # شبكة، تخزين، مراقبة، نسخ احتياطي
    gpu_utilization_rate: float = 1.0     # نسبة الوقت الفعلي للاستخدام (أهم متغير!)

    @property
    def monthly_hours(self) -> float:
        return 730.0  # متوسط ساعات الشهر

    @property
    def total_monthly_cost(self) -> float:
        gpu_cost = (
            self.num_gpus * self.gpu_hourly_cost_usd *
            self.monthly_hours * self.gpu_utilization_rate
        )
        eng_cost = self.engineering_fte_monthly_cost * self.engineering_fte_fraction
        return gpu_cost + eng_cost + self.overhead_monthly_usd


@dataclass
class APIConfig:
    """معاملات تكلفة استخدام API خارجي (أسعار تقريبية توضيحية — تحقق من المزوّد الفعلي)"""
    price_per_million_input_tokens: float
    price_per_million_output_tokens: float
    avg_input_tokens_per_request: int
    avg_output_tokens_per_request: int

    def monthly_cost(self, monthly_requests: int) -> float:
        input_cost = (
            monthly_requests * self.avg_input_tokens_per_request / 1_000_000
            * self.price_per_million_input_tokens
        )
        output_cost = (
            monthly_requests * self.avg_output_tokens_per_request / 1_000_000
            * self.price_per_million_output_tokens
        )
        return input_cost + output_cost


class CostComparisonEngine:
    """
    محرك مقارنة شامل يحسب نقطة التعادل (Break-even Point)
    ويُولّد توصية واضحة بناءً على الحجم المتوقع للطلبات.
    """

    def __init__(self, self_hosting: SelfHostingConfig, api: APIConfig) -> None:
        self.self_hosting = self_hosting
        self.api = api

    def break_even_requests(self) -> float:
        """عند كم طلب شهري تتساوى التكلفتان؟"""
        cost_per_request = (
            self.api.avg_input_tokens_per_request / 1_000_000 * self.api.price_per_million_input_tokens
            + self.api.avg_output_tokens_per_request / 1_000_000 * self.api.price_per_million_output_tokens
        )
        if cost_per_request == 0:
            return float("inf")
        return self.self_hosting.total_monthly_cost / cost_per_request

    def compare_at_volume(self, monthly_requests: int) -> dict:
        self_cost = self.self_hosting.total_monthly_cost
        api_cost = self.api.monthly_cost(monthly_requests)
        cheaper = "self_hosting" if self_cost < api_cost else "api"
        savings = abs(self_cost - api_cost)

        return {
            "monthly_requests": monthly_requests,
            "self_hosting_cost_usd": round(self_cost, 2),
            "api_cost_usd": round(api_cost, 2),
            "cheaper_option": cheaper,
            "monthly_savings_usd": round(savings, 2),
            "break_even_requests": round(self.break_even_requests()),
        }

    def generate_recommendation(self, projected_monthly_requests: int, growth_rate: float = 0.1) -> str:
        """
        توصية استراتيجية تأخذ بالحسبان ليس فقط التكلفة الحالية
        بل اتجاه النمو المتوقع — قرار ديناميكي وليس لحظيًا فقط.
        """
        current = self.compare_at_volume(projected_monthly_requests)
        future_volume = int(projected_monthly_requests * (1 + growth_rate) ** 6)  # توقع 6 أشهر
        future = self.compare_at_volume(future_volume)

        lines = [
            f"📊 عند الحجم الحالي ({projected_monthly_requests:,} طلب/شهر):",
            f"   Self-Hosting: ${current['self_hosting_cost_usd']:,.2f} | "
            f"API: ${current['api_cost_usd']:,.2f}",
            f"   الأرخص: {current['cheaper_option']} "
            f"(توفير ${current['monthly_savings_usd']:,.2f})",
            "",
            f"📈 عند الحجم المتوقع بعد 6 أشهر ({future_volume:,} طلب/شهر "
            f"بافتراض نمو {growth_rate:.0%} شهريًا):",
            f"   Self-Hosting: ${future['self_hosting_cost_usd']:,.2f} | "
            f"API: ${future['api_cost_usd']:,.2f}",
            f"   الأرخص: {future['cheaper_option']}",
            "",
            f"⚖️ نقطة التعادل: {current['break_even_requests']:,} طلب/شهر",
        ]

        if current['cheaper_option'] != future['cheaper_option']:
            lines.append(
                "\n🔄 تنبيه استراتيجي: التوصية المثلى تتغيّر مع النمو! "
                "خطّط لهجرة تدريجية بدل قرار نهائي الآن."
            )

        return "\n".join(lines)


# ───────────────────────── مثال تطبيقي واقعي ─────────────────────────
if __name__ == "__main__":
    # سيناريو: شركة تفكر باستضافة LLaMA 3 70B على 4x H100
    self_hosting_config = SelfHostingConfig(
        num_gpus=4,
        gpu_hourly_cost_usd=5.5,              # تقريبي لـ H100 على سحابة تجارية
        engineering_fte_monthly_cost=12_000,   # راتب مهندس LLMOps متوسط
        engineering_fte_fraction=0.3,          # 30% من وقته فقط مخصص لهذا النظام
        overhead_monthly_usd=1500,
        gpu_utilization_rate=0.6,              # ⚠️ الأجهزة ليست مُستغَلّة 100% من الوقت!
    )

    api_config = APIConfig(
        price_per_million_input_tokens=3.0,    # مثال توضيحي قريب من أسعار GPT-4-class
        price_per_million_output_tokens=15.0,
        avg_input_tokens_per_request=800,
        avg_output_tokens_per_request=300,
    )

    engine = CostComparisonEngine(self_hosting_config, api_config)

    print(engine.generate_recommendation(projected_monthly_requests=500_000, growth_rate=0.15))
    print("\n" + "=" * 60)
    print("📋 مقارنات عند أحجام مختلفة:")
    for volume in [50_000, 200_000, 500_000, 2_000_000]:
        result = engine.compare_at_volume(volume)
        print(f"  {volume:>10,} طلب/شهر → الأرخص: {result['cheaper_option']:12} "
              f"(توفير ${result['monthly_savings_usd']:>10,.2f})")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-64-chapter-64.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
model_cascade_router.py
نظام توجيه هرمي (Cascading) يوجّه كل طلب للنموذج الأرخص الكافي،
ويُصعّد فقط عند عدم الثقة الكافية في الإجابة.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol


class ModelTier(Enum):
    TIER_1_SMALL = "small"      # مثال: نموذج محلي صغير (Phi-3, Gemma-2B) — شبه مجاني
    TIER_2_MEDIUM = "medium"    # مثال: GPT-4o-mini / Claude Haiku — رخيص
    TIER_3_LARGE = "large"       # مثال: GPT-4o / Claude Opus — الأغلى والأقوى


@dataclass
class TierConfig:
    tier: ModelTier
    cost_per_1k_tokens_usd: float
    confidence_threshold: float   # الحد الأدنى للثقة قبل القبول دون تصعيد
    call_fn: Callable[[str], "ModelResponse"]


@dataclass
class ModelResponse:
    text: str
    confidence_score: float       # 0.0-1.0، كيف نحصل عليها؟ (راجع القسم التالي)
    tokens_used: int
    tier_used: ModelTier


@dataclass
class CascadeResult:
    final_response: ModelResponse
    escalation_path: list[ModelTier] = field(default_factory=list)
    total_cost_usd: float = 0.0


class ModelCascadeRouter:
    """
    يوجّه الطلبات عبر سلسلة نماذج متصاعدة التكلفة والقوة.
    يتوقف عند أول Tier يُحقق ثقة كافية، موفرًا التكلفة في الحالات السهلة
    (وهي الأغلبية الساحقة من الطلبات الواقعية في معظم التطبيقات).
    """

    def __init__(self, tiers: list[TierConfig]) -> None:
        # التأكد أن الـ Tiers مرتبة من الأرخص للأغلى
        self.tiers = sorted(tiers, key=lambda t: t.cost_per_1k_tokens_usd)
        self.stats = {tier.tier: 0 for tier in self.tiers}
        self.total_cost_saved_estimate = 0.0

    def route(self, prompt: str) -> CascadeResult:
        escalation_path: list[ModelTier] = []
        total_cost = 0.0

        for i, tier_config in enumerate(self.tiers):
            response = tier_config.call_fn(prompt)
            escalation_path.append(tier_config.tier)

            cost = (response.tokens_used / 1000) * tier_config.cost_per_1k_tokens_usd
            total_cost += cost
            self.stats[tier_config.tier] += 1

            # هل هذا الـ Tier واثق بما يكفي من إجابته؟
            if response.confidence_score >= tier_config.confidence_threshold:
                # حساب التوفير التقديري: ماذا لو ذهبنا مباشرة للـ Tier الأعلى؟
                if i < len(self.tiers) - 1:
                    hypothetical_largest_cost = (
                        response.tokens_used / 1000 * self.tiers[-1].cost_per_1k_tokens_usd
                    )
                    self.total_cost_saved_estimate += (hypothetical_largest_cost - cost)

                return CascadeResult(
                    final_response=response,
                    escalation_path=escalation_path,
                    total_cost_usd=round(total_cost, 6),
                )

            # غير واثق بما يكفي → صعّد للـ Tier التالي (إن وُجد)

        # وصلنا لآخر Tier دون ثقة كافية — نُعيد أفضل ما لدينا على أي حال
        return CascadeResult(
            final_response=response,
            escalation_path=escalation_path,
            total_cost_usd=round(total_cost, 6),
        )

    def report(self) -> str:
        total_requests = sum(self.stats.values())
        lines = [f"📊 توزيع الطلبات عبر {total_requests} استدعاء:"]
        for tier, count in self.stats.items():
            pct = count / total_requests * 100 if total_requests else 0
            lines.append(f"   {tier.value:10}: {count:5} طلب ({pct:.1f}%)")
        lines.append(f"💰 التوفير التقديري الإجمالي: ${self.total_cost_saved_estimate:.4f}")
        return "\n".join(lines)


# ───────────────── محاكاة دوال استدعاء كل Tier ─────────────────
def simulate_small_model_call(prompt: str) -> ModelResponse:
    """نموذج صغير محلي — سريع ورخيص جدًا، لكن ثقته منخفضة للمهام المعقدة"""
    is_simple = len(prompt.split()) < 15  # قاعدة تبسيطية؛ في الواقع نستخدم heuristics أذكى
    confidence = 0.9 if is_simple else 0.4
    return ModelResponse(
        text=f"[Tier1] إجابة سريعة لـ: {prompt[:30]}",
        confidence_score=confidence,
        tokens_used=50,
        tier_used=ModelTier.TIER_1_SMALL,
    )


def simulate_medium_model_call(prompt: str) -> ModelResponse:
    confidence = 0.85 if "معقد جدًا" not in prompt else 0.5
    return ModelResponse(
        text=f"[Tier2] إجابة متوسطة لـ: {prompt[:30]}",
        confidence_score=confidence,
        tokens_used=150,
        tier_used=ModelTier.TIER_2_MEDIUM,
    )


def simulate_large_model_call(prompt: str) -> ModelResponse:
    return ModelResponse(
        text=f"[Tier3] إجابة شاملة وقوية لـ: {prompt[:30]}",
        confidence_score=0.98,  # النموذج الأقوى، ثقة عالية دائمًا تقريبًا
        tokens_used=300,
        tier_used=ModelTier.TIER_3_LARGE,
    )


# ───────────────────────── مثال تطبيقي ─────────────────────────
if __name__ == "__main__":
    router = ModelCascadeRouter(tiers=[
        TierConfig(ModelTier.TIER_1_SMALL, cost_per_1k_tokens_usd=0.0001,
                   confidence_threshold=0.8, call_fn=simulate_small_model_call),
        TierConfig(ModelTier.TIER_2_MEDIUM, cost_per_1k_tokens_usd=0.002,
                   confidence_threshold=0.75, call_fn=simulate_medium_model_call),
        TierConfig(ModelTier.TIER_3_LARGE, cost_per_1k_tokens_usd=0.03,
                   confidence_threshold=0.0, call_fn=simulate_large_model_call),  # آخر محطة، يقبل دائمًا
    ])

    test_prompts = [
        "ما هي ساعات العمل؟",                                    # بسيط → يُحل في Tier 1
        "كيف ألغي اشتراكي في الخطة السنوية؟",                      # متوسط → قد يحتاج Tier 2
        "حلّل لي استراتيجية تسعير معقدة جدًا لمنتج SaaS متعدد الطبقات مع مراعاة المنافسين",  # معقد → Tier 3
    ]

    for prompt in test_prompts:
        result = router.route(prompt)
        print(f"\nالطلب: {prompt[:50]}...")
        print(f"  مسار التصعيد: {[t.value for t in result.escalation_path]}")
        print(f"  التكلفة: ${result.total_cost_usd}")
        print(f"  الإجابة النهائية: {result.final_response.text}")

    print("\n" + router.report())

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-64-chapter-64.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
cost_governance_service.py
خدمة حوكمة تكلفة متكاملة: توجيه Cascading + تتبع لحظي + تنبيهات تجاوز الميزانية
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_cascade_router import (
    ModelCascadeRouter, TierConfig, ModelTier,
    simulate_small_model_call, simulate_medium_model_call, simulate_large_model_call,
)


@dataclass
class BudgetTracker:
    """يُراقب الإنفاق التراكمي مقابل ميزانية شهرية محددة، ويُنذر عند الاقتراب من الحد"""
    monthly_budget_usd: float
    current_spend_usd: float = 0.0
    alert_threshold_pct: float = 0.8  # نبّه عند 80% من الميزانية

    def record_spend(self, amount: float) -> str | None:
        self.current_spend_usd += amount
        usage_pct = self.current_spend_usd / self.monthly_budget_usd

        if usage_pct >= 1.0:
            return "🔴 تجاوزت الميزانية الشهرية بالكامل!"
        elif usage_pct >= self.alert_threshold_pct:
            return f"🟡 تحذير: استُهلك {usage_pct:.0%} من الميزانية الشهرية"
        return None

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.monthly_budget_usd - self.current_spend_usd)


class ChatRequest(BaseModel):
    prompt: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.router = ModelCascadeRouter(tiers=[
        TierConfig(ModelTier.TIER_1_SMALL, 0.0001, 0.8, simulate_small_model_call),
        TierConfig(ModelTier.TIER_2_MEDIUM, 0.002, 0.75, simulate_medium_model_call),
        TierConfig(ModelTier.TIER_3_LARGE, 0.03, 0.0, simulate_large_model_call),
    ])
    app.state.budget = BudgetTracker(monthly_budget_usd=5000.0)
    app.state.alerts_log: list[str] = []
    yield


app = FastAPI(title="LLM Cost Governance Service", lifespan=lifespan)


@app.post("/chat")
async def chat(req: ChatRequest) -> dict:
    budget: BudgetTracker = app.state.budget

    # 🚨 حارس صارم: ارفض الطلب تمامًا إذا تجاوزت الميزانية بالكامل
    if budget.remaining_budget <= 0:
        raise HTTPException(
            status_code=429,
            detail="تم تجاوز الميزانية الشهرية المحددة. يرجى مراجعة الفريق المالي.",
        )

    result = app.state.router.route(req.prompt)
    alert = budget.record_spend(result.total_cost_usd)

    if alert:
        app.state.alerts_log.append(f"{datetime.now(timezone.utc).isoformat()} | {alert}")
        print(f"[BUDGET ALERT] {alert}")

    return {
        "response": result.final_response.text,
        "cost_usd": result.total_cost_usd,
        "escalation_path": [t.value for t in result.escalation_path],
        "remaining_budget_usd": round(budget.remaining_budget, 2),
    }


@app.get("/cost-report")
async def cost_report() -> dict:
    """تقرير تنفيذي جاهز للعرض على الإدارة — إجابة مباشرة لسؤال 'أين يذهب المال؟'"""
    router: ModelCascadeRouter = app.state.router
    budget: BudgetTracker = app.state.budget

    total_requests = sum(router.stats.values())
    tier_breakdown = {
        tier.value: {
            "requests": count,
            "pct": round(count / total_requests * 100, 1) if total_requests else 0,
        }
        for tier, count in router.stats.items()
    }

    return {
        "budget_summary": {
            "monthly_budget_usd": budget.monthly_budget_usd,
            "spent_usd": round(budget.current_spend_usd, 2),
            "remaining_usd": round(budget.remaining_budget, 2),
            "usage_pct": round(budget.current_spend_usd / budget.monthly_budget_usd * 100, 1),
        },
        "tier_distribution": tier_breakdown,
        "estimated_cascade_savings_usd": round(router.total_cost_saved_estimate, 2),
        "recent_alerts": app.state.alerts_log[-5:],
    }

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-64-chapter-64.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — هندسة التكلفة — Self-Hosting مقابل API، والتوجيه الذكي للنماذج
Large Language Model Engineering — Self-Hosting

Book slug : llm-engineering-69-chapters
Chapter   : ch-64-chapter-64 (lesson 64)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_64.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 64: الفصل 64: هندسة التكلفة — Self-Hosting مقابل API، والتوجيه الذكي للنماذج."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_cost() -> None:
    options = [{'name': 'local-small', 'input_per_m': 0.2, 'output_per_m': 0.4, 'latency': 0.12}, {'name': 'hosted-large', 'input_per_m': 2.0, 'output_per_m': 8.0, 'latency': 0.8}]
    tokens_in, tokens_out = (1000, 300)
    budget, latency_sla = (0.01, 0.5)
    for option in options:
        cost = (tokens_in * option['input_per_m'] + tokens_out * option['output_per_m']) / 1000000.0
        eligible = cost <= budget and option['latency'] <= latency_sla
        print(option['name'], 'cost:', round(cost, 6), '| eligible:', eligible)

if __name__ == "__main__":
    print("Chapter 64 lab: Route requests using latency and cost budgets")
    run_cost()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-11-deployment-llmops/chapter-64.md
# ────────────────────────────────────────────────────────────────────────
"""
cost_calculator.py
حاسبة تكلفة شاملة تقارن Self-Hosting مقابل API الخارجي
بناءً على معاملات حقيقية قابلة للتخصيص حسب سيناريو كل شركة.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass
class SelfHostingConfig:
    """معاملات تكلفة الاستضافة الذاتية"""
    num_gpus: int
    gpu_hourly_cost_usd: float          # مثال: H100 على AWS ≈ $4-8/ساعة حسب النوع والمنطقة
    engineering_fte_monthly_cost: float  # تكلفة مهندس LLMOps جزئي/كامل شهريًا
    engineering_fte_fraction: float      # أي جزء من وقت المهندس مخصص لهذا (0.0-1.0)
    overhead_monthly_usd: float = 2000.0  # شبكة، تخزين، مراقبة، نسخ احتياطي
    gpu_utilization_rate: float = 1.0     # نسبة الوقت الفعلي للاستخدام (أهم متغير!)

    @property
    def monthly_hours(self) -> float:
        return 730.0  # متوسط ساعات الشهر

    @property
    def total_monthly_cost(self) -> float:
        gpu_cost = (
            self.num_gpus * self.gpu_hourly_cost_usd *
            self.monthly_hours * self.gpu_utilization_rate
        )
        eng_cost = self.engineering_fte_monthly_cost * self.engineering_fte_fraction
        return gpu_cost + eng_cost + self.overhead_monthly_usd


@dataclass
class APIConfig:
    """معاملات تكلفة استخدام API خارجي (أسعار تقريبية توضيحية — تحقق من المزوّد الفعلي)"""
    price_per_million_input_tokens: float
    price_per_million_output_tokens: float
    avg_input_tokens_per_request: int
    avg_output_tokens_per_request: int

    def monthly_cost(self, monthly_requests: int) -> float:
        input_cost = (
            monthly_requests * self.avg_input_tokens_per_request / 1_000_000
            * self.price_per_million_input_tokens
        )
        output_cost = (
            monthly_requests * self.avg_output_tokens_per_request / 1_000_000
            * self.price_per_million_output_tokens
        )
        return input_cost + output_cost


class CostComparisonEngine:
    """
    محرك مقارنة شامل يحسب نقطة التعادل (Break-even Point)
    ويُولّد توصية واضحة بناءً على الحجم المتوقع للطلبات.
    """

    def __init__(self, self_hosting: SelfHostingConfig, api: APIConfig) -> None:
        self.self_hosting = self_hosting
        self.api = api

    def break_even_requests(self) -> float:
        """عند كم طلب شهري تتساوى التكلفتان؟"""
        cost_per_request = (
            self.api.avg_input_tokens_per_request / 1_000_000 * self.api.price_per_million_input_tokens
            + self.api.avg_output_tokens_per_request / 1_000_000 * self.api.price_per_million_output_tokens
        )
        if cost_per_request == 0:
            return float("inf")
        return self.self_hosting.total_monthly_cost / cost_per_request

    def compare_at_volume(self, monthly_requests: int) -> dict:
        self_cost = self.self_hosting.total_monthly_cost
        api_cost = self.api.monthly_cost(monthly_requests)
        cheaper = "self_hosting" if self_cost < api_cost else "api"
        savings = abs(self_cost - api_cost)

        return {
            "monthly_requests": monthly_requests,
            "self_hosting_cost_usd": round(self_cost, 2),
            "api_cost_usd": round(api_cost, 2),
            "cheaper_option": cheaper,
            "monthly_savings_usd": round(savings, 2),
            "break_even_requests": round(self.break_even_requests()),
        }

    def generate_recommendation(self, projected_monthly_requests: int, growth_rate: float = 0.1) -> str:
        """
        توصية استراتيجية تأخذ بالحسبان ليس فقط التكلفة الحالية
        بل اتجاه النمو المتوقع — قرار ديناميكي وليس لحظيًا فقط.
        """
        current = self.compare_at_volume(projected_monthly_requests)
        future_volume = int(projected_monthly_requests * (1 + growth_rate) ** 6)  # توقع 6 أشهر
        future = self.compare_at_volume(future_volume)

        lines = [
            f"📊 عند الحجم الحالي ({projected_monthly_requests:,} طلب/شهر):",
            f"   Self-Hosting: ${current['self_hosting_cost_usd']:,.2f} | "
            f"API: ${current['api_cost_usd']:,.2f}",
            f"   الأرخص: {current['cheaper_option']} "
            f"(توفير ${current['monthly_savings_usd']:,.2f})",
            "",
            f"📈 عند الحجم المتوقع بعد 6 أشهر ({future_volume:,} طلب/شهر "
            f"بافتراض نمو {growth_rate:.0%} شهريًا):",
            f"   Self-Hosting: ${future['self_hosting_cost_usd']:,.2f} | "
            f"API: ${future['api_cost_usd']:,.2f}",
            f"   الأرخص: {future['cheaper_option']}",
            "",
            f"⚖️ نقطة التعادل: {current['break_even_requests']:,} طلب/شهر",
        ]

        if current['cheaper_option'] != future['cheaper_option']:
            lines.append(
                "\n🔄 تنبيه استراتيجي: التوصية المثلى تتغيّر مع النمو! "
                "خطّط لهجرة تدريجية بدل قرار نهائي الآن."
            )

        return "\n".join(lines)


# ───────────────────────── مثال تطبيقي واقعي ─────────────────────────
if __name__ == "__main__":
    # سيناريو: شركة تفكر باستضافة LLaMA 3 70B على 4x H100
    self_hosting_config = SelfHostingConfig(
        num_gpus=4,
        gpu_hourly_cost_usd=5.5,              # تقريبي لـ H100 على سحابة تجارية
        engineering_fte_monthly_cost=12_000,   # راتب مهندس LLMOps متوسط
        engineering_fte_fraction=0.3,          # 30% من وقته فقط مخصص لهذا النظام
        overhead_monthly_usd=1500,
        gpu_utilization_rate=0.6,              # ⚠️ الأجهزة ليست مُستغَلّة 100% من الوقت!
    )

    api_config = APIConfig(
        price_per_million_input_tokens=3.0,    # مثال توضيحي قريب من أسعار GPT-4-class
        price_per_million_output_tokens=15.0,
        avg_input_tokens_per_request=800,
        avg_output_tokens_per_request=300,
    )

    engine = CostComparisonEngine(self_hosting_config, api_config)

    print(engine.generate_recommendation(projected_monthly_requests=500_000, growth_rate=0.15))
    print("\n" + "=" * 60)
    print("📋 مقارنات عند أحجام مختلفة:")
    for volume in [50_000, 200_000, 500_000, 2_000_000]:
        result = engine.compare_at_volume(volume)
        print(f"  {volume:>10,} طلب/شهر → الأرخص: {result['cheaper_option']:12} "
              f"(توفير ${result['monthly_savings_usd']:>10,.2f})")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-11-deployment-llmops/chapter-64.md
# ────────────────────────────────────────────────────────────────────────
"""
model_cascade_router.py
نظام توجيه هرمي (Cascading) يوجّه كل طلب للنموذج الأرخص الكافي،
ويُصعّد فقط عند عدم الثقة الكافية في الإجابة.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol


class ModelTier(Enum):
    TIER_1_SMALL = "small"      # مثال: نموذج محلي صغير (Phi-3, Gemma-2B) — شبه مجاني
    TIER_2_MEDIUM = "medium"    # مثال: GPT-4o-mini / Claude Haiku — رخيص
    TIER_3_LARGE = "large"       # مثال: GPT-4o / Claude Opus — الأغلى والأقوى


@dataclass
class TierConfig:
    tier: ModelTier
    cost_per_1k_tokens_usd: float
    confidence_threshold: float   # الحد الأدنى للثقة قبل القبول دون تصعيد
    call_fn: Callable[[str], "ModelResponse"]


@dataclass
class ModelResponse:
    text: str
    confidence_score: float       # 0.0-1.0، كيف نحصل عليها؟ (راجع القسم التالي)
    tokens_used: int
    tier_used: ModelTier


@dataclass
class CascadeResult:
    final_response: ModelResponse
    escalation_path: list[ModelTier] = field(default_factory=list)
    total_cost_usd: float = 0.0


class ModelCascadeRouter:
    """
    يوجّه الطلبات عبر سلسلة نماذج متصاعدة التكلفة والقوة.
    يتوقف عند أول Tier يُحقق ثقة كافية، موفرًا التكلفة في الحالات السهلة
    (وهي الأغلبية الساحقة من الطلبات الواقعية في معظم التطبيقات).
    """

    def __init__(self, tiers: list[TierConfig]) -> None:
        # التأكد أن الـ Tiers مرتبة من الأرخص للأغلى
        self.tiers = sorted(tiers, key=lambda t: t.cost_per_1k_tokens_usd)
        self.stats = {tier.tier: 0 for tier in self.tiers}
        self.total_cost_saved_estimate = 0.0

    def route(self, prompt: str) -> CascadeResult:
        escalation_path: list[ModelTier] = []
        total_cost = 0.0

        for i, tier_config in enumerate(self.tiers):
            response = tier_config.call_fn(prompt)
            escalation_path.append(tier_config.tier)

            cost = (response.tokens_used / 1000) * tier_config.cost_per_1k_tokens_usd
            total_cost += cost
            self.stats[tier_config.tier] += 1

            # هل هذا الـ Tier واثق بما يكفي من إجابته؟
            if response.confidence_score >= tier_config.confidence_threshold:
                # حساب التوفير التقديري: ماذا لو ذهبنا مباشرة للـ Tier الأعلى؟
                if i < len(self.tiers) - 1:
                    hypothetical_largest_cost = (
                        response.tokens_used / 1000 * self.tiers[-1].cost_per_1k_tokens_usd
                    )
                    self.total_cost_saved_estimate += (hypothetical_largest_cost - cost)

                return CascadeResult(
                    final_response=response,
                    escalation_path=escalation_path,
                    total_cost_usd=round(total_cost, 6),
                )

            # غير واثق بما يكفي → صعّد للـ Tier التالي (إن وُجد)

        # وصلنا لآخر Tier دون ثقة كافية — نُعيد أفضل ما لدينا على أي حال
        return CascadeResult(
            final_response=response,
            escalation_path=escalation_path,
            total_cost_usd=round(total_cost, 6),
        )

    def report(self) -> str:
        total_requests = sum(self.stats.values())
        lines = [f"📊 توزيع الطلبات عبر {total_requests} استدعاء:"]
        for tier, count in self.stats.items():
            pct = count / total_requests * 100 if total_requests else 0
            lines.append(f"   {tier.value:10}: {count:5} طلب ({pct:.1f}%)")
        lines.append(f"💰 التوفير التقديري الإجمالي: ${self.total_cost_saved_estimate:.4f}")
        return "\n".join(lines)


# ───────────────── محاكاة دوال استدعاء كل Tier ─────────────────
def simulate_small_model_call(prompt: str) -> ModelResponse:
    """نموذج صغير محلي — سريع ورخيص جدًا، لكن ثقته منخفضة للمهام المعقدة"""
    is_simple = len(prompt.split()) < 15  # قاعدة تبسيطية؛ في الواقع نستخدم heuristics أذكى
    confidence = 0.9 if is_simple else 0.4
    return ModelResponse(
        text=f"[Tier1] إجابة سريعة لـ: {prompt[:30]}",
        confidence_score=confidence,
        tokens_used=50,
        tier_used=ModelTier.TIER_1_SMALL,
    )


def simulate_medium_model_call(prompt: str) -> ModelResponse:
    confidence = 0.85 if "معقد جدًا" not in prompt else 0.5
    return ModelResponse(
        text=f"[Tier2] إجابة متوسطة لـ: {prompt[:30]}",
        confidence_score=confidence,
        tokens_used=150,
        tier_used=ModelTier.TIER_2_MEDIUM,
    )


def simulate_large_model_call(prompt: str) -> ModelResponse:
    return ModelResponse(
        text=f"[Tier3] إجابة شاملة وقوية لـ: {prompt[:30]}",
        confidence_score=0.98,  # النموذج الأقوى، ثقة عالية دائمًا تقريبًا
        tokens_used=300,
        tier_used=ModelTier.TIER_3_LARGE,
    )


# ───────────────────────── مثال تطبيقي ─────────────────────────
if __name__ == "__main__":
    router = ModelCascadeRouter(tiers=[
        TierConfig(ModelTier.TIER_1_SMALL, cost_per_1k_tokens_usd=0.0001,
                   confidence_threshold=0.8, call_fn=simulate_small_model_call),
        TierConfig(ModelTier.TIER_2_MEDIUM, cost_per_1k_tokens_usd=0.002,
                   confidence_threshold=0.75, call_fn=simulate_medium_model_call),
        TierConfig(ModelTier.TIER_3_LARGE, cost_per_1k_tokens_usd=0.03,
                   confidence_threshold=0.0, call_fn=simulate_large_model_call),  # آخر محطة، يقبل دائمًا
    ])

    test_prompts = [
        "ما هي ساعات العمل؟",                                    # بسيط → يُحل في Tier 1
        "كيف ألغي اشتراكي في الخطة السنوية؟",                      # متوسط → قد يحتاج Tier 2
        "حلّل لي استراتيجية تسعير معقدة جدًا لمنتج SaaS متعدد الطبقات مع مراعاة المنافسين",  # معقد → Tier 3
    ]

    for prompt in test_prompts:
        result = router.route(prompt)
        print(f"\nالطلب: {prompt[:50]}...")
        print(f"  مسار التصعيد: {[t.value for t in result.escalation_path]}")
        print(f"  التكلفة: ${result.total_cost_usd}")
        print(f"  الإجابة النهائية: {result.final_response.text}")

    print("\n" + router.report())

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-11-deployment-llmops/chapter-64.md
# ────────────────────────────────────────────────────────────────────────
"""
cost_governance_service.py
خدمة حوكمة تكلفة متكاملة: توجيه Cascading + تتبع لحظي + تنبيهات تجاوز الميزانية
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_cascade_router import (
    ModelCascadeRouter, TierConfig, ModelTier,
    simulate_small_model_call, simulate_medium_model_call, simulate_large_model_call,
)


@dataclass
class BudgetTracker:
    """يُراقب الإنفاق التراكمي مقابل ميزانية شهرية محددة، ويُنذر عند الاقتراب من الحد"""
    monthly_budget_usd: float
    current_spend_usd: float = 0.0
    alert_threshold_pct: float = 0.8  # نبّه عند 80% من الميزانية

    def record_spend(self, amount: float) -> str | None:
        self.current_spend_usd += amount
        usage_pct = self.current_spend_usd / self.monthly_budget_usd

        if usage_pct >= 1.0:
            return "🔴 تجاوزت الميزانية الشهرية بالكامل!"
        elif usage_pct >= self.alert_threshold_pct:
            return f"🟡 تحذير: استُهلك {usage_pct:.0%} من الميزانية الشهرية"
        return None

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.monthly_budget_usd - self.current_spend_usd)


class ChatRequest(BaseModel):
    prompt: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.router = ModelCascadeRouter(tiers=[
        TierConfig(ModelTier.TIER_1_SMALL, 0.0001, 0.8, simulate_small_model_call),
        TierConfig(ModelTier.TIER_2_MEDIUM, 0.002, 0.75, simulate_medium_model_call),
        TierConfig(ModelTier.TIER_3_LARGE, 0.03, 0.0, simulate_large_model_call),
    ])
    app.state.budget = BudgetTracker(monthly_budget_usd=5000.0)
    app.state.alerts_log: list[str] = []
    yield


app = FastAPI(title="LLM Cost Governance Service", lifespan=lifespan)


@app.post("/chat")
async def chat(req: ChatRequest) -> dict:
    budget: BudgetTracker = app.state.budget

    # 🚨 حارس صارم: ارفض الطلب تمامًا إذا تجاوزت الميزانية بالكامل
    if budget.remaining_budget <= 0:
        raise HTTPException(
            status_code=429,
            detail="تم تجاوز الميزانية الشهرية المحددة. يرجى مراجعة الفريق المالي.",
        )

    result = app.state.router.route(req.prompt)
    alert = budget.record_spend(result.total_cost_usd)

    if alert:
        app.state.alerts_log.append(f"{datetime.now(timezone.utc).isoformat()} | {alert}")
        print(f"[BUDGET ALERT] {alert}")

    return {
        "response": result.final_response.text,
        "cost_usd": result.total_cost_usd,
        "escalation_path": [t.value for t in result.escalation_path],
        "remaining_budget_usd": round(budget.remaining_budget, 2),
    }


@app.get("/cost-report")
async def cost_report() -> dict:
    """تقرير تنفيذي جاهز للعرض على الإدارة — إجابة مباشرة لسؤال 'أين يذهب المال؟'"""
    router: ModelCascadeRouter = app.state.router
    budget: BudgetTracker = app.state.budget

    total_requests = sum(router.stats.values())
    tier_breakdown = {
        tier.value: {
            "requests": count,
            "pct": round(count / total_requests * 100, 1) if total_requests else 0,
        }
        for tier, count in router.stats.items()
    }

    return {
        "budget_summary": {
            "monthly_budget_usd": budget.monthly_budget_usd,
            "spent_usd": round(budget.current_spend_usd, 2),
            "remaining_usd": round(budget.remaining_budget, 2),
            "usage_pct": round(budget.current_spend_usd / budget.monthly_budget_usd * 100, 1),
        },
        "tier_distribution": tier_breakdown,
        "estimated_cascade_savings_usd": round(router.total_cost_saved_estimate, 2),
        "recent_alerts": app.state.alerts_log[-5:],
    }

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-64-chapter-64.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 64: الفصل 64: هندسة التكلفة — Self-Hosting مقابل API، والتوجيه الذكي للنماذج."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_cost() -> None:
    options = [{'name': 'local-small', 'input_per_m': 0.2, 'output_per_m': 0.4, 'latency': 0.12}, {'name': 'hosted-large', 'input_per_m': 2.0, 'output_per_m': 8.0, 'latency': 0.8}]
    tokens_in, tokens_out = (1000, 300)
    budget, latency_sla = (0.01, 0.5)
    for option in options:
        cost = (tokens_in * option['input_per_m'] + tokens_out * option['output_per_m']) / 1000000.0
        eligible = cost <= budget and option['latency'] <= latency_sla
        print(option['name'], 'cost:', round(cost, 6), '| eligible:', eligible)

if __name__ == "__main__":
    print("Chapter 64 lab: Route requests using latency and cost budgets")
    run_cost()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-64-chapter-64.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
cost_calculator.py
حاسبة تكلفة شاملة تقارن Self-Hosting مقابل API الخارجي
بناءً على معاملات حقيقية قابلة للتخصيص حسب سيناريو كل شركة.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass
class SelfHostingConfig:
    """معاملات تكلفة الاستضافة الذاتية"""
    num_gpus: int
    gpu_hourly_cost_usd: float          # مثال: H100 على AWS ≈ $4-8/ساعة حسب النوع والمنطقة
    engineering_fte_monthly_cost: float  # تكلفة مهندس LLMOps جزئي/كامل شهريًا
    engineering_fte_fraction: float      # أي جزء من وقت المهندس مخصص لهذا (0.0-1.0)
    overhead_monthly_usd: float = 2000.0  # شبكة، تخزين، مراقبة، نسخ احتياطي
    gpu_utilization_rate: float = 1.0     # نسبة الوقت الفعلي للاستخدام (أهم متغير!)

    @property
    def monthly_hours(self) -> float:
        return 730.0  # متوسط ساعات الشهر

    @property
    def total_monthly_cost(self) -> float:
        gpu_cost = (
            self.num_gpus * self.gpu_hourly_cost_usd *
            self.monthly_hours * self.gpu_utilization_rate
        )
        eng_cost = self.engineering_fte_monthly_cost * self.engineering_fte_fraction
        return gpu_cost + eng_cost + self.overhead_monthly_usd


@dataclass
class APIConfig:
    """معاملات تكلفة استخدام API خارجي (أسعار تقريبية توضيحية — تحقق من المزوّد الفعلي)"""
    price_per_million_input_tokens: float
    price_per_million_output_tokens: float
    avg_input_tokens_per_request: int
    avg_output_tokens_per_request: int

    def monthly_cost(self, monthly_requests: int) -> float:
        input_cost = (
            monthly_requests * self.avg_input_tokens_per_request / 1_000_000
            * self.price_per_million_input_tokens
        )
        output_cost = (
            monthly_requests * self.avg_output_tokens_per_request / 1_000_000
            * self.price_per_million_output_tokens
        )
        return input_cost + output_cost


class CostComparisonEngine:
    """
    محرك مقارنة شامل يحسب نقطة التعادل (Break-even Point)
    ويُولّد توصية واضحة بناءً على الحجم المتوقع للطلبات.
    """

    def __init__(self, self_hosting: SelfHostingConfig, api: APIConfig) -> None:
        self.self_hosting = self_hosting
        self.api = api

    def break_even_requests(self) -> float:
        """عند كم طلب شهري تتساوى التكلفتان؟"""
        cost_per_request = (
            self.api.avg_input_tokens_per_request / 1_000_000 * self.api.price_per_million_input_tokens
            + self.api.avg_output_tokens_per_request / 1_000_000 * self.api.price_per_million_output_tokens
        )
        if cost_per_request == 0:
            return float("inf")
        return self.self_hosting.total_monthly_cost / cost_per_request

    def compare_at_volume(self, monthly_requests: int) -> dict:
        self_cost = self.self_hosting.total_monthly_cost
        api_cost = self.api.monthly_cost(monthly_requests)
        cheaper = "self_hosting" if self_cost < api_cost else "api"
        savings = abs(self_cost - api_cost)

        return {
            "monthly_requests": monthly_requests,
            "self_hosting_cost_usd": round(self_cost, 2),
            "api_cost_usd": round(api_cost, 2),
            "cheaper_option": cheaper,
            "monthly_savings_usd": round(savings, 2),
            "break_even_requests": round(self.break_even_requests()),
        }

    def generate_recommendation(self, projected_monthly_requests: int, growth_rate: float = 0.1) -> str:
        """
        توصية استراتيجية تأخذ بالحسبان ليس فقط التكلفة الحالية
        بل اتجاه النمو المتوقع — قرار ديناميكي وليس لحظيًا فقط.
        """
        current = self.compare_at_volume(projected_monthly_requests)
        future_volume = int(projected_monthly_requests * (1 + growth_rate) ** 6)  # توقع 6 أشهر
        future = self.compare_at_volume(future_volume)

        lines = [
            f"📊 عند الحجم الحالي ({projected_monthly_requests:,} طلب/شهر):",
            f"   Self-Hosting: ${current['self_hosting_cost_usd']:,.2f} | "
            f"API: ${current['api_cost_usd']:,.2f}",
            f"   الأرخص: {current['cheaper_option']} "
            f"(توفير ${current['monthly_savings_usd']:,.2f})",
            "",
            f"📈 عند الحجم المتوقع بعد 6 أشهر ({future_volume:,} طلب/شهر "
            f"بافتراض نمو {growth_rate:.0%} شهريًا):",
            f"   Self-Hosting: ${future['self_hosting_cost_usd']:,.2f} | "
            f"API: ${future['api_cost_usd']:,.2f}",
            f"   الأرخص: {future['cheaper_option']}",
            "",
            f"⚖️ نقطة التعادل: {current['break_even_requests']:,} طلب/شهر",
        ]

        if current['cheaper_option'] != future['cheaper_option']:
            lines.append(
                "\n🔄 تنبيه استراتيجي: التوصية المثلى تتغيّر مع النمو! "
                "خطّط لهجرة تدريجية بدل قرار نهائي الآن."
            )

        return "\n".join(lines)


# ───────────────────────── مثال تطبيقي واقعي ─────────────────────────
if __name__ == "__main__":
    # سيناريو: شركة تفكر باستضافة LLaMA 3 70B على 4x H100
    self_hosting_config = SelfHostingConfig(
        num_gpus=4,
        gpu_hourly_cost_usd=5.5,              # تقريبي لـ H100 على سحابة تجارية
        engineering_fte_monthly_cost=12_000,   # راتب مهندس LLMOps متوسط
        engineering_fte_fraction=0.3,          # 30% من وقته فقط مخصص لهذا النظام
        overhead_monthly_usd=1500,
        gpu_utilization_rate=0.6,              # ⚠️ الأجهزة ليست مُستغَلّة 100% من الوقت!
    )

    api_config = APIConfig(
        price_per_million_input_tokens=3.0,    # مثال توضيحي قريب من أسعار GPT-4-class
        price_per_million_output_tokens=15.0,
        avg_input_tokens_per_request=800,
        avg_output_tokens_per_request=300,
    )

    engine = CostComparisonEngine(self_hosting_config, api_config)

    print(engine.generate_recommendation(projected_monthly_requests=500_000, growth_rate=0.15))
    print("\n" + "=" * 60)
    print("📋 مقارنات عند أحجام مختلفة:")
    for volume in [50_000, 200_000, 500_000, 2_000_000]:
        result = engine.compare_at_volume(volume)
        print(f"  {volume:>10,} طلب/شهر → الأرخص: {result['cheaper_option']:12} "
              f"(توفير ${result['monthly_savings_usd']:>10,.2f})")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-64-chapter-64.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
model_cascade_router.py
نظام توجيه هرمي (Cascading) يوجّه كل طلب للنموذج الأرخص الكافي،
ويُصعّد فقط عند عدم الثقة الكافية في الإجابة.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol


class ModelTier(Enum):
    TIER_1_SMALL = "small"      # مثال: نموذج محلي صغير (Phi-3, Gemma-2B) — شبه مجاني
    TIER_2_MEDIUM = "medium"    # مثال: GPT-4o-mini / Claude Haiku — رخيص
    TIER_3_LARGE = "large"       # مثال: GPT-4o / Claude Opus — الأغلى والأقوى


@dataclass
class TierConfig:
    tier: ModelTier
    cost_per_1k_tokens_usd: float
    confidence_threshold: float   # الحد الأدنى للثقة قبل القبول دون تصعيد
    call_fn: Callable[[str], "ModelResponse"]


@dataclass
class ModelResponse:
    text: str
    confidence_score: float       # 0.0-1.0، كيف نحصل عليها؟ (راجع القسم التالي)
    tokens_used: int
    tier_used: ModelTier


@dataclass
class CascadeResult:
    final_response: ModelResponse
    escalation_path: list[ModelTier] = field(default_factory=list)
    total_cost_usd: float = 0.0


class ModelCascadeRouter:
    """
    يوجّه الطلبات عبر سلسلة نماذج متصاعدة التكلفة والقوة.
    يتوقف عند أول Tier يُحقق ثقة كافية، موفرًا التكلفة في الحالات السهلة
    (وهي الأغلبية الساحقة من الطلبات الواقعية في معظم التطبيقات).
    """

    def __init__(self, tiers: list[TierConfig]) -> None:
        # التأكد أن الـ Tiers مرتبة من الأرخص للأغلى
        self.tiers = sorted(tiers, key=lambda t: t.cost_per_1k_tokens_usd)
        self.stats = {tier.tier: 0 for tier in self.tiers}
        self.total_cost_saved_estimate = 0.0

    def route(self, prompt: str) -> CascadeResult:
        escalation_path: list[ModelTier] = []
        total_cost = 0.0

        for i, tier_config in enumerate(self.tiers):
            response = tier_config.call_fn(prompt)
            escalation_path.append(tier_config.tier)

            cost = (response.tokens_used / 1000) * tier_config.cost_per_1k_tokens_usd
            total_cost += cost
            self.stats[tier_config.tier] += 1

            # هل هذا الـ Tier واثق بما يكفي من إجابته؟
            if response.confidence_score >= tier_config.confidence_threshold:
                # حساب التوفير التقديري: ماذا لو ذهبنا مباشرة للـ Tier الأعلى؟
                if i < len(self.tiers) - 1:
                    hypothetical_largest_cost = (
                        response.tokens_used / 1000 * self.tiers[-1].cost_per_1k_tokens_usd
                    )
                    self.total_cost_saved_estimate += (hypothetical_largest_cost - cost)

                return CascadeResult(
                    final_response=response,
                    escalation_path=escalation_path,
                    total_cost_usd=round(total_cost, 6),
                )

            # غير واثق بما يكفي → صعّد للـ Tier التالي (إن وُجد)

        # وصلنا لآخر Tier دون ثقة كافية — نُعيد أفضل ما لدينا على أي حال
        return CascadeResult(
            final_response=response,
            escalation_path=escalation_path,
            total_cost_usd=round(total_cost, 6),
        )

    def report(self) -> str:
        total_requests = sum(self.stats.values())
        lines = [f"📊 توزيع الطلبات عبر {total_requests} استدعاء:"]
        for tier, count in self.stats.items():
            pct = count / total_requests * 100 if total_requests else 0
            lines.append(f"   {tier.value:10}: {count:5} طلب ({pct:.1f}%)")
        lines.append(f"💰 التوفير التقديري الإجمالي: ${self.total_cost_saved_estimate:.4f}")
        return "\n".join(lines)


# ───────────────── محاكاة دوال استدعاء كل Tier ─────────────────
def simulate_small_model_call(prompt: str) -> ModelResponse:
    """نموذج صغير محلي — سريع ورخيص جدًا، لكن ثقته منخفضة للمهام المعقدة"""
    is_simple = len(prompt.split()) < 15  # قاعدة تبسيطية؛ في الواقع نستخدم heuristics أذكى
    confidence = 0.9 if is_simple else 0.4
    return ModelResponse(
        text=f"[Tier1] إجابة سريعة لـ: {prompt[:30]}",
        confidence_score=confidence,
        tokens_used=50,
        tier_used=ModelTier.TIER_1_SMALL,
    )


def simulate_medium_model_call(prompt: str) -> ModelResponse:
    confidence = 0.85 if "معقد جدًا" not in prompt else 0.5
    return ModelResponse(
        text=f"[Tier2] إجابة متوسطة لـ: {prompt[:30]}",
        confidence_score=confidence,
        tokens_used=150,
        tier_used=ModelTier.TIER_2_MEDIUM,
    )


def simulate_large_model_call(prompt: str) -> ModelResponse:
    return ModelResponse(
        text=f"[Tier3] إجابة شاملة وقوية لـ: {prompt[:30]}",
        confidence_score=0.98,  # النموذج الأقوى، ثقة عالية دائمًا تقريبًا
        tokens_used=300,
        tier_used=ModelTier.TIER_3_LARGE,
    )


# ───────────────────────── مثال تطبيقي ─────────────────────────
if __name__ == "__main__":
    router = ModelCascadeRouter(tiers=[
        TierConfig(ModelTier.TIER_1_SMALL, cost_per_1k_tokens_usd=0.0001,
                   confidence_threshold=0.8, call_fn=simulate_small_model_call),
        TierConfig(ModelTier.TIER_2_MEDIUM, cost_per_1k_tokens_usd=0.002,
                   confidence_threshold=0.75, call_fn=simulate_medium_model_call),
        TierConfig(ModelTier.TIER_3_LARGE, cost_per_1k_tokens_usd=0.03,
                   confidence_threshold=0.0, call_fn=simulate_large_model_call),  # آخر محطة، يقبل دائمًا
    ])

    test_prompts = [
        "ما هي ساعات العمل؟",                                    # بسيط → يُحل في Tier 1
        "كيف ألغي اشتراكي في الخطة السنوية؟",                      # متوسط → قد يحتاج Tier 2
        "حلّل لي استراتيجية تسعير معقدة جدًا لمنتج SaaS متعدد الطبقات مع مراعاة المنافسين",  # معقد → Tier 3
    ]

    for prompt in test_prompts:
        result = router.route(prompt)
        print(f"\nالطلب: {prompt[:50]}...")
        print(f"  مسار التصعيد: {[t.value for t in result.escalation_path]}")
        print(f"  التكلفة: ${result.total_cost_usd}")
        print(f"  الإجابة النهائية: {result.final_response.text}")

    print("\n" + router.report())

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-64-chapter-64.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
"""
cost_governance_service.py
خدمة حوكمة تكلفة متكاملة: توجيه Cascading + تتبع لحظي + تنبيهات تجاوز الميزانية
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_cascade_router import (
    ModelCascadeRouter, TierConfig, ModelTier,
    simulate_small_model_call, simulate_medium_model_call, simulate_large_model_call,
)


@dataclass
class BudgetTracker:
    """يُراقب الإنفاق التراكمي مقابل ميزانية شهرية محددة، ويُنذر عند الاقتراب من الحد"""
    monthly_budget_usd: float
    current_spend_usd: float = 0.0
    alert_threshold_pct: float = 0.8  # نبّه عند 80% من الميزانية

    def record_spend(self, amount: float) -> str | None:
        self.current_spend_usd += amount
        usage_pct = self.current_spend_usd / self.monthly_budget_usd

        if usage_pct >= 1.0:
            return "🔴 تجاوزت الميزانية الشهرية بالكامل!"
        elif usage_pct >= self.alert_threshold_pct:
            return f"🟡 تحذير: استُهلك {usage_pct:.0%} من الميزانية الشهرية"
        return None

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.monthly_budget_usd - self.current_spend_usd)


class ChatRequest(BaseModel):
    prompt: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.router = ModelCascadeRouter(tiers=[
        TierConfig(ModelTier.TIER_1_SMALL, 0.0001, 0.8, simulate_small_model_call),
        TierConfig(ModelTier.TIER_2_MEDIUM, 0.002, 0.75, simulate_medium_model_call),
        TierConfig(ModelTier.TIER_3_LARGE, 0.03, 0.0, simulate_large_model_call),
    ])
    app.state.budget = BudgetTracker(monthly_budget_usd=5000.0)
    app.state.alerts_log: list[str] = []
    yield


app = FastAPI(title="LLM Cost Governance Service", lifespan=lifespan)


@app.post("/chat")
async def chat(req: ChatRequest) -> dict:
    budget: BudgetTracker = app.state.budget

    # 🚨 حارس صارم: ارفض الطلب تمامًا إذا تجاوزت الميزانية بالكامل
    if budget.remaining_budget <= 0:
        raise HTTPException(
            status_code=429,
            detail="تم تجاوز الميزانية الشهرية المحددة. يرجى مراجعة الفريق المالي.",
        )

    result = app.state.router.route(req.prompt)
    alert = budget.record_spend(result.total_cost_usd)

    if alert:
        app.state.alerts_log.append(f"{datetime.now(timezone.utc).isoformat()} | {alert}")
        print(f"[BUDGET ALERT] {alert}")

    return {
        "response": result.final_response.text,
        "cost_usd": result.total_cost_usd,
        "escalation_path": [t.value for t in result.escalation_path],
        "remaining_budget_usd": round(budget.remaining_budget, 2),
    }


@app.get("/cost-report")
async def cost_report() -> dict:
    """تقرير تنفيذي جاهز للعرض على الإدارة — إجابة مباشرة لسؤال 'أين يذهب المال؟'"""
    router: ModelCascadeRouter = app.state.router
    budget: BudgetTracker = app.state.budget

    total_requests = sum(router.stats.values())
    tier_breakdown = {
        tier.value: {
            "requests": count,
            "pct": round(count / total_requests * 100, 1) if total_requests else 0,
        }
        for tier, count in router.stats.items()
    }

    return {
        "budget_summary": {
            "monthly_budget_usd": budget.monthly_budget_usd,
            "spent_usd": round(budget.current_spend_usd, 2),
            "remaining_usd": round(budget.remaining_budget, 2),
            "usage_pct": round(budget.current_spend_usd / budget.monthly_budget_usd * 100, 1),
        },
        "tier_distribution": tier_breakdown,
        "estimated_cascade_savings_usd": round(router.total_cost_saved_estimate, 2),
        "recent_alerts": app.state.alerts_log[-5:],
    }
