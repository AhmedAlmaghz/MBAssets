# -*- coding: utf-8 -*-
"""
from dataclasses import dataclass

مقتطف من الفصل 13 — «المراقبة العميقة والتتبع الموزع وبث الأحداث».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch13_observability_sse.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, Field

# نفترض استيراد SpanKind و TelemetrySpan من الفصل الثالث عشر
from enum import StrEnum
from typing import Any


class SpanKind(StrEnum):
    AGENT_RUN = "agent.run"
    REASONING_TURN = "agent.turn"
    LLM_GENERATION = "gen_ai.chat"
    TOOL_EXECUTION = "agent.tool"


class TelemetrySpan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: SpanKind
    duration_ms: float = Field(ge=0.0)
    attributes: dict[str, Any] = Field(default_factory=dict)


class TenantBudgetExhaustedError(RuntimeError):
    """يُرفع فوراً عندما يتجاوز المستأجر السقف المالي اليومي المسموح به."""


class TenantFinOpsReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    total_input_tokens: int = Field(ge=0)
    total_cached_tokens: int = Field(ge=0)
    total_cost_usd: float = Field(ge=0.0)
    overall_cache_hit_ratio: float = Field(ge=0.0, le=1.0)
    cache_degradation_warning: bool = Field(
        description="تنبيه عند استهلاك >= 1000 توكن بنسبة إصابة كاش أقل من 50%"
    )
    budget_exceeded: bool


@dataclass
class _TenantAccumulator:
    input_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0


class FinOpsTelemetryGuard:
    """
    حارس مالي لحظي (Real-Time FinOps Guard) يحلل فواصل OpenTelemetry
    لكشف تدهور الكاش وإيقاف نزيف الميزانية لكل مستأجر.
    """

    def __init__(self, tenant_daily_budget_usd: float = 5.0) -> None:
        if tenant_daily_budget_usd <= 0:
            raise ValueError("يجب أن تكون الميزانية اليومية أكبر من الصفر.")
        self._daily_budget_usd = tenant_daily_budget_usd
        self._ledger: dict[str, _TenantAccumulator] = {}

    def ingest_span(
        self, tenant_id: str, span: TelemetrySpan
    ) -> TenantFinOpsReport:
        acc = self._ledger.setdefault(tenant_id, _TenantAccumulator())

        # لا نراكم سوى الفواصل الزمنية الخاصة باستدعاءات النماذج اللغوية
        if span.kind == SpanKind.LLM_GENERATION:
            acc.input_tokens += int(
                span.attributes.get("gen_ai.usage.input_tokens", 0)
            )
            acc.cached_tokens += int(
                span.attributes.get("gen_ai.usage.cached_tokens", 0)
            )
            acc.cost_usd = round(
                acc.cost_usd + float(span.attributes.get("gen_ai.cost.usd", 0.0)),
                6,
            )

        hit_ratio = (
            round(acc.cached_tokens / acc.input_tokens, 4)
            if acc.input_tokens > 0
            else 0.0
        )
        # كشف تدهور الكاش إذا تجاوز الاستهلاك 1000 توكن وهبطت النسبة عن 50%
        degraded = (acc.input_tokens >= 1000) and (hit_ratio < 0.50)
        exceeded = acc.cost_usd > self._daily_budget_usd

        report = TenantFinOpsReport(
            tenant_id=tenant_id,
            total_input_tokens=acc.input_tokens,
            total_cached_tokens=acc.cached_tokens,
            total_cost_usd=acc.cost_usd,
            overall_cache_hit_ratio=hit_ratio,
            cache_degradation_warning=degraded,
            budget_exceeded=exceeded,
        )

        if exceeded:
            raise TenantBudgetExhaustedError(
                f"تم إيقاف الوكيل للمستأجر '{tenant_id}': التكلفة التراكمية "
                f"(${acc.cost_usd:.4f}) تجاوزت السقف اليومي (${self._daily_budget_usd:.4f})."
            )

        return report
