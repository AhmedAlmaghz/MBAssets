#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
الفصل 13: المراقبة العميقة والتتبع الموزع وبث الأحداث
==============================================================================

البرنامج الكامل للفصل 13 من كتاب «هندسة وكلاء الذكاء الاصطناعي».
المراقبة العميقة والتتبع الموزع وبث الأحداث — Observability, OpenTelemetry & Real-Time SSE.

التشغيل:
    python ch13_observability_sse.py

المتطلبات: Python 3.12+ (الكتاب يستهدف 3.14+) و pydantic>=2.9
"""
# ملاحظة توافق: تُقيَّم الأنواع كسولاً في Python 3.14 (PEP 649)؛
# هذا السطر يمنح السلوك نفسه على Python 3.12/3.13.
from __future__ import annotations

import asyncio
import contextvars
import json
import re
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from enum import StrEnum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


# =====================================================================
# 1. جدار تطهير البيانات الحساسة (Telemetry PII & Secret Scrubber)
# =====================================================================

class PIITelemetryScrubber:
    """
    يطهّر القواميس والنصوص من الأسرار والبيانات الشخصية (PII)
    قبل تسجيلها في التتبع (Spans) أو بثها عبر SSE.
    """

    _SENSITIVE_KEYS: frozenset[str] = frozenset({
        "api_key", "secret", "password", "token", "authorization", "ssn", "cvv"
    })
    _PATTERNS: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"sk-[a-zA-Z0-9]{10,}"), "[REDACTED_API_KEY]"),
        (re.compile(r"Bearer\s+[a-zA-Z0-9\-._~+/]+=*"), "Bearer [REDACTED_TOKEN]"),
        (
            re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
            "[REDACTED_EMAIL]",
        ),
        (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[REDACTED_CARD]"),
    ]

    @classmethod
    def scrub_text(cls, text: str) -> str:
        cleaned = text
        for pattern, replacement in cls._PATTERNS:
            cleaned = pattern.sub(replacement, cleaned)
        return cleaned

    @classmethod
    def scrub_value(cls, value: Any) -> Any:
        if isinstance(value, str):
            return cls.scrub_text(value)
        if isinstance(value, dict):
            return {
                k: (
                    "[REDACTED_SECRET_FIELD]"
                    if str(k).lower() in cls._SENSITIVE_KEYS
                    else cls.scrub_value(v)
                )
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [cls.scrub_value(item) for item in value]
        return value


# =====================================================================
# 2. نماذج التتبع الموزع المتوافقة مع OpenTelemetry GenAI
# =====================================================================

class SpanKind(StrEnum):
    AGENT_RUN = "agent.run"
    REASONING_TURN = "agent.turn"
    LLM_GENERATION = "gen_ai.chat"
    TOOL_EXECUTION = "agent.tool"


class SpanStatus(StrEnum):
    OK = "OK"
    ERROR = "ERROR"


class TelemetrySpan(BaseModel):
    """يمثل فاصل تتبع زمني مهيكل (OpenTelemetry-compatible Span)."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: SpanKind
    start_time_ns: int
    end_time_ns: int
    duration_ms: float = Field(ge=0.0)
    status: SpanStatus = SpanStatus.OK
    attributes: dict[str, Any] = Field(default_factory=dict)


# متغيرات السياق المعزولة (ContextVars) التي تنتقل تلقائياً عبر asyncio.TaskGroup
_current_trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_trace_id", default=None
)
_current_span_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_span_id", default=None
)


class _MutableSpanBuilder:
    """باني داخلي لتجميع بيانات الـ Span أثناء التنفيذ قبل تجميده في TelemetrySpan."""

    def __init__(
        self,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None,
        name: str,
        kind: SpanKind,
        initial_attributes: dict[str, Any] | None = None,
    ) -> None:
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.name = name
        self.kind = kind
        self.start_time_ns = time.perf_counter_ns()
        self.status = SpanStatus.OK
        self.attributes: dict[str, Any] = dict(initial_attributes or {})

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def record_llm_usage(
        self,
        model: str,
        input_tokens: int,
        cached_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        """تسجيل سمات OpenTelemetry GenAI القياسية لاستهلاك التوكنات والكاش."""
        cache_hit_ratio = (
            round(cached_tokens / input_tokens, 4) if input_tokens > 0 else 0.0
        )
        self.attributes.update({
            "gen_ai.request.model": model,
            "gen_ai.usage.input_tokens": input_tokens,
            "gen_ai.usage.cached_tokens": cached_tokens,
            "gen_ai.usage.cache_hit_ratio": cache_hit_ratio,
            "gen_ai.usage.output_tokens": output_tokens,
            "gen_ai.cost.usd": round(cost_usd, 6),
        })

    def finalize(self) -> TelemetrySpan:
        end_ns = time.perf_counter_ns()
        duration_ms = round((end_ns - self.start_time_ns) / 1_000_000, 2)
        scrubbed_attrs = PIITelemetryScrubber.scrub_value(self.attributes)
        return TelemetrySpan(
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            name=self.name,
            kind=self.kind,
            start_time_ns=self.start_time_ns,
            end_time_ns=end_ns,
            duration_ms=duration_ms,
            status=self.status,
            attributes=scrubbed_attrs,
        )


class AgentTracer:
    """
    محرك تتبع موزع يحافظ على شجرة الأبناء والآباء عبر المهام المتوازية
    باستخدام contextvars دون أي تلويث للحالة العامة.
    """

    def __init__(self) -> None:
        self._recorded_spans: list[TelemetrySpan] = []

    @property
    def recorded_spans(self) -> list[TelemetrySpan]:
        return sorted(self._recorded_spans, key=lambda s: s.start_time_ns)

    @asynccontextmanager
    async def start_span(
        self,
        name: str,
        kind: SpanKind,
        attributes: dict[str, Any] | None = None,
    ) -> AsyncGenerator[_MutableSpanBuilder, None]:
        trace_id = _current_trace_id_var.get() or uuid.uuid4().hex[:16]
        parent_id = _current_span_id_var.get()
        span_id = uuid.uuid4().hex[:12]

        token_trace = _current_trace_id_var.set(trace_id)
        token_span = _current_span_id_var.set(span_id)

        builder = _MutableSpanBuilder(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_id,
            name=name,
            kind=kind,
            initial_attributes=attributes,
        )
        try:
            yield builder
        except Exception as exc:
            builder.status = SpanStatus.ERROR
            builder.set_attribute("error.type", type(exc).__name__)
            builder.set_attribute("error.message", str(exc))
            raise
        finally:
            self._recorded_spans.append(builder.finalize())
            _current_span_id_var.reset(token_span)
            _current_trace_id_var.reset(token_trace)

    def render_ascii_waterfall(self) -> str:
        """يرسم شجرة التتبع الزمنية بشكل مرئي (ASCII Waterfall Tree)."""
        spans = self.recorded_spans
        if not spans:
            return "No spans recorded."

        children_map: dict[str | None, list[TelemetrySpan]] = {}
        for s in spans:
            children_map.setdefault(s.parent_span_id, []).append(s)

        lines: list[str] = []

        def _walk(parent_id: str | None, prefix: str = "") -> None:
            children = children_map.get(parent_id, [])
            for idx, span in enumerate(children):
                is_last = idx == len(children) - 1
                connector = "└── " if is_last else "├── "
                extra_info = ""
                if span.kind == SpanKind.LLM_GENERATION:
                    in_t = span.attributes.get("gen_ai.usage.input_tokens", 0)
                    cache_r = span.attributes.get("gen_ai.usage.cache_hit_ratio", 0.0)
                    cost = span.attributes.get("gen_ai.cost.usd", 0.0)
                    extra_info = f" [in={in_t}, cache_hit={cache_r:.0%}, cost=${cost:.4f}]"
                elif span.kind == SpanKind.TOOL_EXECUTION:
                    args = span.attributes.get("tool.args", {})
                    extra_info = f" [args={args}]"

                lines.append(
                    f"{prefix}{connector}[{span.kind.value}] {span.name} "
                    f"({span.duration_ms:.1f}ms | {span.status.value}){extra_info}"
                )
                next_prefix = prefix + ("    " if is_last else "│   ")
                _walk(span.span_id, next_prefix)

        _walk(None)
        return "\n".join(lines)


# =====================================================================
# 3. نماذج أحداث البث الحي (Typed SSE Stream Events - Discriminated Union)
# =====================================================================

class TurnStartedEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    event_type: Literal["turn_started"] = "turn_started"
    turn_number: int
    trace_id: str


class ToolStatusEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    event_type: Literal["tool_status"] = "tool_status"
    tool_name: str
    phase: Literal["started", "completed"]
    summary: str


class TokenDeltaEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    event_type: Literal["token_delta"] = "token_delta"
    delta_text: str


class RunCompletedEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    event_type: Literal["run_completed"] = "run_completed"
    trace_id: str
    total_duration_ms: float
    total_cost_usd: float
    total_input_tokens: int
    total_cached_tokens: int


type AgentStreamEvent = (
    TurnStartedEvent | ToolStatusEvent | TokenDeltaEvent | RunCompletedEvent
)

_StreamEventAdapter = TypeAdapter(AgentStreamEvent)


def format_sse_frame(seq_id: int, event: AgentStreamEvent) -> str:
    """
    تحويل الحدث المهيكل إلى إطار Server-Sent Events (SSE) قياسي متوافق مع W3C،
    مع تطهير أي بيانات حساسة قبل البث.
    """
    raw_dict = _StreamEventAdapter.dump_python(event)
    scrubbed_dict = PIITelemetryScrubber.scrub_value(raw_dict)
    json_payload = json.dumps(scrubbed_dict, ensure_ascii=False)
    return f"id: {seq_id}\nevent: {event.event_type}\ndata: {json_payload}\n\n"


# =====================================================================
# 4. إطار تشغيل الوكيل المرئي والمتدفق (Observable Streaming Agent Harness)
# =====================================================================

class ObservableStreamingAgentHarness:
    """
    يدير تنفيذ الوكيل، وينشئ شجرة تتبع OpenTelemetry، ويبث الأحداث الحية
    عبر طابور محدود السعة (Backpressure-Bounded Queue) مع إلغاء آمن للمهام.
    """

    def __init__(self, tracer: AgentTracer, queue_maxsize: int = 32) -> None:
        self._tracer = tracer
        self._queue_maxsize = queue_maxsize

    async def _simulate_parallel_tool(
        self,
        queue: asyncio.Queue[AgentStreamEvent | None],
        tool_name: str,
        args: dict[str, Any],
        delay_sec: float,
    ) -> str:
        """محاكاة تنفيذ أداة فرعية متوازية مع تسجيل Span خاص بها وبث حالتها."""
        async with self._tracer.start_span(
            name=f"tool.{tool_name}",
            kind=SpanKind.TOOL_EXECUTION,
            attributes={"tool.name": tool_name, "tool.args": args},
        ) as span:
            await queue.put(
                ToolStatusEvent(
                    tool_name=tool_name,
                    phase="started",
                    summary=f"جاري تشغيل الأداة {tool_name}...",
                )
            )
            await asyncio.sleep(delay_sec)
            result_text = f"نتيجة {tool_name}: تم التحقق بنجاح"
            span.set_attribute("tool.result_preview", result_text)

            await queue.put(
                ToolStatusEvent(
                    tool_name=tool_name,
                    phase="completed",
                    summary=result_text,
                )
            )
            return result_text

    async def _agent_execution_worker(
        self,
        user_query: str,
        queue: asyncio.Queue[AgentStreamEvent | None],
    ) -> None:
        """المهمة المنتجة (Producer) التي تشغل منطق الوكيل وتدفع الأحداث للطابور."""
        try:
            async with self._tracer.start_span(
                name="financial_support_agent",
                kind=SpanKind.AGENT_RUN,
                attributes={"user.query": user_query},
            ) as root_span:
                trace_id = root_span.trace_id

                # --- الدورة الأولى: التخطيط واستدعاء الأدوات بالتوازي ---
                async with self._tracer.start_span(
                    name="turn_1_planning", kind=SpanKind.REASONING_TURN
                ):
                    await queue.put(
                        TurnStartedEvent(turn_number=1, trace_id=trace_id)
                    )

                    async with self._tracer.start_span(
                        name="claude-3-7-sonnet", kind=SpanKind.LLM_GENERATION
                    ) as llm1:
                        await asyncio.sleep(0.03)
                        llm1.record_llm_usage(
                            model="claude-3-7-sonnet",
                            input_tokens=1500,
                            cached_tokens=1200,  # 80% KV-Cache Hit!
                            output_tokens=85,
                            cost_usd=0.0024,
                        )

                    # تشغيل أداتين بالتوازي المهيكل: لاحظ كيف يرث كل Span أباه تلقائياً!
                    async with asyncio.TaskGroup() as tg:
                        tg.create_task(
                            self._simulate_parallel_tool(
                                queue,
                                "verify_kyc_token",
                                {
                                    "customer_email": "ali.hassan@example.com",
                                    "api_key": "sk-live9988776655443322",  # سيتم حجبه تلقائياً!
                                },
                                delay_sec=0.04,
                            )
                        )
                        tg.create_task(
                            self._simulate_parallel_tool(
                                queue,
                                "check_ledger_balance",
                                {"account_id": "ACC-7741"},
                                delay_sec=0.06,
                            )
                        )

                # --- الدورة الثانية: توليد الإجابة النهائية المتدفقة (Token Streaming) ---
                async with self._tracer.start_span(
                    name="turn_2_synthesis", kind=SpanKind.REASONING_TURN
                ):
                    await queue.put(
                        TurnStartedEvent(turn_number=2, trace_id=trace_id)
                    )

                    async with self._tracer.start_span(
                        name="claude-3-7-sonnet-stream",
                        kind=SpanKind.LLM_GENERATION,
                    ) as llm2:
                        chunks = [
                            "تم ",
                            "التحقق ",
                            "من ",
                            "الحساب ",
                            "ACC-7741 ",
                            "ورصيدكم ",
                            "المتاح ",
                            "هو ",
                            "14,500 ",
                            "دولار.",
                        ]
                        for token_str in chunks:
                            await asyncio.sleep(0.01)
                            await queue.put(TokenDeltaEvent(delta_text=token_str))

                        llm2.record_llm_usage(
                            model="claude-3-7-sonnet",
                            input_tokens=1850,
                            cached_tokens=1500,
                            output_tokens=42,
                            cost_usd=0.0021,
                        )

                # تجميع إحصائيات التشغيل الكلي من الـ Spans المسجلة
                llm_spans = [
                    s
                    for s in self._tracer.recorded_spans
                    if s.kind == SpanKind.LLM_GENERATION
                ]
                total_cost = sum(
                    float(s.attributes.get("gen_ai.cost.usd", 0.0))
                    for s in llm_spans
                )
                total_in = sum(
                    int(s.attributes.get("gen_ai.usage.input_tokens", 0))
                    for s in llm_spans
                )
                total_cached = sum(
                    int(s.attributes.get("gen_ai.usage.cached_tokens", 0))
                    for s in llm_spans
                )
                elapsed_ms = round(
                    (time.perf_counter_ns() - root_span.start_time_ns) / 1_000_000,
                    2,
                )

                await queue.put(
                    RunCompletedEvent(
                        trace_id=trace_id,
                        total_duration_ms=elapsed_ms,
                        total_cost_usd=round(total_cost, 6),
                        total_input_tokens=total_in,
                        total_cached_tokens=total_cached,
                    )
                )
        finally:
            # إشارة نهاية البث (Sentinel)
            await queue.put(None)

    async def stream_sse(self, user_query: str) -> AsyncGenerator[str, None]:
        """
        المولد غير المتزامن الذي تستهلكه نقطة النهاية (FastAPI StreamingResponse).
        يضمن تنظيف وإلغاء مهمة الوكيل الخلفية فوراً إذا قطع العميل الاتصال!
        """
        queue: asyncio.Queue[AgentStreamEvent | None] = asyncio.Queue(
            maxsize=self._queue_maxsize
        )
        worker_task = asyncio.create_task(
            self._agent_execution_worker(user_query, queue)
        )
        seq_id = 1
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield format_sse_frame(seq_id, event)
                seq_id += 1
        finally:
            if not worker_task.done():
                worker_task.cancel()
                await asyncio.gather(worker_task, return_exceptions=True)


# =====================================================================
# 5. تشغيل المحاكاة واستعراض البث الحي وشجرة التتبع (Main Simulation)
# =====================================================================

async def main() -> None:
    tracer = AgentTracer()
    harness = ObservableStreamingAgentHarness(tracer=tracer, queue_maxsize=16)

    query = (
        "مرحباً، أنا العميل ali.hassan@example.com ومفتاحي sk-live9988776655443322، "
        "كم يبلغ رصيدي في الحساب ACC-7741؟"
    )

    print("=" * 80)
    print("📡 1. بث الأحداث الحية بصيغة W3C Server-Sent Events (SSE Wire Stream)")
    print("=" * 80)

    streamed_answer_parts: list[str] = []
    async for sse_frame in harness.stream_sse(query):
        # طباعة الإطارات المهمة وإظهار تدفق التوكنات
        if "event: token_delta" in sse_frame:
            data_line = [
                line for line in sse_frame.splitlines() if line.startswith("data: ")
            ][0]
            payload = json.loads(data_line.removeprefix("data: "))
            streamed_answer_parts.append(payload["delta_text"])
        else:
            print(sse_frame.strip())
            print("-" * 50)

    print(f"\n💬 النص المتدفق المجمّع لدى العميل: {''.join(streamed_answer_parts)}\n")

    print("=" * 80)
    print("🌲 2. شجرة التتبع الموزع الهرمية (OpenTelemetry ASCII Trace Waterfall)")
    print("=" * 80)
    print(tracer.render_ascii_waterfall())

    print("\n" + "=" * 80)
    print("🛡️ 3. فحص تطهير البيانات الحساسة (PII & Secret Scrubbing Verification)")
    print("=" * 80)
    for span in tracer.recorded_spans:
        if span.kind in (SpanKind.AGENT_RUN, SpanKind.TOOL_EXECUTION):
            print(f"• Span [{span.name}] Attributes -> {span.attributes}")


if __name__ == "__main__":
    asyncio.run(main())
