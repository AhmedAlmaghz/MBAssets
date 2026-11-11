#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
الفصل 1: ما هو وكيل الذكاء الاصطناعي؟ وما هو إطار التحكم؟
==============================================================================

البرنامج الكامل للفصل 1 من كتاب «هندسة وكلاء الذكاء الاصطناعي».
ما هو وكيل الذكاء الاصطناعي؟ وما هو إطار التحكم؟ — What Is an AI Agent — and What Is The Agent Harness?.

التشغيل:
    python ch01_agent_harness_foundations.py

المتطلبات: Python 3.12+ (الكتاب يستهدف 3.14+) و pydantic>=2.9
"""
# ملاحظة توافق: تُقيَّم الأنواع كسولاً في Python 3.14 (PEP 649)؛
# هذا السطر يمنح السلوك نفسه على Python 3.12/3.13.
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum, auto
import time
from typing import Any, Protocol

# =====================================================================
# 1. تعريف الأنواع والحالات الأساسية (Modern Python 3.14+ Typing & Enums)
# =====================================================================
type ToolName = str
type ToolArgs = dict[str, Any]


class StopReason(StrEnum):
    """الأسباب المحتملة لتوقف الـ Harness عن العمل (حتمية وواضحة)"""

    COMPLETED = auto()  # الوكيل أنجز المهمة بنجاح
    MAX_STEPS_REACHED = auto()  # تدخل الـ Harness: تجاوز الحد الأقصى للخطوات
    BUDGET_EXCEEDED = auto()  # تدخل الـ Harness: تجاوز ميزانية التوكنز
    TIMEOUT = auto()  # تدخل الـ Harness: انتهاء الوقت المسموح
    ERROR = auto()  # خطأ غير قابل للتعافي


class Role(StrEnum):
    SYSTEM = auto()
    USER = auto()
    ASSISTANT = auto()
    TOOL = auto()


@dataclass(slots=True, frozen=True)
class Message:
    """تمثيل غير قابل للتغيير (Immutable) لأي رسالة داخل ذاكرة الوكيل"""

    role: Role
    content: str
    tool_name: ToolName | None = None


@dataclass(slots=True, frozen=True)
class ToolAction:
    """طلب من النموذج اللغوي لتنفيذ أداة معينة"""

    name: ToolName
    arguments: ToolArgs


@dataclass(slots=True, frozen=True)
class LLMResponse:
    """المخرجات الموحدة التي يتوقعها الـ Harness من أي نموذج لغوي"""

    thought: str  # تفكير الوكيل الداخلي (Reasoning)
    final_answer: str | None = None  # الإجابة النهائية (إن انتهى)
    action: ToolAction | None = None  # الأداة المطلوب تشغيلها (إن لم ينتهِ)
    tokens_used: int = 0  # عدد التوكنز المستهلكة في هذه الخطوة


# =====================================================================
# 2. العقود البرمجية (Protocols) لفصل الـ Harness عن المزودين الخارجيين
# =====================================================================
class LLMProvider(Protocol):
    """عقد برمجي: أي نموذج لغوي (OpenAI, Claude, Local Ollama) يجب أن يحقق هذه الدالة"""

    async def generate(self, messages: list[Message]) -> LLMResponse:
        ...


class ToolExecutor(Protocol):
    """عقد برمجي: المسؤول عن تنفيذ الأداة وإرجاع النتيجة كنص"""

    async def execute(self, action: ToolAction) -> str:
        ...


# =====================================================================
# 3. إعدادات وحالة الـ Harness (Configuration & State Management)
# =====================================================================
@dataclass(slots=True, kw_only=True)
class HarnessConfig:
    """القيود الصارمة التي يفرضها المهندس على الوكيل"""

    system_prompt: str
    max_steps: int = 10
    max_tokens_budget: int = 8_000
    execution_timeout_sec: float = 30.0


@dataclass(slots=True)
class ExecutionMetrics:
    """مقاييس الأداء والمراقبة (Observability Telemetry)"""

    steps_taken: int = 0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0


@dataclass(slots=True)
class HarnessResult[ContextT]:
    """النتيجة النهائية الشاملة التي يعيدها الـ Harness لتطبيقك"""

    stop_reason: StopReason
    final_output: str
    metrics: ExecutionMetrics
    context: ContextT
    history: list[Message] = field(default_factory=list)


# =====================================================================
# 4. قلب النظام: The Agent Harness Engine (باستخدام PEP 695 Generics)
# =====================================================================
class AgentHarness[ContextT]:
    """إطار التحكم والتشغيل للوكيل.

    يدير دورة حياة الوكيل بأمان، ويراقب الموارد، ويعزل منطق التنفيذ عن النموذج.
    """

    def __init__(
        self,
        config: HarnessConfig,
        llm: LLMProvider,
        tools: ToolExecutor,
    ) -> None:
        self.config = config
        self.llm = llm
        self.tools = tools

    async def run(
        self, user_goal: str, context: ContextT
    ) -> HarnessResult[ContextT]:
        """نقطة الدخول الرئيسية لتشغيل الوكيل تحت رقابة الـ Harness"""
        start_time = time.monotonic()
        metrics = ExecutionMetrics()
        history: list[Message] = [
            Message(role=Role.SYSTEM, content=self.config.system_prompt),
            Message(role=Role.USER, content=user_goal),
        ]

        try:
            # فرض مهلة زمنية شاملة على دورة حياة الوكيل بالكامل
            async with asyncio.timeout(self.config.execution_timeout_sec):
                while metrics.steps_taken < self.config.max_steps:
                    metrics.steps_taken += 1

                    # 1. خطوة التفكير (Think): استدعاء النموذج اللغوي
                    response = await self.llm.generate(history)
                    metrics.total_tokens += response.tokens_used

                    # 2. فحص حواجز الميزانية (Budget Guardrail)
                    if metrics.total_tokens > self.config.max_tokens_budget:
                        metrics.elapsed_seconds = time.monotonic() - start_time
                        return HarnessResult(
                            stop_reason=StopReason.BUDGET_EXCEEDED,
                            final_output="تم إيقاف الوكيل لتجاوز ميزانية التوكنز المسموحة.",
                            metrics=metrics,
                            context=context,
                            history=history,
                        )

                    # 3. تحليل قرار النموذج باستخدام Structural Pattern Matching الحديث
                    match response:
                        # الحالة أ: الوكيل قرر أنه وصل للإجابة النهائية
                        case LLMResponse(
                            final_answer=str(answer), action=None
                        ) if answer.strip():
                            history.append(
                                Message(role=Role.ASSISTANT, content=answer)
                            )
                            metrics.elapsed_seconds = (
                                time.monotonic() - start_time
                            )
                            return HarnessResult(
                                stop_reason=StopReason.COMPLETED,
                                final_output=answer,
                                metrics=metrics,
                                context=context,
                                history=history,
                            )

                        # الحالة ب: الوكيل يطلب تشغيل أداة (Act -> Observe)
                        case LLMResponse(
                            thought=thought, action=ToolAction() as action
                        ):
                            history.append(
                                Message(
                                    role=Role.ASSISTANT,
                                    content=f"Thought: {thought} | Calling: {action.name}",
                                )
                            )
                            # تنفيذ الأداة عبر المحرك المعزول وإضافة الملاحظة للذاكرة
                            observation = await self.tools.execute(action)
                            history.append(
                                Message(
                                    role=Role.TOOL,
                                    content=observation,
                                    tool_name=action.name,
                                )
                            )

                        # الحالة ج: مخرجات مشوهة من النموذج (Self-Correction Hook)
                        case _:
                            history.append(
                                Message(
                                    role=Role.USER,
                                    content="System Harness Error: يجب أن تحدد إما أداة للتنفيذ (action) أو إجابة نهائية (final_answer).",
                                )
                            )

                # إذا خرجنا من الحلقة، فهذا يعني أن الوكيل استنفد الحد الأقصى للخطوات
                metrics.elapsed_seconds = time.monotonic() - start_time
                return HarnessResult(
                    stop_reason=StopReason.MAX_STEPS_REACHED,
                    final_output="لم يتمكن الوكيل من حسم المهمة ضمن عدد الخطوات المسموح به.",
                    metrics=metrics,
                    context=context,
                    history=history,
                )

        except TimeoutError:
            metrics.elapsed_seconds = time.monotonic() - start_time
            return HarnessResult(
                stop_reason=StopReason.TIMEOUT,
                final_output=f"تجاوز الوكيل المهلة الزمنية القصوى ({self.config.execution_timeout_sec}s).",
                metrics=metrics,
                context=context,
                history=history,
            )


# =====================================================================
# 5. محاكاة تشغيلية كاملة للـ Harness (Runnable Simulation)
# =====================================================================
class MockServerDiagnosticLLM:
    """محاكاة لنموذج لغوي يقوم بفحص حالة الخادم على خطوتين ثم يعطي التقرير"""

    def __init__(self) -> None:
        self._call_count = 0

    async def generate(self, messages: list[Message]) -> LLMResponse:
        await asyncio.sleep(0.1)  # محاكاة زمن اتصال الشبكة
        self._call_count += 1

        match self._call_count:
            case 1:
                return LLMResponse(
                    thought="أحتاج أولاً لفحص استهلاك المعالج والذاكرة للخادم.",
                    action=ToolAction(
                        name="check_server_metrics",
                        arguments={"host": "prod-db-01"},
                    ),
                    tokens_used=450,
                )
            case 2:
                return LLMResponse(
                    thought="الذاكرة ممتلئة بنسبة 98%. سأقوم بتفريغ الذاكرة المؤقتة (Cache).",
                    action=ToolAction(
                        name="clear_redis_cache",
                        arguments={"host": "prod-db-01"},
                    ),
                    tokens_used=620,
                )
            case _:
                return LLMResponse(
                    thought="تم تفريغ الذاكرة بنجاح وعاد الخادم للعمل الطبيعي.",
                    final_answer="تم حل المشكلة: كان الخادم prod-db-01 يعاني من امتلاء الذاكرة بنسبة 98%، وتم تنظيف الـ Cache بنجاح.",
                    tokens_used=310,
                )


class MockDevOpsTools:
    """محاكاة لمحرك تنفيذ أدوات البنية التحتية"""

    async def execute(self, action: ToolAction) -> str:
        await asyncio.sleep(0.05)
        match action.name:
            case "check_server_metrics":
                return (
                    "{'cpu': '42%', 'ram': '98%', 'status': 'CRITICAL_MEMORY'}"
                )
            case "clear_redis_cache":
                return "{'status': 'CACHE_CLEARED', 'ram_after': '35%'}"
            case unknown:
                return f"Error: Unknown tool '{unknown}'"


async def main() -> None:
    config = HarnessConfig(
        system_prompt="أنت وكيل هندسة نظم (SRE Agent) مسؤول عن تشخيص وإصلاح الخوادم.",
        max_steps=5,
        max_tokens_budget=5_000,
        execution_timeout_sec=10.0,
    )

    # سياق مخصص (Custom Context) يمكن تمريره بأمان مع الحفاظ على النوع (Type-Safe)
    user_context = {"engineer_id": "ENG-99", "environment": "production"}

    harness = AgentHarness[dict[str, str]](
        config=config,
        llm=MockServerDiagnosticLLM(),
        tools=MockDevOpsTools(),
    )

    result = await harness.run(
        user_goal="الخادم prod-db-01 بطيء جداً، افحصه وعالج المشكلة.",
        context=user_context,
    )

    print(f"🏁 حالة التوقف     : {result.stop_reason.upper()}")
    print(f"📊 الخطوات المنفذة : {result.metrics.steps_taken}")
    print(f"🪙 التوكنز المستهلكة: {result.metrics.total_tokens}")
    print(f"⏱️ الزمن المستغرق  : {result.metrics.elapsed_seconds:.2f} ثانية")
    print(f"💡 المخرج النهائي  : {result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
