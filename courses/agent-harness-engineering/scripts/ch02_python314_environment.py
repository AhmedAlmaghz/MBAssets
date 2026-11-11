#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
الفصل 2: تسخير قوة Python 3.14+ لهندسة الوكلاء
==============================================================================

البرنامج الكامل للفصل 2 من كتاب «هندسة وكلاء الذكاء الاصطناعي».
تسخير قوة Python 3.14+ لهندسة الوكلاء — Harnessing Python 3.14+ for Agent Engineering.

التشغيل:
    python ch02_python314_environment.py

المتطلبات: Python 3.12+ (الكتاب يستهدف 3.14+) و pydantic>=2.9
"""
# ملاحظة توافق: تُقيَّم الأنواع كسولاً في Python 3.14 (PEP 649)؛
# هذا السطر يمنح السلوك نفسه على Python 3.12/3.13.
from __future__ import annotations

try:
    import annotationlib  # Python 3.14+ (PEP 649)
except ModuleNotFoundError:  # توافق Python 3.12/3.13
    import harness_compat as annotationlib
import ast
import asyncio
from collections.abc import Callable
import sys
import time
from typing import Annotated, Any, Literal
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    TypeAdapter,
    ValidationError,
    field_validator,
)


# =====================================================================
# 1. فحص بيئة تشغيل Python 3.14+ (التحقق من حالة الـ GIL)
# =====================================================================
def check_runtime_capabilities() -> dict[str, Any]:
    """فحص قدرات مفسر بايثون 3.14 أثناء التشغيل"""
    gil_enabled = True
    if hasattr(sys, "_is_gil_enabled"):
        gil_enabled = sys._is_gil_enabled()

    return {
        "python_version": sys.version.split()[0],
        "free_threaded_no_gil": not gil_enabled,
        "concurrency_mode": (
            "True Multi-Core Parallelism (No-GIL)"
            if not gil_enabled
            else "Standard GIL Threading"
        ),
    }


# =====================================================================
# 2. نمذجة قرارات الوكيل الصارمة باستخدام Pydantic v2 Discriminated Unions
# =====================================================================
class BaseAgentEvent(BaseModel):
    """النموذج الأب لجميع قرارات الوكيل - محمي وغير قابل للتعديل بعد الإنشاء"""

    model_config = ConfigDict(
        frozen=True, extra="forbid", str_strip_whitespace=True
    )


class ExecuteToolDecision(BaseAgentEvent):
    """قرار بتشغيل أداة واحدة أو أكثر بالتوازي"""

    decision_type: Literal["execute_tools"] = "execute_tools"
    thought: str = Field(min_length=5, description="التبرير المنطقي لاختيار هذه الأدوات")
    tools_to_call: list[dict[str, Any]] = Field(min_length=1, max_length=5)


class AskHumanDecision(BaseAgentEvent):
    """قرار بطلب مساعدة أو موافقة من المهندس البشري"""

    decision_type: Literal["ask_human"] = "ask_human"
    question: str = Field(min_length=10)
    urgency: Literal["low", "medium", "critical"] = "medium"


class FinalAnswerDecision(BaseAgentEvent):
    """قرار بإنهاء المهمة وتقديم الإجابة النهائية"""

    decision_type: Literal["final_answer"] = "final_answer"
    summary: str = Field(min_length=10)
    confidence_score: float = Field(ge=0.0, le=1.0)

    @field_validator("confidence_score")
    @classmethod
    def check_high_confidence(cls, v: float) -> float:
        if v < 0.5:
            raise ValueError(
                "لا يُسمح بتقديم إجابة نهائية بدرجة ثقة أقل من 0.5! اطلب مساعدة بشرية أو ابحث أكثر."
            )
        return v


# استخدام الاتحادات المميزة (Discriminated Union) لتحليل قرار النموذج بسرعة Rust الفائقة
type AgentDecision = Annotated[
    Annotated[ExecuteToolDecision, Tag("execute_tools")]
    | Annotated[AskHumanDecision, Tag("ask_human")]
    | Annotated[FinalAnswerDecision, Tag("final_answer")],
    Discriminator("decision_type"),
]

# محول الأنواع فائق السرعة من Pydantic v2
DecisionAdapter: TypeAdapter[AgentDecision] = TypeAdapter(AgentDecision)


# =====================================================================
# 3. بناء محرك الأدوات الهجين (I/O Async + CPU Free-Threaded)
# =====================================================================
class HybridToolRunner:
    """محرك أدوات ذكي يميز بين الأدوات الشبكية (Async I/O)

    والأدوات الحسابية الثقيلة (CPU-Bound) ليشغلها على أنوية المعالج دون تجميد النظام.
    """

    def __init__(self) -> None:
        self._async_tools: dict[str, Callable[..., Any]] = {}
        self._cpu_tools: dict[str, Callable[..., Any]] = {}

    def register_io_tool(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """تسجيل أداة شبكية غير متزامنة (Async Def)"""
        # استخدام annotationlib الجديدة في Python 3.14 لفحص الأنواع بأمان
        annotations = annotationlib.get_annotations(
            func, format=annotationlib.Format.VALUE
        )
        print(
            f"📌 [Registry] تم تسجيل أداة I/O: '{func.__name__}' | الأنواع: {annotations}"
        )
        self._async_tools[func.__name__] = func
        return func

    def register_cpu_tool(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """تسجيل أداة حسابية ثقيلة (Sync Def) ليتم تفويضها لخيوط المعالج"""
        annotations = annotationlib.get_annotations(
            func, format=annotationlib.Format.VALUE
        )
        print(
            f"⚙️ [Registry] تم تسجيل أداة CPU: '{func.__name__}' | الأنواع: {annotations}"
        )
        self._cpu_tools[func.__name__] = func
        return func

    async def _run_single(self, name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """تشغيل أداة واحدة وقياس زمنها بدقة مع احتواء أي استثناء"""
        t0 = time.perf_counter()
        try:
            if name in self._async_tools:
                # تشغيل مباشر على حلقة أحداث asyncio
                res = await self._async_tools[name](**kwargs)
                mode = "Async-IO"
            elif name in self._cpu_tools:
                # تفويض المهمة الثقيلة لخيط منفصل (يستغل No-GIL في Python 3.14t!)
                res = await asyncio.to_thread(self._cpu_tools[name], **kwargs)
                mode = "CPU-Thread"
            else:
                return {"tool": name, "error": f"الأداة '{name}' غير مسجلة!", "ok": False}

            elapsed = (time.perf_counter() - t0) * 1000
            return {
                "tool": name,
                "mode": mode,
                "result": res,
                "ms": round(elapsed, 2),
                "ok": True,
            }

        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            return {
                "tool": name,
                "error": f"{type(exc).__name__}: {exc}",
                "ms": round(elapsed, 2),
                "ok": False,
            }

    async def execute_parallel(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """تشغيل جميع الأدوات المطلوبة بالتوازي المهيكل عبر TaskGroup"""
        tasks: list[asyncio.Task[dict[str, Any]]] = []
        async with asyncio.TaskGroup() as tg:
            for call in tool_calls:
                t_name = str(call.get("name", ""))
                t_args = dict(call.get("args", {}))
                tasks.append(tg.create_task(self._run_single(t_name, t_args)))

        return [t.result() for t in tasks]


# =====================================================================
# 4. تعريف أدوات حقيقية واختبار النظام بالكامل
# =====================================================================
runner = HybridToolRunner()


@runner.register_io_tool
async def fetch_security_advisory(package_name: str) -> str:
    """أداة شبكية تحاكي الاتصال بقاعدة بيانات الثغرات الأمنية"""
    await asyncio.sleep(0.4)  # محاكاة انتظار الشبكة 400ms
    return f"Package '{package_name}' is clean. No CVEs found in version 3.14."


@runner.register_cpu_tool
def analyze_ast_nodes(source_code: str, iterations: int = 5_000) -> dict[str, int]:
    """أداة ثقيلة على المعالج (CPU-Bound): تقوم بتحليل الكود المصدري إلى شجرة AST

    آلاف المرات لمحاكاة فحص أمني عميق للكود دون تجميد الـ Event Loop!
    """
    node_count = 0
    func_count = 0
    for _ in range(iterations):
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            node_count += 1
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_count += 1
    return {"total_ast_nodes_scanned": node_count, "functions_detected": func_count}


async def main() -> None:
    print("🚀 فحص بيئة التشغيل:", check_runtime_capabilities(), "\n")

    # محاكاة مخرج JSON قادم من النموذج اللغوي (LLM) يطلب تشغيل أداتي I/O و CPU معاً
    raw_llm_json = """
    {
        "decision_type": "execute_tools",
        "thought": "سأقوم بفحص الثغرات عبر الشبكة وفي نفس الوقت أحلل شجرة الكود محلياً.",
        "tools_to_call": [
            {"name": "fetch_security_advisory", "args": {"package_name": "pydantic"}},
            {"name": "analyze_ast_nodes", "args": {"source_code": "async def agent():\\n    return 42", "iterations": 8000}}
        ]
    }
    """

    # 1. التحقق الصارم عبر Pydantic v2 Discriminated Union
    decision = DecisionAdapter.validate_json(raw_llm_json)

    # 2. التوجيه النظيف عبر Structural Pattern Matching
    match decision:
        case ExecuteToolDecision(thought=thought, tools_to_call=calls):
            print(f"🧠 تفكير الوكيل: {thought}")
            t_start = time.perf_counter()
            results = await runner.execute_parallel(calls)
            total_ms = (time.perf_counter() - t_start) * 1000

            print(f"\n⚡ نتائج التنفيذ المتوازي (الزمن الكلي: {total_ms:.2f} ms):")
            for r in results:
                print(f"   -> {r}")

        case AskHumanDecision(question=q, urgency=u):
            print(f"🙋 مطلوب تدخل بشري [{u}]: {q}")

        case FinalAnswerDecision(summary=s, confidence_score=c):
            print(f"✅ الإجابة النهائية (ثقة {c:.0%}): {s}")

    # 3. تجربة درع الحماية: ماذا لو حاول النموذج إعطاء إجابة نهائية بدرجة ثقة ضعيفة (0.2)؟
    print("\n🛡️ اختبار درع Pydantic ضد الهلوسة وانخفاض الثقة:")
    bad_llm_json = '{"decision_type": "final_answer", "summary": "أعتقد أن المشكلة في الشبكة لست متأكداً", "confidence_score": 0.2}'
    try:
        DecisionAdapter.validate_json(bad_llm_json)
    except ValidationError as err:
        print(f"   ❌ تم صد المخرج غير الموثوق بنجاح! الرسالة التي ستعود للنموذج:\n   {err.errors()[0]['msg']}")


if __name__ == "__main__":
    asyncio.run(main())
