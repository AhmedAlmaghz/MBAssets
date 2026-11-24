#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — تسخير قوة Python 3.14+ لهندسة الوكلاء
AI Agent Engineering — Harnessing Python 3.14+ for Agent Engineering

Book slug : agent-harness-engineering
Chapter   : ch-02-ch02 (lesson 2)
Code files: 17 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/agent-harness-engineering/code/ch-02-ch02.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — تسخير قوة Python 3.14+ لهندسة الوكلاء
AI Agent Engineering — Harnessing Python 3.14+ for Agent Engineering

Book slug : agent-harness-engineering
Chapter   : ch-02-ch02 (lesson 2)
Code files: 16 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/agent-harness-engineering/code/ch-02-ch02.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — تسخير قوة Python 3.14+ لهندسة الوكلاء
AI Agent Engineering — Harnessing Python 3.14+ for Agent Engineering

Book slug : agent-harness-engineering
Chapter   : ch-02-ch02 (lesson 2)
Code files: 15 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/ch02_python314_environment/ch02_python314_environment.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ⚙️ إعداد البيئة — يعمل على Google Colab و Kaggle ومحلياً
from __future__ import annotations   # ينتقل تلقائياً إلى كل خلايا الدفتر التالية

import sys, subprocess, types, typing, warnings

if sys.version_info < (3, 12):
    raise RuntimeError(
        "هذا الدفتر يتطلب Python 3.12 أو أحدث (الكتاب يستهدف 3.14+). "
        f"الإصدار الحالي: {sys.version.split()[0]}"
    )

try:
    import pydantic  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pydantic>=2.9"], check=True)

# طبقة توافق: annotationlib متاحة أصلاً في Python 3.14 فقط
if sys.version_info < (3, 14):
    _al = types.ModuleType("annotationlib")

    class _Format:
        VALUE, FORWARDREF, STRING = 1, 2, 3

    _al.Format = _Format
    _al.get_annotations = lambda obj, **kw: typing.get_type_hints(obj, include_extras=True)
    sys.modules.setdefault("annotationlib", _al)

warnings.filterwarnings("ignore", category=DeprecationWarning)
print("✅ البيئة جاهزة — Python", sys.version.split()[0])

# ────────────────────────────────────────────────────────────────────────
# [002] code/ch02_python314_environment/ch02_python314_environment.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] code/ch02_python314_environment/ch02_python314_environment.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] code/ch02_python314_environment/ch02_python314_environment.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [005] code/ch02_python314_environment/ch02_python314_environment.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] code/ch02_python314_environment/ch02_python314_environment.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
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

# في الدفتر لا نستخدم asyncio.run: الحلقة قائمة بالفعل، لذا ننتظر التنفيذ مباشرة.
await main()

# ────────────────────────────────────────────────────────────────────────
# [007] code/ch02_python314_environment/ch02_python314_environment.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [008] code/ch02_python314_environment/harness_compat.py
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
توافق خلفي (Backport) لمكتبة `annotationlib` القياسية في Python 3.14.

الكتاب يستهدف Python 3.14+ حيث تُقيَّم التعليقات النوعية كسولاً (PEP 649/749).
هذا الملف يوفّر واجهة مطابقة مبسّطة تعمل على Python 3.12/3.13 كي تبقى أمثلة
الفصلين الثاني والرابع قابلة للتشغيل على Colab وKaggle وأي بيئة أقدم.
"""
from __future__ import annotations

import sys
import typing

__all__ = ["Format", "get_annotations"]


class Format:
    """تنسيقات القراءة الثلاثة (مطابقة لأسماء Python 3.14)."""

    VALUE = 1
    FORWARDREF = 2
    STRING = 3


def get_annotations(
    obj: object,
    *,
    format: int = Format.VALUE,  # noqa: A002 - مطابقة لتوقيع المكتبة القياسية
    globals: dict | None = None,  # noqa: A002
    locals: dict | None = None,  # noqa: A002
    eval_str: bool = False,
) -> dict[str, typing.Any]:
    """يعيد تعليقات الكائن النوعية؛ يحلّ الإشارات الأمامية عند طلب VALUE."""
    if format == Format.STRING and not eval_str:
        return dict(getattr(obj, "__annotations__", {}) or {})

    try:
        return typing.get_type_hints(
            obj, globalns=globals, localns=locals, include_extras=True
        )
    except Exception:  # pragma: no cover - سلوك احتياطي متسامح
        return dict(getattr(obj, "__annotations__", {}) or {})


if sys.version_info >= (3, 14):  # pragma: no cover
    from annotationlib import Format, get_annotations  # type: ignore[no-redef]

# ────────────────────────────────────────────────────────────────────────
# [009] code/ch02_python314_environment/snippets/anti_pattern_01.py
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
❌ ANTI-PATTERN: خداع النفس بـ async def مع كود متزامن يجمد النظام بالكامل!

مقتطف من الفصل 2 — «تسخير قوة Python 3.14+ لهندسة الوكلاء».
النوع: anti_pattern. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch02_python314_environment.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ❌ ANTI-PATTERN: خداع النفس بـ async def مع كود متزامن يجمد النظام بالكامل!
import requests


async def bad_web_search_tool(query: str) -> str:
    # كارثة: مكتبة requests متزامنة! هذا السطر سيجمد كل الوكلاء والمستخدمين على الخادم
    # حتى يرد الموقع الخارجي بعد 5 ثوانٍ!
    response = requests.get(f"https://api.example.com/search?q={query}", timeout=5)
    return response.text

# ────────────────────────────────────────────────────────────────────────
# [010] code/ch02_python314_environment/snippets/example_02.py
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
✅ BEST PRACTICE: استخدام مكتبات غير متزامنة أصيلة (مثل httpx) للـ I/O، و to_thread للـ CPU

مقتطف من الفصل 2 — «تسخير قوة Python 3.14+ لهندسة الوكلاء».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch02_python314_environment.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ✅ BEST PRACTICE: استخدام مكتبات غير متزامنة أصيلة (مثل httpx) للـ I/O، و to_thread للـ CPU
import httpx


async def good_web_search_tool(query: str, client: httpx.AsyncClient) -> str:
    # ممتاز: httpx غير متزامن ويحرر حلقة الأحداث لخدمة مهام أخرى أثناء انتظار الشبكة
    response = await client.get(f"https://api.example.com/search?q={query}", timeout=5.0)
    return response.text

# ────────────────────────────────────────────────────────────────────────
# [011] code/ch02_python314_environment/snippets/example_03.py
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
from typing import Any, Literal, Self

مقتطف من الفصل 2 — «تسخير قوة Python 3.14+ لهندسة الوكلاء».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch02_python314_environment.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

from typing import Any, Literal, Self
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BaseAgentEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class ToolCallItem(BaseAgentEvent):
    """نموذج صارم لكل أداة يطلبها الوكيل"""

    name: str = Field(
        min_length=2,
        max_length=64,
        pattern=r"^[a-z0-9_]+$",  # يمنع أي محاولة لحقن رموز غريبة في اسم الأداة
        description="اسم الأداة بالأحرف الإنجليزية الصغيرة والشرطة السفلية فقط",
    )
    args: dict[str, Any] = Field(default_factory=dict)


class ExecuteToolDecision(BaseAgentEvent):
    decision_type: Literal["execute_tools"] = "execute_tools"
    thought: str = Field(min_length=5)
    tools_to_call: list[ToolCallItem] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def prevent_duplicate_parallel_calls(self) -> Self:
        """يمنع الوكيل من طلب تنفيذ نفس الأداة بنفس المعطيات مرتين في نفس اللحظة"""
        seen_signatures: set[str] = set()
        for item in self.tools_to_call:
            # تكوين بصمة فريدة للأداة ومدخلاتها المرتبة
            signature = f"{item.name}:{sorted(item.args.items())}"
            if signature in seen_signatures:
                raise ValueError(
                    f"Duplicate parallel tool call detected for '{item.name}' with identical args. "
                    "Call each unique tool/argument combination only once per batch."
                )
            seen_signatures.add(signature)
        return self

# ────────────────────────────────────────────────────────────────────────
# [012] dist/md/ch02.md
# ────────────────────────────────────────────────────────────────────────
import annotationlib  # ميزة جديدة أصيلة في Python 3.14 لفحص الأنواع بذكاء!
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

# ────────────────────────────────────────────────────────────────────────
# [013] dist/md/ch02.md
# ────────────────────────────────────────────────────────────────────────
# ❌ ANTI-PATTERN: خداع النفس بـ async def مع كود متزامن يجمد النظام بالكامل!
import requests


async def bad_web_search_tool(query: str) -> str:
    # كارثة: مكتبة requests متزامنة! هذا السطر سيجمد كل الوكلاء والمستخدمين على الخادم
    # حتى يرد الموقع الخارجي بعد 5 ثوانٍ!
    response = requests.get(f"https://api.example.com/search?q={query}", timeout=5)
    return response.text

# ────────────────────────────────────────────────────────────────────────
# [014] dist/md/ch02.md
# ────────────────────────────────────────────────────────────────────────
# ✅ BEST PRACTICE: استخدام مكتبات غير متزامنة أصيلة (مثل httpx) للـ I/O، و to_thread للـ CPU
import httpx


async def good_web_search_tool(query: str, client: httpx.AsyncClient) -> str:
    # ممتاز: httpx غير متزامن ويحرر حلقة الأحداث لخدمة مهام أخرى أثناء انتظار الشبكة
    response = await client.get(f"https://api.example.com/search?q={query}", timeout=5.0)
    return response.text

# ────────────────────────────────────────────────────────────────────────
# [015] dist/md/ch02.md
# ────────────────────────────────────────────────────────────────────────
from typing import Any, Literal, Self
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BaseAgentEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class ToolCallItem(BaseAgentEvent):
    """نموذج صارم لكل أداة يطلبها الوكيل"""

    name: str = Field(
        min_length=2,
        max_length=64,
        pattern=r"^[a-z0-9_]+$",  # يمنع أي محاولة لحقن رموز غريبة في اسم الأداة
        description="اسم الأداة بالأحرف الإنجليزية الصغيرة والشرطة السفلية فقط",
    )
    args: dict[str, Any] = Field(default_factory=dict)


class ExecuteToolDecision(BaseAgentEvent):
    decision_type: Literal["execute_tools"] = "execute_tools"
    thought: str = Field(min_length=5)
    tools_to_call: list[ToolCallItem] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def prevent_duplicate_parallel_calls(self) -> Self:
        """يمنع الوكيل من طلب تنفيذ نفس الأداة بنفس المعطيات مرتين في نفس اللحظة"""
        seen_signatures: set[str] = set()
        for item in self.tools_to_call:
            # تكوين بصمة فريدة للأداة ومدخلاتها المرتبة
            signature = f"{item.name}:{sorted(item.args.items())}"
            if signature in seen_signatures:
                raise ValueError(
                    f"Duplicate parallel tool call detected for '{item.name}' with identical args. "
                    "Call each unique tool/argument combination only once per batch."
                )
            seen_signatures.add(signature)
        return self

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ⚙️ إعداد البيئة — يعمل على Google Colab و Kaggle ومحلياً
from __future__ import annotations   # ينتقل تلقائياً إلى كل خلايا الدفتر التالية

import sys, subprocess, types, typing, warnings

if sys.version_info < (3, 12):
    raise RuntimeError(
        "هذا الدفتر يتطلب Python 3.12 أو أحدث (الكتاب يستهدف 3.14+). "
        f"الإصدار الحالي: {sys.version.split()[0]}"
    )

try:
    import pydantic  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pydantic>=2.9"], check=True)

# طبقة توافق: annotationlib متاحة أصلاً في Python 3.14 فقط
if sys.version_info < (3, 14):
    _al = types.ModuleType("annotationlib")

    class _Format:
        VALUE, FORWARDREF, STRING = 1, 2, 3

    _al.Format = _Format
    _al.get_annotations = lambda obj, **kw: typing.get_type_hints(obj, include_extras=True)
    sys.modules.setdefault("annotationlib", _al)

warnings.filterwarnings("ignore", category=DeprecationWarning)
print("✅ البيئة جاهزة — Python", sys.version.split()[0])

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
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

# في الدفتر لا نستخدم asyncio.run: الحلقة قائمة بالفعل، لذا ننتظر التنفيذ مباشرة.
await main()

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
توافق خلفي (Backport) لمكتبة `annotationlib` القياسية في Python 3.14.

الكتاب يستهدف Python 3.14+ حيث تُقيَّم التعليقات النوعية كسولاً (PEP 649/749).
هذا الملف يوفّر واجهة مطابقة مبسّطة تعمل على Python 3.12/3.13 كي تبقى أمثلة
الفصلين الثاني والرابع قابلة للتشغيل على Colab وKaggle وأي بيئة أقدم.
"""
from __future__ import annotations

import sys
import typing

__all__ = ["Format", "get_annotations"]


class Format:
    """تنسيقات القراءة الثلاثة (مطابقة لأسماء Python 3.14)."""

    VALUE = 1
    FORWARDREF = 2
    STRING = 3


def get_annotations(
    obj: object,
    *,
    format: int = Format.VALUE,  # noqa: A002 - مطابقة لتوقيع المكتبة القياسية
    globals: dict | None = None,  # noqa: A002
    locals: dict | None = None,  # noqa: A002
    eval_str: bool = False,
) -> dict[str, typing.Any]:
    """يعيد تعليقات الكائن النوعية؛ يحلّ الإشارات الأمامية عند طلب VALUE."""
    if format == Format.STRING and not eval_str:
        return dict(getattr(obj, "__annotations__", {}) or {})

    try:
        return typing.get_type_hints(
            obj, globalns=globals, localns=locals, include_extras=True
        )
    except Exception:  # pragma: no cover - سلوك احتياطي متسامح
        return dict(getattr(obj, "__annotations__", {}) or {})


if sys.version_info >= (3, 14):  # pragma: no cover
    from annotationlib import Format, get_annotations  # type: ignore[no-redef]

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
❌ ANTI-PATTERN: خداع النفس بـ async def مع كود متزامن يجمد النظام بالكامل!

مقتطف من الفصل 2 — «تسخير قوة Python 3.14+ لهندسة الوكلاء».
النوع: anti_pattern. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch02_python314_environment.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ❌ ANTI-PATTERN: خداع النفس بـ async def مع كود متزامن يجمد النظام بالكامل!
import requests


async def bad_web_search_tool(query: str) -> str:
    # كارثة: مكتبة requests متزامنة! هذا السطر سيجمد كل الوكلاء والمستخدمين على الخادم
    # حتى يرد الموقع الخارجي بعد 5 ثوانٍ!
    response = requests.get(f"https://api.example.com/search?q={query}", timeout=5)
    return response.text

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
✅ BEST PRACTICE: استخدام مكتبات غير متزامنة أصيلة (مثل httpx) للـ I/O، و to_thread للـ CPU

مقتطف من الفصل 2 — «تسخير قوة Python 3.14+ لهندسة الوكلاء».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch02_python314_environment.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ✅ BEST PRACTICE: استخدام مكتبات غير متزامنة أصيلة (مثل httpx) للـ I/O، و to_thread للـ CPU
import httpx


async def good_web_search_tool(query: str, client: httpx.AsyncClient) -> str:
    # ممتاز: httpx غير متزامن ويحرر حلقة الأحداث لخدمة مهام أخرى أثناء انتظار الشبكة
    response = await client.get(f"https://api.example.com/search?q={query}", timeout=5.0)
    return response.text

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
from typing import Any, Literal, Self

مقتطف من الفصل 2 — «تسخير قوة Python 3.14+ لهندسة الوكلاء».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch02_python314_environment.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

from typing import Any, Literal, Self
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BaseAgentEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class ToolCallItem(BaseAgentEvent):
    """نموذج صارم لكل أداة يطلبها الوكيل"""

    name: str = Field(
        min_length=2,
        max_length=64,
        pattern=r"^[a-z0-9_]+$",  # يمنع أي محاولة لحقن رموز غريبة في اسم الأداة
        description="اسم الأداة بالأحرف الإنجليزية الصغيرة والشرطة السفلية فقط",
    )
    args: dict[str, Any] = Field(default_factory=dict)


class ExecuteToolDecision(BaseAgentEvent):
    decision_type: Literal["execute_tools"] = "execute_tools"
    thought: str = Field(min_length=5)
    tools_to_call: list[ToolCallItem] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def prevent_duplicate_parallel_calls(self) -> Self:
        """يمنع الوكيل من طلب تنفيذ نفس الأداة بنفس المعطيات مرتين في نفس اللحظة"""
        seen_signatures: set[str] = set()
        for item in self.tools_to_call:
            # تكوين بصمة فريدة للأداة ومدخلاتها المرتبة
            signature = f"{item.name}:{sorted(item.args.items())}"
            if signature in seen_signatures:
                raise ValueError(
                    f"Duplicate parallel tool call detected for '{item.name}' with identical args. "
                    "Call each unique tool/argument combination only once per batch."
                )
            seen_signatures.add(signature)
        return self

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import annotationlib  # ميزة جديدة أصيلة في Python 3.14 لفحص الأنواع بذكاء!
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

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
# ❌ ANTI-PATTERN: خداع النفس بـ async def مع كود متزامن يجمد النظام بالكامل!
import requests


async def bad_web_search_tool(query: str) -> str:
    # كارثة: مكتبة requests متزامنة! هذا السطر سيجمد كل الوكلاء والمستخدمين على الخادم
    # حتى يرد الموقع الخارجي بعد 5 ثوانٍ!
    response = requests.get(f"https://api.example.com/search?q={query}", timeout=5)
    return response.text

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
# ✅ BEST PRACTICE: استخدام مكتبات غير متزامنة أصيلة (مثل httpx) للـ I/O، و to_thread للـ CPU
import httpx


async def good_web_search_tool(query: str, client: httpx.AsyncClient) -> str:
    # ممتاز: httpx غير متزامن ويحرر حلقة الأحداث لخدمة مهام أخرى أثناء انتظار الشبكة
    response = await client.get(f"https://api.example.com/search?q={query}", timeout=5.0)
    return response.text

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
from typing import Any, Literal, Self
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BaseAgentEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class ToolCallItem(BaseAgentEvent):
    """نموذج صارم لكل أداة يطلبها الوكيل"""

    name: str = Field(
        min_length=2,
        max_length=64,
        pattern=r"^[a-z0-9_]+$",  # يمنع أي محاولة لحقن رموز غريبة في اسم الأداة
        description="اسم الأداة بالأحرف الإنجليزية الصغيرة والشرطة السفلية فقط",
    )
    args: dict[str, Any] = Field(default_factory=dict)


class ExecuteToolDecision(BaseAgentEvent):
    decision_type: Literal["execute_tools"] = "execute_tools"
    thought: str = Field(min_length=5)
    tools_to_call: list[ToolCallItem] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def prevent_duplicate_parallel_calls(self) -> Self:
        """يمنع الوكيل من طلب تنفيذ نفس الأداة بنفس المعطيات مرتين في نفس اللحظة"""
        seen_signatures: set[str] = set()
        for item in self.tools_to_call:
            # تكوين بصمة فريدة للأداة ومدخلاتها المرتبة
            signature = f"{item.name}:{sorted(item.args.items())}"
            if signature in seen_signatures:
                raise ValueError(
                    f"Duplicate parallel tool call detected for '{item.name}' with identical args. "
                    "Call each unique tool/argument combination only once per batch."
                )
            seen_signatures.add(signature)
        return self

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — تسخير قوة Python 3.14+ لهندسة الوكلاء
AI Agent Engineering — Harnessing Python 3.14+ for Agent Engineering

Book slug : agent-harness-engineering
Chapter   : ch-02-ch02 (lesson 2)
Code files: 15 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/ch02_python314_environment/ch02_python314_environment.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ⚙️ إعداد البيئة — يعمل على Google Colab و Kaggle ومحلياً
from __future__ import annotations   # ينتقل تلقائياً إلى كل خلايا الدفتر التالية

import sys, subprocess, types, typing, warnings

if sys.version_info < (3, 12):
    raise RuntimeError(
        "هذا الدفتر يتطلب Python 3.12 أو أحدث (الكتاب يستهدف 3.14+). "
        f"الإصدار الحالي: {sys.version.split()[0]}"
    )

try:
    import pydantic  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pydantic>=2.9"], check=True)

# طبقة توافق: annotationlib متاحة أصلاً في Python 3.14 فقط
if sys.version_info < (3, 14):
    _al = types.ModuleType("annotationlib")

    class _Format:
        VALUE, FORWARDREF, STRING = 1, 2, 3

    _al.Format = _Format
    _al.get_annotations = lambda obj, **kw: typing.get_type_hints(obj, include_extras=True)
    sys.modules.setdefault("annotationlib", _al)

warnings.filterwarnings("ignore", category=DeprecationWarning)
print("✅ البيئة جاهزة — Python", sys.version.split()[0])

# ────────────────────────────────────────────────────────────────────────
# [002] code/ch02_python314_environment/ch02_python314_environment.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] code/ch02_python314_environment/ch02_python314_environment.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] code/ch02_python314_environment/ch02_python314_environment.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [005] code/ch02_python314_environment/ch02_python314_environment.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] code/ch02_python314_environment/ch02_python314_environment.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
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

# في الدفتر لا نستخدم asyncio.run: الحلقة قائمة بالفعل، لذا ننتظر التنفيذ مباشرة.
await main()

# ────────────────────────────────────────────────────────────────────────
# [007] code/ch02_python314_environment/ch02_python314_environment.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [008] code/ch02_python314_environment/harness_compat.py
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
توافق خلفي (Backport) لمكتبة `annotationlib` القياسية في Python 3.14.

الكتاب يستهدف Python 3.14+ حيث تُقيَّم التعليقات النوعية كسولاً (PEP 649/749).
هذا الملف يوفّر واجهة مطابقة مبسّطة تعمل على Python 3.12/3.13 كي تبقى أمثلة
الفصلين الثاني والرابع قابلة للتشغيل على Colab وKaggle وأي بيئة أقدم.
"""
from __future__ import annotations

import sys
import typing

__all__ = ["Format", "get_annotations"]


class Format:
    """تنسيقات القراءة الثلاثة (مطابقة لأسماء Python 3.14)."""

    VALUE = 1
    FORWARDREF = 2
    STRING = 3


def get_annotations(
    obj: object,
    *,
    format: int = Format.VALUE,  # noqa: A002 - مطابقة لتوقيع المكتبة القياسية
    globals: dict | None = None,  # noqa: A002
    locals: dict | None = None,  # noqa: A002
    eval_str: bool = False,
) -> dict[str, typing.Any]:
    """يعيد تعليقات الكائن النوعية؛ يحلّ الإشارات الأمامية عند طلب VALUE."""
    if format == Format.STRING and not eval_str:
        return dict(getattr(obj, "__annotations__", {}) or {})

    try:
        return typing.get_type_hints(
            obj, globalns=globals, localns=locals, include_extras=True
        )
    except Exception:  # pragma: no cover - سلوك احتياطي متسامح
        return dict(getattr(obj, "__annotations__", {}) or {})


if sys.version_info >= (3, 14):  # pragma: no cover
    from annotationlib import Format, get_annotations  # type: ignore[no-redef]

# ────────────────────────────────────────────────────────────────────────
# [009] code/ch02_python314_environment/snippets/anti_pattern_01.py
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
❌ ANTI-PATTERN: خداع النفس بـ async def مع كود متزامن يجمد النظام بالكامل!

مقتطف من الفصل 2 — «تسخير قوة Python 3.14+ لهندسة الوكلاء».
النوع: anti_pattern. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch02_python314_environment.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ❌ ANTI-PATTERN: خداع النفس بـ async def مع كود متزامن يجمد النظام بالكامل!
import requests


async def bad_web_search_tool(query: str) -> str:
    # كارثة: مكتبة requests متزامنة! هذا السطر سيجمد كل الوكلاء والمستخدمين على الخادم
    # حتى يرد الموقع الخارجي بعد 5 ثوانٍ!
    response = requests.get(f"https://api.example.com/search?q={query}", timeout=5)
    return response.text

# ────────────────────────────────────────────────────────────────────────
# [010] code/ch02_python314_environment/snippets/example_02.py
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
✅ BEST PRACTICE: استخدام مكتبات غير متزامنة أصيلة (مثل httpx) للـ I/O، و to_thread للـ CPU

مقتطف من الفصل 2 — «تسخير قوة Python 3.14+ لهندسة الوكلاء».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch02_python314_environment.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ✅ BEST PRACTICE: استخدام مكتبات غير متزامنة أصيلة (مثل httpx) للـ I/O، و to_thread للـ CPU
import httpx


async def good_web_search_tool(query: str, client: httpx.AsyncClient) -> str:
    # ممتاز: httpx غير متزامن ويحرر حلقة الأحداث لخدمة مهام أخرى أثناء انتظار الشبكة
    response = await client.get(f"https://api.example.com/search?q={query}", timeout=5.0)
    return response.text

# ────────────────────────────────────────────────────────────────────────
# [011] code/ch02_python314_environment/snippets/example_03.py
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
from typing import Any, Literal, Self

مقتطف من الفصل 2 — «تسخير قوة Python 3.14+ لهندسة الوكلاء».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch02_python314_environment.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

from typing import Any, Literal, Self
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BaseAgentEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class ToolCallItem(BaseAgentEvent):
    """نموذج صارم لكل أداة يطلبها الوكيل"""

    name: str = Field(
        min_length=2,
        max_length=64,
        pattern=r"^[a-z0-9_]+$",  # يمنع أي محاولة لحقن رموز غريبة في اسم الأداة
        description="اسم الأداة بالأحرف الإنجليزية الصغيرة والشرطة السفلية فقط",
    )
    args: dict[str, Any] = Field(default_factory=dict)


class ExecuteToolDecision(BaseAgentEvent):
    decision_type: Literal["execute_tools"] = "execute_tools"
    thought: str = Field(min_length=5)
    tools_to_call: list[ToolCallItem] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def prevent_duplicate_parallel_calls(self) -> Self:
        """يمنع الوكيل من طلب تنفيذ نفس الأداة بنفس المعطيات مرتين في نفس اللحظة"""
        seen_signatures: set[str] = set()
        for item in self.tools_to_call:
            # تكوين بصمة فريدة للأداة ومدخلاتها المرتبة
            signature = f"{item.name}:{sorted(item.args.items())}"
            if signature in seen_signatures:
                raise ValueError(
                    f"Duplicate parallel tool call detected for '{item.name}' with identical args. "
                    "Call each unique tool/argument combination only once per batch."
                )
            seen_signatures.add(signature)
        return self

# ────────────────────────────────────────────────────────────────────────
# [012] dist/md/ch02.md
# ────────────────────────────────────────────────────────────────────────
import annotationlib  # ميزة جديدة أصيلة في Python 3.14 لفحص الأنواع بذكاء!
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

# ────────────────────────────────────────────────────────────────────────
# [013] dist/md/ch02.md
# ────────────────────────────────────────────────────────────────────────
# ❌ ANTI-PATTERN: خداع النفس بـ async def مع كود متزامن يجمد النظام بالكامل!
import requests


async def bad_web_search_tool(query: str) -> str:
    # كارثة: مكتبة requests متزامنة! هذا السطر سيجمد كل الوكلاء والمستخدمين على الخادم
    # حتى يرد الموقع الخارجي بعد 5 ثوانٍ!
    response = requests.get(f"https://api.example.com/search?q={query}", timeout=5)
    return response.text

# ────────────────────────────────────────────────────────────────────────
# [014] dist/md/ch02.md
# ────────────────────────────────────────────────────────────────────────
# ✅ BEST PRACTICE: استخدام مكتبات غير متزامنة أصيلة (مثل httpx) للـ I/O، و to_thread للـ CPU
import httpx


async def good_web_search_tool(query: str, client: httpx.AsyncClient) -> str:
    # ممتاز: httpx غير متزامن ويحرر حلقة الأحداث لخدمة مهام أخرى أثناء انتظار الشبكة
    response = await client.get(f"https://api.example.com/search?q={query}", timeout=5.0)
    return response.text

# ────────────────────────────────────────────────────────────────────────
# [015] dist/md/ch02.md
# ────────────────────────────────────────────────────────────────────────
from typing import Any, Literal, Self
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BaseAgentEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class ToolCallItem(BaseAgentEvent):
    """نموذج صارم لكل أداة يطلبها الوكيل"""

    name: str = Field(
        min_length=2,
        max_length=64,
        pattern=r"^[a-z0-9_]+$",  # يمنع أي محاولة لحقن رموز غريبة في اسم الأداة
        description="اسم الأداة بالأحرف الإنجليزية الصغيرة والشرطة السفلية فقط",
    )
    args: dict[str, Any] = Field(default_factory=dict)


class ExecuteToolDecision(BaseAgentEvent):
    decision_type: Literal["execute_tools"] = "execute_tools"
    thought: str = Field(min_length=5)
    tools_to_call: list[ToolCallItem] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def prevent_duplicate_parallel_calls(self) -> Self:
        """يمنع الوكيل من طلب تنفيذ نفس الأداة بنفس المعطيات مرتين في نفس اللحظة"""
        seen_signatures: set[str] = set()
        for item in self.tools_to_call:
            # تكوين بصمة فريدة للأداة ومدخلاتها المرتبة
            signature = f"{item.name}:{sorted(item.args.items())}"
            if signature in seen_signatures:
                raise ValueError(
                    f"Duplicate parallel tool call detected for '{item.name}' with identical args. "
                    "Call each unique tool/argument combination only once per batch."
                )
            seen_signatures.add(signature)
        return self

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ⚙️ إعداد البيئة — يعمل على Google Colab و Kaggle ومحلياً
from __future__ import annotations   # ينتقل تلقائياً إلى كل خلايا الدفتر التالية

import sys, subprocess, types, typing, warnings

if sys.version_info < (3, 12):
    raise RuntimeError(
        "هذا الدفتر يتطلب Python 3.12 أو أحدث (الكتاب يستهدف 3.14+). "
        f"الإصدار الحالي: {sys.version.split()[0]}"
    )

try:
    import pydantic  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pydantic>=2.9"], check=True)

# طبقة توافق: annotationlib متاحة أصلاً في Python 3.14 فقط
if sys.version_info < (3, 14):
    _al = types.ModuleType("annotationlib")

    class _Format:
        VALUE, FORWARDREF, STRING = 1, 2, 3

    _al.Format = _Format
    _al.get_annotations = lambda obj, **kw: typing.get_type_hints(obj, include_extras=True)
    sys.modules.setdefault("annotationlib", _al)

warnings.filterwarnings("ignore", category=DeprecationWarning)
print("✅ البيئة جاهزة — Python", sys.version.split()[0])

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
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

# في الدفتر لا نستخدم asyncio.run: الحلقة قائمة بالفعل، لذا ننتظر التنفيذ مباشرة.
await main()

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
توافق خلفي (Backport) لمكتبة `annotationlib` القياسية في Python 3.14.

الكتاب يستهدف Python 3.14+ حيث تُقيَّم التعليقات النوعية كسولاً (PEP 649/749).
هذا الملف يوفّر واجهة مطابقة مبسّطة تعمل على Python 3.12/3.13 كي تبقى أمثلة
الفصلين الثاني والرابع قابلة للتشغيل على Colab وKaggle وأي بيئة أقدم.
"""
from __future__ import annotations

import sys
import typing

__all__ = ["Format", "get_annotations"]


class Format:
    """تنسيقات القراءة الثلاثة (مطابقة لأسماء Python 3.14)."""

    VALUE = 1
    FORWARDREF = 2
    STRING = 3


def get_annotations(
    obj: object,
    *,
    format: int = Format.VALUE,  # noqa: A002 - مطابقة لتوقيع المكتبة القياسية
    globals: dict | None = None,  # noqa: A002
    locals: dict | None = None,  # noqa: A002
    eval_str: bool = False,
) -> dict[str, typing.Any]:
    """يعيد تعليقات الكائن النوعية؛ يحلّ الإشارات الأمامية عند طلب VALUE."""
    if format == Format.STRING and not eval_str:
        return dict(getattr(obj, "__annotations__", {}) or {})

    try:
        return typing.get_type_hints(
            obj, globalns=globals, localns=locals, include_extras=True
        )
    except Exception:  # pragma: no cover - سلوك احتياطي متسامح
        return dict(getattr(obj, "__annotations__", {}) or {})


if sys.version_info >= (3, 14):  # pragma: no cover
    from annotationlib import Format, get_annotations  # type: ignore[no-redef]

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
❌ ANTI-PATTERN: خداع النفس بـ async def مع كود متزامن يجمد النظام بالكامل!

مقتطف من الفصل 2 — «تسخير قوة Python 3.14+ لهندسة الوكلاء».
النوع: anti_pattern. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch02_python314_environment.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ❌ ANTI-PATTERN: خداع النفس بـ async def مع كود متزامن يجمد النظام بالكامل!
import requests


async def bad_web_search_tool(query: str) -> str:
    # كارثة: مكتبة requests متزامنة! هذا السطر سيجمد كل الوكلاء والمستخدمين على الخادم
    # حتى يرد الموقع الخارجي بعد 5 ثوانٍ!
    response = requests.get(f"https://api.example.com/search?q={query}", timeout=5)
    return response.text

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
✅ BEST PRACTICE: استخدام مكتبات غير متزامنة أصيلة (مثل httpx) للـ I/O، و to_thread للـ CPU

مقتطف من الفصل 2 — «تسخير قوة Python 3.14+ لهندسة الوكلاء».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch02_python314_environment.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ✅ BEST PRACTICE: استخدام مكتبات غير متزامنة أصيلة (مثل httpx) للـ I/O، و to_thread للـ CPU
import httpx


async def good_web_search_tool(query: str, client: httpx.AsyncClient) -> str:
    # ممتاز: httpx غير متزامن ويحرر حلقة الأحداث لخدمة مهام أخرى أثناء انتظار الشبكة
    response = await client.get(f"https://api.example.com/search?q={query}", timeout=5.0)
    return response.text

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
from typing import Any, Literal, Self

مقتطف من الفصل 2 — «تسخير قوة Python 3.14+ لهندسة الوكلاء».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch02_python314_environment.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

from typing import Any, Literal, Self
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BaseAgentEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class ToolCallItem(BaseAgentEvent):
    """نموذج صارم لكل أداة يطلبها الوكيل"""

    name: str = Field(
        min_length=2,
        max_length=64,
        pattern=r"^[a-z0-9_]+$",  # يمنع أي محاولة لحقن رموز غريبة في اسم الأداة
        description="اسم الأداة بالأحرف الإنجليزية الصغيرة والشرطة السفلية فقط",
    )
    args: dict[str, Any] = Field(default_factory=dict)


class ExecuteToolDecision(BaseAgentEvent):
    decision_type: Literal["execute_tools"] = "execute_tools"
    thought: str = Field(min_length=5)
    tools_to_call: list[ToolCallItem] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def prevent_duplicate_parallel_calls(self) -> Self:
        """يمنع الوكيل من طلب تنفيذ نفس الأداة بنفس المعطيات مرتين في نفس اللحظة"""
        seen_signatures: set[str] = set()
        for item in self.tools_to_call:
            # تكوين بصمة فريدة للأداة ومدخلاتها المرتبة
            signature = f"{item.name}:{sorted(item.args.items())}"
            if signature in seen_signatures:
                raise ValueError(
                    f"Duplicate parallel tool call detected for '{item.name}' with identical args. "
                    "Call each unique tool/argument combination only once per batch."
                )
            seen_signatures.add(signature)
        return self

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import annotationlib  # ميزة جديدة أصيلة في Python 3.14 لفحص الأنواع بذكاء!
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

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
# ❌ ANTI-PATTERN: خداع النفس بـ async def مع كود متزامن يجمد النظام بالكامل!
import requests


async def bad_web_search_tool(query: str) -> str:
    # كارثة: مكتبة requests متزامنة! هذا السطر سيجمد كل الوكلاء والمستخدمين على الخادم
    # حتى يرد الموقع الخارجي بعد 5 ثوانٍ!
    response = requests.get(f"https://api.example.com/search?q={query}", timeout=5)
    return response.text

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
# ✅ BEST PRACTICE: استخدام مكتبات غير متزامنة أصيلة (مثل httpx) للـ I/O، و to_thread للـ CPU
import httpx


async def good_web_search_tool(query: str, client: httpx.AsyncClient) -> str:
    # ممتاز: httpx غير متزامن ويحرر حلقة الأحداث لخدمة مهام أخرى أثناء انتظار الشبكة
    response = await client.get(f"https://api.example.com/search?q={query}", timeout=5.0)
    return response.text

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/agent-harness-engineering/notebooks/ch-02-ch02.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
from typing import Any, Literal, Self
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BaseAgentEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class ToolCallItem(BaseAgentEvent):
    """نموذج صارم لكل أداة يطلبها الوكيل"""

    name: str = Field(
        min_length=2,
        max_length=64,
        pattern=r"^[a-z0-9_]+$",  # يمنع أي محاولة لحقن رموز غريبة في اسم الأداة
        description="اسم الأداة بالأحرف الإنجليزية الصغيرة والشرطة السفلية فقط",
    )
    args: dict[str, Any] = Field(default_factory=dict)


class ExecuteToolDecision(BaseAgentEvent):
    decision_type: Literal["execute_tools"] = "execute_tools"
    thought: str = Field(min_length=5)
    tools_to_call: list[ToolCallItem] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def prevent_duplicate_parallel_calls(self) -> Self:
        """يمنع الوكيل من طلب تنفيذ نفس الأداة بنفس المعطيات مرتين في نفس اللحظة"""
        seen_signatures: set[str] = set()
        for item in self.tools_to_call:
            # تكوين بصمة فريدة للأداة ومدخلاتها المرتبة
            signature = f"{item.name}:{sorted(item.args.items())}"
            if signature in seen_signatures:
                raise ValueError(
                    f"Duplicate parallel tool call detected for '{item.name}' with identical args. "
                    "Call each unique tool/argument combination only once per batch."
                )
            seen_signatures.add(signature)
        return self
