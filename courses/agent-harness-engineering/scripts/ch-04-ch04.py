#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — هندسة الأدوات واستدعاء الدوال
AI Agent Engineering — Tooling & Function Calling Architecture

Book slug : agent-harness-engineering
Chapter   : ch-04-ch04 (lesson 4)
Code files: 17 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/agent-harness-engineering/code/ch-04-ch04.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — هندسة الأدوات واستدعاء الدوال
AI Agent Engineering — Tooling & Function Calling Architecture

Book slug : agent-harness-engineering
Chapter   : ch-04-ch04 (lesson 4)
Code files: 16 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/agent-harness-engineering/code/ch-04-ch04.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — هندسة الأدوات واستدعاء الدوال
AI Agent Engineering — Tooling & Function Calling Architecture

Book slug : agent-harness-engineering
Chapter   : ch-04-ch04 (lesson 4)
Code files: 15 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/ch04_tool_registry/ch04_tool_registry.ipynb#cell-1
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
# [002] code/ch04_tool_registry/ch04_tool_registry.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
try:
    import annotationlib  # Python 3.14+ (PEP 649)
except ModuleNotFoundError:  # توافق Python 3.12/3.13
    import harness_compat as annotationlib
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import inspect
import json
import time
from typing import Annotated, Any
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

# ────────────────────────────────────────────────────────────────────────
# [003] code/ch04_tool_registry/ch04_tool_registry.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class _InjectedMarker:
    """علامة داخلية لتمييز المعاملات التي يجب إخفاؤها عن الـ LLM وحقنها من الـ Harness"""

    pass


# نوع مساعد أنيق يستخدمه المبرمج في توقيع الدالة: ctx: Injected[AppContext]
type Injected[T] = Annotated[T, _InjectedMarker()]


def _is_injected_annotation(annotation: Any) -> bool:
    """يفحص ما إذا كان تلميح النوع مغلفاً بـ Injected[T]"""
    # فك تغليف TypeAliasType في حال استخدام `type Injected[T]` في Python 3.12-3.14+
    origin = getattr(annotation, "__value__", annotation)
    metadata = getattr(origin, "__metadata__", ())
    return any(isinstance(m, _InjectedMarker) for m in metadata)

# ────────────────────────────────────────────────────────────────────────
# [004] code/ch04_tool_registry/ch04_tool_registry.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
@dataclass(slots=True, frozen=True)
class ToolCallRequest:
    """الطلب القادم من النموذج اللغوي لتشغيل أداة"""

    call_id: str
    tool_name: str
    raw_arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ToolExecutionOutput:
    """المخرج الموحد والمؤمّن الذي يعود من الـ Harness إلى ذاكرة الوكيل"""

    call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    was_truncated: bool = False
    execution_ms: float = 0.0


@dataclass(slots=True)
class ToolSpec:
    """المواصفات الكاملة للأداة بعد فحصها ذاتياً (Introspection)"""

    name: str
    description: str
    func: Callable[..., Awaitable[Any]]
    args_model: type[BaseModel]  # نموذج Pydantic المولد ديناميكياً للمعاملات العلنية فقط
    injected_param_names: list[str]  # أسماء المعاملات السرية التي سيحقنها الـ Harness
    timeout_sec: float
    max_output_chars: int

    def to_llm_json_schema(self) -> dict[str, Any]:
        """توليد المخطط القياسي المتوافق مع OpenAI / Anthropic / Gemini"""
        schema = self.args_model.model_json_schema()
        # تنظيف الحقول الإضافية لتقليل استهلاك التوكنز
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }

# ────────────────────────────────────────────────────────────────────────
# [005] code/ch04_tool_registry/ch04_tool_registry.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class EnterpriseToolHarness[ContextT]:
    """إطار التحكم بالأدوات من الدرجة الإنتاجية.

    يولد المخططات تلقائياً، يحقن السياق الموثوق خفيةً، ويحمي نافذة السياق من الانفجار.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def tool(
        self,
        *,
        name: str | None = None,
        timeout_sec: float = 15.0,
        max_output_chars: int = 4_000,
    ) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        """Decorator ذكي يفحص الدالة ويبني لها نموذج Pydantic ومخطط LLM تلقائياً"""

        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            if not inspect.iscoroutinefunction(func):
                raise TypeError(
                    f"الأداة '{func.__name__}' يجب أن تكون غير متزامنة (async def)."
                )

            tool_name = name or func.__name__
            doc = inspect.getdoc(func)
            if not doc or len(doc.strip()) < 10:
                raise ValueError(
                    f"الأداة '{tool_name}' تفتقر إلى Docstring تفصيلي! "
                    "تذكر أن الـ Docstring هو الموجه الذي يقرؤه النموذج اللغوي."
                )

            # 1. استخراج الأنواع باستخدام annotationlib الحديثة في Python 3.14
            type_hints = annotationlib.get_annotations(
                func, format=annotationlib.Format.VALUE
            )
            sig = inspect.signature(func)

            public_fields: dict[str, Any] = {}
            injected_params: list[str] = []

            for param_name, param in sig.parameters.items():
                if param_name == "return":
                    continue

                hint = type_hints.get(param_name, Any)

                # 2. فحص هل المعامل سري يُحقن من الـ Harness أم علني يملؤه الـ LLM؟
                if _is_injected_annotation(hint):
                    injected_params.append(param_name)
                    continue

                # 3. إعداد الحقل لنموذج Pydantic الديناميكي
                default_val = ... if param.default is inspect.Parameter.empty else param.default
                public_fields[param_name] = (hint, default_val)

            # 4. إنشاء كلاس Pydantic v2 ديناميكي صارم للمعاملات العلنية فقط
            dynamic_model = create_model(
                f"{tool_name}_ArgsSchema",
                __config__=ConfigDict(extra="forbid", str_strip_whitespace=True),
                **public_fields,
            )

            self._tools[tool_name] = ToolSpec(
                name=tool_name,
                description=doc.strip(),
                func=func,
                args_model=dynamic_model,
                injected_param_names=injected_params,
                timeout_sec=timeout_sec,
                max_output_chars=max_output_chars,
            )
            return func

        return decorator

    def get_llm_schemas(self) -> list[dict[str, Any]]:
        """إرجاع كتالوج الأدوات العلني الجاهز للإرسال للنموذج اللغوي"""
        return [spec.to_llm_json_schema() for spec in self._tools.values()]

    @staticmethod
    def _bound_output(raw_text: str, max_chars: int) -> tuple[str, bool]:
        """حاجز تقليم المخرجات لحماية نافذة السياق من القنابل النصية"""
        if len(raw_text) <= max_chars:
            return raw_text, False

        truncated = raw_text[:max_chars]
        omitted_count = len(raw_text) - max_chars
        notice = (
            f"\n\n... [⚠️ HARNESS TRUNCATION NOTICE: تم اقتصاص {omitted_count} حرفاً إضافياً "
            "لحماية نافذة السياق. استخدم فلاتر أدق أو pagination في استدعائك القادم]."
        )
        return truncated + notice, True

    async def execute_single(
        self,
        request: ToolCallRequest,
        trusted_context: ContextT,
    ) -> ToolExecutionOutput:
        """تنفيذ أداة واحدة مع التحقق، وحقن السياق الموثوق، وتقليم المخرجات"""
        t0 = time.perf_counter()
        spec = self._tools.get(request.tool_name)

        if spec is None:
            available = ", ".join(self._tools.keys())
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{request.tool_name}' not found. Available tools: [{available}]",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        try:
            # 1. التحقق الصارم من المدخلات القادمة من النموذج اللغوي
            validated_public_args = spec.args_model.model_validate(request.raw_arguments)
            final_kwargs = validated_public_args.model_dump()

            # 2. حقن السياق الموثوق (Zero-Trust Injection) دون علم النموذج!
            for injected_name in spec.injected_param_names:
                final_kwargs[injected_name] = trusted_context

            # 3. التنفيذ المحمي بمهلة زمنية
            async with asyncio.timeout(spec.timeout_sec):
                raw_result = await spec.func(**final_kwargs)

            # 4. تنسيق المخرج وتطبيق حاجز التقليم (Output Bounding)
            serialized = (
                json.dumps(raw_result, ensure_ascii=False)
                if isinstance(raw_result, (dict, list))
                else str(raw_result)
            )
            bounded_text, was_truncated = self._bound_output(
                serialized, spec.max_output_chars
            )

            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=bounded_text,
                is_error=False,
                was_truncated=was_truncated,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        except ValidationError as val_err:
            # تغذية راجعة واضحة للنموذج ليصلح معطيات الأداة في الخطوة القادمة
            compact_errors = [
                f"`{'.'.join(str(x) for x in e['loc'])}`: {e['msg']}"
                for e in val_err.errors(include_url=False)
            ]
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Tool Argument Validation Error: {'; '.join(compact_errors)}",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except TimeoutError:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{spec.name}' timed out after {spec.timeout_sec}s.",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except Exception as exc:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Tool Runtime Error ({type(exc).__name__}): {exc}",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

    async def execute_batch(
        self,
        requests: list[ToolCallRequest],
        trusted_context: ContextT,
    ) -> list[ToolExecutionOutput]:
        """تنفيذ دفعة من الأدوات بالتوازي المهيكل عبر TaskGroup"""
        tasks: list[asyncio.Task[ToolExecutionOutput]] = []
        async with asyncio.TaskGroup() as tg:
            for req in requests:
                tasks.append(tg.create_task(self.execute_single(req, trusted_context)))
        return [t.result() for t in tasks]

# ────────────────────────────────────────────────────────────────────────
# [006] code/ch04_tool_registry/ch04_tool_registry.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
@dataclass(slots=True, frozen=True)
class SecurityContext:
    """سياق الأمان الموثوق الذي يملكه الخادم ولا يراه النموذج اللغوي أبداً"""

    authenticated_user_id: str
    tenant_org_id: str
    db_connection_string: str


# إنشاء نسخة من الـ Harness مخصصة لنوع السياق الأمني الخاص بنا
tool_harness = EnterpriseToolHarness[SecurityContext]()


@tool_harness.tool(timeout_sec=5.0, max_output_chars=500)
async def list_organization_invoices(
    status: Annotated[str, Field(pattern=r"^(paid|unpaid|overdue)$", description="حالة الفاتورة المطلوبة")],
    limit: Annotated[int, Field(ge=1, le=50, description="الحد الأقصى لعدد الفواتير")] = 10,
    # لاحظ هنا: المعامل ctx موسوم بـ Injected! لن يظهر في JSON Schema المرسل للـ LLM أبداً!
    ctx: Injected[SecurityContext] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    """جلب قائمة الفواتير المالية الخاصة بالمؤسسة الحالية بناءً على حالتها.

    استخدم هذه الأداة فقط عندما يسأل المستخدم عن الفواتير أو المستحقات المالية.
    """
    await asyncio.sleep(0.05)  # محاكاة استعلام قاعدة البيانات باستخدام ctx.tenant_org_id
    return {
        "verified_tenant": ctx.tenant_org_id,
        "queried_by_user": ctx.authenticated_user_id,
        "filter_status": status,
        "invoices": [
            {"inv_id": f"INV-{i}", "amount_usd": 1250.0 * i, "status": status}
            for i in range(1, limit + 1)
        ],
    }


@tool_harness.tool(timeout_sec=3.0, max_output_chars=180)
async def fetch_raw_server_logs(
    service_name: Annotated[str, Field(min_length=2, description="اسم الخدمة البرمجية")],
) -> str:
    """قراءة سجلات النظام الخام لخدمة معينة لتشخيص الأعطال التقنية."""
    await asyncio.sleep(0.02)
    # محاكاة مخرج ضخم جداً لاختبار حاجز تقليم المخرجات (Output Bounding)
    return "\n".join(
        f"[2025-05-01 12:00:{i:02d}] INFO ({service_name}): Worker processed batch #{i} successfully."
        for i in range(1, 30)
    )


async def main() -> None:
    # 1. لنفحص المخطط (JSON Schema) الذي سيعطيه الـ Harness للنموذج اللغوي!
    schemas = tool_harness.get_llm_schemas()
    print("🔍 المخطط العلني المولد تلقائياً لأداة الفواتير (لاحظ اختفاء ctx تماماً!):")
    print(json.dumps(schemas[0], indent=2, ensure_ascii=False), "\n")

    # 2. إعداد السياق الموثوق على الخادم (قادم من JWT Token مثلاً)
    server_context = SecurityContext(
        authenticated_user_id="USR-9981",
        tenant_org_id="ORG-ACME-CORP",
        db_connection_string="postgresql://prod-internal:5432/fintech",
    )

    # 3. محاكاة 3 استدعاءات متوازية من النموذج اللغوي:
    # - الأول: استدعاء سليم لأداة الفواتير (سيتم حقن الهوية تلقائياً)
    # - الثاني: محاولة خبيثة من النموذج لتمرير حقل `ctx` أو `tenant_org_id` لتزوير الهوية!
    # - الثالث: استدعاء لأداة السجلات التي تعيد نصاً ضخماً (سيتم تقليمه تلقائياً)
    batch_requests = [
        ToolCallRequest(
            call_id="call_01",
            tool_name="list_organization_invoices",
            raw_arguments={"status": "overdue", "limit": 2},
        ),
        ToolCallRequest(
            call_id="call_02",
            tool_name="list_organization_invoices",
            raw_arguments={"status": "paid", "ctx": "HACKED_CTX"},  # سيرفضه extra="forbid"!
        ),
        ToolCallRequest(
            call_id="call_03",
            tool_name="fetch_raw_server_logs",
            raw_arguments={"service_name": "payment-gateway"},
        ),
    ]

    outputs = await tool_harness.execute_batch(batch_requests, trusted_context=server_context)

    for out in outputs:
        icon = "❌" if out.is_error else ("✂️" if out.was_truncated else "✅")
        print(f"{icon} [{out.tool_name}] (ID: {out.call_id} | {out.execution_ms}ms):")
        print(f"   {out.content}\n")

# في الدفتر لا نستخدم asyncio.run: الحلقة قائمة بالفعل، لذا ننتظر التنفيذ مباشرة.
await main()

# ────────────────────────────────────────────────────────────────────────
# [007] code/ch04_tool_registry/ch04_tool_registry.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
الفصل 4: هندسة الأدوات واستدعاء الدوال
==============================================================================

البرنامج الكامل للفصل 4 من كتاب «هندسة وكلاء الذكاء الاصطناعي».
هندسة الأدوات واستدعاء الدوال — Tooling & Function Calling Architecture.

التشغيل:
    python ch04_tool_registry.py

المتطلبات: Python 3.12+ (الكتاب يستهدف 3.14+) و pydantic>=2.9
"""
# ملاحظة توافق: تُقيَّم الأنواع كسولاً في Python 3.14 (PEP 649)؛
# هذا السطر يمنح السلوك نفسه على Python 3.12/3.13.
from __future__ import annotations

try:
    import annotationlib  # Python 3.14+ (PEP 649)
except ModuleNotFoundError:  # توافق Python 3.12/3.13
    import harness_compat as annotationlib
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import inspect
import json
import time
from typing import Annotated, Any
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model


# =====================================================================
# 1. آلية حقن الاعتماديات المخفية (Zero-Trust Dependency Injection Marker)
# =====================================================================
class _InjectedMarker:
    """علامة داخلية لتمييز المعاملات التي يجب إخفاؤها عن الـ LLM وحقنها من الـ Harness"""

    pass


# نوع مساعد أنيق يستخدمه المبرمج في توقيع الدالة: ctx: Injected[AppContext]
type Injected[T] = Annotated[T, _InjectedMarker()]


def _is_injected_annotation(annotation: Any) -> bool:
    """يفحص ما إذا كان تلميح النوع مغلفاً بـ Injected[T]"""
    # فك تغليف TypeAliasType في حال استخدام `type Injected[T]` في Python 3.12-3.14+
    origin = getattr(annotation, "__value__", annotation)
    metadata = getattr(origin, "__metadata__", ())
    return any(isinstance(m, _InjectedMarker) for m in metadata)


# =====================================================================
# 2. هياكل بيانات التشغيل والمخرجات داخل الـ Tool Harness
# =====================================================================
@dataclass(slots=True, frozen=True)
class ToolCallRequest:
    """الطلب القادم من النموذج اللغوي لتشغيل أداة"""

    call_id: str
    tool_name: str
    raw_arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ToolExecutionOutput:
    """المخرج الموحد والمؤمّن الذي يعود من الـ Harness إلى ذاكرة الوكيل"""

    call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    was_truncated: bool = False
    execution_ms: float = 0.0


@dataclass(slots=True)
class ToolSpec:
    """المواصفات الكاملة للأداة بعد فحصها ذاتياً (Introspection)"""

    name: str
    description: str
    func: Callable[..., Awaitable[Any]]
    args_model: type[BaseModel]  # نموذج Pydantic المولد ديناميكياً للمعاملات العلنية فقط
    injected_param_names: list[str]  # أسماء المعاملات السرية التي سيحقنها الـ Harness
    timeout_sec: float
    max_output_chars: int

    def to_llm_json_schema(self) -> dict[str, Any]:
        """توليد المخطط القياسي المتوافق مع OpenAI / Anthropic / Gemini"""
        schema = self.args_model.model_json_schema()
        # تنظيف الحقول الإضافية لتقليل استهلاك التوكنز
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


# =====================================================================
# 3. قلب النظام: EnterpriseToolHarness (باستخدام PEP 695 Generics)
# =====================================================================
class EnterpriseToolHarness[ContextT]:
    """إطار التحكم بالأدوات من الدرجة الإنتاجية.

    يولد المخططات تلقائياً، يحقن السياق الموثوق خفيةً، ويحمي نافذة السياق من الانفجار.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def tool(
        self,
        *,
        name: str | None = None,
        timeout_sec: float = 15.0,
        max_output_chars: int = 4_000,
    ) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        """Decorator ذكي يفحص الدالة ويبني لها نموذج Pydantic ومخطط LLM تلقائياً"""

        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            if not inspect.iscoroutinefunction(func):
                raise TypeError(
                    f"الأداة '{func.__name__}' يجب أن تكون غير متزامنة (async def)."
                )

            tool_name = name or func.__name__
            doc = inspect.getdoc(func)
            if not doc or len(doc.strip()) < 10:
                raise ValueError(
                    f"الأداة '{tool_name}' تفتقر إلى Docstring تفصيلي! "
                    "تذكر أن الـ Docstring هو الموجه الذي يقرؤه النموذج اللغوي."
                )

            # 1. استخراج الأنواع باستخدام annotationlib الحديثة في Python 3.14
            type_hints = annotationlib.get_annotations(
                func, format=annotationlib.Format.VALUE
            )
            sig = inspect.signature(func)

            public_fields: dict[str, Any] = {}
            injected_params: list[str] = []

            for param_name, param in sig.parameters.items():
                if param_name == "return":
                    continue

                hint = type_hints.get(param_name, Any)

                # 2. فحص هل المعامل سري يُحقن من الـ Harness أم علني يملؤه الـ LLM؟
                if _is_injected_annotation(hint):
                    injected_params.append(param_name)
                    continue

                # 3. إعداد الحقل لنموذج Pydantic الديناميكي
                default_val = ... if param.default is inspect.Parameter.empty else param.default
                public_fields[param_name] = (hint, default_val)

            # 4. إنشاء كلاس Pydantic v2 ديناميكي صارم للمعاملات العلنية فقط
            dynamic_model = create_model(
                f"{tool_name}_ArgsSchema",
                __config__=ConfigDict(extra="forbid", str_strip_whitespace=True),
                **public_fields,
            )

            self._tools[tool_name] = ToolSpec(
                name=tool_name,
                description=doc.strip(),
                func=func,
                args_model=dynamic_model,
                injected_param_names=injected_params,
                timeout_sec=timeout_sec,
                max_output_chars=max_output_chars,
            )
            return func

        return decorator

    def get_llm_schemas(self) -> list[dict[str, Any]]:
        """إرجاع كتالوج الأدوات العلني الجاهز للإرسال للنموذج اللغوي"""
        return [spec.to_llm_json_schema() for spec in self._tools.values()]

    @staticmethod
    def _bound_output(raw_text: str, max_chars: int) -> tuple[str, bool]:
        """حاجز تقليم المخرجات لحماية نافذة السياق من القنابل النصية"""
        if len(raw_text) <= max_chars:
            return raw_text, False

        truncated = raw_text[:max_chars]
        omitted_count = len(raw_text) - max_chars
        notice = (
            f"\n\n... [⚠️ HARNESS TRUNCATION NOTICE: تم اقتصاص {omitted_count} حرفاً إضافياً "
            "لحماية نافذة السياق. استخدم فلاتر أدق أو pagination في استدعائك القادم]."
        )
        return truncated + notice, True

    async def execute_single(
        self,
        request: ToolCallRequest,
        trusted_context: ContextT,
    ) -> ToolExecutionOutput:
        """تنفيذ أداة واحدة مع التحقق، وحقن السياق الموثوق، وتقليم المخرجات"""
        t0 = time.perf_counter()
        spec = self._tools.get(request.tool_name)

        if spec is None:
            available = ", ".join(self._tools.keys())
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{request.tool_name}' not found. Available tools: [{available}]",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        try:
            # 1. التحقق الصارم من المدخلات القادمة من النموذج اللغوي
            validated_public_args = spec.args_model.model_validate(request.raw_arguments)
            final_kwargs = validated_public_args.model_dump()

            # 2. حقن السياق الموثوق (Zero-Trust Injection) دون علم النموذج!
            for injected_name in spec.injected_param_names:
                final_kwargs[injected_name] = trusted_context

            # 3. التنفيذ المحمي بمهلة زمنية
            async with asyncio.timeout(spec.timeout_sec):
                raw_result = await spec.func(**final_kwargs)

            # 4. تنسيق المخرج وتطبيق حاجز التقليم (Output Bounding)
            serialized = (
                json.dumps(raw_result, ensure_ascii=False)
                if isinstance(raw_result, (dict, list))
                else str(raw_result)
            )
            bounded_text, was_truncated = self._bound_output(
                serialized, spec.max_output_chars
            )

            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=bounded_text,
                is_error=False,
                was_truncated=was_truncated,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        except ValidationError as val_err:
            # تغذية راجعة واضحة للنموذج ليصلح معطيات الأداة في الخطوة القادمة
            compact_errors = [
                f"`{'.'.join(str(x) for x in e['loc'])}`: {e['msg']}"
                for e in val_err.errors(include_url=False)
            ]
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Tool Argument Validation Error: {'; '.join(compact_errors)}",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except TimeoutError:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{spec.name}' timed out after {spec.timeout_sec}s.",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except Exception as exc:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Tool Runtime Error ({type(exc).__name__}): {exc}",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

    async def execute_batch(
        self,
        requests: list[ToolCallRequest],
        trusted_context: ContextT,
    ) -> list[ToolExecutionOutput]:
        """تنفيذ دفعة من الأدوات بالتوازي المهيكل عبر TaskGroup"""
        tasks: list[asyncio.Task[ToolExecutionOutput]] = []
        async with asyncio.TaskGroup() as tg:
            for req in requests:
                tasks.append(tg.create_task(self.execute_single(req, trusted_context)))
        return [t.result() for t in tasks]


# =====================================================================
# 4. مثال تطبيقي: أدوات نظام مالي متعددة المستأجرين (Multi-Tenant FinTech)
# =====================================================================
@dataclass(slots=True, frozen=True)
class SecurityContext:
    """سياق الأمان الموثوق الذي يملكه الخادم ولا يراه النموذج اللغوي أبداً"""

    authenticated_user_id: str
    tenant_org_id: str
    db_connection_string: str


# إنشاء نسخة من الـ Harness مخصصة لنوع السياق الأمني الخاص بنا
tool_harness = EnterpriseToolHarness[SecurityContext]()


@tool_harness.tool(timeout_sec=5.0, max_output_chars=500)
async def list_organization_invoices(
    status: Annotated[str, Field(pattern=r"^(paid|unpaid|overdue)$", description="حالة الفاتورة المطلوبة")],
    limit: Annotated[int, Field(ge=1, le=50, description="الحد الأقصى لعدد الفواتير")] = 10,
    # لاحظ هنا: المعامل ctx موسوم بـ Injected! لن يظهر في JSON Schema المرسل للـ LLM أبداً!
    ctx: Injected[SecurityContext] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    """جلب قائمة الفواتير المالية الخاصة بالمؤسسة الحالية بناءً على حالتها.

    استخدم هذه الأداة فقط عندما يسأل المستخدم عن الفواتير أو المستحقات المالية.
    """
    await asyncio.sleep(0.05)  # محاكاة استعلام قاعدة البيانات باستخدام ctx.tenant_org_id
    return {
        "verified_tenant": ctx.tenant_org_id,
        "queried_by_user": ctx.authenticated_user_id,
        "filter_status": status,
        "invoices": [
            {"inv_id": f"INV-{i}", "amount_usd": 1250.0 * i, "status": status}
            for i in range(1, limit + 1)
        ],
    }


@tool_harness.tool(timeout_sec=3.0, max_output_chars=180)
async def fetch_raw_server_logs(
    service_name: Annotated[str, Field(min_length=2, description="اسم الخدمة البرمجية")],
) -> str:
    """قراءة سجلات النظام الخام لخدمة معينة لتشخيص الأعطال التقنية."""
    await asyncio.sleep(0.02)
    # محاكاة مخرج ضخم جداً لاختبار حاجز تقليم المخرجات (Output Bounding)
    return "\n".join(
        f"[2025-05-01 12:00:{i:02d}] INFO ({service_name}): Worker processed batch #{i} successfully."
        for i in range(1, 30)
    )


async def main() -> None:
    # 1. لنفحص المخطط (JSON Schema) الذي سيعطيه الـ Harness للنموذج اللغوي!
    schemas = tool_harness.get_llm_schemas()
    print("🔍 المخطط العلني المولد تلقائياً لأداة الفواتير (لاحظ اختفاء ctx تماماً!):")
    print(json.dumps(schemas[0], indent=2, ensure_ascii=False), "\n")

    # 2. إعداد السياق الموثوق على الخادم (قادم من JWT Token مثلاً)
    server_context = SecurityContext(
        authenticated_user_id="USR-9981",
        tenant_org_id="ORG-ACME-CORP",
        db_connection_string="postgresql://prod-internal:5432/fintech",
    )

    # 3. محاكاة 3 استدعاءات متوازية من النموذج اللغوي:
    # - الأول: استدعاء سليم لأداة الفواتير (سيتم حقن الهوية تلقائياً)
    # - الثاني: محاولة خبيثة من النموذج لتمرير حقل `ctx` أو `tenant_org_id` لتزوير الهوية!
    # - الثالث: استدعاء لأداة السجلات التي تعيد نصاً ضخماً (سيتم تقليمه تلقائياً)
    batch_requests = [
        ToolCallRequest(
            call_id="call_01",
            tool_name="list_organization_invoices",
            raw_arguments={"status": "overdue", "limit": 2},
        ),
        ToolCallRequest(
            call_id="call_02",
            tool_name="list_organization_invoices",
            raw_arguments={"status": "paid", "ctx": "HACKED_CTX"},  # سيرفضه extra="forbid"!
        ),
        ToolCallRequest(
            call_id="call_03",
            tool_name="fetch_raw_server_logs",
            raw_arguments={"service_name": "payment-gateway"},
        ),
    ]

    outputs = await tool_harness.execute_batch(batch_requests, trusted_context=server_context)

    for out in outputs:
        icon = "❌" if out.is_error else ("✂️" if out.was_truncated else "✅")
        print(f"{icon} [{out.tool_name}] (ID: {out.call_id} | {out.execution_ms}ms):")
        print(f"   {out.content}\n")


if __name__ == "__main__":
    asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [008] code/ch04_tool_registry/harness_compat.py
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
# [009] code/ch04_tool_registry/snippets/anti_pattern_02.py
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
❌ ANTI-PATTERN: كارثة أمنية محققة!

مقتطف من الفصل 4 — «هندسة الأدوات واستدعاء الدوال».
النوع: anti_pattern. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch04_tool_registry.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ❌ ANTI-PATTERN: كارثة أمنية محققة!
@app.tool()
async def delete_user_document(user_id: str, doc_id: str) -> str:
    # إذا خدع المهاجم النموذج اللغوي عبر Prompt Injection، سيمرر النموذج user_id الخاص بضحية أخرى!
    ...

# ────────────────────────────────────────────────────────────────────────
# [010] code/ch04_tool_registry/snippets/example_01.py
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
async def get_customer_invoices(user_id: int, status: str, db: DatabaseSession) -> list[dict]:

مقتطف من الفصل 4 — «هندسة الأدوات واستدعاء الدوال».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch04_tool_registry.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

async def get_customer_invoices(user_id: int, status: str, db: DatabaseSession) -> list[dict]:
    ...

# ────────────────────────────────────────────────────────────────────────
# [011] code/ch04_tool_registry/snippets/example_03.py
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
from typing import Protocol

مقتطف من الفصل 4 — «هندسة الأدوات واستدعاء الدوال».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch04_tool_registry.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

from typing import Protocol


# 1. تعريف عقد برمجي (Protocol) يضمن أن أي سياق يمرر للـ Harness يمتلك خاصية user_scopes
class HasSecurityScopes(Protocol):
    @property
    def user_scopes(self) -> set[str]:
        ...


# 2. تقييد الـ Generic Type في الـ Harness ليقبل فقط السياقات التي تحقق HasSecurityScopes
class EnterpriseToolHarness[ContextT: HasSecurityScopes]:
    # ... (داخل المزخرف @tool نضيف المعامل required_scopes: set[str] | None = None)
    # ونحفظه داخل ToolSpec كـ required_scopes: frozenset[str] = field(default_factory=frozenset)

    def get_llm_schemas(self, trusted_context: ContextT) -> list[dict[str, object]]:
        """المستوى الأول (حاجز الرؤية): إخفاء الأدوات التي لا يملك المستخدم صلاحياتها عن الـ LLM تماماً"""
        authorized_schemas: list[dict[str, object]] = []
        for spec in self._tools.values():
            if spec.required_scopes.issubset(trusted_context.user_scopes):
                authorized_schemas.append(spec.to_llm_json_schema())
        return authorized_schemas

    async def execute_single(
        self,
        request: ToolCallRequest,
        trusted_context: ContextT,
    ) -> ToolExecutionOutput:
        """المستوى الثاني (الدفاع في العمق - Defense-in-Depth): التحقق الصارم قبل التنفيذ الفعلي"""
        spec = self._tools.get(request.tool_name)
        if spec is None:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{request.tool_name}' is not available.",
                is_error=True,
            )

        # فحص الصلاحيات وقت التشغيل لمنع استدعاء الأدوات المخفية عبر التخمين أو الحقن!
        missing_scopes = spec.required_scopes - trusted_context.user_scopes
        if missing_scopes:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Authorization Denied: Missing required scopes {sorted(missing_scopes)} to run '{spec.name}'.",
                is_error=True,
            )

        # ... استكمال التحقق من المعطيات والتنفيذ كالمعتاد

# ────────────────────────────────────────────────────────────────────────
# [012] dist/md/ch04.md
# ────────────────────────────────────────────────────────────────────────
async def get_customer_invoices(user_id: int, status: str, db: DatabaseSession) -> list[dict]:
    ...

# ────────────────────────────────────────────────────────────────────────
# [013] dist/md/ch04.md
# ────────────────────────────────────────────────────────────────────────
import annotationlib  # مكتبة Python 3.14 القياسية لفحص الأنواع بدقة وأمان
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import inspect
import json
import time
from typing import Annotated, Any
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model


# =====================================================================
# 1. آلية حقن الاعتماديات المخفية (Zero-Trust Dependency Injection Marker)
# =====================================================================
class _InjectedMarker:
    """علامة داخلية لتمييز المعاملات التي يجب إخفاؤها عن الـ LLM وحقنها من الـ Harness"""

    pass


# نوع مساعد أنيق يستخدمه المبرمج في توقيع الدالة: ctx: Injected[AppContext]
type Injected[T] = Annotated[T, _InjectedMarker()]


def _is_injected_annotation(annotation: Any) -> bool:
    """يفحص ما إذا كان تلميح النوع مغلفاً بـ Injected[T]"""
    # فك تغليف TypeAliasType في حال استخدام `type Injected[T]` في Python 3.12-3.14+
    origin = getattr(annotation, "__value__", annotation)
    metadata = getattr(origin, "__metadata__", ())
    return any(isinstance(m, _InjectedMarker) for m in metadata)


# =====================================================================
# 2. هياكل بيانات التشغيل والمخرجات داخل الـ Tool Harness
# =====================================================================
@dataclass(slots=True, frozen=True)
class ToolCallRequest:
    """الطلب القادم من النموذج اللغوي لتشغيل أداة"""

    call_id: str
    tool_name: str
    raw_arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ToolExecutionOutput:
    """المخرج الموحد والمؤمّن الذي يعود من الـ Harness إلى ذاكرة الوكيل"""

    call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    was_truncated: bool = False
    execution_ms: float = 0.0


@dataclass(slots=True)
class ToolSpec:
    """المواصفات الكاملة للأداة بعد فحصها ذاتياً (Introspection)"""

    name: str
    description: str
    func: Callable[..., Awaitable[Any]]
    args_model: type[BaseModel]  # نموذج Pydantic المولد ديناميكياً للمعاملات العلنية فقط
    injected_param_names: list[str]  # أسماء المعاملات السرية التي سيحقنها الـ Harness
    timeout_sec: float
    max_output_chars: int

    def to_llm_json_schema(self) -> dict[str, Any]:
        """توليد المخطط القياسي المتوافق مع OpenAI / Anthropic / Gemini"""
        schema = self.args_model.model_json_schema()
        # تنظيف الحقول الإضافية لتقليل استهلاك التوكنز
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


# =====================================================================
# 3. قلب النظام: EnterpriseToolHarness (باستخدام PEP 695 Generics)
# =====================================================================
class EnterpriseToolHarness[ContextT]:
    """إطار التحكم بالأدوات من الدرجة الإنتاجية.

    يولد المخططات تلقائياً، يحقن السياق الموثوق خفيةً، ويحمي نافذة السياق من الانفجار.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def tool(
        self,
        *,
        name: str | None = None,
        timeout_sec: float = 15.0,
        max_output_chars: int = 4_000,
    ) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        """Decorator ذكي يفحص الدالة ويبني لها نموذج Pydantic ومخطط LLM تلقائياً"""

        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            if not inspect.iscoroutinefunction(func):
                raise TypeError(
                    f"الأداة '{func.__name__}' يجب أن تكون غير متزامنة (async def)."
                )

            tool_name = name or func.__name__
            doc = inspect.getdoc(func)
            if not doc or len(doc.strip()) < 10:
                raise ValueError(
                    f"الأداة '{tool_name}' تفتقر إلى Docstring تفصيلي! "
                    "تذكر أن الـ Docstring هو الموجه الذي يقرؤه النموذج اللغوي."
                )

            # 1. استخراج الأنواع باستخدام annotationlib الحديثة في Python 3.14
            type_hints = annotationlib.get_annotations(
                func, format=annotationlib.Format.VALUE
            )
            sig = inspect.signature(func)

            public_fields: dict[str, Any] = {}
            injected_params: list[str] = []

            for param_name, param in sig.parameters.items():
                if param_name == "return":
                    continue

                hint = type_hints.get(param_name, Any)

                # 2. فحص هل المعامل سري يُحقن من الـ Harness أم علني يملؤه الـ LLM؟
                if _is_injected_annotation(hint):
                    injected_params.append(param_name)
                    continue

                # 3. إعداد الحقل لنموذج Pydantic الديناميكي
                default_val = ... if param.default is inspect.Parameter.empty else param.default
                public_fields[param_name] = (hint, default_val)

            # 4. إنشاء كلاس Pydantic v2 ديناميكي صارم للمعاملات العلنية فقط
            dynamic_model = create_model(
                f"{tool_name}_ArgsSchema",
                __config__=ConfigDict(extra="forbid", str_strip_whitespace=True),
                **public_fields,
            )

            self._tools[tool_name] = ToolSpec(
                name=tool_name,
                description=doc.strip(),
                func=func,
                args_model=dynamic_model,
                injected_param_names=injected_params,
                timeout_sec=timeout_sec,
                max_output_chars=max_output_chars,
            )
            return func

        return decorator

    def get_llm_schemas(self) -> list[dict[str, Any]]:
        """إرجاع كتالوج الأدوات العلني الجاهز للإرسال للنموذج اللغوي"""
        return [spec.to_llm_json_schema() for spec in self._tools.values()]

    @staticmethod
    def _bound_output(raw_text: str, max_chars: int) -> tuple[str, bool]:
        """حاجز تقليم المخرجات لحماية نافذة السياق من القنابل النصية"""
        if len(raw_text) <= max_chars:
            return raw_text, False

        truncated = raw_text[:max_chars]
        omitted_count = len(raw_text) - max_chars
        notice = (
            f"\n\n... [⚠️ HARNESS TRUNCATION NOTICE: تم اقتصاص {omitted_count} حرفاً إضافياً "
            "لحماية نافذة السياق. استخدم فلاتر أدق أو pagination في استدعائك القادم]."
        )
        return truncated + notice, True

    async def execute_single(
        self,
        request: ToolCallRequest,
        trusted_context: ContextT,
    ) -> ToolExecutionOutput:
        """تنفيذ أداة واحدة مع التحقق، وحقن السياق الموثوق، وتقليم المخرجات"""
        t0 = time.perf_counter()
        spec = self._tools.get(request.tool_name)

        if spec is None:
            available = ", ".join(self._tools.keys())
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{request.tool_name}' not found. Available tools: [{available}]",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        try:
            # 1. التحقق الصارم من المدخلات القادمة من النموذج اللغوي
            validated_public_args = spec.args_model.model_validate(request.raw_arguments)
            final_kwargs = validated_public_args.model_dump()

            # 2. حقن السياق الموثوق (Zero-Trust Injection) دون علم النموذج!
            for injected_name in spec.injected_param_names:
                final_kwargs[injected_name] = trusted_context

            # 3. التنفيذ المحمي بمهلة زمنية
            async with asyncio.timeout(spec.timeout_sec):
                raw_result = await spec.func(**final_kwargs)

            # 4. تنسيق المخرج وتطبيق حاجز التقليم (Output Bounding)
            serialized = (
                json.dumps(raw_result, ensure_ascii=False)
                if isinstance(raw_result, (dict, list))
                else str(raw_result)
            )
            bounded_text, was_truncated = self._bound_output(
                serialized, spec.max_output_chars
            )

            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=bounded_text,
                is_error=False,
                was_truncated=was_truncated,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        except ValidationError as val_err:
            # تغذية راجعة واضحة للنموذج ليصلح معطيات الأداة في الخطوة القادمة
            compact_errors = [
                f"`{'.'.join(str(x) for x in e['loc'])}`: {e['msg']}"
                for e in val_err.errors(include_url=False)
            ]
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Tool Argument Validation Error: {'; '.join(compact_errors)}",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except TimeoutError:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{spec.name}' timed out after {spec.timeout_sec}s.",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except Exception as exc:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Tool Runtime Error ({type(exc).__name__}): {exc}",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

    async def execute_batch(
        self,
        requests: list[ToolCallRequest],
        trusted_context: ContextT,
    ) -> list[ToolExecutionOutput]:
        """تنفيذ دفعة من الأدوات بالتوازي المهيكل عبر TaskGroup"""
        tasks: list[asyncio.Task[ToolExecutionOutput]] = []
        async with asyncio.TaskGroup() as tg:
            for req in requests:
                tasks.append(tg.create_task(self.execute_single(req, trusted_context)))
        return [t.result() for t in tasks]


# =====================================================================
# 4. مثال تطبيقي: أدوات نظام مالي متعددة المستأجرين (Multi-Tenant FinTech)
# =====================================================================
@dataclass(slots=True, frozen=True)
class SecurityContext:
    """سياق الأمان الموثوق الذي يملكه الخادم ولا يراه النموذج اللغوي أبداً"""

    authenticated_user_id: str
    tenant_org_id: str
    db_connection_string: str


# إنشاء نسخة من الـ Harness مخصصة لنوع السياق الأمني الخاص بنا
tool_harness = EnterpriseToolHarness[SecurityContext]()


@tool_harness.tool(timeout_sec=5.0, max_output_chars=500)
async def list_organization_invoices(
    status: Annotated[str, Field(pattern=r"^(paid|unpaid|overdue)$", description="حالة الفاتورة المطلوبة")],
    limit: Annotated[int, Field(ge=1, le=50, description="الحد الأقصى لعدد الفواتير")] = 10,
    # لاحظ هنا: المعامل ctx موسوم بـ Injected! لن يظهر في JSON Schema المرسل للـ LLM أبداً!
    ctx: Injected[SecurityContext] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    """جلب قائمة الفواتير المالية الخاصة بالمؤسسة الحالية بناءً على حالتها.

    استخدم هذه الأداة فقط عندما يسأل المستخدم عن الفواتير أو المستحقات المالية.
    """
    await asyncio.sleep(0.05)  # محاكاة استعلام قاعدة البيانات باستخدام ctx.tenant_org_id
    return {
        "verified_tenant": ctx.tenant_org_id,
        "queried_by_user": ctx.authenticated_user_id,
        "filter_status": status,
        "invoices": [
            {"inv_id": f"INV-{i}", "amount_usd": 1250.0 * i, "status": status}
            for i in range(1, limit + 1)
        ],
    }


@tool_harness.tool(timeout_sec=3.0, max_output_chars=180)
async def fetch_raw_server_logs(
    service_name: Annotated[str, Field(min_length=2, description="اسم الخدمة البرمجية")],
) -> str:
    """قراءة سجلات النظام الخام لخدمة معينة لتشخيص الأعطال التقنية."""
    await asyncio.sleep(0.02)
    # محاكاة مخرج ضخم جداً لاختبار حاجز تقليم المخرجات (Output Bounding)
    return "\n".join(
        f"[2025-05-01 12:00:{i:02d}] INFO ({service_name}): Worker processed batch #{i} successfully."
        for i in range(1, 30)
    )


async def main() -> None:
    # 1. لنفحص المخطط (JSON Schema) الذي سيعطيه الـ Harness للنموذج اللغوي!
    schemas = tool_harness.get_llm_schemas()
    print("🔍 المخطط العلني المولد تلقائياً لأداة الفواتير (لاحظ اختفاء ctx تماماً!):")
    print(json.dumps(schemas[0], indent=2, ensure_ascii=False), "\n")

    # 2. إعداد السياق الموثوق على الخادم (قادم من JWT Token مثلاً)
    server_context = SecurityContext(
        authenticated_user_id="USR-9981",
        tenant_org_id="ORG-ACME-CORP",
        db_connection_string="postgresql://prod-internal:5432/fintech",
    )

    # 3. محاكاة 3 استدعاءات متوازية من النموذج اللغوي:
    # - الأول: استدعاء سليم لأداة الفواتير (سيتم حقن الهوية تلقائياً)
    # - الثاني: محاولة خبيثة من النموذج لتمرير حقل `ctx` أو `tenant_org_id` لتزوير الهوية!
    # - الثالث: استدعاء لأداة السجلات التي تعيد نصاً ضخماً (سيتم تقليمه تلقائياً)
    batch_requests = [
        ToolCallRequest(
            call_id="call_01",
            tool_name="list_organization_invoices",
            raw_arguments={"status": "overdue", "limit": 2},
        ),
        ToolCallRequest(
            call_id="call_02",
            tool_name="list_organization_invoices",
            raw_arguments={"status": "paid", "ctx": "HACKED_CTX"},  # سيرفضه extra="forbid"!
        ),
        ToolCallRequest(
            call_id="call_03",
            tool_name="fetch_raw_server_logs",
            raw_arguments={"service_name": "payment-gateway"},
        ),
    ]

    outputs = await tool_harness.execute_batch(batch_requests, trusted_context=server_context)

    for out in outputs:
        icon = "❌" if out.is_error else ("✂️" if out.was_truncated else "✅")
        print(f"{icon} [{out.tool_name}] (ID: {out.call_id} | {out.execution_ms}ms):")
        print(f"   {out.content}\n")


if __name__ == "__main__":
    asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [014] dist/md/ch04.md
# ────────────────────────────────────────────────────────────────────────
# ❌ ANTI-PATTERN: كارثة أمنية محققة!
@app.tool()
async def delete_user_document(user_id: str, doc_id: str) -> str:
    # إذا خدع المهاجم النموذج اللغوي عبر Prompt Injection، سيمرر النموذج user_id الخاص بضحية أخرى!
    ...

# ────────────────────────────────────────────────────────────────────────
# [015] dist/md/ch04.md
# ────────────────────────────────────────────────────────────────────────
from typing import Protocol


# 1. تعريف عقد برمجي (Protocol) يضمن أن أي سياق يمرر للـ Harness يمتلك خاصية user_scopes
class HasSecurityScopes(Protocol):
    @property
    def user_scopes(self) -> set[str]:
        ...


# 2. تقييد الـ Generic Type في الـ Harness ليقبل فقط السياقات التي تحقق HasSecurityScopes
class EnterpriseToolHarness[ContextT: HasSecurityScopes]:
    # ... (داخل المزخرف @tool نضيف المعامل required_scopes: set[str] | None = None)
    # ونحفظه داخل ToolSpec كـ required_scopes: frozenset[str] = field(default_factory=frozenset)

    def get_llm_schemas(self, trusted_context: ContextT) -> list[dict[str, object]]:
        """المستوى الأول (حاجز الرؤية): إخفاء الأدوات التي لا يملك المستخدم صلاحياتها عن الـ LLM تماماً"""
        authorized_schemas: list[dict[str, object]] = []
        for spec in self._tools.values():
            if spec.required_scopes.issubset(trusted_context.user_scopes):
                authorized_schemas.append(spec.to_llm_json_schema())
        return authorized_schemas

    async def execute_single(
        self,
        request: ToolCallRequest,
        trusted_context: ContextT,
    ) -> ToolExecutionOutput:
        """المستوى الثاني (الدفاع في العمق - Defense-in-Depth): التحقق الصارم قبل التنفيذ الفعلي"""
        spec = self._tools.get(request.tool_name)
        if spec is None:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{request.tool_name}' is not available.",
                is_error=True,
            )

        # فحص الصلاحيات وقت التشغيل لمنع استدعاء الأدوات المخفية عبر التخمين أو الحقن!
        missing_scopes = spec.required_scopes - trusted_context.user_scopes
        if missing_scopes:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Authorization Denied: Missing required scopes {sorted(missing_scopes)} to run '{spec.name}'.",
                is_error=True,
            )

        # ... استكمال التحقق من المعطيات والتنفيذ كالمعتاد

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-1
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
# [003] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
try:
    import annotationlib  # Python 3.14+ (PEP 649)
except ModuleNotFoundError:  # توافق Python 3.12/3.13
    import harness_compat as annotationlib
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import inspect
import json
import time
from typing import Annotated, Any
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class _InjectedMarker:
    """علامة داخلية لتمييز المعاملات التي يجب إخفاؤها عن الـ LLM وحقنها من الـ Harness"""

    pass


# نوع مساعد أنيق يستخدمه المبرمج في توقيع الدالة: ctx: Injected[AppContext]
type Injected[T] = Annotated[T, _InjectedMarker()]


def _is_injected_annotation(annotation: Any) -> bool:
    """يفحص ما إذا كان تلميح النوع مغلفاً بـ Injected[T]"""
    # فك تغليف TypeAliasType في حال استخدام `type Injected[T]` في Python 3.12-3.14+
    origin = getattr(annotation, "__value__", annotation)
    metadata = getattr(origin, "__metadata__", ())
    return any(isinstance(m, _InjectedMarker) for m in metadata)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
@dataclass(slots=True, frozen=True)
class ToolCallRequest:
    """الطلب القادم من النموذج اللغوي لتشغيل أداة"""

    call_id: str
    tool_name: str
    raw_arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ToolExecutionOutput:
    """المخرج الموحد والمؤمّن الذي يعود من الـ Harness إلى ذاكرة الوكيل"""

    call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    was_truncated: bool = False
    execution_ms: float = 0.0


@dataclass(slots=True)
class ToolSpec:
    """المواصفات الكاملة للأداة بعد فحصها ذاتياً (Introspection)"""

    name: str
    description: str
    func: Callable[..., Awaitable[Any]]
    args_model: type[BaseModel]  # نموذج Pydantic المولد ديناميكياً للمعاملات العلنية فقط
    injected_param_names: list[str]  # أسماء المعاملات السرية التي سيحقنها الـ Harness
    timeout_sec: float
    max_output_chars: int

    def to_llm_json_schema(self) -> dict[str, Any]:
        """توليد المخطط القياسي المتوافق مع OpenAI / Anthropic / Gemini"""
        schema = self.args_model.model_json_schema()
        # تنظيف الحقول الإضافية لتقليل استهلاك التوكنز
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class EnterpriseToolHarness[ContextT]:
    """إطار التحكم بالأدوات من الدرجة الإنتاجية.

    يولد المخططات تلقائياً، يحقن السياق الموثوق خفيةً، ويحمي نافذة السياق من الانفجار.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def tool(
        self,
        *,
        name: str | None = None,
        timeout_sec: float = 15.0,
        max_output_chars: int = 4_000,
    ) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        """Decorator ذكي يفحص الدالة ويبني لها نموذج Pydantic ومخطط LLM تلقائياً"""

        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            if not inspect.iscoroutinefunction(func):
                raise TypeError(
                    f"الأداة '{func.__name__}' يجب أن تكون غير متزامنة (async def)."
                )

            tool_name = name or func.__name__
            doc = inspect.getdoc(func)
            if not doc or len(doc.strip()) < 10:
                raise ValueError(
                    f"الأداة '{tool_name}' تفتقر إلى Docstring تفصيلي! "
                    "تذكر أن الـ Docstring هو الموجه الذي يقرؤه النموذج اللغوي."
                )

            # 1. استخراج الأنواع باستخدام annotationlib الحديثة في Python 3.14
            type_hints = annotationlib.get_annotations(
                func, format=annotationlib.Format.VALUE
            )
            sig = inspect.signature(func)

            public_fields: dict[str, Any] = {}
            injected_params: list[str] = []

            for param_name, param in sig.parameters.items():
                if param_name == "return":
                    continue

                hint = type_hints.get(param_name, Any)

                # 2. فحص هل المعامل سري يُحقن من الـ Harness أم علني يملؤه الـ LLM؟
                if _is_injected_annotation(hint):
                    injected_params.append(param_name)
                    continue

                # 3. إعداد الحقل لنموذج Pydantic الديناميكي
                default_val = ... if param.default is inspect.Parameter.empty else param.default
                public_fields[param_name] = (hint, default_val)

            # 4. إنشاء كلاس Pydantic v2 ديناميكي صارم للمعاملات العلنية فقط
            dynamic_model = create_model(
                f"{tool_name}_ArgsSchema",
                __config__=ConfigDict(extra="forbid", str_strip_whitespace=True),
                **public_fields,
            )

            self._tools[tool_name] = ToolSpec(
                name=tool_name,
                description=doc.strip(),
                func=func,
                args_model=dynamic_model,
                injected_param_names=injected_params,
                timeout_sec=timeout_sec,
                max_output_chars=max_output_chars,
            )
            return func

        return decorator

    def get_llm_schemas(self) -> list[dict[str, Any]]:
        """إرجاع كتالوج الأدوات العلني الجاهز للإرسال للنموذج اللغوي"""
        return [spec.to_llm_json_schema() for spec in self._tools.values()]

    @staticmethod
    def _bound_output(raw_text: str, max_chars: int) -> tuple[str, bool]:
        """حاجز تقليم المخرجات لحماية نافذة السياق من القنابل النصية"""
        if len(raw_text) <= max_chars:
            return raw_text, False

        truncated = raw_text[:max_chars]
        omitted_count = len(raw_text) - max_chars
        notice = (
            f"\n\n... [⚠️ HARNESS TRUNCATION NOTICE: تم اقتصاص {omitted_count} حرفاً إضافياً "
            "لحماية نافذة السياق. استخدم فلاتر أدق أو pagination في استدعائك القادم]."
        )
        return truncated + notice, True

    async def execute_single(
        self,
        request: ToolCallRequest,
        trusted_context: ContextT,
    ) -> ToolExecutionOutput:
        """تنفيذ أداة واحدة مع التحقق، وحقن السياق الموثوق، وتقليم المخرجات"""
        t0 = time.perf_counter()
        spec = self._tools.get(request.tool_name)

        if spec is None:
            available = ", ".join(self._tools.keys())
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{request.tool_name}' not found. Available tools: [{available}]",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        try:
            # 1. التحقق الصارم من المدخلات القادمة من النموذج اللغوي
            validated_public_args = spec.args_model.model_validate(request.raw_arguments)
            final_kwargs = validated_public_args.model_dump()

            # 2. حقن السياق الموثوق (Zero-Trust Injection) دون علم النموذج!
            for injected_name in spec.injected_param_names:
                final_kwargs[injected_name] = trusted_context

            # 3. التنفيذ المحمي بمهلة زمنية
            async with asyncio.timeout(spec.timeout_sec):
                raw_result = await spec.func(**final_kwargs)

            # 4. تنسيق المخرج وتطبيق حاجز التقليم (Output Bounding)
            serialized = (
                json.dumps(raw_result, ensure_ascii=False)
                if isinstance(raw_result, (dict, list))
                else str(raw_result)
            )
            bounded_text, was_truncated = self._bound_output(
                serialized, spec.max_output_chars
            )

            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=bounded_text,
                is_error=False,
                was_truncated=was_truncated,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        except ValidationError as val_err:
            # تغذية راجعة واضحة للنموذج ليصلح معطيات الأداة في الخطوة القادمة
            compact_errors = [
                f"`{'.'.join(str(x) for x in e['loc'])}`: {e['msg']}"
                for e in val_err.errors(include_url=False)
            ]
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Tool Argument Validation Error: {'; '.join(compact_errors)}",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except TimeoutError:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{spec.name}' timed out after {spec.timeout_sec}s.",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except Exception as exc:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Tool Runtime Error ({type(exc).__name__}): {exc}",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

    async def execute_batch(
        self,
        requests: list[ToolCallRequest],
        trusted_context: ContextT,
    ) -> list[ToolExecutionOutput]:
        """تنفيذ دفعة من الأدوات بالتوازي المهيكل عبر TaskGroup"""
        tasks: list[asyncio.Task[ToolExecutionOutput]] = []
        async with asyncio.TaskGroup() as tg:
            for req in requests:
                tasks.append(tg.create_task(self.execute_single(req, trusted_context)))
        return [t.result() for t in tasks]

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
@dataclass(slots=True, frozen=True)
class SecurityContext:
    """سياق الأمان الموثوق الذي يملكه الخادم ولا يراه النموذج اللغوي أبداً"""

    authenticated_user_id: str
    tenant_org_id: str
    db_connection_string: str


# إنشاء نسخة من الـ Harness مخصصة لنوع السياق الأمني الخاص بنا
tool_harness = EnterpriseToolHarness[SecurityContext]()


@tool_harness.tool(timeout_sec=5.0, max_output_chars=500)
async def list_organization_invoices(
    status: Annotated[str, Field(pattern=r"^(paid|unpaid|overdue)$", description="حالة الفاتورة المطلوبة")],
    limit: Annotated[int, Field(ge=1, le=50, description="الحد الأقصى لعدد الفواتير")] = 10,
    # لاحظ هنا: المعامل ctx موسوم بـ Injected! لن يظهر في JSON Schema المرسل للـ LLM أبداً!
    ctx: Injected[SecurityContext] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    """جلب قائمة الفواتير المالية الخاصة بالمؤسسة الحالية بناءً على حالتها.

    استخدم هذه الأداة فقط عندما يسأل المستخدم عن الفواتير أو المستحقات المالية.
    """
    await asyncio.sleep(0.05)  # محاكاة استعلام قاعدة البيانات باستخدام ctx.tenant_org_id
    return {
        "verified_tenant": ctx.tenant_org_id,
        "queried_by_user": ctx.authenticated_user_id,
        "filter_status": status,
        "invoices": [
            {"inv_id": f"INV-{i}", "amount_usd": 1250.0 * i, "status": status}
            for i in range(1, limit + 1)
        ],
    }


@tool_harness.tool(timeout_sec=3.0, max_output_chars=180)
async def fetch_raw_server_logs(
    service_name: Annotated[str, Field(min_length=2, description="اسم الخدمة البرمجية")],
) -> str:
    """قراءة سجلات النظام الخام لخدمة معينة لتشخيص الأعطال التقنية."""
    await asyncio.sleep(0.02)
    # محاكاة مخرج ضخم جداً لاختبار حاجز تقليم المخرجات (Output Bounding)
    return "\n".join(
        f"[2025-05-01 12:00:{i:02d}] INFO ({service_name}): Worker processed batch #{i} successfully."
        for i in range(1, 30)
    )


async def main() -> None:
    # 1. لنفحص المخطط (JSON Schema) الذي سيعطيه الـ Harness للنموذج اللغوي!
    schemas = tool_harness.get_llm_schemas()
    print("🔍 المخطط العلني المولد تلقائياً لأداة الفواتير (لاحظ اختفاء ctx تماماً!):")
    print(json.dumps(schemas[0], indent=2, ensure_ascii=False), "\n")

    # 2. إعداد السياق الموثوق على الخادم (قادم من JWT Token مثلاً)
    server_context = SecurityContext(
        authenticated_user_id="USR-9981",
        tenant_org_id="ORG-ACME-CORP",
        db_connection_string="postgresql://prod-internal:5432/fintech",
    )

    # 3. محاكاة 3 استدعاءات متوازية من النموذج اللغوي:
    # - الأول: استدعاء سليم لأداة الفواتير (سيتم حقن الهوية تلقائياً)
    # - الثاني: محاولة خبيثة من النموذج لتمرير حقل `ctx` أو `tenant_org_id` لتزوير الهوية!
    # - الثالث: استدعاء لأداة السجلات التي تعيد نصاً ضخماً (سيتم تقليمه تلقائياً)
    batch_requests = [
        ToolCallRequest(
            call_id="call_01",
            tool_name="list_organization_invoices",
            raw_arguments={"status": "overdue", "limit": 2},
        ),
        ToolCallRequest(
            call_id="call_02",
            tool_name="list_organization_invoices",
            raw_arguments={"status": "paid", "ctx": "HACKED_CTX"},  # سيرفضه extra="forbid"!
        ),
        ToolCallRequest(
            call_id="call_03",
            tool_name="fetch_raw_server_logs",
            raw_arguments={"service_name": "payment-gateway"},
        ),
    ]

    outputs = await tool_harness.execute_batch(batch_requests, trusted_context=server_context)

    for out in outputs:
        icon = "❌" if out.is_error else ("✂️" if out.was_truncated else "✅")
        print(f"{icon} [{out.tool_name}] (ID: {out.call_id} | {out.execution_ms}ms):")
        print(f"   {out.content}\n")

# في الدفتر لا نستخدم asyncio.run: الحلقة قائمة بالفعل، لذا ننتظر التنفيذ مباشرة.
await main()

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
الفصل 4: هندسة الأدوات واستدعاء الدوال
==============================================================================

البرنامج الكامل للفصل 4 من كتاب «هندسة وكلاء الذكاء الاصطناعي».
هندسة الأدوات واستدعاء الدوال — Tooling & Function Calling Architecture.

التشغيل:
    python ch04_tool_registry.py

المتطلبات: Python 3.12+ (الكتاب يستهدف 3.14+) و pydantic>=2.9
"""
# ملاحظة توافق: تُقيَّم الأنواع كسولاً في Python 3.14 (PEP 649)؛
# هذا السطر يمنح السلوك نفسه على Python 3.12/3.13.
from __future__ import annotations

try:
    import annotationlib  # Python 3.14+ (PEP 649)
except ModuleNotFoundError:  # توافق Python 3.12/3.13
    import harness_compat as annotationlib
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import inspect
import json
import time
from typing import Annotated, Any
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model


# =====================================================================
# 1. آلية حقن الاعتماديات المخفية (Zero-Trust Dependency Injection Marker)
# =====================================================================
class _InjectedMarker:
    """علامة داخلية لتمييز المعاملات التي يجب إخفاؤها عن الـ LLM وحقنها من الـ Harness"""

    pass


# نوع مساعد أنيق يستخدمه المبرمج في توقيع الدالة: ctx: Injected[AppContext]
type Injected[T] = Annotated[T, _InjectedMarker()]


def _is_injected_annotation(annotation: Any) -> bool:
    """يفحص ما إذا كان تلميح النوع مغلفاً بـ Injected[T]"""
    # فك تغليف TypeAliasType في حال استخدام `type Injected[T]` في Python 3.12-3.14+
    origin = getattr(annotation, "__value__", annotation)
    metadata = getattr(origin, "__metadata__", ())
    return any(isinstance(m, _InjectedMarker) for m in metadata)


# =====================================================================
# 2. هياكل بيانات التشغيل والمخرجات داخل الـ Tool Harness
# =====================================================================
@dataclass(slots=True, frozen=True)
class ToolCallRequest:
    """الطلب القادم من النموذج اللغوي لتشغيل أداة"""

    call_id: str
    tool_name: str
    raw_arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ToolExecutionOutput:
    """المخرج الموحد والمؤمّن الذي يعود من الـ Harness إلى ذاكرة الوكيل"""

    call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    was_truncated: bool = False
    execution_ms: float = 0.0


@dataclass(slots=True)
class ToolSpec:
    """المواصفات الكاملة للأداة بعد فحصها ذاتياً (Introspection)"""

    name: str
    description: str
    func: Callable[..., Awaitable[Any]]
    args_model: type[BaseModel]  # نموذج Pydantic المولد ديناميكياً للمعاملات العلنية فقط
    injected_param_names: list[str]  # أسماء المعاملات السرية التي سيحقنها الـ Harness
    timeout_sec: float
    max_output_chars: int

    def to_llm_json_schema(self) -> dict[str, Any]:
        """توليد المخطط القياسي المتوافق مع OpenAI / Anthropic / Gemini"""
        schema = self.args_model.model_json_schema()
        # تنظيف الحقول الإضافية لتقليل استهلاك التوكنز
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


# =====================================================================
# 3. قلب النظام: EnterpriseToolHarness (باستخدام PEP 695 Generics)
# =====================================================================
class EnterpriseToolHarness[ContextT]:
    """إطار التحكم بالأدوات من الدرجة الإنتاجية.

    يولد المخططات تلقائياً، يحقن السياق الموثوق خفيةً، ويحمي نافذة السياق من الانفجار.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def tool(
        self,
        *,
        name: str | None = None,
        timeout_sec: float = 15.0,
        max_output_chars: int = 4_000,
    ) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        """Decorator ذكي يفحص الدالة ويبني لها نموذج Pydantic ومخطط LLM تلقائياً"""

        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            if not inspect.iscoroutinefunction(func):
                raise TypeError(
                    f"الأداة '{func.__name__}' يجب أن تكون غير متزامنة (async def)."
                )

            tool_name = name or func.__name__
            doc = inspect.getdoc(func)
            if not doc or len(doc.strip()) < 10:
                raise ValueError(
                    f"الأداة '{tool_name}' تفتقر إلى Docstring تفصيلي! "
                    "تذكر أن الـ Docstring هو الموجه الذي يقرؤه النموذج اللغوي."
                )

            # 1. استخراج الأنواع باستخدام annotationlib الحديثة في Python 3.14
            type_hints = annotationlib.get_annotations(
                func, format=annotationlib.Format.VALUE
            )
            sig = inspect.signature(func)

            public_fields: dict[str, Any] = {}
            injected_params: list[str] = []

            for param_name, param in sig.parameters.items():
                if param_name == "return":
                    continue

                hint = type_hints.get(param_name, Any)

                # 2. فحص هل المعامل سري يُحقن من الـ Harness أم علني يملؤه الـ LLM؟
                if _is_injected_annotation(hint):
                    injected_params.append(param_name)
                    continue

                # 3. إعداد الحقل لنموذج Pydantic الديناميكي
                default_val = ... if param.default is inspect.Parameter.empty else param.default
                public_fields[param_name] = (hint, default_val)

            # 4. إنشاء كلاس Pydantic v2 ديناميكي صارم للمعاملات العلنية فقط
            dynamic_model = create_model(
                f"{tool_name}_ArgsSchema",
                __config__=ConfigDict(extra="forbid", str_strip_whitespace=True),
                **public_fields,
            )

            self._tools[tool_name] = ToolSpec(
                name=tool_name,
                description=doc.strip(),
                func=func,
                args_model=dynamic_model,
                injected_param_names=injected_params,
                timeout_sec=timeout_sec,
                max_output_chars=max_output_chars,
            )
            return func

        return decorator

    def get_llm_schemas(self) -> list[dict[str, Any]]:
        """إرجاع كتالوج الأدوات العلني الجاهز للإرسال للنموذج اللغوي"""
        return [spec.to_llm_json_schema() for spec in self._tools.values()]

    @staticmethod
    def _bound_output(raw_text: str, max_chars: int) -> tuple[str, bool]:
        """حاجز تقليم المخرجات لحماية نافذة السياق من القنابل النصية"""
        if len(raw_text) <= max_chars:
            return raw_text, False

        truncated = raw_text[:max_chars]
        omitted_count = len(raw_text) - max_chars
        notice = (
            f"\n\n... [⚠️ HARNESS TRUNCATION NOTICE: تم اقتصاص {omitted_count} حرفاً إضافياً "
            "لحماية نافذة السياق. استخدم فلاتر أدق أو pagination في استدعائك القادم]."
        )
        return truncated + notice, True

    async def execute_single(
        self,
        request: ToolCallRequest,
        trusted_context: ContextT,
    ) -> ToolExecutionOutput:
        """تنفيذ أداة واحدة مع التحقق، وحقن السياق الموثوق، وتقليم المخرجات"""
        t0 = time.perf_counter()
        spec = self._tools.get(request.tool_name)

        if spec is None:
            available = ", ".join(self._tools.keys())
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{request.tool_name}' not found. Available tools: [{available}]",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        try:
            # 1. التحقق الصارم من المدخلات القادمة من النموذج اللغوي
            validated_public_args = spec.args_model.model_validate(request.raw_arguments)
            final_kwargs = validated_public_args.model_dump()

            # 2. حقن السياق الموثوق (Zero-Trust Injection) دون علم النموذج!
            for injected_name in spec.injected_param_names:
                final_kwargs[injected_name] = trusted_context

            # 3. التنفيذ المحمي بمهلة زمنية
            async with asyncio.timeout(spec.timeout_sec):
                raw_result = await spec.func(**final_kwargs)

            # 4. تنسيق المخرج وتطبيق حاجز التقليم (Output Bounding)
            serialized = (
                json.dumps(raw_result, ensure_ascii=False)
                if isinstance(raw_result, (dict, list))
                else str(raw_result)
            )
            bounded_text, was_truncated = self._bound_output(
                serialized, spec.max_output_chars
            )

            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=bounded_text,
                is_error=False,
                was_truncated=was_truncated,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        except ValidationError as val_err:
            # تغذية راجعة واضحة للنموذج ليصلح معطيات الأداة في الخطوة القادمة
            compact_errors = [
                f"`{'.'.join(str(x) for x in e['loc'])}`: {e['msg']}"
                for e in val_err.errors(include_url=False)
            ]
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Tool Argument Validation Error: {'; '.join(compact_errors)}",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except TimeoutError:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{spec.name}' timed out after {spec.timeout_sec}s.",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except Exception as exc:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Tool Runtime Error ({type(exc).__name__}): {exc}",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

    async def execute_batch(
        self,
        requests: list[ToolCallRequest],
        trusted_context: ContextT,
    ) -> list[ToolExecutionOutput]:
        """تنفيذ دفعة من الأدوات بالتوازي المهيكل عبر TaskGroup"""
        tasks: list[asyncio.Task[ToolExecutionOutput]] = []
        async with asyncio.TaskGroup() as tg:
            for req in requests:
                tasks.append(tg.create_task(self.execute_single(req, trusted_context)))
        return [t.result() for t in tasks]


# =====================================================================
# 4. مثال تطبيقي: أدوات نظام مالي متعددة المستأجرين (Multi-Tenant FinTech)
# =====================================================================
@dataclass(slots=True, frozen=True)
class SecurityContext:
    """سياق الأمان الموثوق الذي يملكه الخادم ولا يراه النموذج اللغوي أبداً"""

    authenticated_user_id: str
    tenant_org_id: str
    db_connection_string: str


# إنشاء نسخة من الـ Harness مخصصة لنوع السياق الأمني الخاص بنا
tool_harness = EnterpriseToolHarness[SecurityContext]()


@tool_harness.tool(timeout_sec=5.0, max_output_chars=500)
async def list_organization_invoices(
    status: Annotated[str, Field(pattern=r"^(paid|unpaid|overdue)$", description="حالة الفاتورة المطلوبة")],
    limit: Annotated[int, Field(ge=1, le=50, description="الحد الأقصى لعدد الفواتير")] = 10,
    # لاحظ هنا: المعامل ctx موسوم بـ Injected! لن يظهر في JSON Schema المرسل للـ LLM أبداً!
    ctx: Injected[SecurityContext] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    """جلب قائمة الفواتير المالية الخاصة بالمؤسسة الحالية بناءً على حالتها.

    استخدم هذه الأداة فقط عندما يسأل المستخدم عن الفواتير أو المستحقات المالية.
    """
    await asyncio.sleep(0.05)  # محاكاة استعلام قاعدة البيانات باستخدام ctx.tenant_org_id
    return {
        "verified_tenant": ctx.tenant_org_id,
        "queried_by_user": ctx.authenticated_user_id,
        "filter_status": status,
        "invoices": [
            {"inv_id": f"INV-{i}", "amount_usd": 1250.0 * i, "status": status}
            for i in range(1, limit + 1)
        ],
    }


@tool_harness.tool(timeout_sec=3.0, max_output_chars=180)
async def fetch_raw_server_logs(
    service_name: Annotated[str, Field(min_length=2, description="اسم الخدمة البرمجية")],
) -> str:
    """قراءة سجلات النظام الخام لخدمة معينة لتشخيص الأعطال التقنية."""
    await asyncio.sleep(0.02)
    # محاكاة مخرج ضخم جداً لاختبار حاجز تقليم المخرجات (Output Bounding)
    return "\n".join(
        f"[2025-05-01 12:00:{i:02d}] INFO ({service_name}): Worker processed batch #{i} successfully."
        for i in range(1, 30)
    )


async def main() -> None:
    # 1. لنفحص المخطط (JSON Schema) الذي سيعطيه الـ Harness للنموذج اللغوي!
    schemas = tool_harness.get_llm_schemas()
    print("🔍 المخطط العلني المولد تلقائياً لأداة الفواتير (لاحظ اختفاء ctx تماماً!):")
    print(json.dumps(schemas[0], indent=2, ensure_ascii=False), "\n")

    # 2. إعداد السياق الموثوق على الخادم (قادم من JWT Token مثلاً)
    server_context = SecurityContext(
        authenticated_user_id="USR-9981",
        tenant_org_id="ORG-ACME-CORP",
        db_connection_string="postgresql://prod-internal:5432/fintech",
    )

    # 3. محاكاة 3 استدعاءات متوازية من النموذج اللغوي:
    # - الأول: استدعاء سليم لأداة الفواتير (سيتم حقن الهوية تلقائياً)
    # - الثاني: محاولة خبيثة من النموذج لتمرير حقل `ctx` أو `tenant_org_id` لتزوير الهوية!
    # - الثالث: استدعاء لأداة السجلات التي تعيد نصاً ضخماً (سيتم تقليمه تلقائياً)
    batch_requests = [
        ToolCallRequest(
            call_id="call_01",
            tool_name="list_organization_invoices",
            raw_arguments={"status": "overdue", "limit": 2},
        ),
        ToolCallRequest(
            call_id="call_02",
            tool_name="list_organization_invoices",
            raw_arguments={"status": "paid", "ctx": "HACKED_CTX"},  # سيرفضه extra="forbid"!
        ),
        ToolCallRequest(
            call_id="call_03",
            tool_name="fetch_raw_server_logs",
            raw_arguments={"service_name": "payment-gateway"},
        ),
    ]

    outputs = await tool_harness.execute_batch(batch_requests, trusted_context=server_context)

    for out in outputs:
        icon = "❌" if out.is_error else ("✂️" if out.was_truncated else "✅")
        print(f"{icon} [{out.tool_name}] (ID: {out.call_id} | {out.execution_ms}ms):")
        print(f"   {out.content}\n")


if __name__ == "__main__":
    asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-8
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
# [010] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
❌ ANTI-PATTERN: كارثة أمنية محققة!

مقتطف من الفصل 4 — «هندسة الأدوات واستدعاء الدوال».
النوع: anti_pattern. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch04_tool_registry.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ❌ ANTI-PATTERN: كارثة أمنية محققة!
@app.tool()
async def delete_user_document(user_id: str, doc_id: str) -> str:
    # إذا خدع المهاجم النموذج اللغوي عبر Prompt Injection، سيمرر النموذج user_id الخاص بضحية أخرى!
    ...

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
async def get_customer_invoices(user_id: int, status: str, db: DatabaseSession) -> list[dict]:

مقتطف من الفصل 4 — «هندسة الأدوات واستدعاء الدوال».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch04_tool_registry.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

async def get_customer_invoices(user_id: int, status: str, db: DatabaseSession) -> list[dict]:
    ...

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
from typing import Protocol

مقتطف من الفصل 4 — «هندسة الأدوات واستدعاء الدوال».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch04_tool_registry.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

from typing import Protocol


# 1. تعريف عقد برمجي (Protocol) يضمن أن أي سياق يمرر للـ Harness يمتلك خاصية user_scopes
class HasSecurityScopes(Protocol):
    @property
    def user_scopes(self) -> set[str]:
        ...


# 2. تقييد الـ Generic Type في الـ Harness ليقبل فقط السياقات التي تحقق HasSecurityScopes
class EnterpriseToolHarness[ContextT: HasSecurityScopes]:
    # ... (داخل المزخرف @tool نضيف المعامل required_scopes: set[str] | None = None)
    # ونحفظه داخل ToolSpec كـ required_scopes: frozenset[str] = field(default_factory=frozenset)

    def get_llm_schemas(self, trusted_context: ContextT) -> list[dict[str, object]]:
        """المستوى الأول (حاجز الرؤية): إخفاء الأدوات التي لا يملك المستخدم صلاحياتها عن الـ LLM تماماً"""
        authorized_schemas: list[dict[str, object]] = []
        for spec in self._tools.values():
            if spec.required_scopes.issubset(trusted_context.user_scopes):
                authorized_schemas.append(spec.to_llm_json_schema())
        return authorized_schemas

    async def execute_single(
        self,
        request: ToolCallRequest,
        trusted_context: ContextT,
    ) -> ToolExecutionOutput:
        """المستوى الثاني (الدفاع في العمق - Defense-in-Depth): التحقق الصارم قبل التنفيذ الفعلي"""
        spec = self._tools.get(request.tool_name)
        if spec is None:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{request.tool_name}' is not available.",
                is_error=True,
            )

        # فحص الصلاحيات وقت التشغيل لمنع استدعاء الأدوات المخفية عبر التخمين أو الحقن!
        missing_scopes = spec.required_scopes - trusted_context.user_scopes
        if missing_scopes:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Authorization Denied: Missing required scopes {sorted(missing_scopes)} to run '{spec.name}'.",
                is_error=True,
            )

        # ... استكمال التحقق من المعطيات والتنفيذ كالمعتاد

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
async def get_customer_invoices(user_id: int, status: str, db: DatabaseSession) -> list[dict]:
    ...

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import annotationlib  # مكتبة Python 3.14 القياسية لفحص الأنواع بدقة وأمان
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import inspect
import json
import time
from typing import Annotated, Any
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model


# =====================================================================
# 1. آلية حقن الاعتماديات المخفية (Zero-Trust Dependency Injection Marker)
# =====================================================================
class _InjectedMarker:
    """علامة داخلية لتمييز المعاملات التي يجب إخفاؤها عن الـ LLM وحقنها من الـ Harness"""

    pass


# نوع مساعد أنيق يستخدمه المبرمج في توقيع الدالة: ctx: Injected[AppContext]
type Injected[T] = Annotated[T, _InjectedMarker()]


def _is_injected_annotation(annotation: Any) -> bool:
    """يفحص ما إذا كان تلميح النوع مغلفاً بـ Injected[T]"""
    # فك تغليف TypeAliasType في حال استخدام `type Injected[T]` في Python 3.12-3.14+
    origin = getattr(annotation, "__value__", annotation)
    metadata = getattr(origin, "__metadata__", ())
    return any(isinstance(m, _InjectedMarker) for m in metadata)


# =====================================================================
# 2. هياكل بيانات التشغيل والمخرجات داخل الـ Tool Harness
# =====================================================================
@dataclass(slots=True, frozen=True)
class ToolCallRequest:
    """الطلب القادم من النموذج اللغوي لتشغيل أداة"""

    call_id: str
    tool_name: str
    raw_arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ToolExecutionOutput:
    """المخرج الموحد والمؤمّن الذي يعود من الـ Harness إلى ذاكرة الوكيل"""

    call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    was_truncated: bool = False
    execution_ms: float = 0.0


@dataclass(slots=True)
class ToolSpec:
    """المواصفات الكاملة للأداة بعد فحصها ذاتياً (Introspection)"""

    name: str
    description: str
    func: Callable[..., Awaitable[Any]]
    args_model: type[BaseModel]  # نموذج Pydantic المولد ديناميكياً للمعاملات العلنية فقط
    injected_param_names: list[str]  # أسماء المعاملات السرية التي سيحقنها الـ Harness
    timeout_sec: float
    max_output_chars: int

    def to_llm_json_schema(self) -> dict[str, Any]:
        """توليد المخطط القياسي المتوافق مع OpenAI / Anthropic / Gemini"""
        schema = self.args_model.model_json_schema()
        # تنظيف الحقول الإضافية لتقليل استهلاك التوكنز
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


# =====================================================================
# 3. قلب النظام: EnterpriseToolHarness (باستخدام PEP 695 Generics)
# =====================================================================
class EnterpriseToolHarness[ContextT]:
    """إطار التحكم بالأدوات من الدرجة الإنتاجية.

    يولد المخططات تلقائياً، يحقن السياق الموثوق خفيةً، ويحمي نافذة السياق من الانفجار.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def tool(
        self,
        *,
        name: str | None = None,
        timeout_sec: float = 15.0,
        max_output_chars: int = 4_000,
    ) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        """Decorator ذكي يفحص الدالة ويبني لها نموذج Pydantic ومخطط LLM تلقائياً"""

        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            if not inspect.iscoroutinefunction(func):
                raise TypeError(
                    f"الأداة '{func.__name__}' يجب أن تكون غير متزامنة (async def)."
                )

            tool_name = name or func.__name__
            doc = inspect.getdoc(func)
            if not doc or len(doc.strip()) < 10:
                raise ValueError(
                    f"الأداة '{tool_name}' تفتقر إلى Docstring تفصيلي! "
                    "تذكر أن الـ Docstring هو الموجه الذي يقرؤه النموذج اللغوي."
                )

            # 1. استخراج الأنواع باستخدام annotationlib الحديثة في Python 3.14
            type_hints = annotationlib.get_annotations(
                func, format=annotationlib.Format.VALUE
            )
            sig = inspect.signature(func)

            public_fields: dict[str, Any] = {}
            injected_params: list[str] = []

            for param_name, param in sig.parameters.items():
                if param_name == "return":
                    continue

                hint = type_hints.get(param_name, Any)

                # 2. فحص هل المعامل سري يُحقن من الـ Harness أم علني يملؤه الـ LLM؟
                if _is_injected_annotation(hint):
                    injected_params.append(param_name)
                    continue

                # 3. إعداد الحقل لنموذج Pydantic الديناميكي
                default_val = ... if param.default is inspect.Parameter.empty else param.default
                public_fields[param_name] = (hint, default_val)

            # 4. إنشاء كلاس Pydantic v2 ديناميكي صارم للمعاملات العلنية فقط
            dynamic_model = create_model(
                f"{tool_name}_ArgsSchema",
                __config__=ConfigDict(extra="forbid", str_strip_whitespace=True),
                **public_fields,
            )

            self._tools[tool_name] = ToolSpec(
                name=tool_name,
                description=doc.strip(),
                func=func,
                args_model=dynamic_model,
                injected_param_names=injected_params,
                timeout_sec=timeout_sec,
                max_output_chars=max_output_chars,
            )
            return func

        return decorator

    def get_llm_schemas(self) -> list[dict[str, Any]]:
        """إرجاع كتالوج الأدوات العلني الجاهز للإرسال للنموذج اللغوي"""
        return [spec.to_llm_json_schema() for spec in self._tools.values()]

    @staticmethod
    def _bound_output(raw_text: str, max_chars: int) -> tuple[str, bool]:
        """حاجز تقليم المخرجات لحماية نافذة السياق من القنابل النصية"""
        if len(raw_text) <= max_chars:
            return raw_text, False

        truncated = raw_text[:max_chars]
        omitted_count = len(raw_text) - max_chars
        notice = (
            f"\n\n... [⚠️ HARNESS TRUNCATION NOTICE: تم اقتصاص {omitted_count} حرفاً إضافياً "
            "لحماية نافذة السياق. استخدم فلاتر أدق أو pagination في استدعائك القادم]."
        )
        return truncated + notice, True

    async def execute_single(
        self,
        request: ToolCallRequest,
        trusted_context: ContextT,
    ) -> ToolExecutionOutput:
        """تنفيذ أداة واحدة مع التحقق، وحقن السياق الموثوق، وتقليم المخرجات"""
        t0 = time.perf_counter()
        spec = self._tools.get(request.tool_name)

        if spec is None:
            available = ", ".join(self._tools.keys())
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{request.tool_name}' not found. Available tools: [{available}]",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        try:
            # 1. التحقق الصارم من المدخلات القادمة من النموذج اللغوي
            validated_public_args = spec.args_model.model_validate(request.raw_arguments)
            final_kwargs = validated_public_args.model_dump()

            # 2. حقن السياق الموثوق (Zero-Trust Injection) دون علم النموذج!
            for injected_name in spec.injected_param_names:
                final_kwargs[injected_name] = trusted_context

            # 3. التنفيذ المحمي بمهلة زمنية
            async with asyncio.timeout(spec.timeout_sec):
                raw_result = await spec.func(**final_kwargs)

            # 4. تنسيق المخرج وتطبيق حاجز التقليم (Output Bounding)
            serialized = (
                json.dumps(raw_result, ensure_ascii=False)
                if isinstance(raw_result, (dict, list))
                else str(raw_result)
            )
            bounded_text, was_truncated = self._bound_output(
                serialized, spec.max_output_chars
            )

            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=bounded_text,
                is_error=False,
                was_truncated=was_truncated,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        except ValidationError as val_err:
            # تغذية راجعة واضحة للنموذج ليصلح معطيات الأداة في الخطوة القادمة
            compact_errors = [
                f"`{'.'.join(str(x) for x in e['loc'])}`: {e['msg']}"
                for e in val_err.errors(include_url=False)
            ]
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Tool Argument Validation Error: {'; '.join(compact_errors)}",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except TimeoutError:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{spec.name}' timed out after {spec.timeout_sec}s.",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except Exception as exc:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Tool Runtime Error ({type(exc).__name__}): {exc}",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

    async def execute_batch(
        self,
        requests: list[ToolCallRequest],
        trusted_context: ContextT,
    ) -> list[ToolExecutionOutput]:
        """تنفيذ دفعة من الأدوات بالتوازي المهيكل عبر TaskGroup"""
        tasks: list[asyncio.Task[ToolExecutionOutput]] = []
        async with asyncio.TaskGroup() as tg:
            for req in requests:
                tasks.append(tg.create_task(self.execute_single(req, trusted_context)))
        return [t.result() for t in tasks]


# =====================================================================
# 4. مثال تطبيقي: أدوات نظام مالي متعددة المستأجرين (Multi-Tenant FinTech)
# =====================================================================
@dataclass(slots=True, frozen=True)
class SecurityContext:
    """سياق الأمان الموثوق الذي يملكه الخادم ولا يراه النموذج اللغوي أبداً"""

    authenticated_user_id: str
    tenant_org_id: str
    db_connection_string: str


# إنشاء نسخة من الـ Harness مخصصة لنوع السياق الأمني الخاص بنا
tool_harness = EnterpriseToolHarness[SecurityContext]()


@tool_harness.tool(timeout_sec=5.0, max_output_chars=500)
async def list_organization_invoices(
    status: Annotated[str, Field(pattern=r"^(paid|unpaid|overdue)$", description="حالة الفاتورة المطلوبة")],
    limit: Annotated[int, Field(ge=1, le=50, description="الحد الأقصى لعدد الفواتير")] = 10,
    # لاحظ هنا: المعامل ctx موسوم بـ Injected! لن يظهر في JSON Schema المرسل للـ LLM أبداً!
    ctx: Injected[SecurityContext] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    """جلب قائمة الفواتير المالية الخاصة بالمؤسسة الحالية بناءً على حالتها.

    استخدم هذه الأداة فقط عندما يسأل المستخدم عن الفواتير أو المستحقات المالية.
    """
    await asyncio.sleep(0.05)  # محاكاة استعلام قاعدة البيانات باستخدام ctx.tenant_org_id
    return {
        "verified_tenant": ctx.tenant_org_id,
        "queried_by_user": ctx.authenticated_user_id,
        "filter_status": status,
        "invoices": [
            {"inv_id": f"INV-{i}", "amount_usd": 1250.0 * i, "status": status}
            for i in range(1, limit + 1)
        ],
    }


@tool_harness.tool(timeout_sec=3.0, max_output_chars=180)
async def fetch_raw_server_logs(
    service_name: Annotated[str, Field(min_length=2, description="اسم الخدمة البرمجية")],
) -> str:
    """قراءة سجلات النظام الخام لخدمة معينة لتشخيص الأعطال التقنية."""
    await asyncio.sleep(0.02)
    # محاكاة مخرج ضخم جداً لاختبار حاجز تقليم المخرجات (Output Bounding)
    return "\n".join(
        f"[2025-05-01 12:00:{i:02d}] INFO ({service_name}): Worker processed batch #{i} successfully."
        for i in range(1, 30)
    )


async def main() -> None:
    # 1. لنفحص المخطط (JSON Schema) الذي سيعطيه الـ Harness للنموذج اللغوي!
    schemas = tool_harness.get_llm_schemas()
    print("🔍 المخطط العلني المولد تلقائياً لأداة الفواتير (لاحظ اختفاء ctx تماماً!):")
    print(json.dumps(schemas[0], indent=2, ensure_ascii=False), "\n")

    # 2. إعداد السياق الموثوق على الخادم (قادم من JWT Token مثلاً)
    server_context = SecurityContext(
        authenticated_user_id="USR-9981",
        tenant_org_id="ORG-ACME-CORP",
        db_connection_string="postgresql://prod-internal:5432/fintech",
    )

    # 3. محاكاة 3 استدعاءات متوازية من النموذج اللغوي:
    # - الأول: استدعاء سليم لأداة الفواتير (سيتم حقن الهوية تلقائياً)
    # - الثاني: محاولة خبيثة من النموذج لتمرير حقل `ctx` أو `tenant_org_id` لتزوير الهوية!
    # - الثالث: استدعاء لأداة السجلات التي تعيد نصاً ضخماً (سيتم تقليمه تلقائياً)
    batch_requests = [
        ToolCallRequest(
            call_id="call_01",
            tool_name="list_organization_invoices",
            raw_arguments={"status": "overdue", "limit": 2},
        ),
        ToolCallRequest(
            call_id="call_02",
            tool_name="list_organization_invoices",
            raw_arguments={"status": "paid", "ctx": "HACKED_CTX"},  # سيرفضه extra="forbid"!
        ),
        ToolCallRequest(
            call_id="call_03",
            tool_name="fetch_raw_server_logs",
            raw_arguments={"service_name": "payment-gateway"},
        ),
    ]

    outputs = await tool_harness.execute_batch(batch_requests, trusted_context=server_context)

    for out in outputs:
        icon = "❌" if out.is_error else ("✂️" if out.was_truncated else "✅")
        print(f"{icon} [{out.tool_name}] (ID: {out.call_id} | {out.execution_ms}ms):")
        print(f"   {out.content}\n")


if __name__ == "__main__":
    asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
# ❌ ANTI-PATTERN: كارثة أمنية محققة!
@app.tool()
async def delete_user_document(user_id: str, doc_id: str) -> str:
    # إذا خدع المهاجم النموذج اللغوي عبر Prompt Injection، سيمرر النموذج user_id الخاص بضحية أخرى!
    ...

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
from typing import Protocol


# 1. تعريف عقد برمجي (Protocol) يضمن أن أي سياق يمرر للـ Harness يمتلك خاصية user_scopes
class HasSecurityScopes(Protocol):
    @property
    def user_scopes(self) -> set[str]:
        ...


# 2. تقييد الـ Generic Type في الـ Harness ليقبل فقط السياقات التي تحقق HasSecurityScopes
class EnterpriseToolHarness[ContextT: HasSecurityScopes]:
    # ... (داخل المزخرف @tool نضيف المعامل required_scopes: set[str] | None = None)
    # ونحفظه داخل ToolSpec كـ required_scopes: frozenset[str] = field(default_factory=frozenset)

    def get_llm_schemas(self, trusted_context: ContextT) -> list[dict[str, object]]:
        """المستوى الأول (حاجز الرؤية): إخفاء الأدوات التي لا يملك المستخدم صلاحياتها عن الـ LLM تماماً"""
        authorized_schemas: list[dict[str, object]] = []
        for spec in self._tools.values():
            if spec.required_scopes.issubset(trusted_context.user_scopes):
                authorized_schemas.append(spec.to_llm_json_schema())
        return authorized_schemas

    async def execute_single(
        self,
        request: ToolCallRequest,
        trusted_context: ContextT,
    ) -> ToolExecutionOutput:
        """المستوى الثاني (الدفاع في العمق - Defense-in-Depth): التحقق الصارم قبل التنفيذ الفعلي"""
        spec = self._tools.get(request.tool_name)
        if spec is None:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{request.tool_name}' is not available.",
                is_error=True,
            )

        # فحص الصلاحيات وقت التشغيل لمنع استدعاء الأدوات المخفية عبر التخمين أو الحقن!
        missing_scopes = spec.required_scopes - trusted_context.user_scopes
        if missing_scopes:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Authorization Denied: Missing required scopes {sorted(missing_scopes)} to run '{spec.name}'.",
                is_error=True,
            )

        # ... استكمال التحقق من المعطيات والتنفيذ كالمعتاد

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — هندسة الأدوات واستدعاء الدوال
AI Agent Engineering — Tooling & Function Calling Architecture

Book slug : agent-harness-engineering
Chapter   : ch-04-ch04 (lesson 4)
Code files: 15 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/ch04_tool_registry/ch04_tool_registry.ipynb#cell-1
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
# [002] code/ch04_tool_registry/ch04_tool_registry.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
try:
    import annotationlib  # Python 3.14+ (PEP 649)
except ModuleNotFoundError:  # توافق Python 3.12/3.13
    import harness_compat as annotationlib
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import inspect
import json
import time
from typing import Annotated, Any
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

# ────────────────────────────────────────────────────────────────────────
# [003] code/ch04_tool_registry/ch04_tool_registry.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class _InjectedMarker:
    """علامة داخلية لتمييز المعاملات التي يجب إخفاؤها عن الـ LLM وحقنها من الـ Harness"""

    pass


# نوع مساعد أنيق يستخدمه المبرمج في توقيع الدالة: ctx: Injected[AppContext]
type Injected[T] = Annotated[T, _InjectedMarker()]


def _is_injected_annotation(annotation: Any) -> bool:
    """يفحص ما إذا كان تلميح النوع مغلفاً بـ Injected[T]"""
    # فك تغليف TypeAliasType في حال استخدام `type Injected[T]` في Python 3.12-3.14+
    origin = getattr(annotation, "__value__", annotation)
    metadata = getattr(origin, "__metadata__", ())
    return any(isinstance(m, _InjectedMarker) for m in metadata)

# ────────────────────────────────────────────────────────────────────────
# [004] code/ch04_tool_registry/ch04_tool_registry.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
@dataclass(slots=True, frozen=True)
class ToolCallRequest:
    """الطلب القادم من النموذج اللغوي لتشغيل أداة"""

    call_id: str
    tool_name: str
    raw_arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ToolExecutionOutput:
    """المخرج الموحد والمؤمّن الذي يعود من الـ Harness إلى ذاكرة الوكيل"""

    call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    was_truncated: bool = False
    execution_ms: float = 0.0


@dataclass(slots=True)
class ToolSpec:
    """المواصفات الكاملة للأداة بعد فحصها ذاتياً (Introspection)"""

    name: str
    description: str
    func: Callable[..., Awaitable[Any]]
    args_model: type[BaseModel]  # نموذج Pydantic المولد ديناميكياً للمعاملات العلنية فقط
    injected_param_names: list[str]  # أسماء المعاملات السرية التي سيحقنها الـ Harness
    timeout_sec: float
    max_output_chars: int

    def to_llm_json_schema(self) -> dict[str, Any]:
        """توليد المخطط القياسي المتوافق مع OpenAI / Anthropic / Gemini"""
        schema = self.args_model.model_json_schema()
        # تنظيف الحقول الإضافية لتقليل استهلاك التوكنز
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }

# ────────────────────────────────────────────────────────────────────────
# [005] code/ch04_tool_registry/ch04_tool_registry.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class EnterpriseToolHarness[ContextT]:
    """إطار التحكم بالأدوات من الدرجة الإنتاجية.

    يولد المخططات تلقائياً، يحقن السياق الموثوق خفيةً، ويحمي نافذة السياق من الانفجار.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def tool(
        self,
        *,
        name: str | None = None,
        timeout_sec: float = 15.0,
        max_output_chars: int = 4_000,
    ) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        """Decorator ذكي يفحص الدالة ويبني لها نموذج Pydantic ومخطط LLM تلقائياً"""

        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            if not inspect.iscoroutinefunction(func):
                raise TypeError(
                    f"الأداة '{func.__name__}' يجب أن تكون غير متزامنة (async def)."
                )

            tool_name = name or func.__name__
            doc = inspect.getdoc(func)
            if not doc or len(doc.strip()) < 10:
                raise ValueError(
                    f"الأداة '{tool_name}' تفتقر إلى Docstring تفصيلي! "
                    "تذكر أن الـ Docstring هو الموجه الذي يقرؤه النموذج اللغوي."
                )

            # 1. استخراج الأنواع باستخدام annotationlib الحديثة في Python 3.14
            type_hints = annotationlib.get_annotations(
                func, format=annotationlib.Format.VALUE
            )
            sig = inspect.signature(func)

            public_fields: dict[str, Any] = {}
            injected_params: list[str] = []

            for param_name, param in sig.parameters.items():
                if param_name == "return":
                    continue

                hint = type_hints.get(param_name, Any)

                # 2. فحص هل المعامل سري يُحقن من الـ Harness أم علني يملؤه الـ LLM؟
                if _is_injected_annotation(hint):
                    injected_params.append(param_name)
                    continue

                # 3. إعداد الحقل لنموذج Pydantic الديناميكي
                default_val = ... if param.default is inspect.Parameter.empty else param.default
                public_fields[param_name] = (hint, default_val)

            # 4. إنشاء كلاس Pydantic v2 ديناميكي صارم للمعاملات العلنية فقط
            dynamic_model = create_model(
                f"{tool_name}_ArgsSchema",
                __config__=ConfigDict(extra="forbid", str_strip_whitespace=True),
                **public_fields,
            )

            self._tools[tool_name] = ToolSpec(
                name=tool_name,
                description=doc.strip(),
                func=func,
                args_model=dynamic_model,
                injected_param_names=injected_params,
                timeout_sec=timeout_sec,
                max_output_chars=max_output_chars,
            )
            return func

        return decorator

    def get_llm_schemas(self) -> list[dict[str, Any]]:
        """إرجاع كتالوج الأدوات العلني الجاهز للإرسال للنموذج اللغوي"""
        return [spec.to_llm_json_schema() for spec in self._tools.values()]

    @staticmethod
    def _bound_output(raw_text: str, max_chars: int) -> tuple[str, bool]:
        """حاجز تقليم المخرجات لحماية نافذة السياق من القنابل النصية"""
        if len(raw_text) <= max_chars:
            return raw_text, False

        truncated = raw_text[:max_chars]
        omitted_count = len(raw_text) - max_chars
        notice = (
            f"\n\n... [⚠️ HARNESS TRUNCATION NOTICE: تم اقتصاص {omitted_count} حرفاً إضافياً "
            "لحماية نافذة السياق. استخدم فلاتر أدق أو pagination في استدعائك القادم]."
        )
        return truncated + notice, True

    async def execute_single(
        self,
        request: ToolCallRequest,
        trusted_context: ContextT,
    ) -> ToolExecutionOutput:
        """تنفيذ أداة واحدة مع التحقق، وحقن السياق الموثوق، وتقليم المخرجات"""
        t0 = time.perf_counter()
        spec = self._tools.get(request.tool_name)

        if spec is None:
            available = ", ".join(self._tools.keys())
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{request.tool_name}' not found. Available tools: [{available}]",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        try:
            # 1. التحقق الصارم من المدخلات القادمة من النموذج اللغوي
            validated_public_args = spec.args_model.model_validate(request.raw_arguments)
            final_kwargs = validated_public_args.model_dump()

            # 2. حقن السياق الموثوق (Zero-Trust Injection) دون علم النموذج!
            for injected_name in spec.injected_param_names:
                final_kwargs[injected_name] = trusted_context

            # 3. التنفيذ المحمي بمهلة زمنية
            async with asyncio.timeout(spec.timeout_sec):
                raw_result = await spec.func(**final_kwargs)

            # 4. تنسيق المخرج وتطبيق حاجز التقليم (Output Bounding)
            serialized = (
                json.dumps(raw_result, ensure_ascii=False)
                if isinstance(raw_result, (dict, list))
                else str(raw_result)
            )
            bounded_text, was_truncated = self._bound_output(
                serialized, spec.max_output_chars
            )

            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=bounded_text,
                is_error=False,
                was_truncated=was_truncated,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        except ValidationError as val_err:
            # تغذية راجعة واضحة للنموذج ليصلح معطيات الأداة في الخطوة القادمة
            compact_errors = [
                f"`{'.'.join(str(x) for x in e['loc'])}`: {e['msg']}"
                for e in val_err.errors(include_url=False)
            ]
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Tool Argument Validation Error: {'; '.join(compact_errors)}",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except TimeoutError:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{spec.name}' timed out after {spec.timeout_sec}s.",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except Exception as exc:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Tool Runtime Error ({type(exc).__name__}): {exc}",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

    async def execute_batch(
        self,
        requests: list[ToolCallRequest],
        trusted_context: ContextT,
    ) -> list[ToolExecutionOutput]:
        """تنفيذ دفعة من الأدوات بالتوازي المهيكل عبر TaskGroup"""
        tasks: list[asyncio.Task[ToolExecutionOutput]] = []
        async with asyncio.TaskGroup() as tg:
            for req in requests:
                tasks.append(tg.create_task(self.execute_single(req, trusted_context)))
        return [t.result() for t in tasks]

# ────────────────────────────────────────────────────────────────────────
# [006] code/ch04_tool_registry/ch04_tool_registry.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
@dataclass(slots=True, frozen=True)
class SecurityContext:
    """سياق الأمان الموثوق الذي يملكه الخادم ولا يراه النموذج اللغوي أبداً"""

    authenticated_user_id: str
    tenant_org_id: str
    db_connection_string: str


# إنشاء نسخة من الـ Harness مخصصة لنوع السياق الأمني الخاص بنا
tool_harness = EnterpriseToolHarness[SecurityContext]()


@tool_harness.tool(timeout_sec=5.0, max_output_chars=500)
async def list_organization_invoices(
    status: Annotated[str, Field(pattern=r"^(paid|unpaid|overdue)$", description="حالة الفاتورة المطلوبة")],
    limit: Annotated[int, Field(ge=1, le=50, description="الحد الأقصى لعدد الفواتير")] = 10,
    # لاحظ هنا: المعامل ctx موسوم بـ Injected! لن يظهر في JSON Schema المرسل للـ LLM أبداً!
    ctx: Injected[SecurityContext] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    """جلب قائمة الفواتير المالية الخاصة بالمؤسسة الحالية بناءً على حالتها.

    استخدم هذه الأداة فقط عندما يسأل المستخدم عن الفواتير أو المستحقات المالية.
    """
    await asyncio.sleep(0.05)  # محاكاة استعلام قاعدة البيانات باستخدام ctx.tenant_org_id
    return {
        "verified_tenant": ctx.tenant_org_id,
        "queried_by_user": ctx.authenticated_user_id,
        "filter_status": status,
        "invoices": [
            {"inv_id": f"INV-{i}", "amount_usd": 1250.0 * i, "status": status}
            for i in range(1, limit + 1)
        ],
    }


@tool_harness.tool(timeout_sec=3.0, max_output_chars=180)
async def fetch_raw_server_logs(
    service_name: Annotated[str, Field(min_length=2, description="اسم الخدمة البرمجية")],
) -> str:
    """قراءة سجلات النظام الخام لخدمة معينة لتشخيص الأعطال التقنية."""
    await asyncio.sleep(0.02)
    # محاكاة مخرج ضخم جداً لاختبار حاجز تقليم المخرجات (Output Bounding)
    return "\n".join(
        f"[2025-05-01 12:00:{i:02d}] INFO ({service_name}): Worker processed batch #{i} successfully."
        for i in range(1, 30)
    )


async def main() -> None:
    # 1. لنفحص المخطط (JSON Schema) الذي سيعطيه الـ Harness للنموذج اللغوي!
    schemas = tool_harness.get_llm_schemas()
    print("🔍 المخطط العلني المولد تلقائياً لأداة الفواتير (لاحظ اختفاء ctx تماماً!):")
    print(json.dumps(schemas[0], indent=2, ensure_ascii=False), "\n")

    # 2. إعداد السياق الموثوق على الخادم (قادم من JWT Token مثلاً)
    server_context = SecurityContext(
        authenticated_user_id="USR-9981",
        tenant_org_id="ORG-ACME-CORP",
        db_connection_string="postgresql://prod-internal:5432/fintech",
    )

    # 3. محاكاة 3 استدعاءات متوازية من النموذج اللغوي:
    # - الأول: استدعاء سليم لأداة الفواتير (سيتم حقن الهوية تلقائياً)
    # - الثاني: محاولة خبيثة من النموذج لتمرير حقل `ctx` أو `tenant_org_id` لتزوير الهوية!
    # - الثالث: استدعاء لأداة السجلات التي تعيد نصاً ضخماً (سيتم تقليمه تلقائياً)
    batch_requests = [
        ToolCallRequest(
            call_id="call_01",
            tool_name="list_organization_invoices",
            raw_arguments={"status": "overdue", "limit": 2},
        ),
        ToolCallRequest(
            call_id="call_02",
            tool_name="list_organization_invoices",
            raw_arguments={"status": "paid", "ctx": "HACKED_CTX"},  # سيرفضه extra="forbid"!
        ),
        ToolCallRequest(
            call_id="call_03",
            tool_name="fetch_raw_server_logs",
            raw_arguments={"service_name": "payment-gateway"},
        ),
    ]

    outputs = await tool_harness.execute_batch(batch_requests, trusted_context=server_context)

    for out in outputs:
        icon = "❌" if out.is_error else ("✂️" if out.was_truncated else "✅")
        print(f"{icon} [{out.tool_name}] (ID: {out.call_id} | {out.execution_ms}ms):")
        print(f"   {out.content}\n")

# في الدفتر لا نستخدم asyncio.run: الحلقة قائمة بالفعل، لذا ننتظر التنفيذ مباشرة.
await main()

# ────────────────────────────────────────────────────────────────────────
# [007] code/ch04_tool_registry/ch04_tool_registry.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
الفصل 4: هندسة الأدوات واستدعاء الدوال
==============================================================================

البرنامج الكامل للفصل 4 من كتاب «هندسة وكلاء الذكاء الاصطناعي».
هندسة الأدوات واستدعاء الدوال — Tooling & Function Calling Architecture.

التشغيل:
    python ch04_tool_registry.py

المتطلبات: Python 3.12+ (الكتاب يستهدف 3.14+) و pydantic>=2.9
"""
# ملاحظة توافق: تُقيَّم الأنواع كسولاً في Python 3.14 (PEP 649)؛
# هذا السطر يمنح السلوك نفسه على Python 3.12/3.13.
from __future__ import annotations

try:
    import annotationlib  # Python 3.14+ (PEP 649)
except ModuleNotFoundError:  # توافق Python 3.12/3.13
    import harness_compat as annotationlib
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import inspect
import json
import time
from typing import Annotated, Any
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model


# =====================================================================
# 1. آلية حقن الاعتماديات المخفية (Zero-Trust Dependency Injection Marker)
# =====================================================================
class _InjectedMarker:
    """علامة داخلية لتمييز المعاملات التي يجب إخفاؤها عن الـ LLM وحقنها من الـ Harness"""

    pass


# نوع مساعد أنيق يستخدمه المبرمج في توقيع الدالة: ctx: Injected[AppContext]
type Injected[T] = Annotated[T, _InjectedMarker()]


def _is_injected_annotation(annotation: Any) -> bool:
    """يفحص ما إذا كان تلميح النوع مغلفاً بـ Injected[T]"""
    # فك تغليف TypeAliasType في حال استخدام `type Injected[T]` في Python 3.12-3.14+
    origin = getattr(annotation, "__value__", annotation)
    metadata = getattr(origin, "__metadata__", ())
    return any(isinstance(m, _InjectedMarker) for m in metadata)


# =====================================================================
# 2. هياكل بيانات التشغيل والمخرجات داخل الـ Tool Harness
# =====================================================================
@dataclass(slots=True, frozen=True)
class ToolCallRequest:
    """الطلب القادم من النموذج اللغوي لتشغيل أداة"""

    call_id: str
    tool_name: str
    raw_arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ToolExecutionOutput:
    """المخرج الموحد والمؤمّن الذي يعود من الـ Harness إلى ذاكرة الوكيل"""

    call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    was_truncated: bool = False
    execution_ms: float = 0.0


@dataclass(slots=True)
class ToolSpec:
    """المواصفات الكاملة للأداة بعد فحصها ذاتياً (Introspection)"""

    name: str
    description: str
    func: Callable[..., Awaitable[Any]]
    args_model: type[BaseModel]  # نموذج Pydantic المولد ديناميكياً للمعاملات العلنية فقط
    injected_param_names: list[str]  # أسماء المعاملات السرية التي سيحقنها الـ Harness
    timeout_sec: float
    max_output_chars: int

    def to_llm_json_schema(self) -> dict[str, Any]:
        """توليد المخطط القياسي المتوافق مع OpenAI / Anthropic / Gemini"""
        schema = self.args_model.model_json_schema()
        # تنظيف الحقول الإضافية لتقليل استهلاك التوكنز
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


# =====================================================================
# 3. قلب النظام: EnterpriseToolHarness (باستخدام PEP 695 Generics)
# =====================================================================
class EnterpriseToolHarness[ContextT]:
    """إطار التحكم بالأدوات من الدرجة الإنتاجية.

    يولد المخططات تلقائياً، يحقن السياق الموثوق خفيةً، ويحمي نافذة السياق من الانفجار.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def tool(
        self,
        *,
        name: str | None = None,
        timeout_sec: float = 15.0,
        max_output_chars: int = 4_000,
    ) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        """Decorator ذكي يفحص الدالة ويبني لها نموذج Pydantic ومخطط LLM تلقائياً"""

        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            if not inspect.iscoroutinefunction(func):
                raise TypeError(
                    f"الأداة '{func.__name__}' يجب أن تكون غير متزامنة (async def)."
                )

            tool_name = name or func.__name__
            doc = inspect.getdoc(func)
            if not doc or len(doc.strip()) < 10:
                raise ValueError(
                    f"الأداة '{tool_name}' تفتقر إلى Docstring تفصيلي! "
                    "تذكر أن الـ Docstring هو الموجه الذي يقرؤه النموذج اللغوي."
                )

            # 1. استخراج الأنواع باستخدام annotationlib الحديثة في Python 3.14
            type_hints = annotationlib.get_annotations(
                func, format=annotationlib.Format.VALUE
            )
            sig = inspect.signature(func)

            public_fields: dict[str, Any] = {}
            injected_params: list[str] = []

            for param_name, param in sig.parameters.items():
                if param_name == "return":
                    continue

                hint = type_hints.get(param_name, Any)

                # 2. فحص هل المعامل سري يُحقن من الـ Harness أم علني يملؤه الـ LLM؟
                if _is_injected_annotation(hint):
                    injected_params.append(param_name)
                    continue

                # 3. إعداد الحقل لنموذج Pydantic الديناميكي
                default_val = ... if param.default is inspect.Parameter.empty else param.default
                public_fields[param_name] = (hint, default_val)

            # 4. إنشاء كلاس Pydantic v2 ديناميكي صارم للمعاملات العلنية فقط
            dynamic_model = create_model(
                f"{tool_name}_ArgsSchema",
                __config__=ConfigDict(extra="forbid", str_strip_whitespace=True),
                **public_fields,
            )

            self._tools[tool_name] = ToolSpec(
                name=tool_name,
                description=doc.strip(),
                func=func,
                args_model=dynamic_model,
                injected_param_names=injected_params,
                timeout_sec=timeout_sec,
                max_output_chars=max_output_chars,
            )
            return func

        return decorator

    def get_llm_schemas(self) -> list[dict[str, Any]]:
        """إرجاع كتالوج الأدوات العلني الجاهز للإرسال للنموذج اللغوي"""
        return [spec.to_llm_json_schema() for spec in self._tools.values()]

    @staticmethod
    def _bound_output(raw_text: str, max_chars: int) -> tuple[str, bool]:
        """حاجز تقليم المخرجات لحماية نافذة السياق من القنابل النصية"""
        if len(raw_text) <= max_chars:
            return raw_text, False

        truncated = raw_text[:max_chars]
        omitted_count = len(raw_text) - max_chars
        notice = (
            f"\n\n... [⚠️ HARNESS TRUNCATION NOTICE: تم اقتصاص {omitted_count} حرفاً إضافياً "
            "لحماية نافذة السياق. استخدم فلاتر أدق أو pagination في استدعائك القادم]."
        )
        return truncated + notice, True

    async def execute_single(
        self,
        request: ToolCallRequest,
        trusted_context: ContextT,
    ) -> ToolExecutionOutput:
        """تنفيذ أداة واحدة مع التحقق، وحقن السياق الموثوق، وتقليم المخرجات"""
        t0 = time.perf_counter()
        spec = self._tools.get(request.tool_name)

        if spec is None:
            available = ", ".join(self._tools.keys())
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{request.tool_name}' not found. Available tools: [{available}]",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        try:
            # 1. التحقق الصارم من المدخلات القادمة من النموذج اللغوي
            validated_public_args = spec.args_model.model_validate(request.raw_arguments)
            final_kwargs = validated_public_args.model_dump()

            # 2. حقن السياق الموثوق (Zero-Trust Injection) دون علم النموذج!
            for injected_name in spec.injected_param_names:
                final_kwargs[injected_name] = trusted_context

            # 3. التنفيذ المحمي بمهلة زمنية
            async with asyncio.timeout(spec.timeout_sec):
                raw_result = await spec.func(**final_kwargs)

            # 4. تنسيق المخرج وتطبيق حاجز التقليم (Output Bounding)
            serialized = (
                json.dumps(raw_result, ensure_ascii=False)
                if isinstance(raw_result, (dict, list))
                else str(raw_result)
            )
            bounded_text, was_truncated = self._bound_output(
                serialized, spec.max_output_chars
            )

            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=bounded_text,
                is_error=False,
                was_truncated=was_truncated,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        except ValidationError as val_err:
            # تغذية راجعة واضحة للنموذج ليصلح معطيات الأداة في الخطوة القادمة
            compact_errors = [
                f"`{'.'.join(str(x) for x in e['loc'])}`: {e['msg']}"
                for e in val_err.errors(include_url=False)
            ]
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Tool Argument Validation Error: {'; '.join(compact_errors)}",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except TimeoutError:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{spec.name}' timed out after {spec.timeout_sec}s.",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except Exception as exc:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Tool Runtime Error ({type(exc).__name__}): {exc}",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

    async def execute_batch(
        self,
        requests: list[ToolCallRequest],
        trusted_context: ContextT,
    ) -> list[ToolExecutionOutput]:
        """تنفيذ دفعة من الأدوات بالتوازي المهيكل عبر TaskGroup"""
        tasks: list[asyncio.Task[ToolExecutionOutput]] = []
        async with asyncio.TaskGroup() as tg:
            for req in requests:
                tasks.append(tg.create_task(self.execute_single(req, trusted_context)))
        return [t.result() for t in tasks]


# =====================================================================
# 4. مثال تطبيقي: أدوات نظام مالي متعددة المستأجرين (Multi-Tenant FinTech)
# =====================================================================
@dataclass(slots=True, frozen=True)
class SecurityContext:
    """سياق الأمان الموثوق الذي يملكه الخادم ولا يراه النموذج اللغوي أبداً"""

    authenticated_user_id: str
    tenant_org_id: str
    db_connection_string: str


# إنشاء نسخة من الـ Harness مخصصة لنوع السياق الأمني الخاص بنا
tool_harness = EnterpriseToolHarness[SecurityContext]()


@tool_harness.tool(timeout_sec=5.0, max_output_chars=500)
async def list_organization_invoices(
    status: Annotated[str, Field(pattern=r"^(paid|unpaid|overdue)$", description="حالة الفاتورة المطلوبة")],
    limit: Annotated[int, Field(ge=1, le=50, description="الحد الأقصى لعدد الفواتير")] = 10,
    # لاحظ هنا: المعامل ctx موسوم بـ Injected! لن يظهر في JSON Schema المرسل للـ LLM أبداً!
    ctx: Injected[SecurityContext] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    """جلب قائمة الفواتير المالية الخاصة بالمؤسسة الحالية بناءً على حالتها.

    استخدم هذه الأداة فقط عندما يسأل المستخدم عن الفواتير أو المستحقات المالية.
    """
    await asyncio.sleep(0.05)  # محاكاة استعلام قاعدة البيانات باستخدام ctx.tenant_org_id
    return {
        "verified_tenant": ctx.tenant_org_id,
        "queried_by_user": ctx.authenticated_user_id,
        "filter_status": status,
        "invoices": [
            {"inv_id": f"INV-{i}", "amount_usd": 1250.0 * i, "status": status}
            for i in range(1, limit + 1)
        ],
    }


@tool_harness.tool(timeout_sec=3.0, max_output_chars=180)
async def fetch_raw_server_logs(
    service_name: Annotated[str, Field(min_length=2, description="اسم الخدمة البرمجية")],
) -> str:
    """قراءة سجلات النظام الخام لخدمة معينة لتشخيص الأعطال التقنية."""
    await asyncio.sleep(0.02)
    # محاكاة مخرج ضخم جداً لاختبار حاجز تقليم المخرجات (Output Bounding)
    return "\n".join(
        f"[2025-05-01 12:00:{i:02d}] INFO ({service_name}): Worker processed batch #{i} successfully."
        for i in range(1, 30)
    )


async def main() -> None:
    # 1. لنفحص المخطط (JSON Schema) الذي سيعطيه الـ Harness للنموذج اللغوي!
    schemas = tool_harness.get_llm_schemas()
    print("🔍 المخطط العلني المولد تلقائياً لأداة الفواتير (لاحظ اختفاء ctx تماماً!):")
    print(json.dumps(schemas[0], indent=2, ensure_ascii=False), "\n")

    # 2. إعداد السياق الموثوق على الخادم (قادم من JWT Token مثلاً)
    server_context = SecurityContext(
        authenticated_user_id="USR-9981",
        tenant_org_id="ORG-ACME-CORP",
        db_connection_string="postgresql://prod-internal:5432/fintech",
    )

    # 3. محاكاة 3 استدعاءات متوازية من النموذج اللغوي:
    # - الأول: استدعاء سليم لأداة الفواتير (سيتم حقن الهوية تلقائياً)
    # - الثاني: محاولة خبيثة من النموذج لتمرير حقل `ctx` أو `tenant_org_id` لتزوير الهوية!
    # - الثالث: استدعاء لأداة السجلات التي تعيد نصاً ضخماً (سيتم تقليمه تلقائياً)
    batch_requests = [
        ToolCallRequest(
            call_id="call_01",
            tool_name="list_organization_invoices",
            raw_arguments={"status": "overdue", "limit": 2},
        ),
        ToolCallRequest(
            call_id="call_02",
            tool_name="list_organization_invoices",
            raw_arguments={"status": "paid", "ctx": "HACKED_CTX"},  # سيرفضه extra="forbid"!
        ),
        ToolCallRequest(
            call_id="call_03",
            tool_name="fetch_raw_server_logs",
            raw_arguments={"service_name": "payment-gateway"},
        ),
    ]

    outputs = await tool_harness.execute_batch(batch_requests, trusted_context=server_context)

    for out in outputs:
        icon = "❌" if out.is_error else ("✂️" if out.was_truncated else "✅")
        print(f"{icon} [{out.tool_name}] (ID: {out.call_id} | {out.execution_ms}ms):")
        print(f"   {out.content}\n")


if __name__ == "__main__":
    asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [008] code/ch04_tool_registry/harness_compat.py
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
# [009] code/ch04_tool_registry/snippets/anti_pattern_02.py
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
❌ ANTI-PATTERN: كارثة أمنية محققة!

مقتطف من الفصل 4 — «هندسة الأدوات واستدعاء الدوال».
النوع: anti_pattern. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch04_tool_registry.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ❌ ANTI-PATTERN: كارثة أمنية محققة!
@app.tool()
async def delete_user_document(user_id: str, doc_id: str) -> str:
    # إذا خدع المهاجم النموذج اللغوي عبر Prompt Injection، سيمرر النموذج user_id الخاص بضحية أخرى!
    ...

# ────────────────────────────────────────────────────────────────────────
# [010] code/ch04_tool_registry/snippets/example_01.py
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
async def get_customer_invoices(user_id: int, status: str, db: DatabaseSession) -> list[dict]:

مقتطف من الفصل 4 — «هندسة الأدوات واستدعاء الدوال».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch04_tool_registry.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

async def get_customer_invoices(user_id: int, status: str, db: DatabaseSession) -> list[dict]:
    ...

# ────────────────────────────────────────────────────────────────────────
# [011] code/ch04_tool_registry/snippets/example_03.py
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
from typing import Protocol

مقتطف من الفصل 4 — «هندسة الأدوات واستدعاء الدوال».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch04_tool_registry.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

from typing import Protocol


# 1. تعريف عقد برمجي (Protocol) يضمن أن أي سياق يمرر للـ Harness يمتلك خاصية user_scopes
class HasSecurityScopes(Protocol):
    @property
    def user_scopes(self) -> set[str]:
        ...


# 2. تقييد الـ Generic Type في الـ Harness ليقبل فقط السياقات التي تحقق HasSecurityScopes
class EnterpriseToolHarness[ContextT: HasSecurityScopes]:
    # ... (داخل المزخرف @tool نضيف المعامل required_scopes: set[str] | None = None)
    # ونحفظه داخل ToolSpec كـ required_scopes: frozenset[str] = field(default_factory=frozenset)

    def get_llm_schemas(self, trusted_context: ContextT) -> list[dict[str, object]]:
        """المستوى الأول (حاجز الرؤية): إخفاء الأدوات التي لا يملك المستخدم صلاحياتها عن الـ LLM تماماً"""
        authorized_schemas: list[dict[str, object]] = []
        for spec in self._tools.values():
            if spec.required_scopes.issubset(trusted_context.user_scopes):
                authorized_schemas.append(spec.to_llm_json_schema())
        return authorized_schemas

    async def execute_single(
        self,
        request: ToolCallRequest,
        trusted_context: ContextT,
    ) -> ToolExecutionOutput:
        """المستوى الثاني (الدفاع في العمق - Defense-in-Depth): التحقق الصارم قبل التنفيذ الفعلي"""
        spec = self._tools.get(request.tool_name)
        if spec is None:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{request.tool_name}' is not available.",
                is_error=True,
            )

        # فحص الصلاحيات وقت التشغيل لمنع استدعاء الأدوات المخفية عبر التخمين أو الحقن!
        missing_scopes = spec.required_scopes - trusted_context.user_scopes
        if missing_scopes:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Authorization Denied: Missing required scopes {sorted(missing_scopes)} to run '{spec.name}'.",
                is_error=True,
            )

        # ... استكمال التحقق من المعطيات والتنفيذ كالمعتاد

# ────────────────────────────────────────────────────────────────────────
# [012] dist/md/ch04.md
# ────────────────────────────────────────────────────────────────────────
async def get_customer_invoices(user_id: int, status: str, db: DatabaseSession) -> list[dict]:
    ...

# ────────────────────────────────────────────────────────────────────────
# [013] dist/md/ch04.md
# ────────────────────────────────────────────────────────────────────────
import annotationlib  # مكتبة Python 3.14 القياسية لفحص الأنواع بدقة وأمان
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import inspect
import json
import time
from typing import Annotated, Any
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model


# =====================================================================
# 1. آلية حقن الاعتماديات المخفية (Zero-Trust Dependency Injection Marker)
# =====================================================================
class _InjectedMarker:
    """علامة داخلية لتمييز المعاملات التي يجب إخفاؤها عن الـ LLM وحقنها من الـ Harness"""

    pass


# نوع مساعد أنيق يستخدمه المبرمج في توقيع الدالة: ctx: Injected[AppContext]
type Injected[T] = Annotated[T, _InjectedMarker()]


def _is_injected_annotation(annotation: Any) -> bool:
    """يفحص ما إذا كان تلميح النوع مغلفاً بـ Injected[T]"""
    # فك تغليف TypeAliasType في حال استخدام `type Injected[T]` في Python 3.12-3.14+
    origin = getattr(annotation, "__value__", annotation)
    metadata = getattr(origin, "__metadata__", ())
    return any(isinstance(m, _InjectedMarker) for m in metadata)


# =====================================================================
# 2. هياكل بيانات التشغيل والمخرجات داخل الـ Tool Harness
# =====================================================================
@dataclass(slots=True, frozen=True)
class ToolCallRequest:
    """الطلب القادم من النموذج اللغوي لتشغيل أداة"""

    call_id: str
    tool_name: str
    raw_arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ToolExecutionOutput:
    """المخرج الموحد والمؤمّن الذي يعود من الـ Harness إلى ذاكرة الوكيل"""

    call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    was_truncated: bool = False
    execution_ms: float = 0.0


@dataclass(slots=True)
class ToolSpec:
    """المواصفات الكاملة للأداة بعد فحصها ذاتياً (Introspection)"""

    name: str
    description: str
    func: Callable[..., Awaitable[Any]]
    args_model: type[BaseModel]  # نموذج Pydantic المولد ديناميكياً للمعاملات العلنية فقط
    injected_param_names: list[str]  # أسماء المعاملات السرية التي سيحقنها الـ Harness
    timeout_sec: float
    max_output_chars: int

    def to_llm_json_schema(self) -> dict[str, Any]:
        """توليد المخطط القياسي المتوافق مع OpenAI / Anthropic / Gemini"""
        schema = self.args_model.model_json_schema()
        # تنظيف الحقول الإضافية لتقليل استهلاك التوكنز
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


# =====================================================================
# 3. قلب النظام: EnterpriseToolHarness (باستخدام PEP 695 Generics)
# =====================================================================
class EnterpriseToolHarness[ContextT]:
    """إطار التحكم بالأدوات من الدرجة الإنتاجية.

    يولد المخططات تلقائياً، يحقن السياق الموثوق خفيةً، ويحمي نافذة السياق من الانفجار.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def tool(
        self,
        *,
        name: str | None = None,
        timeout_sec: float = 15.0,
        max_output_chars: int = 4_000,
    ) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        """Decorator ذكي يفحص الدالة ويبني لها نموذج Pydantic ومخطط LLM تلقائياً"""

        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            if not inspect.iscoroutinefunction(func):
                raise TypeError(
                    f"الأداة '{func.__name__}' يجب أن تكون غير متزامنة (async def)."
                )

            tool_name = name or func.__name__
            doc = inspect.getdoc(func)
            if not doc or len(doc.strip()) < 10:
                raise ValueError(
                    f"الأداة '{tool_name}' تفتقر إلى Docstring تفصيلي! "
                    "تذكر أن الـ Docstring هو الموجه الذي يقرؤه النموذج اللغوي."
                )

            # 1. استخراج الأنواع باستخدام annotationlib الحديثة في Python 3.14
            type_hints = annotationlib.get_annotations(
                func, format=annotationlib.Format.VALUE
            )
            sig = inspect.signature(func)

            public_fields: dict[str, Any] = {}
            injected_params: list[str] = []

            for param_name, param in sig.parameters.items():
                if param_name == "return":
                    continue

                hint = type_hints.get(param_name, Any)

                # 2. فحص هل المعامل سري يُحقن من الـ Harness أم علني يملؤه الـ LLM؟
                if _is_injected_annotation(hint):
                    injected_params.append(param_name)
                    continue

                # 3. إعداد الحقل لنموذج Pydantic الديناميكي
                default_val = ... if param.default is inspect.Parameter.empty else param.default
                public_fields[param_name] = (hint, default_val)

            # 4. إنشاء كلاس Pydantic v2 ديناميكي صارم للمعاملات العلنية فقط
            dynamic_model = create_model(
                f"{tool_name}_ArgsSchema",
                __config__=ConfigDict(extra="forbid", str_strip_whitespace=True),
                **public_fields,
            )

            self._tools[tool_name] = ToolSpec(
                name=tool_name,
                description=doc.strip(),
                func=func,
                args_model=dynamic_model,
                injected_param_names=injected_params,
                timeout_sec=timeout_sec,
                max_output_chars=max_output_chars,
            )
            return func

        return decorator

    def get_llm_schemas(self) -> list[dict[str, Any]]:
        """إرجاع كتالوج الأدوات العلني الجاهز للإرسال للنموذج اللغوي"""
        return [spec.to_llm_json_schema() for spec in self._tools.values()]

    @staticmethod
    def _bound_output(raw_text: str, max_chars: int) -> tuple[str, bool]:
        """حاجز تقليم المخرجات لحماية نافذة السياق من القنابل النصية"""
        if len(raw_text) <= max_chars:
            return raw_text, False

        truncated = raw_text[:max_chars]
        omitted_count = len(raw_text) - max_chars
        notice = (
            f"\n\n... [⚠️ HARNESS TRUNCATION NOTICE: تم اقتصاص {omitted_count} حرفاً إضافياً "
            "لحماية نافذة السياق. استخدم فلاتر أدق أو pagination في استدعائك القادم]."
        )
        return truncated + notice, True

    async def execute_single(
        self,
        request: ToolCallRequest,
        trusted_context: ContextT,
    ) -> ToolExecutionOutput:
        """تنفيذ أداة واحدة مع التحقق، وحقن السياق الموثوق، وتقليم المخرجات"""
        t0 = time.perf_counter()
        spec = self._tools.get(request.tool_name)

        if spec is None:
            available = ", ".join(self._tools.keys())
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{request.tool_name}' not found. Available tools: [{available}]",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        try:
            # 1. التحقق الصارم من المدخلات القادمة من النموذج اللغوي
            validated_public_args = spec.args_model.model_validate(request.raw_arguments)
            final_kwargs = validated_public_args.model_dump()

            # 2. حقن السياق الموثوق (Zero-Trust Injection) دون علم النموذج!
            for injected_name in spec.injected_param_names:
                final_kwargs[injected_name] = trusted_context

            # 3. التنفيذ المحمي بمهلة زمنية
            async with asyncio.timeout(spec.timeout_sec):
                raw_result = await spec.func(**final_kwargs)

            # 4. تنسيق المخرج وتطبيق حاجز التقليم (Output Bounding)
            serialized = (
                json.dumps(raw_result, ensure_ascii=False)
                if isinstance(raw_result, (dict, list))
                else str(raw_result)
            )
            bounded_text, was_truncated = self._bound_output(
                serialized, spec.max_output_chars
            )

            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=bounded_text,
                is_error=False,
                was_truncated=was_truncated,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        except ValidationError as val_err:
            # تغذية راجعة واضحة للنموذج ليصلح معطيات الأداة في الخطوة القادمة
            compact_errors = [
                f"`{'.'.join(str(x) for x in e['loc'])}`: {e['msg']}"
                for e in val_err.errors(include_url=False)
            ]
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Tool Argument Validation Error: {'; '.join(compact_errors)}",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except TimeoutError:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{spec.name}' timed out after {spec.timeout_sec}s.",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except Exception as exc:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Tool Runtime Error ({type(exc).__name__}): {exc}",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

    async def execute_batch(
        self,
        requests: list[ToolCallRequest],
        trusted_context: ContextT,
    ) -> list[ToolExecutionOutput]:
        """تنفيذ دفعة من الأدوات بالتوازي المهيكل عبر TaskGroup"""
        tasks: list[asyncio.Task[ToolExecutionOutput]] = []
        async with asyncio.TaskGroup() as tg:
            for req in requests:
                tasks.append(tg.create_task(self.execute_single(req, trusted_context)))
        return [t.result() for t in tasks]


# =====================================================================
# 4. مثال تطبيقي: أدوات نظام مالي متعددة المستأجرين (Multi-Tenant FinTech)
# =====================================================================
@dataclass(slots=True, frozen=True)
class SecurityContext:
    """سياق الأمان الموثوق الذي يملكه الخادم ولا يراه النموذج اللغوي أبداً"""

    authenticated_user_id: str
    tenant_org_id: str
    db_connection_string: str


# إنشاء نسخة من الـ Harness مخصصة لنوع السياق الأمني الخاص بنا
tool_harness = EnterpriseToolHarness[SecurityContext]()


@tool_harness.tool(timeout_sec=5.0, max_output_chars=500)
async def list_organization_invoices(
    status: Annotated[str, Field(pattern=r"^(paid|unpaid|overdue)$", description="حالة الفاتورة المطلوبة")],
    limit: Annotated[int, Field(ge=1, le=50, description="الحد الأقصى لعدد الفواتير")] = 10,
    # لاحظ هنا: المعامل ctx موسوم بـ Injected! لن يظهر في JSON Schema المرسل للـ LLM أبداً!
    ctx: Injected[SecurityContext] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    """جلب قائمة الفواتير المالية الخاصة بالمؤسسة الحالية بناءً على حالتها.

    استخدم هذه الأداة فقط عندما يسأل المستخدم عن الفواتير أو المستحقات المالية.
    """
    await asyncio.sleep(0.05)  # محاكاة استعلام قاعدة البيانات باستخدام ctx.tenant_org_id
    return {
        "verified_tenant": ctx.tenant_org_id,
        "queried_by_user": ctx.authenticated_user_id,
        "filter_status": status,
        "invoices": [
            {"inv_id": f"INV-{i}", "amount_usd": 1250.0 * i, "status": status}
            for i in range(1, limit + 1)
        ],
    }


@tool_harness.tool(timeout_sec=3.0, max_output_chars=180)
async def fetch_raw_server_logs(
    service_name: Annotated[str, Field(min_length=2, description="اسم الخدمة البرمجية")],
) -> str:
    """قراءة سجلات النظام الخام لخدمة معينة لتشخيص الأعطال التقنية."""
    await asyncio.sleep(0.02)
    # محاكاة مخرج ضخم جداً لاختبار حاجز تقليم المخرجات (Output Bounding)
    return "\n".join(
        f"[2025-05-01 12:00:{i:02d}] INFO ({service_name}): Worker processed batch #{i} successfully."
        for i in range(1, 30)
    )


async def main() -> None:
    # 1. لنفحص المخطط (JSON Schema) الذي سيعطيه الـ Harness للنموذج اللغوي!
    schemas = tool_harness.get_llm_schemas()
    print("🔍 المخطط العلني المولد تلقائياً لأداة الفواتير (لاحظ اختفاء ctx تماماً!):")
    print(json.dumps(schemas[0], indent=2, ensure_ascii=False), "\n")

    # 2. إعداد السياق الموثوق على الخادم (قادم من JWT Token مثلاً)
    server_context = SecurityContext(
        authenticated_user_id="USR-9981",
        tenant_org_id="ORG-ACME-CORP",
        db_connection_string="postgresql://prod-internal:5432/fintech",
    )

    # 3. محاكاة 3 استدعاءات متوازية من النموذج اللغوي:
    # - الأول: استدعاء سليم لأداة الفواتير (سيتم حقن الهوية تلقائياً)
    # - الثاني: محاولة خبيثة من النموذج لتمرير حقل `ctx` أو `tenant_org_id` لتزوير الهوية!
    # - الثالث: استدعاء لأداة السجلات التي تعيد نصاً ضخماً (سيتم تقليمه تلقائياً)
    batch_requests = [
        ToolCallRequest(
            call_id="call_01",
            tool_name="list_organization_invoices",
            raw_arguments={"status": "overdue", "limit": 2},
        ),
        ToolCallRequest(
            call_id="call_02",
            tool_name="list_organization_invoices",
            raw_arguments={"status": "paid", "ctx": "HACKED_CTX"},  # سيرفضه extra="forbid"!
        ),
        ToolCallRequest(
            call_id="call_03",
            tool_name="fetch_raw_server_logs",
            raw_arguments={"service_name": "payment-gateway"},
        ),
    ]

    outputs = await tool_harness.execute_batch(batch_requests, trusted_context=server_context)

    for out in outputs:
        icon = "❌" if out.is_error else ("✂️" if out.was_truncated else "✅")
        print(f"{icon} [{out.tool_name}] (ID: {out.call_id} | {out.execution_ms}ms):")
        print(f"   {out.content}\n")


if __name__ == "__main__":
    asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [014] dist/md/ch04.md
# ────────────────────────────────────────────────────────────────────────
# ❌ ANTI-PATTERN: كارثة أمنية محققة!
@app.tool()
async def delete_user_document(user_id: str, doc_id: str) -> str:
    # إذا خدع المهاجم النموذج اللغوي عبر Prompt Injection، سيمرر النموذج user_id الخاص بضحية أخرى!
    ...

# ────────────────────────────────────────────────────────────────────────
# [015] dist/md/ch04.md
# ────────────────────────────────────────────────────────────────────────
from typing import Protocol


# 1. تعريف عقد برمجي (Protocol) يضمن أن أي سياق يمرر للـ Harness يمتلك خاصية user_scopes
class HasSecurityScopes(Protocol):
    @property
    def user_scopes(self) -> set[str]:
        ...


# 2. تقييد الـ Generic Type في الـ Harness ليقبل فقط السياقات التي تحقق HasSecurityScopes
class EnterpriseToolHarness[ContextT: HasSecurityScopes]:
    # ... (داخل المزخرف @tool نضيف المعامل required_scopes: set[str] | None = None)
    # ونحفظه داخل ToolSpec كـ required_scopes: frozenset[str] = field(default_factory=frozenset)

    def get_llm_schemas(self, trusted_context: ContextT) -> list[dict[str, object]]:
        """المستوى الأول (حاجز الرؤية): إخفاء الأدوات التي لا يملك المستخدم صلاحياتها عن الـ LLM تماماً"""
        authorized_schemas: list[dict[str, object]] = []
        for spec in self._tools.values():
            if spec.required_scopes.issubset(trusted_context.user_scopes):
                authorized_schemas.append(spec.to_llm_json_schema())
        return authorized_schemas

    async def execute_single(
        self,
        request: ToolCallRequest,
        trusted_context: ContextT,
    ) -> ToolExecutionOutput:
        """المستوى الثاني (الدفاع في العمق - Defense-in-Depth): التحقق الصارم قبل التنفيذ الفعلي"""
        spec = self._tools.get(request.tool_name)
        if spec is None:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{request.tool_name}' is not available.",
                is_error=True,
            )

        # فحص الصلاحيات وقت التشغيل لمنع استدعاء الأدوات المخفية عبر التخمين أو الحقن!
        missing_scopes = spec.required_scopes - trusted_context.user_scopes
        if missing_scopes:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Authorization Denied: Missing required scopes {sorted(missing_scopes)} to run '{spec.name}'.",
                is_error=True,
            )

        # ... استكمال التحقق من المعطيات والتنفيذ كالمعتاد

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-2
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
# [004] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
try:
    import annotationlib  # Python 3.14+ (PEP 649)
except ModuleNotFoundError:  # توافق Python 3.12/3.13
    import harness_compat as annotationlib
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import inspect
import json
import time
from typing import Annotated, Any
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class _InjectedMarker:
    """علامة داخلية لتمييز المعاملات التي يجب إخفاؤها عن الـ LLM وحقنها من الـ Harness"""

    pass


# نوع مساعد أنيق يستخدمه المبرمج في توقيع الدالة: ctx: Injected[AppContext]
type Injected[T] = Annotated[T, _InjectedMarker()]


def _is_injected_annotation(annotation: Any) -> bool:
    """يفحص ما إذا كان تلميح النوع مغلفاً بـ Injected[T]"""
    # فك تغليف TypeAliasType في حال استخدام `type Injected[T]` في Python 3.12-3.14+
    origin = getattr(annotation, "__value__", annotation)
    metadata = getattr(origin, "__metadata__", ())
    return any(isinstance(m, _InjectedMarker) for m in metadata)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
@dataclass(slots=True, frozen=True)
class ToolCallRequest:
    """الطلب القادم من النموذج اللغوي لتشغيل أداة"""

    call_id: str
    tool_name: str
    raw_arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ToolExecutionOutput:
    """المخرج الموحد والمؤمّن الذي يعود من الـ Harness إلى ذاكرة الوكيل"""

    call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    was_truncated: bool = False
    execution_ms: float = 0.0


@dataclass(slots=True)
class ToolSpec:
    """المواصفات الكاملة للأداة بعد فحصها ذاتياً (Introspection)"""

    name: str
    description: str
    func: Callable[..., Awaitable[Any]]
    args_model: type[BaseModel]  # نموذج Pydantic المولد ديناميكياً للمعاملات العلنية فقط
    injected_param_names: list[str]  # أسماء المعاملات السرية التي سيحقنها الـ Harness
    timeout_sec: float
    max_output_chars: int

    def to_llm_json_schema(self) -> dict[str, Any]:
        """توليد المخطط القياسي المتوافق مع OpenAI / Anthropic / Gemini"""
        schema = self.args_model.model_json_schema()
        # تنظيف الحقول الإضافية لتقليل استهلاك التوكنز
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
class EnterpriseToolHarness[ContextT]:
    """إطار التحكم بالأدوات من الدرجة الإنتاجية.

    يولد المخططات تلقائياً، يحقن السياق الموثوق خفيةً، ويحمي نافذة السياق من الانفجار.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def tool(
        self,
        *,
        name: str | None = None,
        timeout_sec: float = 15.0,
        max_output_chars: int = 4_000,
    ) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        """Decorator ذكي يفحص الدالة ويبني لها نموذج Pydantic ومخطط LLM تلقائياً"""

        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            if not inspect.iscoroutinefunction(func):
                raise TypeError(
                    f"الأداة '{func.__name__}' يجب أن تكون غير متزامنة (async def)."
                )

            tool_name = name or func.__name__
            doc = inspect.getdoc(func)
            if not doc or len(doc.strip()) < 10:
                raise ValueError(
                    f"الأداة '{tool_name}' تفتقر إلى Docstring تفصيلي! "
                    "تذكر أن الـ Docstring هو الموجه الذي يقرؤه النموذج اللغوي."
                )

            # 1. استخراج الأنواع باستخدام annotationlib الحديثة في Python 3.14
            type_hints = annotationlib.get_annotations(
                func, format=annotationlib.Format.VALUE
            )
            sig = inspect.signature(func)

            public_fields: dict[str, Any] = {}
            injected_params: list[str] = []

            for param_name, param in sig.parameters.items():
                if param_name == "return":
                    continue

                hint = type_hints.get(param_name, Any)

                # 2. فحص هل المعامل سري يُحقن من الـ Harness أم علني يملؤه الـ LLM؟
                if _is_injected_annotation(hint):
                    injected_params.append(param_name)
                    continue

                # 3. إعداد الحقل لنموذج Pydantic الديناميكي
                default_val = ... if param.default is inspect.Parameter.empty else param.default
                public_fields[param_name] = (hint, default_val)

            # 4. إنشاء كلاس Pydantic v2 ديناميكي صارم للمعاملات العلنية فقط
            dynamic_model = create_model(
                f"{tool_name}_ArgsSchema",
                __config__=ConfigDict(extra="forbid", str_strip_whitespace=True),
                **public_fields,
            )

            self._tools[tool_name] = ToolSpec(
                name=tool_name,
                description=doc.strip(),
                func=func,
                args_model=dynamic_model,
                injected_param_names=injected_params,
                timeout_sec=timeout_sec,
                max_output_chars=max_output_chars,
            )
            return func

        return decorator

    def get_llm_schemas(self) -> list[dict[str, Any]]:
        """إرجاع كتالوج الأدوات العلني الجاهز للإرسال للنموذج اللغوي"""
        return [spec.to_llm_json_schema() for spec in self._tools.values()]

    @staticmethod
    def _bound_output(raw_text: str, max_chars: int) -> tuple[str, bool]:
        """حاجز تقليم المخرجات لحماية نافذة السياق من القنابل النصية"""
        if len(raw_text) <= max_chars:
            return raw_text, False

        truncated = raw_text[:max_chars]
        omitted_count = len(raw_text) - max_chars
        notice = (
            f"\n\n... [⚠️ HARNESS TRUNCATION NOTICE: تم اقتصاص {omitted_count} حرفاً إضافياً "
            "لحماية نافذة السياق. استخدم فلاتر أدق أو pagination في استدعائك القادم]."
        )
        return truncated + notice, True

    async def execute_single(
        self,
        request: ToolCallRequest,
        trusted_context: ContextT,
    ) -> ToolExecutionOutput:
        """تنفيذ أداة واحدة مع التحقق، وحقن السياق الموثوق، وتقليم المخرجات"""
        t0 = time.perf_counter()
        spec = self._tools.get(request.tool_name)

        if spec is None:
            available = ", ".join(self._tools.keys())
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{request.tool_name}' not found. Available tools: [{available}]",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        try:
            # 1. التحقق الصارم من المدخلات القادمة من النموذج اللغوي
            validated_public_args = spec.args_model.model_validate(request.raw_arguments)
            final_kwargs = validated_public_args.model_dump()

            # 2. حقن السياق الموثوق (Zero-Trust Injection) دون علم النموذج!
            for injected_name in spec.injected_param_names:
                final_kwargs[injected_name] = trusted_context

            # 3. التنفيذ المحمي بمهلة زمنية
            async with asyncio.timeout(spec.timeout_sec):
                raw_result = await spec.func(**final_kwargs)

            # 4. تنسيق المخرج وتطبيق حاجز التقليم (Output Bounding)
            serialized = (
                json.dumps(raw_result, ensure_ascii=False)
                if isinstance(raw_result, (dict, list))
                else str(raw_result)
            )
            bounded_text, was_truncated = self._bound_output(
                serialized, spec.max_output_chars
            )

            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=bounded_text,
                is_error=False,
                was_truncated=was_truncated,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        except ValidationError as val_err:
            # تغذية راجعة واضحة للنموذج ليصلح معطيات الأداة في الخطوة القادمة
            compact_errors = [
                f"`{'.'.join(str(x) for x in e['loc'])}`: {e['msg']}"
                for e in val_err.errors(include_url=False)
            ]
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Tool Argument Validation Error: {'; '.join(compact_errors)}",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except TimeoutError:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{spec.name}' timed out after {spec.timeout_sec}s.",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except Exception as exc:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Tool Runtime Error ({type(exc).__name__}): {exc}",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

    async def execute_batch(
        self,
        requests: list[ToolCallRequest],
        trusted_context: ContextT,
    ) -> list[ToolExecutionOutput]:
        """تنفيذ دفعة من الأدوات بالتوازي المهيكل عبر TaskGroup"""
        tasks: list[asyncio.Task[ToolExecutionOutput]] = []
        async with asyncio.TaskGroup() as tg:
            for req in requests:
                tasks.append(tg.create_task(self.execute_single(req, trusted_context)))
        return [t.result() for t in tasks]

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
@dataclass(slots=True, frozen=True)
class SecurityContext:
    """سياق الأمان الموثوق الذي يملكه الخادم ولا يراه النموذج اللغوي أبداً"""

    authenticated_user_id: str
    tenant_org_id: str
    db_connection_string: str


# إنشاء نسخة من الـ Harness مخصصة لنوع السياق الأمني الخاص بنا
tool_harness = EnterpriseToolHarness[SecurityContext]()


@tool_harness.tool(timeout_sec=5.0, max_output_chars=500)
async def list_organization_invoices(
    status: Annotated[str, Field(pattern=r"^(paid|unpaid|overdue)$", description="حالة الفاتورة المطلوبة")],
    limit: Annotated[int, Field(ge=1, le=50, description="الحد الأقصى لعدد الفواتير")] = 10,
    # لاحظ هنا: المعامل ctx موسوم بـ Injected! لن يظهر في JSON Schema المرسل للـ LLM أبداً!
    ctx: Injected[SecurityContext] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    """جلب قائمة الفواتير المالية الخاصة بالمؤسسة الحالية بناءً على حالتها.

    استخدم هذه الأداة فقط عندما يسأل المستخدم عن الفواتير أو المستحقات المالية.
    """
    await asyncio.sleep(0.05)  # محاكاة استعلام قاعدة البيانات باستخدام ctx.tenant_org_id
    return {
        "verified_tenant": ctx.tenant_org_id,
        "queried_by_user": ctx.authenticated_user_id,
        "filter_status": status,
        "invoices": [
            {"inv_id": f"INV-{i}", "amount_usd": 1250.0 * i, "status": status}
            for i in range(1, limit + 1)
        ],
    }


@tool_harness.tool(timeout_sec=3.0, max_output_chars=180)
async def fetch_raw_server_logs(
    service_name: Annotated[str, Field(min_length=2, description="اسم الخدمة البرمجية")],
) -> str:
    """قراءة سجلات النظام الخام لخدمة معينة لتشخيص الأعطال التقنية."""
    await asyncio.sleep(0.02)
    # محاكاة مخرج ضخم جداً لاختبار حاجز تقليم المخرجات (Output Bounding)
    return "\n".join(
        f"[2025-05-01 12:00:{i:02d}] INFO ({service_name}): Worker processed batch #{i} successfully."
        for i in range(1, 30)
    )


async def main() -> None:
    # 1. لنفحص المخطط (JSON Schema) الذي سيعطيه الـ Harness للنموذج اللغوي!
    schemas = tool_harness.get_llm_schemas()
    print("🔍 المخطط العلني المولد تلقائياً لأداة الفواتير (لاحظ اختفاء ctx تماماً!):")
    print(json.dumps(schemas[0], indent=2, ensure_ascii=False), "\n")

    # 2. إعداد السياق الموثوق على الخادم (قادم من JWT Token مثلاً)
    server_context = SecurityContext(
        authenticated_user_id="USR-9981",
        tenant_org_id="ORG-ACME-CORP",
        db_connection_string="postgresql://prod-internal:5432/fintech",
    )

    # 3. محاكاة 3 استدعاءات متوازية من النموذج اللغوي:
    # - الأول: استدعاء سليم لأداة الفواتير (سيتم حقن الهوية تلقائياً)
    # - الثاني: محاولة خبيثة من النموذج لتمرير حقل `ctx` أو `tenant_org_id` لتزوير الهوية!
    # - الثالث: استدعاء لأداة السجلات التي تعيد نصاً ضخماً (سيتم تقليمه تلقائياً)
    batch_requests = [
        ToolCallRequest(
            call_id="call_01",
            tool_name="list_organization_invoices",
            raw_arguments={"status": "overdue", "limit": 2},
        ),
        ToolCallRequest(
            call_id="call_02",
            tool_name="list_organization_invoices",
            raw_arguments={"status": "paid", "ctx": "HACKED_CTX"},  # سيرفضه extra="forbid"!
        ),
        ToolCallRequest(
            call_id="call_03",
            tool_name="fetch_raw_server_logs",
            raw_arguments={"service_name": "payment-gateway"},
        ),
    ]

    outputs = await tool_harness.execute_batch(batch_requests, trusted_context=server_context)

    for out in outputs:
        icon = "❌" if out.is_error else ("✂️" if out.was_truncated else "✅")
        print(f"{icon} [{out.tool_name}] (ID: {out.call_id} | {out.execution_ms}ms):")
        print(f"   {out.content}\n")

# في الدفتر لا نستخدم asyncio.run: الحلقة قائمة بالفعل، لذا ننتظر التنفيذ مباشرة.
await main()

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
الفصل 4: هندسة الأدوات واستدعاء الدوال
==============================================================================

البرنامج الكامل للفصل 4 من كتاب «هندسة وكلاء الذكاء الاصطناعي».
هندسة الأدوات واستدعاء الدوال — Tooling & Function Calling Architecture.

التشغيل:
    python ch04_tool_registry.py

المتطلبات: Python 3.12+ (الكتاب يستهدف 3.14+) و pydantic>=2.9
"""
# ملاحظة توافق: تُقيَّم الأنواع كسولاً في Python 3.14 (PEP 649)؛
# هذا السطر يمنح السلوك نفسه على Python 3.12/3.13.
from __future__ import annotations

try:
    import annotationlib  # Python 3.14+ (PEP 649)
except ModuleNotFoundError:  # توافق Python 3.12/3.13
    import harness_compat as annotationlib
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import inspect
import json
import time
from typing import Annotated, Any
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model


# =====================================================================
# 1. آلية حقن الاعتماديات المخفية (Zero-Trust Dependency Injection Marker)
# =====================================================================
class _InjectedMarker:
    """علامة داخلية لتمييز المعاملات التي يجب إخفاؤها عن الـ LLM وحقنها من الـ Harness"""

    pass


# نوع مساعد أنيق يستخدمه المبرمج في توقيع الدالة: ctx: Injected[AppContext]
type Injected[T] = Annotated[T, _InjectedMarker()]


def _is_injected_annotation(annotation: Any) -> bool:
    """يفحص ما إذا كان تلميح النوع مغلفاً بـ Injected[T]"""
    # فك تغليف TypeAliasType في حال استخدام `type Injected[T]` في Python 3.12-3.14+
    origin = getattr(annotation, "__value__", annotation)
    metadata = getattr(origin, "__metadata__", ())
    return any(isinstance(m, _InjectedMarker) for m in metadata)


# =====================================================================
# 2. هياكل بيانات التشغيل والمخرجات داخل الـ Tool Harness
# =====================================================================
@dataclass(slots=True, frozen=True)
class ToolCallRequest:
    """الطلب القادم من النموذج اللغوي لتشغيل أداة"""

    call_id: str
    tool_name: str
    raw_arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ToolExecutionOutput:
    """المخرج الموحد والمؤمّن الذي يعود من الـ Harness إلى ذاكرة الوكيل"""

    call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    was_truncated: bool = False
    execution_ms: float = 0.0


@dataclass(slots=True)
class ToolSpec:
    """المواصفات الكاملة للأداة بعد فحصها ذاتياً (Introspection)"""

    name: str
    description: str
    func: Callable[..., Awaitable[Any]]
    args_model: type[BaseModel]  # نموذج Pydantic المولد ديناميكياً للمعاملات العلنية فقط
    injected_param_names: list[str]  # أسماء المعاملات السرية التي سيحقنها الـ Harness
    timeout_sec: float
    max_output_chars: int

    def to_llm_json_schema(self) -> dict[str, Any]:
        """توليد المخطط القياسي المتوافق مع OpenAI / Anthropic / Gemini"""
        schema = self.args_model.model_json_schema()
        # تنظيف الحقول الإضافية لتقليل استهلاك التوكنز
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


# =====================================================================
# 3. قلب النظام: EnterpriseToolHarness (باستخدام PEP 695 Generics)
# =====================================================================
class EnterpriseToolHarness[ContextT]:
    """إطار التحكم بالأدوات من الدرجة الإنتاجية.

    يولد المخططات تلقائياً، يحقن السياق الموثوق خفيةً، ويحمي نافذة السياق من الانفجار.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def tool(
        self,
        *,
        name: str | None = None,
        timeout_sec: float = 15.0,
        max_output_chars: int = 4_000,
    ) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        """Decorator ذكي يفحص الدالة ويبني لها نموذج Pydantic ومخطط LLM تلقائياً"""

        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            if not inspect.iscoroutinefunction(func):
                raise TypeError(
                    f"الأداة '{func.__name__}' يجب أن تكون غير متزامنة (async def)."
                )

            tool_name = name or func.__name__
            doc = inspect.getdoc(func)
            if not doc or len(doc.strip()) < 10:
                raise ValueError(
                    f"الأداة '{tool_name}' تفتقر إلى Docstring تفصيلي! "
                    "تذكر أن الـ Docstring هو الموجه الذي يقرؤه النموذج اللغوي."
                )

            # 1. استخراج الأنواع باستخدام annotationlib الحديثة في Python 3.14
            type_hints = annotationlib.get_annotations(
                func, format=annotationlib.Format.VALUE
            )
            sig = inspect.signature(func)

            public_fields: dict[str, Any] = {}
            injected_params: list[str] = []

            for param_name, param in sig.parameters.items():
                if param_name == "return":
                    continue

                hint = type_hints.get(param_name, Any)

                # 2. فحص هل المعامل سري يُحقن من الـ Harness أم علني يملؤه الـ LLM؟
                if _is_injected_annotation(hint):
                    injected_params.append(param_name)
                    continue

                # 3. إعداد الحقل لنموذج Pydantic الديناميكي
                default_val = ... if param.default is inspect.Parameter.empty else param.default
                public_fields[param_name] = (hint, default_val)

            # 4. إنشاء كلاس Pydantic v2 ديناميكي صارم للمعاملات العلنية فقط
            dynamic_model = create_model(
                f"{tool_name}_ArgsSchema",
                __config__=ConfigDict(extra="forbid", str_strip_whitespace=True),
                **public_fields,
            )

            self._tools[tool_name] = ToolSpec(
                name=tool_name,
                description=doc.strip(),
                func=func,
                args_model=dynamic_model,
                injected_param_names=injected_params,
                timeout_sec=timeout_sec,
                max_output_chars=max_output_chars,
            )
            return func

        return decorator

    def get_llm_schemas(self) -> list[dict[str, Any]]:
        """إرجاع كتالوج الأدوات العلني الجاهز للإرسال للنموذج اللغوي"""
        return [spec.to_llm_json_schema() for spec in self._tools.values()]

    @staticmethod
    def _bound_output(raw_text: str, max_chars: int) -> tuple[str, bool]:
        """حاجز تقليم المخرجات لحماية نافذة السياق من القنابل النصية"""
        if len(raw_text) <= max_chars:
            return raw_text, False

        truncated = raw_text[:max_chars]
        omitted_count = len(raw_text) - max_chars
        notice = (
            f"\n\n... [⚠️ HARNESS TRUNCATION NOTICE: تم اقتصاص {omitted_count} حرفاً إضافياً "
            "لحماية نافذة السياق. استخدم فلاتر أدق أو pagination في استدعائك القادم]."
        )
        return truncated + notice, True

    async def execute_single(
        self,
        request: ToolCallRequest,
        trusted_context: ContextT,
    ) -> ToolExecutionOutput:
        """تنفيذ أداة واحدة مع التحقق، وحقن السياق الموثوق، وتقليم المخرجات"""
        t0 = time.perf_counter()
        spec = self._tools.get(request.tool_name)

        if spec is None:
            available = ", ".join(self._tools.keys())
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{request.tool_name}' not found. Available tools: [{available}]",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        try:
            # 1. التحقق الصارم من المدخلات القادمة من النموذج اللغوي
            validated_public_args = spec.args_model.model_validate(request.raw_arguments)
            final_kwargs = validated_public_args.model_dump()

            # 2. حقن السياق الموثوق (Zero-Trust Injection) دون علم النموذج!
            for injected_name in spec.injected_param_names:
                final_kwargs[injected_name] = trusted_context

            # 3. التنفيذ المحمي بمهلة زمنية
            async with asyncio.timeout(spec.timeout_sec):
                raw_result = await spec.func(**final_kwargs)

            # 4. تنسيق المخرج وتطبيق حاجز التقليم (Output Bounding)
            serialized = (
                json.dumps(raw_result, ensure_ascii=False)
                if isinstance(raw_result, (dict, list))
                else str(raw_result)
            )
            bounded_text, was_truncated = self._bound_output(
                serialized, spec.max_output_chars
            )

            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=bounded_text,
                is_error=False,
                was_truncated=was_truncated,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        except ValidationError as val_err:
            # تغذية راجعة واضحة للنموذج ليصلح معطيات الأداة في الخطوة القادمة
            compact_errors = [
                f"`{'.'.join(str(x) for x in e['loc'])}`: {e['msg']}"
                for e in val_err.errors(include_url=False)
            ]
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Tool Argument Validation Error: {'; '.join(compact_errors)}",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except TimeoutError:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{spec.name}' timed out after {spec.timeout_sec}s.",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except Exception as exc:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Tool Runtime Error ({type(exc).__name__}): {exc}",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

    async def execute_batch(
        self,
        requests: list[ToolCallRequest],
        trusted_context: ContextT,
    ) -> list[ToolExecutionOutput]:
        """تنفيذ دفعة من الأدوات بالتوازي المهيكل عبر TaskGroup"""
        tasks: list[asyncio.Task[ToolExecutionOutput]] = []
        async with asyncio.TaskGroup() as tg:
            for req in requests:
                tasks.append(tg.create_task(self.execute_single(req, trusted_context)))
        return [t.result() for t in tasks]


# =====================================================================
# 4. مثال تطبيقي: أدوات نظام مالي متعددة المستأجرين (Multi-Tenant FinTech)
# =====================================================================
@dataclass(slots=True, frozen=True)
class SecurityContext:
    """سياق الأمان الموثوق الذي يملكه الخادم ولا يراه النموذج اللغوي أبداً"""

    authenticated_user_id: str
    tenant_org_id: str
    db_connection_string: str


# إنشاء نسخة من الـ Harness مخصصة لنوع السياق الأمني الخاص بنا
tool_harness = EnterpriseToolHarness[SecurityContext]()


@tool_harness.tool(timeout_sec=5.0, max_output_chars=500)
async def list_organization_invoices(
    status: Annotated[str, Field(pattern=r"^(paid|unpaid|overdue)$", description="حالة الفاتورة المطلوبة")],
    limit: Annotated[int, Field(ge=1, le=50, description="الحد الأقصى لعدد الفواتير")] = 10,
    # لاحظ هنا: المعامل ctx موسوم بـ Injected! لن يظهر في JSON Schema المرسل للـ LLM أبداً!
    ctx: Injected[SecurityContext] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    """جلب قائمة الفواتير المالية الخاصة بالمؤسسة الحالية بناءً على حالتها.

    استخدم هذه الأداة فقط عندما يسأل المستخدم عن الفواتير أو المستحقات المالية.
    """
    await asyncio.sleep(0.05)  # محاكاة استعلام قاعدة البيانات باستخدام ctx.tenant_org_id
    return {
        "verified_tenant": ctx.tenant_org_id,
        "queried_by_user": ctx.authenticated_user_id,
        "filter_status": status,
        "invoices": [
            {"inv_id": f"INV-{i}", "amount_usd": 1250.0 * i, "status": status}
            for i in range(1, limit + 1)
        ],
    }


@tool_harness.tool(timeout_sec=3.0, max_output_chars=180)
async def fetch_raw_server_logs(
    service_name: Annotated[str, Field(min_length=2, description="اسم الخدمة البرمجية")],
) -> str:
    """قراءة سجلات النظام الخام لخدمة معينة لتشخيص الأعطال التقنية."""
    await asyncio.sleep(0.02)
    # محاكاة مخرج ضخم جداً لاختبار حاجز تقليم المخرجات (Output Bounding)
    return "\n".join(
        f"[2025-05-01 12:00:{i:02d}] INFO ({service_name}): Worker processed batch #{i} successfully."
        for i in range(1, 30)
    )


async def main() -> None:
    # 1. لنفحص المخطط (JSON Schema) الذي سيعطيه الـ Harness للنموذج اللغوي!
    schemas = tool_harness.get_llm_schemas()
    print("🔍 المخطط العلني المولد تلقائياً لأداة الفواتير (لاحظ اختفاء ctx تماماً!):")
    print(json.dumps(schemas[0], indent=2, ensure_ascii=False), "\n")

    # 2. إعداد السياق الموثوق على الخادم (قادم من JWT Token مثلاً)
    server_context = SecurityContext(
        authenticated_user_id="USR-9981",
        tenant_org_id="ORG-ACME-CORP",
        db_connection_string="postgresql://prod-internal:5432/fintech",
    )

    # 3. محاكاة 3 استدعاءات متوازية من النموذج اللغوي:
    # - الأول: استدعاء سليم لأداة الفواتير (سيتم حقن الهوية تلقائياً)
    # - الثاني: محاولة خبيثة من النموذج لتمرير حقل `ctx` أو `tenant_org_id` لتزوير الهوية!
    # - الثالث: استدعاء لأداة السجلات التي تعيد نصاً ضخماً (سيتم تقليمه تلقائياً)
    batch_requests = [
        ToolCallRequest(
            call_id="call_01",
            tool_name="list_organization_invoices",
            raw_arguments={"status": "overdue", "limit": 2},
        ),
        ToolCallRequest(
            call_id="call_02",
            tool_name="list_organization_invoices",
            raw_arguments={"status": "paid", "ctx": "HACKED_CTX"},  # سيرفضه extra="forbid"!
        ),
        ToolCallRequest(
            call_id="call_03",
            tool_name="fetch_raw_server_logs",
            raw_arguments={"service_name": "payment-gateway"},
        ),
    ]

    outputs = await tool_harness.execute_batch(batch_requests, trusted_context=server_context)

    for out in outputs:
        icon = "❌" if out.is_error else ("✂️" if out.was_truncated else "✅")
        print(f"{icon} [{out.tool_name}] (ID: {out.call_id} | {out.execution_ms}ms):")
        print(f"   {out.content}\n")


if __name__ == "__main__":
    asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-9
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
# [011] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
❌ ANTI-PATTERN: كارثة أمنية محققة!

مقتطف من الفصل 4 — «هندسة الأدوات واستدعاء الدوال».
النوع: anti_pattern. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch04_tool_registry.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ❌ ANTI-PATTERN: كارثة أمنية محققة!
@app.tool()
async def delete_user_document(user_id: str, doc_id: str) -> str:
    # إذا خدع المهاجم النموذج اللغوي عبر Prompt Injection، سيمرر النموذج user_id الخاص بضحية أخرى!
    ...

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
async def get_customer_invoices(user_id: int, status: str, db: DatabaseSession) -> list[dict]:

مقتطف من الفصل 4 — «هندسة الأدوات واستدعاء الدوال».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch04_tool_registry.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

async def get_customer_invoices(user_id: int, status: str, db: DatabaseSession) -> list[dict]:
    ...

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
from typing import Protocol

مقتطف من الفصل 4 — «هندسة الأدوات واستدعاء الدوال».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch04_tool_registry.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

from typing import Protocol


# 1. تعريف عقد برمجي (Protocol) يضمن أن أي سياق يمرر للـ Harness يمتلك خاصية user_scopes
class HasSecurityScopes(Protocol):
    @property
    def user_scopes(self) -> set[str]:
        ...


# 2. تقييد الـ Generic Type في الـ Harness ليقبل فقط السياقات التي تحقق HasSecurityScopes
class EnterpriseToolHarness[ContextT: HasSecurityScopes]:
    # ... (داخل المزخرف @tool نضيف المعامل required_scopes: set[str] | None = None)
    # ونحفظه داخل ToolSpec كـ required_scopes: frozenset[str] = field(default_factory=frozenset)

    def get_llm_schemas(self, trusted_context: ContextT) -> list[dict[str, object]]:
        """المستوى الأول (حاجز الرؤية): إخفاء الأدوات التي لا يملك المستخدم صلاحياتها عن الـ LLM تماماً"""
        authorized_schemas: list[dict[str, object]] = []
        for spec in self._tools.values():
            if spec.required_scopes.issubset(trusted_context.user_scopes):
                authorized_schemas.append(spec.to_llm_json_schema())
        return authorized_schemas

    async def execute_single(
        self,
        request: ToolCallRequest,
        trusted_context: ContextT,
    ) -> ToolExecutionOutput:
        """المستوى الثاني (الدفاع في العمق - Defense-in-Depth): التحقق الصارم قبل التنفيذ الفعلي"""
        spec = self._tools.get(request.tool_name)
        if spec is None:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{request.tool_name}' is not available.",
                is_error=True,
            )

        # فحص الصلاحيات وقت التشغيل لمنع استدعاء الأدوات المخفية عبر التخمين أو الحقن!
        missing_scopes = spec.required_scopes - trusted_context.user_scopes
        if missing_scopes:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Authorization Denied: Missing required scopes {sorted(missing_scopes)} to run '{spec.name}'.",
                is_error=True,
            )

        # ... استكمال التحقق من المعطيات والتنفيذ كالمعتاد

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
async def get_customer_invoices(user_id: int, status: str, db: DatabaseSession) -> list[dict]:
    ...

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
import annotationlib  # مكتبة Python 3.14 القياسية لفحص الأنواع بدقة وأمان
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import inspect
import json
import time
from typing import Annotated, Any
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model


# =====================================================================
# 1. آلية حقن الاعتماديات المخفية (Zero-Trust Dependency Injection Marker)
# =====================================================================
class _InjectedMarker:
    """علامة داخلية لتمييز المعاملات التي يجب إخفاؤها عن الـ LLM وحقنها من الـ Harness"""

    pass


# نوع مساعد أنيق يستخدمه المبرمج في توقيع الدالة: ctx: Injected[AppContext]
type Injected[T] = Annotated[T, _InjectedMarker()]


def _is_injected_annotation(annotation: Any) -> bool:
    """يفحص ما إذا كان تلميح النوع مغلفاً بـ Injected[T]"""
    # فك تغليف TypeAliasType في حال استخدام `type Injected[T]` في Python 3.12-3.14+
    origin = getattr(annotation, "__value__", annotation)
    metadata = getattr(origin, "__metadata__", ())
    return any(isinstance(m, _InjectedMarker) for m in metadata)


# =====================================================================
# 2. هياكل بيانات التشغيل والمخرجات داخل الـ Tool Harness
# =====================================================================
@dataclass(slots=True, frozen=True)
class ToolCallRequest:
    """الطلب القادم من النموذج اللغوي لتشغيل أداة"""

    call_id: str
    tool_name: str
    raw_arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ToolExecutionOutput:
    """المخرج الموحد والمؤمّن الذي يعود من الـ Harness إلى ذاكرة الوكيل"""

    call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    was_truncated: bool = False
    execution_ms: float = 0.0


@dataclass(slots=True)
class ToolSpec:
    """المواصفات الكاملة للأداة بعد فحصها ذاتياً (Introspection)"""

    name: str
    description: str
    func: Callable[..., Awaitable[Any]]
    args_model: type[BaseModel]  # نموذج Pydantic المولد ديناميكياً للمعاملات العلنية فقط
    injected_param_names: list[str]  # أسماء المعاملات السرية التي سيحقنها الـ Harness
    timeout_sec: float
    max_output_chars: int

    def to_llm_json_schema(self) -> dict[str, Any]:
        """توليد المخطط القياسي المتوافق مع OpenAI / Anthropic / Gemini"""
        schema = self.args_model.model_json_schema()
        # تنظيف الحقول الإضافية لتقليل استهلاك التوكنز
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


# =====================================================================
# 3. قلب النظام: EnterpriseToolHarness (باستخدام PEP 695 Generics)
# =====================================================================
class EnterpriseToolHarness[ContextT]:
    """إطار التحكم بالأدوات من الدرجة الإنتاجية.

    يولد المخططات تلقائياً، يحقن السياق الموثوق خفيةً، ويحمي نافذة السياق من الانفجار.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def tool(
        self,
        *,
        name: str | None = None,
        timeout_sec: float = 15.0,
        max_output_chars: int = 4_000,
    ) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        """Decorator ذكي يفحص الدالة ويبني لها نموذج Pydantic ومخطط LLM تلقائياً"""

        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            if not inspect.iscoroutinefunction(func):
                raise TypeError(
                    f"الأداة '{func.__name__}' يجب أن تكون غير متزامنة (async def)."
                )

            tool_name = name or func.__name__
            doc = inspect.getdoc(func)
            if not doc or len(doc.strip()) < 10:
                raise ValueError(
                    f"الأداة '{tool_name}' تفتقر إلى Docstring تفصيلي! "
                    "تذكر أن الـ Docstring هو الموجه الذي يقرؤه النموذج اللغوي."
                )

            # 1. استخراج الأنواع باستخدام annotationlib الحديثة في Python 3.14
            type_hints = annotationlib.get_annotations(
                func, format=annotationlib.Format.VALUE
            )
            sig = inspect.signature(func)

            public_fields: dict[str, Any] = {}
            injected_params: list[str] = []

            for param_name, param in sig.parameters.items():
                if param_name == "return":
                    continue

                hint = type_hints.get(param_name, Any)

                # 2. فحص هل المعامل سري يُحقن من الـ Harness أم علني يملؤه الـ LLM؟
                if _is_injected_annotation(hint):
                    injected_params.append(param_name)
                    continue

                # 3. إعداد الحقل لنموذج Pydantic الديناميكي
                default_val = ... if param.default is inspect.Parameter.empty else param.default
                public_fields[param_name] = (hint, default_val)

            # 4. إنشاء كلاس Pydantic v2 ديناميكي صارم للمعاملات العلنية فقط
            dynamic_model = create_model(
                f"{tool_name}_ArgsSchema",
                __config__=ConfigDict(extra="forbid", str_strip_whitespace=True),
                **public_fields,
            )

            self._tools[tool_name] = ToolSpec(
                name=tool_name,
                description=doc.strip(),
                func=func,
                args_model=dynamic_model,
                injected_param_names=injected_params,
                timeout_sec=timeout_sec,
                max_output_chars=max_output_chars,
            )
            return func

        return decorator

    def get_llm_schemas(self) -> list[dict[str, Any]]:
        """إرجاع كتالوج الأدوات العلني الجاهز للإرسال للنموذج اللغوي"""
        return [spec.to_llm_json_schema() for spec in self._tools.values()]

    @staticmethod
    def _bound_output(raw_text: str, max_chars: int) -> tuple[str, bool]:
        """حاجز تقليم المخرجات لحماية نافذة السياق من القنابل النصية"""
        if len(raw_text) <= max_chars:
            return raw_text, False

        truncated = raw_text[:max_chars]
        omitted_count = len(raw_text) - max_chars
        notice = (
            f"\n\n... [⚠️ HARNESS TRUNCATION NOTICE: تم اقتصاص {omitted_count} حرفاً إضافياً "
            "لحماية نافذة السياق. استخدم فلاتر أدق أو pagination في استدعائك القادم]."
        )
        return truncated + notice, True

    async def execute_single(
        self,
        request: ToolCallRequest,
        trusted_context: ContextT,
    ) -> ToolExecutionOutput:
        """تنفيذ أداة واحدة مع التحقق، وحقن السياق الموثوق، وتقليم المخرجات"""
        t0 = time.perf_counter()
        spec = self._tools.get(request.tool_name)

        if spec is None:
            available = ", ".join(self._tools.keys())
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{request.tool_name}' not found. Available tools: [{available}]",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        try:
            # 1. التحقق الصارم من المدخلات القادمة من النموذج اللغوي
            validated_public_args = spec.args_model.model_validate(request.raw_arguments)
            final_kwargs = validated_public_args.model_dump()

            # 2. حقن السياق الموثوق (Zero-Trust Injection) دون علم النموذج!
            for injected_name in spec.injected_param_names:
                final_kwargs[injected_name] = trusted_context

            # 3. التنفيذ المحمي بمهلة زمنية
            async with asyncio.timeout(spec.timeout_sec):
                raw_result = await spec.func(**final_kwargs)

            # 4. تنسيق المخرج وتطبيق حاجز التقليم (Output Bounding)
            serialized = (
                json.dumps(raw_result, ensure_ascii=False)
                if isinstance(raw_result, (dict, list))
                else str(raw_result)
            )
            bounded_text, was_truncated = self._bound_output(
                serialized, spec.max_output_chars
            )

            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=bounded_text,
                is_error=False,
                was_truncated=was_truncated,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        except ValidationError as val_err:
            # تغذية راجعة واضحة للنموذج ليصلح معطيات الأداة في الخطوة القادمة
            compact_errors = [
                f"`{'.'.join(str(x) for x in e['loc'])}`: {e['msg']}"
                for e in val_err.errors(include_url=False)
            ]
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Tool Argument Validation Error: {'; '.join(compact_errors)}",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except TimeoutError:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{spec.name}' timed out after {spec.timeout_sec}s.",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except Exception as exc:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Tool Runtime Error ({type(exc).__name__}): {exc}",
                is_error=True,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

    async def execute_batch(
        self,
        requests: list[ToolCallRequest],
        trusted_context: ContextT,
    ) -> list[ToolExecutionOutput]:
        """تنفيذ دفعة من الأدوات بالتوازي المهيكل عبر TaskGroup"""
        tasks: list[asyncio.Task[ToolExecutionOutput]] = []
        async with asyncio.TaskGroup() as tg:
            for req in requests:
                tasks.append(tg.create_task(self.execute_single(req, trusted_context)))
        return [t.result() for t in tasks]


# =====================================================================
# 4. مثال تطبيقي: أدوات نظام مالي متعددة المستأجرين (Multi-Tenant FinTech)
# =====================================================================
@dataclass(slots=True, frozen=True)
class SecurityContext:
    """سياق الأمان الموثوق الذي يملكه الخادم ولا يراه النموذج اللغوي أبداً"""

    authenticated_user_id: str
    tenant_org_id: str
    db_connection_string: str


# إنشاء نسخة من الـ Harness مخصصة لنوع السياق الأمني الخاص بنا
tool_harness = EnterpriseToolHarness[SecurityContext]()


@tool_harness.tool(timeout_sec=5.0, max_output_chars=500)
async def list_organization_invoices(
    status: Annotated[str, Field(pattern=r"^(paid|unpaid|overdue)$", description="حالة الفاتورة المطلوبة")],
    limit: Annotated[int, Field(ge=1, le=50, description="الحد الأقصى لعدد الفواتير")] = 10,
    # لاحظ هنا: المعامل ctx موسوم بـ Injected! لن يظهر في JSON Schema المرسل للـ LLM أبداً!
    ctx: Injected[SecurityContext] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    """جلب قائمة الفواتير المالية الخاصة بالمؤسسة الحالية بناءً على حالتها.

    استخدم هذه الأداة فقط عندما يسأل المستخدم عن الفواتير أو المستحقات المالية.
    """
    await asyncio.sleep(0.05)  # محاكاة استعلام قاعدة البيانات باستخدام ctx.tenant_org_id
    return {
        "verified_tenant": ctx.tenant_org_id,
        "queried_by_user": ctx.authenticated_user_id,
        "filter_status": status,
        "invoices": [
            {"inv_id": f"INV-{i}", "amount_usd": 1250.0 * i, "status": status}
            for i in range(1, limit + 1)
        ],
    }


@tool_harness.tool(timeout_sec=3.0, max_output_chars=180)
async def fetch_raw_server_logs(
    service_name: Annotated[str, Field(min_length=2, description="اسم الخدمة البرمجية")],
) -> str:
    """قراءة سجلات النظام الخام لخدمة معينة لتشخيص الأعطال التقنية."""
    await asyncio.sleep(0.02)
    # محاكاة مخرج ضخم جداً لاختبار حاجز تقليم المخرجات (Output Bounding)
    return "\n".join(
        f"[2025-05-01 12:00:{i:02d}] INFO ({service_name}): Worker processed batch #{i} successfully."
        for i in range(1, 30)
    )


async def main() -> None:
    # 1. لنفحص المخطط (JSON Schema) الذي سيعطيه الـ Harness للنموذج اللغوي!
    schemas = tool_harness.get_llm_schemas()
    print("🔍 المخطط العلني المولد تلقائياً لأداة الفواتير (لاحظ اختفاء ctx تماماً!):")
    print(json.dumps(schemas[0], indent=2, ensure_ascii=False), "\n")

    # 2. إعداد السياق الموثوق على الخادم (قادم من JWT Token مثلاً)
    server_context = SecurityContext(
        authenticated_user_id="USR-9981",
        tenant_org_id="ORG-ACME-CORP",
        db_connection_string="postgresql://prod-internal:5432/fintech",
    )

    # 3. محاكاة 3 استدعاءات متوازية من النموذج اللغوي:
    # - الأول: استدعاء سليم لأداة الفواتير (سيتم حقن الهوية تلقائياً)
    # - الثاني: محاولة خبيثة من النموذج لتمرير حقل `ctx` أو `tenant_org_id` لتزوير الهوية!
    # - الثالث: استدعاء لأداة السجلات التي تعيد نصاً ضخماً (سيتم تقليمه تلقائياً)
    batch_requests = [
        ToolCallRequest(
            call_id="call_01",
            tool_name="list_organization_invoices",
            raw_arguments={"status": "overdue", "limit": 2},
        ),
        ToolCallRequest(
            call_id="call_02",
            tool_name="list_organization_invoices",
            raw_arguments={"status": "paid", "ctx": "HACKED_CTX"},  # سيرفضه extra="forbid"!
        ),
        ToolCallRequest(
            call_id="call_03",
            tool_name="fetch_raw_server_logs",
            raw_arguments={"service_name": "payment-gateway"},
        ),
    ]

    outputs = await tool_harness.execute_batch(batch_requests, trusted_context=server_context)

    for out in outputs:
        icon = "❌" if out.is_error else ("✂️" if out.was_truncated else "✅")
        print(f"{icon} [{out.tool_name}] (ID: {out.call_id} | {out.execution_ms}ms):")
        print(f"   {out.content}\n")


if __name__ == "__main__":
    asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
# ❌ ANTI-PATTERN: كارثة أمنية محققة!
@app.tool()
async def delete_user_document(user_id: str, doc_id: str) -> str:
    # إذا خدع المهاجم النموذج اللغوي عبر Prompt Injection، سيمرر النموذج user_id الخاص بضحية أخرى!
    ...

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/agent-harness-engineering/notebooks/ch-04-ch04.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
from typing import Protocol


# 1. تعريف عقد برمجي (Protocol) يضمن أن أي سياق يمرر للـ Harness يمتلك خاصية user_scopes
class HasSecurityScopes(Protocol):
    @property
    def user_scopes(self) -> set[str]:
        ...


# 2. تقييد الـ Generic Type في الـ Harness ليقبل فقط السياقات التي تحقق HasSecurityScopes
class EnterpriseToolHarness[ContextT: HasSecurityScopes]:
    # ... (داخل المزخرف @tool نضيف المعامل required_scopes: set[str] | None = None)
    # ونحفظه داخل ToolSpec كـ required_scopes: frozenset[str] = field(default_factory=frozenset)

    def get_llm_schemas(self, trusted_context: ContextT) -> list[dict[str, object]]:
        """المستوى الأول (حاجز الرؤية): إخفاء الأدوات التي لا يملك المستخدم صلاحياتها عن الـ LLM تماماً"""
        authorized_schemas: list[dict[str, object]] = []
        for spec in self._tools.values():
            if spec.required_scopes.issubset(trusted_context.user_scopes):
                authorized_schemas.append(spec.to_llm_json_schema())
        return authorized_schemas

    async def execute_single(
        self,
        request: ToolCallRequest,
        trusted_context: ContextT,
    ) -> ToolExecutionOutput:
        """المستوى الثاني (الدفاع في العمق - Defense-in-Depth): التحقق الصارم قبل التنفيذ الفعلي"""
        spec = self._tools.get(request.tool_name)
        if spec is None:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{request.tool_name}' is not available.",
                is_error=True,
            )

        # فحص الصلاحيات وقت التشغيل لمنع استدعاء الأدوات المخفية عبر التخمين أو الحقن!
        missing_scopes = spec.required_scopes - trusted_context.user_scopes
        if missing_scopes:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Authorization Denied: Missing required scopes {sorted(missing_scopes)} to run '{spec.name}'.",
                is_error=True,
            )

        # ... استكمال التحقق من المعطيات والتنفيذ كالمعتاد
