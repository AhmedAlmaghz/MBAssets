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
