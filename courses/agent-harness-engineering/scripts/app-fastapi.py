#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — الخادم المرجعي الكامل للإنتاج باستخدام FastAPI
AI Agent Engineering — Production FastAPI Reference Server

Book slug : agent-harness-engineering
Chapter   : app-fastapi (lesson fastapi)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/agent-harness-engineering/code/app-fastapi.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — الخادم المرجعي الكامل للإنتاج باستخدام FastAPI
AI Agent Engineering — Production FastAPI Reference Server

Book slug : agent-harness-engineering
Chapter   : app-fastapi (lesson fastapi)
Code files: 2 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/agent-harness-engineering/code/app-fastapi.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — الخادم المرجعي الكامل للإنتاج باستخدام FastAPI
AI Agent Engineering — Production FastAPI Reference Server

Book slug : agent-harness-engineering
Chapter   : app-fastapi (lesson fastapi)
Code files: 1 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/md/app-fastapi.md
# ────────────────────────────────────────────────────────────────────────
import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, Field


# =====================================================================
# 1. عقود واجهة برمجة التطبيقات المهيكلة (API Request / Response Contracts)
# =====================================================================

class APIErrorDetail(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    error_code: str
    message: str
    trace_id: str


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(min_length=3, max_length=8000)
    max_steps: int = Field(default=6, ge=1, le=15)
    budget_usd: float = Field(default=0.05, gt=0.0, le=1.0)
    idempotency_key: str = Field(min_length=8, max_length=64)


class HITLResumeRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    idempotency_key: str = Field(min_length=8, max_length=64)
    decision: Literal["APPROVE", "MODIFY_AND_APPROVE", "REJECT"]
    modified_arguments: dict[str, Any] | None = None
    reviewer_notes: str = Field(min_length=5, max_length=1000)


class RequestSecurityContext(BaseModel):
    """سياق الأمان المستخرج على حدود الـ API Gateway قبل دخول الـ Harness."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_id: str
    tenant_id: str
    user_id: str
    scopes: frozenset[str]


# =====================================================================
# 2. مدير موارد الخادم ودورة الحياة (Server Resource Lifespan Manager)
# =====================================================================

class ProductionHarnessContainer:
    """
    حاوية الاعتماديات المركزية للخادم (Singleton per Worker Process).
    تُهيأ داخل lifespan لضمان عدم إنشاء اتصالات قواعد بيانات جديدة مع كل طلب.
    """

    def __init__(self) -> None:
        self.is_ready: bool = False
        self.active_streams_count: int = 0
        self._idempotency_cache: dict[str, dict[str, Any]] = {}
        self._shutdown_event = asyncio.Event()

    async def startup(self) -> None:
        # هنا يتم فتح اتصال تجمع قاعدة البيانات (PostgreSQL / SQLite WAL) ومصدر OTel
        await asyncio.sleep(0.01)
        self.is_ready = True

    async def shutdown(self) -> None:
        # إشارة إيقاف رشيق (Graceful Drain) وانتظار انتهاء التدفقات الجارية
        self.is_ready = False
        self._shutdown_event.set()
        while self.active_streams_count > 0:
            await asyncio.sleep(0.02)

    async def stream_agent_execution(
        self,
        ctx: RequestSecurityContext,
        req: AgentRunRequest,
    ) -> AsyncGenerator[str, None]:
        """مولد SSE الإنتاجي المحمي بعداد التدفقات النشطة وإلغاء المهام."""
        if not self.is_ready:
            raise RuntimeError("الخادم في طور الإيقاف ولا يقبل طلبات جديدة.")

        self.active_streams_count += 1
        seq = 1
        try:
            # الحدث 1: بدء الجلسة وإرسال trace_id للواجهة الأمامية
            yield self._sse(
                seq,
                "session_init",
                {
                    "trace_id": ctx.trace_id,
                    "tenant_id": ctx.tenant_id,
                    "idempotency_key": req.idempotency_key,
                },
            )
            seq += 1

            # الحدث 2: إشعار تشغيل الأداة
            await asyncio.sleep(0.02)
            yield self._sse(
                seq,
                "tool_status",
                {"tool": "knowledge_graph_search", "status": "completed"},
            )
            seq += 1

            # الحدث 3: تدفق التوكنات
            for chunk in ["تم ", "تحليل ", "الطلب ", "بنجاح ", "ضمن ", "الميزانية."]:
                await asyncio.sleep(0.01)
                yield self._sse(seq, "token_delta", {"delta": chunk})
                seq += 1

            # الحدث 4: الاكتمال النهائي
            yield self._sse(
                seq,
                "run_completed",
                {"trace_id": ctx.trace_id, "cost_usd": 0.0018, "status": "OK"},
            )
        finally:
            self.active_streams_count -= 1

    @staticmethod
    def _sse(seq_id: int, event_name: str, data: dict[str, Any]) -> str:
        payload = json.dumps(data, ensure_ascii=False)
        return f"id: {seq_id}\nevent: {event_name}\ndata: {payload}\n\n"


# =====================================================================
# 3. مصنع تطبيق FastAPI الإنتاجي (Production FastAPI Factory)
# =====================================================================

SSE_PRODUCTION_HEADERS: dict[str, str] = {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # يمنع Nginx و Cloudflare من حجز البث في الذاكرة
}


def create_production_fastapi_app() -> Any:
    """
    ينشئ تطبيق FastAPI كامل الإعداد إذا كانت مكتبة fastapi مثبتة في البيئة،
    أو يعيد محاكي ASGI خفيف الوزن للاختبار المباشر.
    """
    container = ProductionHarnessContainer()

    @asynccontextmanager
    async def lifespan(_app: Any) -> AsyncGenerator[None, None]:
        await container.startup()
        try:
            yield
        finally:
            await container.shutdown()

    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
        from fastapi.responses import StreamingResponse

        app = FastAPI(
            title="Enterprise Agent Harness API",
            version="1.0.0",
            lifespan=lifespan,
        )

        async def get_security_context(
            x_tenant_id: Annotated[str, Header()],
            x_user_id: Annotated[str, Header()],
            x_scopes: Annotated[str, Header()] = "agent:run",
        ) -> RequestSecurityContext:
            if not x_tenant_id or not x_user_id:
                raise HTTPException(status_code=401, detail="Missing identity headers")
            scope_set = frozenset(s.strip() for s in x_scopes.split(",") if s.strip())
            return RequestSecurityContext(
                trace_id=f"tr-{uuid.uuid4().hex[:12]}",
                tenant_id=x_tenant_id,
                user_id=x_user_id,
                scopes=scope_set,
            )

        @app.get("/health/readiness")
        async def readiness_probe() -> dict[str, Any]:
            return {
                "ready": container.is_ready,
                "active_streams": container.active_streams_count,
            }

        @app.post("/v1/agents/stream")
        async def stream_agent_endpoint(
            req: AgentRunRequest,
            ctx: Annotated[RequestSecurityContext, Depends(get_security_context)],
        ) -> StreamingResponse:
            if "agent:run" not in ctx.scopes:
                raise HTTPException(status_code=403, detail="Missing 'agent:run' scope")
            return StreamingResponse(
                container.stream_agent_execution(ctx, req),
                headers={**SSE_PRODUCTION_HEADERS, "X-Trace-ID": ctx.trace_id},
                media_type="text/event-stream",
            )

        return app
    except ImportError:
        return container


async def _demo_server_lifecycle() -> None:
    container = ProductionHarnessContainer()
    await container.startup()
    ctx = RequestSecurityContext(
        trace_id=f"tr-{uuid.uuid4().hex[:10]}",
        tenant_id="TENANT-PROD-01",
        user_id="usr_992",
        scopes=frozenset({"agent:run"}),
    )
    req = AgentRunRequest(
        query="حلل أداء المحفظة للربع الثالث",
        idempotency_key="IDEMP-2026-0001",
    )
    print("=" * 75)
    print("🌐 اختبار خادم الإنتاج المرجعي (Lifespan + SSE Headers + Graceful Drain)")
    print("=" * 75)
    print(f"• Tls/SSE Anti-Buffering Headers: {SSE_PRODUCTION_HEADERS}")
    async for frame in container.stream_agent_execution(ctx, req):
        print(f"  -> {frame.splitlines()[1]} | {frame.splitlines()[2]}")
    await container.shutdown()
    print(f"• حالة الخادم بعد الإغلاق الرشيق: is_ready={container.is_ready}")


if __name__ == "__main__":
    asyncio.run(_demo_server_lifecycle())

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/agent-harness-engineering/notebooks/app-fastapi.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, Field


# =====================================================================
# 1. عقود واجهة برمجة التطبيقات المهيكلة (API Request / Response Contracts)
# =====================================================================

class APIErrorDetail(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    error_code: str
    message: str
    trace_id: str


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(min_length=3, max_length=8000)
    max_steps: int = Field(default=6, ge=1, le=15)
    budget_usd: float = Field(default=0.05, gt=0.0, le=1.0)
    idempotency_key: str = Field(min_length=8, max_length=64)


class HITLResumeRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    idempotency_key: str = Field(min_length=8, max_length=64)
    decision: Literal["APPROVE", "MODIFY_AND_APPROVE", "REJECT"]
    modified_arguments: dict[str, Any] | None = None
    reviewer_notes: str = Field(min_length=5, max_length=1000)


class RequestSecurityContext(BaseModel):
    """سياق الأمان المستخرج على حدود الـ API Gateway قبل دخول الـ Harness."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_id: str
    tenant_id: str
    user_id: str
    scopes: frozenset[str]


# =====================================================================
# 2. مدير موارد الخادم ودورة الحياة (Server Resource Lifespan Manager)
# =====================================================================

class ProductionHarnessContainer:
    """
    حاوية الاعتماديات المركزية للخادم (Singleton per Worker Process).
    تُهيأ داخل lifespan لضمان عدم إنشاء اتصالات قواعد بيانات جديدة مع كل طلب.
    """

    def __init__(self) -> None:
        self.is_ready: bool = False
        self.active_streams_count: int = 0
        self._idempotency_cache: dict[str, dict[str, Any]] = {}
        self._shutdown_event = asyncio.Event()

    async def startup(self) -> None:
        # هنا يتم فتح اتصال تجمع قاعدة البيانات (PostgreSQL / SQLite WAL) ومصدر OTel
        await asyncio.sleep(0.01)
        self.is_ready = True

    async def shutdown(self) -> None:
        # إشارة إيقاف رشيق (Graceful Drain) وانتظار انتهاء التدفقات الجارية
        self.is_ready = False
        self._shutdown_event.set()
        while self.active_streams_count > 0:
            await asyncio.sleep(0.02)

    async def stream_agent_execution(
        self,
        ctx: RequestSecurityContext,
        req: AgentRunRequest,
    ) -> AsyncGenerator[str, None]:
        """مولد SSE الإنتاجي المحمي بعداد التدفقات النشطة وإلغاء المهام."""
        if not self.is_ready:
            raise RuntimeError("الخادم في طور الإيقاف ولا يقبل طلبات جديدة.")

        self.active_streams_count += 1
        seq = 1
        try:
            # الحدث 1: بدء الجلسة وإرسال trace_id للواجهة الأمامية
            yield self._sse(
                seq,
                "session_init",
                {
                    "trace_id": ctx.trace_id,
                    "tenant_id": ctx.tenant_id,
                    "idempotency_key": req.idempotency_key,
                },
            )
            seq += 1

            # الحدث 2: إشعار تشغيل الأداة
            await asyncio.sleep(0.02)
            yield self._sse(
                seq,
                "tool_status",
                {"tool": "knowledge_graph_search", "status": "completed"},
            )
            seq += 1

            # الحدث 3: تدفق التوكنات
            for chunk in ["تم ", "تحليل ", "الطلب ", "بنجاح ", "ضمن ", "الميزانية."]:
                await asyncio.sleep(0.01)
                yield self._sse(seq, "token_delta", {"delta": chunk})
                seq += 1

            # الحدث 4: الاكتمال النهائي
            yield self._sse(
                seq,
                "run_completed",
                {"trace_id": ctx.trace_id, "cost_usd": 0.0018, "status": "OK"},
            )
        finally:
            self.active_streams_count -= 1

    @staticmethod
    def _sse(seq_id: int, event_name: str, data: dict[str, Any]) -> str:
        payload = json.dumps(data, ensure_ascii=False)
        return f"id: {seq_id}\nevent: {event_name}\ndata: {payload}\n\n"


# =====================================================================
# 3. مصنع تطبيق FastAPI الإنتاجي (Production FastAPI Factory)
# =====================================================================

SSE_PRODUCTION_HEADERS: dict[str, str] = {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # يمنع Nginx و Cloudflare من حجز البث في الذاكرة
}


def create_production_fastapi_app() -> Any:
    """
    ينشئ تطبيق FastAPI كامل الإعداد إذا كانت مكتبة fastapi مثبتة في البيئة،
    أو يعيد محاكي ASGI خفيف الوزن للاختبار المباشر.
    """
    container = ProductionHarnessContainer()

    @asynccontextmanager
    async def lifespan(_app: Any) -> AsyncGenerator[None, None]:
        await container.startup()
        try:
            yield
        finally:
            await container.shutdown()

    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
        from fastapi.responses import StreamingResponse

        app = FastAPI(
            title="Enterprise Agent Harness API",
            version="1.0.0",
            lifespan=lifespan,
        )

        async def get_security_context(
            x_tenant_id: Annotated[str, Header()],
            x_user_id: Annotated[str, Header()],
            x_scopes: Annotated[str, Header()] = "agent:run",
        ) -> RequestSecurityContext:
            if not x_tenant_id or not x_user_id:
                raise HTTPException(status_code=401, detail="Missing identity headers")
            scope_set = frozenset(s.strip() for s in x_scopes.split(",") if s.strip())
            return RequestSecurityContext(
                trace_id=f"tr-{uuid.uuid4().hex[:12]}",
                tenant_id=x_tenant_id,
                user_id=x_user_id,
                scopes=scope_set,
            )

        @app.get("/health/readiness")
        async def readiness_probe() -> dict[str, Any]:
            return {
                "ready": container.is_ready,
                "active_streams": container.active_streams_count,
            }

        @app.post("/v1/agents/stream")
        async def stream_agent_endpoint(
            req: AgentRunRequest,
            ctx: Annotated[RequestSecurityContext, Depends(get_security_context)],
        ) -> StreamingResponse:
            if "agent:run" not in ctx.scopes:
                raise HTTPException(status_code=403, detail="Missing 'agent:run' scope")
            return StreamingResponse(
                container.stream_agent_execution(ctx, req),
                headers={**SSE_PRODUCTION_HEADERS, "X-Trace-ID": ctx.trace_id},
                media_type="text/event-stream",
            )

        return app
    except ImportError:
        return container


async def _demo_server_lifecycle() -> None:
    container = ProductionHarnessContainer()
    await container.startup()
    ctx = RequestSecurityContext(
        trace_id=f"tr-{uuid.uuid4().hex[:10]}",
        tenant_id="TENANT-PROD-01",
        user_id="usr_992",
        scopes=frozenset({"agent:run"}),
    )
    req = AgentRunRequest(
        query="حلل أداء المحفظة للربع الثالث",
        idempotency_key="IDEMP-2026-0001",
    )
    print("=" * 75)
    print("🌐 اختبار خادم الإنتاج المرجعي (Lifespan + SSE Headers + Graceful Drain)")
    print("=" * 75)
    print(f"• Tls/SSE Anti-Buffering Headers: {SSE_PRODUCTION_HEADERS}")
    async for frame in container.stream_agent_execution(ctx, req):
        print(f"  -> {frame.splitlines()[1]} | {frame.splitlines()[2]}")
    await container.shutdown()
    print(f"• حالة الخادم بعد الإغلاق الرشيق: is_ready={container.is_ready}")


if __name__ == "__main__":
    asyncio.run(_demo_server_lifecycle())

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/agent-harness-engineering/notebooks/app-fastapi.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — الخادم المرجعي الكامل للإنتاج باستخدام FastAPI
AI Agent Engineering — Production FastAPI Reference Server

Book slug : agent-harness-engineering
Chapter   : app-fastapi (lesson fastapi)
Code files: 1 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/md/app-fastapi.md
# ────────────────────────────────────────────────────────────────────────
import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, Field


# =====================================================================
# 1. عقود واجهة برمجة التطبيقات المهيكلة (API Request / Response Contracts)
# =====================================================================

class APIErrorDetail(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    error_code: str
    message: str
    trace_id: str


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(min_length=3, max_length=8000)
    max_steps: int = Field(default=6, ge=1, le=15)
    budget_usd: float = Field(default=0.05, gt=0.0, le=1.0)
    idempotency_key: str = Field(min_length=8, max_length=64)


class HITLResumeRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    idempotency_key: str = Field(min_length=8, max_length=64)
    decision: Literal["APPROVE", "MODIFY_AND_APPROVE", "REJECT"]
    modified_arguments: dict[str, Any] | None = None
    reviewer_notes: str = Field(min_length=5, max_length=1000)


class RequestSecurityContext(BaseModel):
    """سياق الأمان المستخرج على حدود الـ API Gateway قبل دخول الـ Harness."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_id: str
    tenant_id: str
    user_id: str
    scopes: frozenset[str]


# =====================================================================
# 2. مدير موارد الخادم ودورة الحياة (Server Resource Lifespan Manager)
# =====================================================================

class ProductionHarnessContainer:
    """
    حاوية الاعتماديات المركزية للخادم (Singleton per Worker Process).
    تُهيأ داخل lifespan لضمان عدم إنشاء اتصالات قواعد بيانات جديدة مع كل طلب.
    """

    def __init__(self) -> None:
        self.is_ready: bool = False
        self.active_streams_count: int = 0
        self._idempotency_cache: dict[str, dict[str, Any]] = {}
        self._shutdown_event = asyncio.Event()

    async def startup(self) -> None:
        # هنا يتم فتح اتصال تجمع قاعدة البيانات (PostgreSQL / SQLite WAL) ومصدر OTel
        await asyncio.sleep(0.01)
        self.is_ready = True

    async def shutdown(self) -> None:
        # إشارة إيقاف رشيق (Graceful Drain) وانتظار انتهاء التدفقات الجارية
        self.is_ready = False
        self._shutdown_event.set()
        while self.active_streams_count > 0:
            await asyncio.sleep(0.02)

    async def stream_agent_execution(
        self,
        ctx: RequestSecurityContext,
        req: AgentRunRequest,
    ) -> AsyncGenerator[str, None]:
        """مولد SSE الإنتاجي المحمي بعداد التدفقات النشطة وإلغاء المهام."""
        if not self.is_ready:
            raise RuntimeError("الخادم في طور الإيقاف ولا يقبل طلبات جديدة.")

        self.active_streams_count += 1
        seq = 1
        try:
            # الحدث 1: بدء الجلسة وإرسال trace_id للواجهة الأمامية
            yield self._sse(
                seq,
                "session_init",
                {
                    "trace_id": ctx.trace_id,
                    "tenant_id": ctx.tenant_id,
                    "idempotency_key": req.idempotency_key,
                },
            )
            seq += 1

            # الحدث 2: إشعار تشغيل الأداة
            await asyncio.sleep(0.02)
            yield self._sse(
                seq,
                "tool_status",
                {"tool": "knowledge_graph_search", "status": "completed"},
            )
            seq += 1

            # الحدث 3: تدفق التوكنات
            for chunk in ["تم ", "تحليل ", "الطلب ", "بنجاح ", "ضمن ", "الميزانية."]:
                await asyncio.sleep(0.01)
                yield self._sse(seq, "token_delta", {"delta": chunk})
                seq += 1

            # الحدث 4: الاكتمال النهائي
            yield self._sse(
                seq,
                "run_completed",
                {"trace_id": ctx.trace_id, "cost_usd": 0.0018, "status": "OK"},
            )
        finally:
            self.active_streams_count -= 1

    @staticmethod
    def _sse(seq_id: int, event_name: str, data: dict[str, Any]) -> str:
        payload = json.dumps(data, ensure_ascii=False)
        return f"id: {seq_id}\nevent: {event_name}\ndata: {payload}\n\n"


# =====================================================================
# 3. مصنع تطبيق FastAPI الإنتاجي (Production FastAPI Factory)
# =====================================================================

SSE_PRODUCTION_HEADERS: dict[str, str] = {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # يمنع Nginx و Cloudflare من حجز البث في الذاكرة
}


def create_production_fastapi_app() -> Any:
    """
    ينشئ تطبيق FastAPI كامل الإعداد إذا كانت مكتبة fastapi مثبتة في البيئة،
    أو يعيد محاكي ASGI خفيف الوزن للاختبار المباشر.
    """
    container = ProductionHarnessContainer()

    @asynccontextmanager
    async def lifespan(_app: Any) -> AsyncGenerator[None, None]:
        await container.startup()
        try:
            yield
        finally:
            await container.shutdown()

    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
        from fastapi.responses import StreamingResponse

        app = FastAPI(
            title="Enterprise Agent Harness API",
            version="1.0.0",
            lifespan=lifespan,
        )

        async def get_security_context(
            x_tenant_id: Annotated[str, Header()],
            x_user_id: Annotated[str, Header()],
            x_scopes: Annotated[str, Header()] = "agent:run",
        ) -> RequestSecurityContext:
            if not x_tenant_id or not x_user_id:
                raise HTTPException(status_code=401, detail="Missing identity headers")
            scope_set = frozenset(s.strip() for s in x_scopes.split(",") if s.strip())
            return RequestSecurityContext(
                trace_id=f"tr-{uuid.uuid4().hex[:12]}",
                tenant_id=x_tenant_id,
                user_id=x_user_id,
                scopes=scope_set,
            )

        @app.get("/health/readiness")
        async def readiness_probe() -> dict[str, Any]:
            return {
                "ready": container.is_ready,
                "active_streams": container.active_streams_count,
            }

        @app.post("/v1/agents/stream")
        async def stream_agent_endpoint(
            req: AgentRunRequest,
            ctx: Annotated[RequestSecurityContext, Depends(get_security_context)],
        ) -> StreamingResponse:
            if "agent:run" not in ctx.scopes:
                raise HTTPException(status_code=403, detail="Missing 'agent:run' scope")
            return StreamingResponse(
                container.stream_agent_execution(ctx, req),
                headers={**SSE_PRODUCTION_HEADERS, "X-Trace-ID": ctx.trace_id},
                media_type="text/event-stream",
            )

        return app
    except ImportError:
        return container


async def _demo_server_lifecycle() -> None:
    container = ProductionHarnessContainer()
    await container.startup()
    ctx = RequestSecurityContext(
        trace_id=f"tr-{uuid.uuid4().hex[:10]}",
        tenant_id="TENANT-PROD-01",
        user_id="usr_992",
        scopes=frozenset({"agent:run"}),
    )
    req = AgentRunRequest(
        query="حلل أداء المحفظة للربع الثالث",
        idempotency_key="IDEMP-2026-0001",
    )
    print("=" * 75)
    print("🌐 اختبار خادم الإنتاج المرجعي (Lifespan + SSE Headers + Graceful Drain)")
    print("=" * 75)
    print(f"• Tls/SSE Anti-Buffering Headers: {SSE_PRODUCTION_HEADERS}")
    async for frame in container.stream_agent_execution(ctx, req):
        print(f"  -> {frame.splitlines()[1]} | {frame.splitlines()[2]}")
    await container.shutdown()
    print(f"• حالة الخادم بعد الإغلاق الرشيق: is_ready={container.is_ready}")


if __name__ == "__main__":
    asyncio.run(_demo_server_lifecycle())

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/agent-harness-engineering/notebooks/app-fastapi.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, Field


# =====================================================================
# 1. عقود واجهة برمجة التطبيقات المهيكلة (API Request / Response Contracts)
# =====================================================================

class APIErrorDetail(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    error_code: str
    message: str
    trace_id: str


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(min_length=3, max_length=8000)
    max_steps: int = Field(default=6, ge=1, le=15)
    budget_usd: float = Field(default=0.05, gt=0.0, le=1.0)
    idempotency_key: str = Field(min_length=8, max_length=64)


class HITLResumeRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    idempotency_key: str = Field(min_length=8, max_length=64)
    decision: Literal["APPROVE", "MODIFY_AND_APPROVE", "REJECT"]
    modified_arguments: dict[str, Any] | None = None
    reviewer_notes: str = Field(min_length=5, max_length=1000)


class RequestSecurityContext(BaseModel):
    """سياق الأمان المستخرج على حدود الـ API Gateway قبل دخول الـ Harness."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_id: str
    tenant_id: str
    user_id: str
    scopes: frozenset[str]


# =====================================================================
# 2. مدير موارد الخادم ودورة الحياة (Server Resource Lifespan Manager)
# =====================================================================

class ProductionHarnessContainer:
    """
    حاوية الاعتماديات المركزية للخادم (Singleton per Worker Process).
    تُهيأ داخل lifespan لضمان عدم إنشاء اتصالات قواعد بيانات جديدة مع كل طلب.
    """

    def __init__(self) -> None:
        self.is_ready: bool = False
        self.active_streams_count: int = 0
        self._idempotency_cache: dict[str, dict[str, Any]] = {}
        self._shutdown_event = asyncio.Event()

    async def startup(self) -> None:
        # هنا يتم فتح اتصال تجمع قاعدة البيانات (PostgreSQL / SQLite WAL) ومصدر OTel
        await asyncio.sleep(0.01)
        self.is_ready = True

    async def shutdown(self) -> None:
        # إشارة إيقاف رشيق (Graceful Drain) وانتظار انتهاء التدفقات الجارية
        self.is_ready = False
        self._shutdown_event.set()
        while self.active_streams_count > 0:
            await asyncio.sleep(0.02)

    async def stream_agent_execution(
        self,
        ctx: RequestSecurityContext,
        req: AgentRunRequest,
    ) -> AsyncGenerator[str, None]:
        """مولد SSE الإنتاجي المحمي بعداد التدفقات النشطة وإلغاء المهام."""
        if not self.is_ready:
            raise RuntimeError("الخادم في طور الإيقاف ولا يقبل طلبات جديدة.")

        self.active_streams_count += 1
        seq = 1
        try:
            # الحدث 1: بدء الجلسة وإرسال trace_id للواجهة الأمامية
            yield self._sse(
                seq,
                "session_init",
                {
                    "trace_id": ctx.trace_id,
                    "tenant_id": ctx.tenant_id,
                    "idempotency_key": req.idempotency_key,
                },
            )
            seq += 1

            # الحدث 2: إشعار تشغيل الأداة
            await asyncio.sleep(0.02)
            yield self._sse(
                seq,
                "tool_status",
                {"tool": "knowledge_graph_search", "status": "completed"},
            )
            seq += 1

            # الحدث 3: تدفق التوكنات
            for chunk in ["تم ", "تحليل ", "الطلب ", "بنجاح ", "ضمن ", "الميزانية."]:
                await asyncio.sleep(0.01)
                yield self._sse(seq, "token_delta", {"delta": chunk})
                seq += 1

            # الحدث 4: الاكتمال النهائي
            yield self._sse(
                seq,
                "run_completed",
                {"trace_id": ctx.trace_id, "cost_usd": 0.0018, "status": "OK"},
            )
        finally:
            self.active_streams_count -= 1

    @staticmethod
    def _sse(seq_id: int, event_name: str, data: dict[str, Any]) -> str:
        payload = json.dumps(data, ensure_ascii=False)
        return f"id: {seq_id}\nevent: {event_name}\ndata: {payload}\n\n"


# =====================================================================
# 3. مصنع تطبيق FastAPI الإنتاجي (Production FastAPI Factory)
# =====================================================================

SSE_PRODUCTION_HEADERS: dict[str, str] = {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # يمنع Nginx و Cloudflare من حجز البث في الذاكرة
}


def create_production_fastapi_app() -> Any:
    """
    ينشئ تطبيق FastAPI كامل الإعداد إذا كانت مكتبة fastapi مثبتة في البيئة،
    أو يعيد محاكي ASGI خفيف الوزن للاختبار المباشر.
    """
    container = ProductionHarnessContainer()

    @asynccontextmanager
    async def lifespan(_app: Any) -> AsyncGenerator[None, None]:
        await container.startup()
        try:
            yield
        finally:
            await container.shutdown()

    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
        from fastapi.responses import StreamingResponse

        app = FastAPI(
            title="Enterprise Agent Harness API",
            version="1.0.0",
            lifespan=lifespan,
        )

        async def get_security_context(
            x_tenant_id: Annotated[str, Header()],
            x_user_id: Annotated[str, Header()],
            x_scopes: Annotated[str, Header()] = "agent:run",
        ) -> RequestSecurityContext:
            if not x_tenant_id or not x_user_id:
                raise HTTPException(status_code=401, detail="Missing identity headers")
            scope_set = frozenset(s.strip() for s in x_scopes.split(",") if s.strip())
            return RequestSecurityContext(
                trace_id=f"tr-{uuid.uuid4().hex[:12]}",
                tenant_id=x_tenant_id,
                user_id=x_user_id,
                scopes=scope_set,
            )

        @app.get("/health/readiness")
        async def readiness_probe() -> dict[str, Any]:
            return {
                "ready": container.is_ready,
                "active_streams": container.active_streams_count,
            }

        @app.post("/v1/agents/stream")
        async def stream_agent_endpoint(
            req: AgentRunRequest,
            ctx: Annotated[RequestSecurityContext, Depends(get_security_context)],
        ) -> StreamingResponse:
            if "agent:run" not in ctx.scopes:
                raise HTTPException(status_code=403, detail="Missing 'agent:run' scope")
            return StreamingResponse(
                container.stream_agent_execution(ctx, req),
                headers={**SSE_PRODUCTION_HEADERS, "X-Trace-ID": ctx.trace_id},
                media_type="text/event-stream",
            )

        return app
    except ImportError:
        return container


async def _demo_server_lifecycle() -> None:
    container = ProductionHarnessContainer()
    await container.startup()
    ctx = RequestSecurityContext(
        trace_id=f"tr-{uuid.uuid4().hex[:10]}",
        tenant_id="TENANT-PROD-01",
        user_id="usr_992",
        scopes=frozenset({"agent:run"}),
    )
    req = AgentRunRequest(
        query="حلل أداء المحفظة للربع الثالث",
        idempotency_key="IDEMP-2026-0001",
    )
    print("=" * 75)
    print("🌐 اختبار خادم الإنتاج المرجعي (Lifespan + SSE Headers + Graceful Drain)")
    print("=" * 75)
    print(f"• Tls/SSE Anti-Buffering Headers: {SSE_PRODUCTION_HEADERS}")
    async for frame in container.stream_agent_execution(ctx, req):
        print(f"  -> {frame.splitlines()[1]} | {frame.splitlines()[2]}")
    await container.shutdown()
    print(f"• حالة الخادم بعد الإغلاق الرشيق: is_ready={container.is_ready}")


if __name__ == "__main__":
    asyncio.run(_demo_server_lifecycle())
