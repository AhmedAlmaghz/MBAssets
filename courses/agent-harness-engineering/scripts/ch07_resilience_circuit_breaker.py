#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
الفصل 7: المرونة والتعافي من الأخطاء وقواطع الدائرة
==============================================================================

البرنامج الكامل للفصل 7 من كتاب «هندسة وكلاء الذكاء الاصطناعي».
المرونة والتعافي من الأخطاء وقواطع الدائرة — Resilience, Smart Retries, Circuit Breakers & Model Fallbacks.

التشغيل:
    python ch07_resilience_circuit_breaker.py

المتطلبات: Python 3.12+ (الكتاب يستهدف 3.14+) و pydantic>=2.9
"""
# ملاحظة توافق: تُقيَّم الأنواع كسولاً في Python 3.14 (PEP 649)؛
# هذا السطر يمنح السلوك نفسه على Python 3.12/3.13.
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum, auto
import random
import time
from typing import Any, Protocol


# =====================================================================
# 1. تصنيف الأخطاء وهياكل الاستجابة الموحدة (Error Taxonomy & Contracts)
# =====================================================================
class ErrorKind(StrEnum):
    """تصنيف هندسي لطبيعة الخطأ ليقرر الـ Harness كيف يتصرف"""

    RATE_LIMIT = auto()      # HTTP 429: قابل لإعادة المحاولة مع احترام Retry-After
    TRANSIENT_5XX = auto()   # HTTP 502/503/504 أو Timeout: قابل لإعادة المحاولة بـ Jitter
    CONTEXT_LENGTH = auto()  # تجاوز نافذة السياق: لا فائدة من إعادة المحاولة لنفس النموذج!
    FATAL_AUTH = auto()      # HTTP 401/403: خطأ قاتل في المزود الحالي، انتقل للبديل فوراً


class LLMProviderError(Exception):
    """استثناء موحد يغلف أخطاء أي مزود نموذج لغوي"""

    def __init__(
        self,
        message: str,
        *,
        provider_name: str,
        kind: ErrorKind,
        retry_after_sec: float | None = None,
    ) -> None:
        super().__init__(message)
        self.provider_name = provider_name
        self.kind = kind
        self.retry_after_sec = retry_after_sec

    @property
    def is_retryable_on_same_provider(self) -> bool:
        return self.kind in {ErrorKind.RATE_LIMIT, ErrorKind.TRANSIENT_5XX}


@dataclass(slots=True, frozen=True)
class UnifiedLLMResult:
    """النتيجة الموحدة التي يعيدها الـ Resilient Harness بغض النظر عن النموذج الذي أجاب"""

    provider_used: str
    model_name: str
    content: str
    used_fallback: bool
    total_attempts: int
    total_latency_ms: float
    failover_notes: tuple[str, ...] = ()


class ModelEndpoint(Protocol):
    """عقد برمجي لأي نقطة اتصال بنموذج لغوي داخل سلسلة البدائل"""

    @property
    def provider_name(self) -> str:
        ...

    @property
    def model_name(self) -> str:
        ...

    async def invoke(self, prompt: str) -> str:
        ...


# =====================================================================
# 2. قاطع الدائرة غير المتزامن (State-Aware Async Circuit Breaker)
# =====================================================================
class CircuitState(StrEnum):
    CLOSED = auto()     # يعمل بشكل طبيعي
    OPEN = auto()       # معطل مؤقتاً: يرفض الطلبات في 0ms ويحول للبديل فوراً
    HALF_OPEN = auto()  # تحت الاختبار: يسمح بطلب استكشافي واحد


class CircuitOpenError(RuntimeError):
    """يُرفع فوراً عندما تكون دائرة المزود مفتوحة لتوفير الوقت والموارد"""

    pass


class AsyncCircuitBreaker:
    """قاطع دائرة هندسي يحمي الـ Harness من الانتظار أمام مزود ساقط"""

    def __init__(
        self,
        provider_name: str,
        *,
        failure_threshold: int = 3,
        cooldown_sec: float = 10.0,
    ) -> None:
        self.provider_name = provider_name
        self.failure_threshold = failure_threshold
        self.cooldown_sec = cooldown_sec

        self._state: CircuitState = CircuitState.CLOSED
        self._consecutive_failures: int = 0
        self._opened_at_monotonic: float = 0.0
        self._half_open_probe_in_flight: bool = False

    @property
    def state(self) -> CircuitState:
        """حساب الحالة اللحظية مع التحول التلقائي من OPEN إلى HALF_OPEN بعد انتهاء التهدئة"""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._opened_at_monotonic
            if elapsed >= self.cooldown_sec:
                self._state = CircuitState.HALF_OPEN
                self._half_open_probe_in_flight = False
        return self._state

    def before_call(self) -> None:
        """يفحص الدائرة قبل إرسال الطلب؛ فإذا كانت مفتوحة يقطع الطريق فوراً"""
        current = self.state
        match current:
            case CircuitState.OPEN:
                remaining = max(
                    0.0, self.cooldown_sec - (time.monotonic() - self._opened_at_monotonic)
                )
                raise CircuitOpenError(
                    f"⚡ Circuit OPEN for '{self.provider_name}' (cooldown: {remaining:.1f}s left)."
                )
            case CircuitState.HALF_OPEN:
                if self._half_open_probe_in_flight:
                    raise CircuitOpenError(
                        f"⚡ Circuit HALF_OPEN for '{self.provider_name}' (probe already in flight)."
                    )
                self._half_open_probe_in_flight = True
            case CircuitState.CLOSED:
                pass

    def record_success(self) -> None:
        """عند نجاح الطلب: إغلاق الدائرة وتصفير عداد الأخطاء"""
        self._consecutive_failures = 0
        self._state = CircuitState.CLOSED
        self._half_open_probe_in_flight = False

    def record_failure(self) -> None:
        """عند فشل المزود: زيادة العداد وفتح الدائرة إذا بلغنا العتبة القصوى"""
        self._consecutive_failures += 1
        self._half_open_probe_in_flight = False
        if (
            self._state == CircuitState.HALF_OPEN
            or self._consecutive_failures >= self.failure_threshold
        ):
            self._state = CircuitState.OPEN
            self._opened_at_monotonic = time.monotonic()


# =====================================================================
# 3. قلب النظام: ResilientModelHarness (Jitter Retry + Circuit Breaker + Fallback)
# =====================================================================
@dataclass(slots=True, kw_only=True)
class RetryPolicy:
    """سياسة إعادة المحاولة الرياضية باستخدام Full Jitter"""

    max_attempts_per_provider: int = 3
    base_delay_sec: float = 0.1
    max_delay_sec: float = 2.0
    per_attempt_timeout_sec: float = 10.0

    def compute_sleep_sec(self, attempt: int, retry_after_hint: float | None = None) -> float:
        """حساب زمن الانتظار مع احترام ترويسة Retry-After وتطبيق Full Jitter"""
        if retry_after_hint is not None and retry_after_hint > 0:
            # إضافة Jitter صغير جداً فوق Retry-After لمنع تصادم الوكلاء بعد انتهاء المهلة
            return min(self.max_delay_sec, retry_after_hint + random.uniform(0.01, 0.05))  # noqa: S311

        exp_cap = min(self.max_delay_sec, self.base_delay_sec * (2 ** (attempt - 1)))
        return random.uniform(0.01, exp_cap)  # noqa: S311


class ResilientModelHarness:
    """إطار التحكم بالمرونة والتوافر العالي للنماذج اللغوية.

    يدير سلسلة من النماذج (Primary -> Fallbacks)، ويحمي كل مزود بقاطع دائرة مستقل.
    """

    def __init__(
        self,
        endpoints: list[ModelEndpoint],
        *,
        retry_policy: RetryPolicy | None = None,
        circuit_failure_threshold: int = 2,
        circuit_cooldown_sec: float = 5.0,
    ) -> None:
        if not endpoints:
            raise ValueError("يجب تزويد الـ Harness بنقطة اتصال واحدة على الأقل.")

        self.endpoints = endpoints
        self.retry_policy = retry_policy or RetryPolicy()
        self.breakers: dict[str, AsyncCircuitBreaker] = {
            ep.provider_name: AsyncCircuitBreaker(
                ep.provider_name,
                failure_threshold=circuit_failure_threshold,
                cooldown_sec=circuit_cooldown_sec,
            )
            for ep in endpoints
        }

    async def generate_resilient(self, prompt: str) -> UnifiedLLMResult:
        """تنفيذ الطلب بأعلى مرونة ممكنة عبر سلسلة النماذج الاحتياطية"""
        t0 = time.perf_counter()
        total_attempts = 0
        notes: list[str] = []

        for ep_index, endpoint in enumerate(self.endpoints):
            breaker = self.breakers[endpoint.provider_name]
            is_fallback = ep_index > 0

            # 1. فحص قاطع الدائرة قبل محاولة لمس المزود
            try:
                breaker.before_call()
            except CircuitOpenError as open_err:
                notes.append(str(open_err))
                continue  # القفز الفوري للمزود البديل في 0 ميلي ثانية!

            # 2. حلقة إعادة المحاولة الذكية داخل المزود الحالي
            for attempt in range(1, self.retry_policy.max_attempts_per_provider + 1):
                total_attempts += 1
                try:
                    async with asyncio.timeout(self.retry_policy.per_attempt_timeout_sec):
                        reply = await endpoint.invoke(prompt)

                    # نجاح! نغلق الدائرة ونعيد النتيجة فوراً
                    breaker.record_success()
                    return UnifiedLLMResult(
                        provider_used=endpoint.provider_name,
                        model_name=endpoint.model_name,
                        content=reply,
                        used_fallback=is_fallback,
                        total_attempts=total_attempts,
                        total_latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                        failover_notes=tuple(notes),
                    )

                except TimeoutError:
                    breaker.record_failure()
                    notes.append(
                        f"⏱️ [{endpoint.provider_name}] Attempt #{attempt} timed out."
                    )
                    if breaker.state == CircuitState.OPEN:
                        notes.append(
                            f"🚨 [{endpoint.provider_name}] Circuit tripped to OPEN! Failing over..."
                        )
                        break

                except LLMProviderError as prov_err:
                    # إذا كان الخطأ قاتلاً أو متعلقاً بنافذة السياق، لا نعيد المحاولة على نفس المزود!
                    if not prov_err.is_retryable_on_same_provider:
                        notes.append(
                            f"⛔ [{endpoint.provider_name}] Non-retryable ({prov_err.kind}): {prov_err}. Switching provider..."
                        )
                        break

                    breaker.record_failure()
                    notes.append(
                        f"⚠️ [{endpoint.provider_name}] Attempt #{attempt} failed ({prov_err.kind}): {prov_err}"
                    )

                    # إذا فتح قاطع الدائرة بسبب تراكم الأخطاء، نخرج فوراً للمزود البديل
                    if breaker.state == CircuitState.OPEN:
                        notes.append(
                            f"🚨 [{endpoint.provider_name}] Circuit tripped to OPEN after {attempt} failures! Failing over..."
                        )
                        break

                    # إذا تبقّت محاولات على نفس المزود، ننتظر باستخدام Full Jitter
                    if attempt < self.retry_policy.max_attempts_per_provider:
                        sleep_s = self.retry_policy.compute_sleep_sec(
                            attempt, prov_err.retry_after_sec
                        )
                        notes.append(
                            f"⏳ [{endpoint.provider_name}] Backing off for {sleep_s * 1000:.1f}ms (Full Jitter)..."
                        )
                        await asyncio.sleep(sleep_s)

        raise RuntimeError(
            f"💥 فشل جميع المزودين في سلسلة الطوارئ بعد {total_attempts} محاولات! السجل: {notes}"
        )


# =====================================================================
# 4. محاكاة تشغيلية واقعية: سقوط المزود الأساسي والإنقاذ اللحظي (Hot Failover)
# =====================================================================
class UnstablePrimaryEndpoint:
    """محاكاة للمزود الأساسي: يعاني من خطأ 429 ثم سقوط 503 مستمر"""

    provider_name = "Anthropic-Primary"
    model_name = "claude-3-7-sonnet"

    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, prompt: str) -> str:
        await asyncio.sleep(0.02)
        self.calls += 1
        if self.calls == 1:
            raise LLMProviderError(
                "Rate limit exceeded (Tokens per minute)",
                provider_name=self.provider_name,
                kind=ErrorKind.RATE_LIMIT,
                retry_after_sec=0.04,
            )
        raise LLMProviderError(
            "HTTP 529 Overloaded: GPU cluster capacity exhausted",
            provider_name=self.provider_name,
            kind=ErrorKind.TRANSIENT_5XX,
        )


class ReliableFallbackEndpoint:
    """محاكاة للمزود الاحتياطي المستقر الذي ينقذ الموقف"""

    provider_name = "OpenAI-Fallback"
    model_name = "gpt-4o"

    async def invoke(self, prompt: str) -> str:
        await asyncio.sleep(0.03)
        return f"تم تحليل الطلب بنجاح عبر النموذج الاحتياطي ({self.model_name})."


async def main() -> None:
    harness = ResilientModelHarness(
        endpoints=[UnstablePrimaryEndpoint(), ReliableFallbackEndpoint()],
        retry_policy=RetryPolicy(max_attempts_per_provider=3, base_delay_sec=0.05),
        circuit_failure_threshold=2,  # يفتح القاطع بعد خطأين متتاليين!
        circuit_cooldown_sec=10.0,
    )

    print("─── 📞 الطلب الأول: المزود الأساسي يفشل مرتين فيفتح القاطع ويتحول للبديل ───")
    res1 = await harness.generate_resilient("حلل الميزانية الربع سنوية للشركة.")
    print(f"✅ المزود المجيب     : {res1.provider_used} ({res1.model_name})")
    print(f"🔄 استخدم البديل؟    : {res1.used_fallback} | إجمالي المحاولات: {res1.total_attempts}")
    print(f"⏱️ الزمن الكلي       : {res1.total_latency_ms} ms")
    print("📜 سجل أحداث المرونة (Failover Telemetry):")
    for note in res1.failover_notes:
        print(f"   -> {note}")

    print("\n─── ⚡ الطلب الثاني (بعد ثانية واحدة): لاحظ سرعة القاطع المفتوح (Fail-Fast 0ms)! ───")
    res2 = await harness.generate_resilient("استخرج أهم المخاطر المالية.")
    print(f"✅ المزود المجيب     : {res2.provider_used} ({res2.model_name})")
    print(f"⏱️ الزمن الكلي       : {res2.total_latency_ms} ms (تخطى المزود المعطل في 0ms!)")
    for note in res2.failover_notes:
        print(f"   -> {note}")


if __name__ == "__main__":
    asyncio.run(main())
