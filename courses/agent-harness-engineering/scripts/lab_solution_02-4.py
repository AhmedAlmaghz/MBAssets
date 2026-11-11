# -*- coding: utf-8 -*-
"""
مقتطف الحل النموذجي لتحدي الفصل السابع داخل ResilientModelHarness.generate_resilient():

مقتطف من الفصل 7 — «المرونة والتعافي من الأخطاء وقواطع الدائرة».
النوع: lab_solution. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch07_resilience_circuit_breaker.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# مقتطف الحل النموذجي لتحدي الفصل السابع داخل ResilientModelHarness.generate_resilient():
async def generate_resilient(
    self,
    prompt: str,
    *,
    remaining_budget_usd: float = 1.0,
    urgent_sla_ms: float | None = None,
) -> UnifiedLLMResult:
    t0 = time.perf_counter()
    total_attempts = 0
    notes: list[str] = []

    # تكييف المهلة الزمنية وعدد المحاولات ديناميكياً إذا كان هناك SLA عاجل!
    effective_max_attempts = (
        1 if (urgent_sla_ms is not None and urgent_sla_ms < 800)
        else self.retry_policy.max_attempts_per_provider
    )
    effective_timeout_sec = (
        (urgent_sla_ms / 1000.0) * 0.8
        if urgent_sla_ms is not None
        else self.retry_policy.per_attempt_timeout_sec
    )

    for ep_index, endpoint in enumerate(self.endpoints):
        # 1. حاجز الميزانية الذكي: تخطي النماذج باهظة التكلفة عندما توشك ميزانية الجلسة على النفاد
        if remaining_budget_usd < 0.01 and endpoint.cost_per_1k_tokens > 0.002:
            notes.append(
                f"💰 [Budget Guardrail] Skipped '{endpoint.model_name}' "
                f"(${endpoint.cost_per_1k_tokens}/1k) due to low remaining budget (${remaining_budget_usd:.4f})."
            )
            continue

        # ... استكمال فحص قاطع الدائرة وحلقة المحاولات باستخدام effective_max_attempts و effective_timeout_sec
