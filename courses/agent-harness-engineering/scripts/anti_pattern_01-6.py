# -*- coding: utf-8 -*-
"""
❌ ANTI-PATTERN: إهدار الوقت والمال على خطأ لن يتغير أبداً!

مقتطف من الفصل 7 — «المرونة والتعافي من الأخطاء وقواطع الدائرة».
النوع: anti_pattern. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch07_resilience_circuit_breaker.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ❌ ANTI-PATTERN: إهدار الوقت والمال على خطأ لن يتغير أبداً!
for _ in range(5):
    try:
        return await call_llm(messages)  # ماذا لو كان الخطأ ContextLengthExceeded (السياق أطول من اللازم)؟!
    except Exception:
        await asyncio.sleep(2)  # سينتظر 10 ثوانٍ ويفشل 5 مرات بنفس الخطأ الحتمي!
