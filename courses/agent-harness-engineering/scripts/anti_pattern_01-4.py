# -*- coding: utf-8 -*-
"""
❌ ANTI-PATTERN: هذا الكود يبطل الـ Prefix KV-Cache في كل خطوة ويزيد فاتورتك 5 أضعاف!

مقتطف من الفصل 5 — «هندسة الذاكرة والسياق».
النوع: anti_pattern. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch05_memory_and_context.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ❌ ANTI-PATTERN: هذا الكود يبطل الـ Prefix KV-Cache في كل خطوة ويزيد فاتورتك 5 أضعاف!
def build_bad_system_prompt(step: int) -> str:
    # وضع الوقت المتغير بالثواني ورقم الخطوة في بداية System Prompt يدمر الـ Cache لكل ما يليه!
    return f"Current Time: {datetime.now().isoformat()} | Step: {step}\nYou are an enterprise agent..."
