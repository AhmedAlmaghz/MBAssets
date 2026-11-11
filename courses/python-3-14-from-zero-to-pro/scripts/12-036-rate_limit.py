#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.7: مشروع تطبيقي: نظام مراقبة أداء مع Decorators وGenerators

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-036-rate_limit.py
"""

import time
from functools import wraps

def rate_limit(calls_per_second: float):
    """decorator factory يمنع استدعاء الدالة أسرع من معدل معين."""
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]  # قائمة بعنصر واحد لتجنب مشاكل nonlocal في closures متداخلة

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.perf_counter() - last_called[0]
            wait_time = min_interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            last_called[0] = time.perf_counter()
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(calls_per_second=2)  # بحد أقصى استدعائين في الثانية
def api_call(n: int) -> str:
    return f"نتيجة الاستدعاء {n}"

start = time.perf_counter()
for i in range(5):
    print(api_call(i))
print(f"الوقت الإجمالي: {time.perf_counter() - start:.2f} ثانية")  # ≈ 2 ثانية
