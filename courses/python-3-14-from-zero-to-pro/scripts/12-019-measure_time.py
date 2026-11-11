#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-019-measure_time.py
"""

import time
from functools import wraps

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"⏱️ {func.__name__} استغرقت {end - start:.4f} ثانية")
        return result
    return wrapper

@measure_time
def slow_calculation(n: int) -> int:
    total = sum(i ** 2 for i in range(n))
    return total

result = slow_calculation(1_000_000)
print(result)
