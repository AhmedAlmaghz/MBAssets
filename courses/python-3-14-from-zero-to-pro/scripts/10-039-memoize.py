#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.10: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-039-memoize.py
"""

def memoize(func):
    """تُرجع نسخة من func تخزّن النتائج السابقة."""
    cache: dict = {}

    def wrapper(*args):
        if args in cache:
            print(f"(من الذاكرة المؤقتة لـ {args})")
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return wrapper

def slow_square(n: int) -> int:
    import time
    time.sleep(1)  # محاكاة عملية بطيئة
    return n ** 2

fast_square = memoize(slow_square)

print(fast_square(5))  # يأخذ ثانية واحدة، النتيجة: 25
print(fast_square(5))  # فوري! (من الذاكرة المؤقتة لـ (5,))
print(fast_square(7))  # يأخذ ثانية واحدة، النتيجة: 49
