#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-020-log_calls.py
"""

from functools import wraps

def log_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"📝 استُدعيت: {func.__name__}({args}, {kwargs})")
        result = func(*args, **kwargs)
        print(f"📝 أُرجعت: {result}")
        return result
    return wrapper

@log_calls
def divide(a: float, b: float) -> float:
    return a / b

divide(10, 2)
