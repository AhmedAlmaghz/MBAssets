#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-015-my_decorator.py
"""

def my_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"🔵 استدعاء {func.__name__} بالمعاملات: {args}, {kwargs}")
        result = func(*args, **kwargs)
        print(f"🔴 النتيجة: {result}")
        return result
    return wrapper

@my_decorator
def add(a: int, b: int) -> int:
    return a + b

result = add(3, 5)
