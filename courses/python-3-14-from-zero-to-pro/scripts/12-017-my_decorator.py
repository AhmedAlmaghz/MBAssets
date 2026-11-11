#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-017-my_decorator.py
"""

def my_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name: str) -> str:
    """تُرجع تحية للمستخدم."""
    return f"مرحباً {name}"

print(greet.__name__)  # wrapper  ⚠️ فقدنا الاسم الأصلي!
print(greet.__doc__)   # None     ⚠️ فقدنا التوثيق الأصلي!
