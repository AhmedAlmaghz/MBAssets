#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-018-my_decorator.py
"""

from functools import wraps

def my_decorator(func):
    @wraps(func)  # يحافظ على __name__ و __doc__ الأصليين
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name: str) -> str:
    """تُرجع تحية للمستخدم."""
    return f"مرحباً {name}"

print(greet.__name__)  # greet  ✅
print(greet.__doc__)   # تُرجع تحية للمستخدم.  ✅
