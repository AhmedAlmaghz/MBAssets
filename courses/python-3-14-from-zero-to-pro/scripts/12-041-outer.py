#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.10: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-041-outer.py
"""

# ============ LAMBDA ============
square = lambda x: x ** 2              # ⚠️ تجنّب تسميتها؛ استخدم def
sorted(data, key=lambda x: x["score"]) # ✅ الاستخدام الصحيح

# ============ CLOSURE ============
def outer(x):
    def inner(y):
        return x + y   # inner "تُغلِق" على x
    return inner

add5 = outer(5)
add5(3)  # 8

# nonlocal لتعديل متغير البيئة الخارجية
def counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment

# ============ DECORATOR ============
from functools import wraps

def my_decorator(func):
    @wraps(func)               # ✅ دائماً!
    def wrapper(*args, **kwargs):
        # كود قبل
        result = func(*args, **kwargs)
        # كود بعد
        return result           # ✅ لا تنسَ return!
    return wrapper

@my_decorator
def my_function(): ...

# Decorator Factory (مع معاملات)
def repeat(times):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(times=3)
def greet(): ...

# ============ GENERATOR ============
def my_generator():
    yield 1
    yield 2
    yield 3

for value in my_generator():
    print(value)

# Generator Expression
gen = (x ** 2 for x in range(1000000))  # lazy - كفاءة ذاكرة عالية

# itertools
from itertools import islice, count, cycle
