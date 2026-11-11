#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.12: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-041-func_name.py
"""

# تعريف دالة أساسية
def func_name(param: type) -> return_type:
    """Docstring."""
    return value

# قيم افتراضية
def func(a: int, b: int = 10) -> int: ...

# ⚠️ لا تستخدم mutable كقيمة افتراضية
def bad(lst: list = []): ...            # ❌
def good(lst: list | None = None):      # ✅
    if lst is None:
        lst = []

# *args و **kwargs
def func(*args, **kwargs): ...
func(1, 2, 3, key="value")

# Keyword-only و Positional-only
def func(a, b, /, c, d, *, e, f): ...
#         ^positional-only  ^keyword-only

# إرجاع قيم متعددة (tuple unpacking)
def minmax(nums: list) -> tuple:
    return min(nums), max(nums)
low, high = minmax([1, 5, 3])

# Scope
x = 10                   # Global
def f():
    y = 20                # Local
    global x; x = 99       # تعديل Global (تجنّبه قدر الإمكان)

# الدوال كقيم (First-Class Functions)
my_func = some_function   # بدون ()
sorted(data, key=my_func) # تمرير كمعامل
