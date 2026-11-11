#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.8: Type Hints المتقدمة في الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-034-format_value.py
"""

# استخدام | للسماح بأكثر من نوع (Python 3.10+)
def format_value(value: int | float | str) -> str:
    return f"القيمة: {value}"

# استخدام Optional (أو | None) لمعامل قد يكون غائباً
def find_user(user_id: int) -> dict | None:
    users = {1: {"name": "Ahmed"}, 2: {"name": "Sara"}}
    return users.get(user_id)

# Callable: Type Hint لمعامل يكون دالة
from collections.abc import Callable

def apply_twice(func: Callable[[int], int], value: int) -> int:
    return func(func(value))

def add_five(x: int) -> int:
    return x + 5

print(apply_twice(add_five, 10))  # 20
