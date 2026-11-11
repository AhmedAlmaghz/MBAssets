#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.7: الدوال كقيم: Callable

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-022-apply_operation.py
"""

from typing import Callable

def apply_operation(
    numbers: list[int],
    operation: Callable[[int], int]
) -> list[int]:
    """يطبّق 'operation' على كل عنصر في القائمة."""
    return [operation(n) for n in numbers]

def double(x: int) -> int:
    return x * 2

def square(x: int) -> int:
    return x * x

print(apply_operation([1, 2, 3], double))  # [2, 4, 6]
print(apply_operation([1, 2, 3], square))  # [1, 4, 9]

# يعمل أيضًا مع lambda
print(apply_operation([1, 2, 3], lambda x: x + 10))  # [11, 12, 13]
