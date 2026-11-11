#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.7: الدوال كقيم: Callable

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-024-measure_time.py
"""

from typing import Callable
import time
import functools

def measure_time(func: Callable[..., float]) -> Callable[..., float]:
    @functools.wraps(func)
    def wrapper(*args: int, **kwargs: int) -> float:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"استغرقت {func.__name__}: {elapsed:.4f} ثانية")
        return result
    return wrapper

@measure_time
def slow_calculation(n: int) -> float:
    return sum(i ** 2 for i in range(n)) / n

slow_calculation(1_000_000)
