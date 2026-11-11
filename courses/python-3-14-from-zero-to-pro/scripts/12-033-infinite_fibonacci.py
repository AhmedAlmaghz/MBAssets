#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-033-infinite_fibonacci.py
"""

def infinite_fibonacci():
    """مولّد لا نهائي لمتتالية فيبوناتشي."""
    a, b = 0, 1
    while True:  # حلقة لا نهائية - آمنة هنا لأن المولّد lazy!
        yield a
        a, b = b, a + b

fib = infinite_fibonacci()
first_10 = [next(fib) for _ in range(10)]
print(first_10)
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
