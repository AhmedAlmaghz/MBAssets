#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-026-get_squares_generator.py
"""

def get_squares_generator(n: int):
    """تُنتج المربعات واحداً تلو الآخر، دون تخزين كل شيء في الذاكرة."""
    for i in range(n):
        yield i ** 2  # yield بدلاً من return

gen = get_squares_generator(5)
print(gen)  # <generator object get_squares_generator at 0x...>

print(next(gen))  # 0
print(next(gen))  # 1
print(next(gen))  # 4
print(next(gen))  # 9
print(next(gen))  # 16
# print(next(gen))  # 💥 StopIteration - انتهت القيم
