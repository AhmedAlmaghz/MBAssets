#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-030-squares_list.py
"""

import sys

squares_list = [x ** 2 for x in range(1_000_000)]
squares_gen = (x ** 2 for x in range(1_000_000))

print(f"حجم القائمة: {sys.getsizeof(squares_list):,} بايت")
print(f"حجم المولّد: {sys.getsizeof(squares_gen):,} بايت")
