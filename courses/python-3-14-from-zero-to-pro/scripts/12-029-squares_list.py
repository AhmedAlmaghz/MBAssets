#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-029-squares_list.py
"""

# List Comprehension - ينشئ القائمة بالكامل فوراً في الذاكرة
squares_list = [x ** 2 for x in range(1_000_000)]

# Generator Expression - ينتج القيم عند الطلب فقط (lazy evaluation)
squares_gen = (x ** 2 for x in range(1_000_000))

print(type(squares_list))  # <class 'list'>
print(type(squares_gen))   # <class 'generator'>
