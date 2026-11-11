#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-031-gen.py
"""

gen = (x for x in range(3))

print(list(gen))  # [0, 1, 2]
print(list(gen))  # []  ⚠️ فارغة! تم استهلاك المولّد بالكامل
