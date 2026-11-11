#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-027-ex12_5.py
"""

# غالباً نستخدم for مباشرة بدلاً من next() اليدوية
for square in get_squares_generator(5):
    print(square)
# 0
# 1
# 4
# 9
# 16
