#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.8: List Comprehension: الأسلوب الأنيق والـ"Pythonic"

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-022-ex8_8.py
"""

squares: list[int] = [number ** 2 for number in range(1, 6)]
print(squares)  # [1, 4, 9, 16, 25]
