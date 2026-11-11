#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.8: List Comprehension: الأسلوب الأنيق والـ"Pythonic"

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-029-ex8_8.py
"""

matrix: list[list[int]] = [[1, 2], [3, 4], [5, 6]]
flattened: list[int] = [num for row in matrix for num in row]
print(flattened)  # [1, 2, 3, 4, 5, 6]
