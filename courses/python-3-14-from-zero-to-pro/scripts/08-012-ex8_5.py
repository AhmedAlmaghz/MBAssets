#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.5: أهم Methods القوائم

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-012-ex8_5.py
"""

numbers: list[int] = [3, 1, 2]
sorted_numbers: list[int] = sorted(numbers)

print(numbers)         # [3, 1, 2]  ← لم تتغيّر!
print(sorted_numbers)  # [1, 2, 3]  ← قائمة جديدة مُرتَّبة
