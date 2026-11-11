#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.9: دوال مُدمجة قوية تعمل مع القوائم

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-031-ex8_9.py
"""

numbers: list[int] = [4, 2, 8, 1, 9, 5]

print(len(numbers))    # 6   ← عدد العناصر
print(sum(numbers))    # 29  ← المجموع
print(max(numbers))    # 9   ← أكبر قيمة
print(min(numbers))    # 1   ← أصغر قيمة
