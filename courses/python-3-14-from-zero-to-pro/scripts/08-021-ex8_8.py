#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.8: List Comprehension: الأسلوب الأنيق والـ"Pythonic"

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-021-ex8_8.py
"""

# الطريقة التقليدية لإنشاء قائمة بمربعات الأرقام
squares: list[int] = []
for number in range(1, 6):
    squares.append(number ** 2)

print(squares)  # [1, 4, 9, 16, 25]
