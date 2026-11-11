#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.6: القوائم المتداخلة (Nested Lists)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-015-ex8_6.py
"""

matrix: list[list[int]] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

for row in matrix:
    for element in row:
        print(element, end=" ")
    print()  # سطر جديد بعد كل صف
