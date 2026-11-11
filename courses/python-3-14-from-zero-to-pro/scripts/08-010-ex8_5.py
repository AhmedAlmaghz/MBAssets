#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.5: أهم Methods القوائم

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-010-ex8_5.py
"""

numbers: list[int] = [5, 2, 8, 1, 9]

numbers.sort()
print(numbers)  # [1, 2, 5, 8, 9]  ← تُعدّل القائمة الأصلية مباشرة!

numbers.sort(reverse=True)
print(numbers)  # [9, 8, 5, 2, 1]

numbers.reverse()
print(numbers)  # [1, 2, 5, 8, 9]  ← عكس الترتيب الحالي
