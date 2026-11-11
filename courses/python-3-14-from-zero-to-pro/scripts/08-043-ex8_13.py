#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.13: أخطاء شائعة ونصائح احترافية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-043-ex8_13.py
"""

numbers: list[int] = [1, 2, 3, 4, 5]
odd_numbers: list[int] = [n for n in numbers if n % 2 != 0]
print(odd_numbers)  # [1, 3, 5]
