#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.13: أخطاء شائعة ونصائح احترافية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-040-sorted_list.py
"""

numbers: list[int] = [3, 1, 2]
sorted_list = numbers.sort()
print(sorted_list)  # None ← خطأ شائع جدًا!
