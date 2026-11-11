#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.5: أهم Methods القوائم

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-011-result.py
"""

numbers: list[int] = [3, 1, 2]
result = numbers.sort()
print(result)   # None  ← فخ شائع جدًا!
print(numbers)  # [1, 2, 3]  ← هنا التعديل الفعلي
