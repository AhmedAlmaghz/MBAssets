#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.7: ⚠️ الفخ الأشهر على الإطلاق: نسخ القوائم (Shallow Copy Trap)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-019-ex8_7.py
"""

original: list[list[int]] = [[1, 2], [3, 4]]
copy: list[list[int]] = original.copy()  # نسخة سطحية فقط!

copy[0].append(99)

print(original)  # [[1, 2, 99], [3, 4]]  ← تأثرت رغم النسخ!
print(copy)       # [[1, 2, 99], [3, 4]]
