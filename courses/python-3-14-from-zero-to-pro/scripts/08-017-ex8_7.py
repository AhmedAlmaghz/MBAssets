#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.7: ⚠️ الفخ الأشهر على الإطلاق: نسخ القوائم (Shallow Copy Trap)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-017-ex8_7.py
"""

original: list[int] = [1, 2, 3]
copy: list[int] = original

print(id(original))  # مثلاً: 140234567890
print(id(copy))      # نفس الرقم بالضبط! 140234567890
