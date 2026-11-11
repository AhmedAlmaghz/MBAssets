#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع: التكرار (Loops)
القسم 7.11: دالة `enumerate()`: الحصول على الفهرس والقيمة معًا

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 07-019-ex7_11.py
"""

# بدء العدّ من 1 بدلاً من 0
for index, fruit in enumerate(fruits, start=1):
    print(f"{index}: {fruit}")
