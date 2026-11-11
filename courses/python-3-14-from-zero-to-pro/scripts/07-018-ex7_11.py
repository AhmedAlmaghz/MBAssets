#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع: التكرار (Loops)
القسم 7.11: دالة `enumerate()`: الحصول على الفهرس والقيمة معًا

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 07-018-ex7_11.py
"""

fruits: list[str] = ["تفاح", "موز", "برتقال"]  # تشويق للفصل القادم

for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
