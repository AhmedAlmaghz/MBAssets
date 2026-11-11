#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع: التكرار (Loops)
القسم 7.3: ⚠️ الفخ الأخطر: الحلقات اللا نهائية (Infinite Loops)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 07-003-ex7_3.py
"""

count: int = 1

while count <= 5:
    print(f"العدّ: {count}")
    # نسيان count += 1 !
