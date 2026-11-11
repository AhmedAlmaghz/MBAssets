#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع: التكرار (Loops)
القسم 7.9: الحلقات المتداخلة (Nested Loops)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 07-016-ex7_9.py
"""

for i in range(1, 4):
    for j in range(1, 4):
        result: int = i * j
        print(f"{i} × {j} = {result}")
    print("-" * 15)  # فاصل بعد كل جدول فرعي
