#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع: التكرار (Loops)
القسم 7.2: حلقة `while`: التكرار الشرطي

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 07-002-ex7_2.py
"""

count: int = 1

while count <= 5:
    print(f"العدّ: {count}")
    count += 1

print("انتهى العد!")
