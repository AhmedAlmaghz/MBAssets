#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع: التكرار (Loops)
القسم 7.17: مشروع تطبيقي 2: «جدول الضرب التفاعلي»

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 07-028-ex7_17.py
"""

# multiplication_table.py
"""طباعة جدول ضرب منسّق من 1 إلى 10."""

SIZE: int = 10

# طباعة رأس الجدول
print("    ", end="")
for col in range(1, SIZE + 1):
    print(f"{col:4}", end="")
print()
print("-" * (SIZE * 4 + 4))

# طباعة الجدول الكامل
for row in range(1, SIZE + 1):
    print(f"{row:3} |", end="")
    for col in range(1, SIZE + 1):
        result: int = row * col
        print(f"{result:4}", end="")
    print()  # سطر جديد بعد كل صف
