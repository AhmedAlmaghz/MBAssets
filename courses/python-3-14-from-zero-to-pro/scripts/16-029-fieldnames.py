#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.5: CSV: البيانات الجدولية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-029-fieldnames.py
"""

import csv

# الكتابة بـ DictWriter
people: list[dict[str, str]] = [
    {"name": "منى", "age": "22", "city": "عمّان"},
    {"name": "طارق", "age": "35", "city": "بيروت"},
]

with open("people2.csv", "w", encoding="utf-8", newline="") as file:
    fieldnames = ["name", "age", "city"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()   # يكتب صف العناوين تلقائيًا
    writer.writerows(people)
