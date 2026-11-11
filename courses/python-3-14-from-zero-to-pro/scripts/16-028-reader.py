#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.5: CSV: البيانات الجدولية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-028-reader.py
"""

import csv

# القراءة بـ DictReader
with open("people.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row)
        # {'name': 'أحمد', 'age': '30', 'city': 'القاهرة'}
        print(f"{row['name']} يعيش في {row['city']}")
