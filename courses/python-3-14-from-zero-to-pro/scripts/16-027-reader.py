#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.5: CSV: البيانات الجدولية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-027-reader.py
"""

import csv

with open("people.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)  # كل صف عبارة عن list[str]
# ['name', 'age', 'city']
# ['أحمد', '30', 'القاهرة']
# ['سارة', '25', 'دبي']
