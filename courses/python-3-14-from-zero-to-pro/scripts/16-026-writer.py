#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.5: CSV: البيانات الجدولية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-026-writer.py
"""

import csv

rows: list[list[str]] = [
    ["name", "age", "city"],
    ["أحمد", "30", "القاهرة"],
    ["سارة", "25", "دبي"],
]

with open("people.csv", "w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(rows)
