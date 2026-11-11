#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.8: حفظ البيانات المُستخرجة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-015-data.py
"""

import csv

data = [
    {"title": "A Light in the Attic", "price": "£51.77", "rating": "Three"},
    {"title": "Tipping the Velvet", "price": "£53.74", "rating": "One"},
]

with open("books.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["title", "price", "rating"])
    writer.writeheader()
    writer.writerows(data)
