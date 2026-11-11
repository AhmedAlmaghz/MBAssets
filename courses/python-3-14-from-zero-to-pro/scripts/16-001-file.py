#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.2: الملفات النصية: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-001-file.py
"""

file = open("notes.txt", "r", encoding="utf-8")
content = file.read()
print(content)
file.close()  # يجب أن نتذكر إغلاق الملف يدويًا!
