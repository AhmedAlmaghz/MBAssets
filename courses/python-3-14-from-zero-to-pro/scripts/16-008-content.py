#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.2: الملفات النصية: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-008-content.py
"""

try:
    with open("missing_file.txt", "r", encoding="utf-8") as file:
        content = file.read()
except FileNotFoundError:
    print("الملف غير موجود!")
except PermissionError:
    print("لا تملك صلاحية الوصول لهذا الملف!")
