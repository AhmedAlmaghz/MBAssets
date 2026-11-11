#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.7: فلسفتان في البرمجة: EAFP مقابل LBYL

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-022-content.py
"""

# نفّذ مباشرة، وتعامل مع الخطأ إن حدث
try:
    with open("data.txt") as f:
        content = f.read()
except FileNotFoundError:
    print("الملف غير موجود")
