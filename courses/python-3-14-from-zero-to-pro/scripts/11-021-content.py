#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.7: فلسفتان في البرمجة: EAFP مقابل LBYL

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-021-content.py
"""

# تحقق أولاً، ثم نفّذ
import os

if os.path.exists("data.txt"):
    with open("data.txt") as f:
        content = f.read()
else:
    print("الملف غير موجود")
