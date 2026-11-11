#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.3: `pathlib`: الأسلوب الحديث للتعامل مع المسارات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-010-ex16_3.py
"""

from pathlib import Path

p = Path("data/2024/report.txt")

print(p.name)        # "report.txt" - اسم الملف كاملاً
print(p.stem)        # "report" - الاسم بدون الامتداد
print(p.suffix)      # ".txt" - الامتداد فقط
print(p.parent)      # "data/2024" - المجلد الأب
print(p.parts)       # ('data', '2024', 'report.txt') - كل الأجزاء

print(p.exists())    # هل المسار موجود فعليًا؟ True/False
print(p.is_file())   # هل هو ملف؟
print(p.is_dir())    # هل هو مجلد؟
print(p.absolute())  # المسار الكامل المطلق
