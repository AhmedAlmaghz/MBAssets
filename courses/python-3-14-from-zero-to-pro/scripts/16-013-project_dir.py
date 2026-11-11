#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.3: `pathlib`: الأسلوب الحديث للتعامل مع المسارات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-013-project_dir.py
"""

from pathlib import Path

project_dir = Path(".")  # المجلد الحالي

# كل الملفات والمجلدات مباشرة (دون تعمق)
for item in project_dir.iterdir():
    print(item)

# البحث عن كل ملفات .py في المجلد الحالي فقط
for py_file in project_dir.glob("*.py"):
    print(py_file)

# البحث التكراري في كل المجلدات الفرعية أيضًا
for py_file in project_dir.glob("**/*.py"):
    print(py_file)

# بديل أحدث وأوضح لنفس الغرض (Python 3.12+)
for py_file in project_dir.rglob("*.py"):
    print(py_file)
