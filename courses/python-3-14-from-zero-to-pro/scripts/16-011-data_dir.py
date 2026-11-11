#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.3: `pathlib`: الأسلوب الحديث للتعامل مع المسارات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-011-data_dir.py
"""

from pathlib import Path

data_dir = Path("data/2024")
data_dir.mkdir(parents=True, exist_ok=True)
