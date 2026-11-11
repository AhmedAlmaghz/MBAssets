#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.6: أفضل الممارسات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-031-file_path.py
"""

# ❌ تجنّب
file_path = "data" + "/" + "2024" + "/" + "report.txt"

# ✅ استخدم
from pathlib import Path
file_path = Path("data") / "2024" / "report.txt"
