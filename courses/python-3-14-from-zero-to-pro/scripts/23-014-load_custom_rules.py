#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث والعشرون: مسار بناء أدوات سطر الأوامر (CLI)
القسم 23.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 23-014-load_custom_rules.py
"""

import json
from pathlib import Path

from file_organizer.rules import FILE_CATEGORIES

def load_custom_rules() -> dict[str, list[str]]:
    """تحميل قواعد تصنيف مخصصة من ملف إعدادات إن وُجد، ودمجها مع الافتراضية."""
    config_locations = [Path.cwd() / ".forganizerc", Path.home() / ".forganizerc"]

    merged_rules = dict(FILE_CATEGORIES)  # نسخة من القواعد الافتراضية

    for config_path in config_locations:
        if config_path.exists():
            custom_rules = json.loads(config_path.read_text(encoding="utf-8"))
            merged_rules.update(custom_rules)  # القواعد المخصصة تطغى على الافتراضية
            break

    return merged_rules
