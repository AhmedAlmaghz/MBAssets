#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.3: `pathlib`: الأسلوب الحديث للتعامل مع المسارات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-009-notes_path.py
"""

from pathlib import Path

# مسار نسبي
notes_path = Path("notes.txt")

# مسار بمجلدات متعددة - لاحظ استخدام / كعامل دمج!
project_path = Path("data") / "2024" / "report.txt"
print(project_path)
# على لينكس/ماك: data/2024/report.txt
# على Windows: data\2024\report.txt  (يتكيف تلقائيًا!)
