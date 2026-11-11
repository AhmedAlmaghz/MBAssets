#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.3: `pathlib`: الأسلوب الحديث للتعامل مع المسارات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-012-notes.py
"""

from pathlib import Path

notes = Path("notes.txt")

# كتابة (تُنشئ الملف تلقائيًا إن لم يكن موجودًا، وتفتح وتغلق تلقائيًا)
notes.write_text("مرحبًا من pathlib!\n", encoding="utf-8")

# قراءة
content: str = notes.read_text(encoding="utf-8")
print(content)
