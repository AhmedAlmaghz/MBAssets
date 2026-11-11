#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-020-encoding.py
"""

import json
from pathlib import Path

# كتابة
Path("user.json").write_text(
    json.dumps(user_data, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

# قراءة
data = json.loads(Path("user.json").read_text(encoding="utf-8"))
