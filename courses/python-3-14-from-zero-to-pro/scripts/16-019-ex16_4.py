#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-019-ex16_4.py
"""

import json

with open("user.json", "r", encoding="utf-8") as file:
    data: dict[str, object] = json.load(file)

print(data["name"])  # "سارة"
