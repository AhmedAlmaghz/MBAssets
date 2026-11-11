#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-018-json_text.py
"""

import json

json_text = '{"name": "خالد", "age": 28}'
data: dict[str, object] = json.loads(json_text)

print(data["name"])  # "خالد"
print(type(data))    # <class 'dict'>
