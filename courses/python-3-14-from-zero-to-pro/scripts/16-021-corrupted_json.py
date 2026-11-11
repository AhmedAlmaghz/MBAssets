#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-021-corrupted_json.py
"""

import json

corrupted_json = '{"name": "أحمد", "age": }'  # صيغة خاطئة

try:
    data = json.loads(corrupted_json)
except json.JSONDecodeError as e:
    print(f"خطأ في تنسيق JSON: {e}")
