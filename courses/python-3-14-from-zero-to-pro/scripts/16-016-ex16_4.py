#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-016-ex16_4.py
"""

import json

user_data: dict[str, object] = {
    "name": "أحمد",
    "age": 30,
    "is_active": True,
    "hobbies": ["قراءة", "برمجة"],
    "spouse": None,
}

# dumps (مع s) تُرجع نصًا (string) يمكن طباعته أو إرساله
json_string: str = json.dumps(user_data, ensure_ascii=False, indent=2)
print(json_string)
