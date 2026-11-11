#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-006-value.py
"""

try:
    value = my_dict[key]
    result = 100 / value
except (KeyError, ZeroDivisionError) as e:
    print(f"حدث خطأ: {e}")
