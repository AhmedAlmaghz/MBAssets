#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.4: الفحص الآلي للأخطاء والجودة: `ruff`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-006-process_data.py
"""

# problematic.py
import os  # مستورد لكن غير مستخدم
import sys

def process_data(data):
    unused_variable = 42  # معرّف لكن غير مستخدم
    result = []
    for item in data:
        if item == None:  # يجب استخدام "is None" وليس "== None"
            continue
        result.append(item)
    return result
