#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-007-ex11_3.py
"""

try:
    risky_operation()
except Exception as e:
    print(f"حدث خطأ غير متوقع: {e}")
