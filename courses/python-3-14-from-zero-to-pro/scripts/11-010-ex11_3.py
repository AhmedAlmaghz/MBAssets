#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-010-ex11_3.py
"""

try:
    x = 1 / 0
except Exception:
    print("خطأ عام")  # هذا ما سيُطبع
except ZeroDivisionError:  # ⚠️ ميت! لن يُنفَّذ أبداً
    print("قسمة على صفر")
