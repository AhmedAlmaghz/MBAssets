#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-004-number.py
"""

try:
    number = int(input("أدخل رقماً: "))
    print(f"ضعف الرقم: {number * 2}")
except ValueError:
    print("❌ خطأ: الرجاء إدخال رقم صحيح فقط.")

print("البرنامج استمر في العمل!")
