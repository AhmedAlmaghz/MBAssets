#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.4: else و finally: إكمال الصورة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-013-number.py
"""

# أقل وضوحاً - أي سطر بالضبط قد يُطلق الخطأ؟
try:
    number = int(text)
    print(f"ضعفه: {number * 2}")  # ماذا لو هذا السطر نفسه فشل لسبب آخر؟
except ValueError:
    print("خطأ")

# أوضح بكثير - try تحتوي فقط على الكود المحتمل فشله
try:
    number = int(text)
except ValueError:
    print("خطأ")
else:
    print(f"ضعفه: {number * 2}")  # نعلم بالتأكيد أن number تم تحويله بنجاح
