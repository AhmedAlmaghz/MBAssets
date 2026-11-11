#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس: اتخاذ القرار (Control Flow)
القسم 6.9: match-case: الأسلوب الحديث الثوري (Python 3.10+)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 06-023-describe_value.py
"""

# مطابقة أنواع البيانات والبنى (سنفهم هذا بعمق أكبر لاحقًا مع القوائم)
def describe_value(value):
    match value:
        case 0:
            return "صفر"
        case int() if value > 0:
            return "رقم صحيح موجب"
        case int() if value < 0:
            return "رقم صحيح سالب"
        case str():
            return "نص"
        case _:
            return "نوع غير معروف"

print(describe_value(5))       # رقم صحيح موجب
print(describe_value(-3))      # رقم صحيح سالب
print(describe_value("hi"))    # نص
