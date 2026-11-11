#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الأول: كيف يفكّر المبرمج؟
القسم 1.6: عقلية حل المشكلات: المهارة الحقيقية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 01-001-is_leap_year.py
"""

def is_leap_year(year: int) -> bool:
    """تحديد ما إذا كانت السنة كبيسة أم لا."""
    if year % 400 == 0:
        return True
    elif year % 100 == 0:
        return False
    elif year % 4 == 0:
        return True
    else:
        return False

# اختبار الدالة
print(is_leap_year(2024))  # True
print(is_leap_year(1900))  # False
print(is_leap_year(2000))  # True
