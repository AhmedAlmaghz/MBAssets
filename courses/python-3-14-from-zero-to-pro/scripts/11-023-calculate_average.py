#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.8: دالة assert: التحقق أثناء التطوير

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-023-calculate_average.py
"""

def calculate_average(numbers: list[float]) -> float:
    assert len(numbers) > 0, "القائمة يجب ألا تكون فارغة"
    return sum(numbers) / len(numbers)

print(calculate_average([10, 20, 30]))  # 20.0
# print(calculate_average([]))  
# 💥 AssertionError: القائمة يجب ألا تكون فارغة
