#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.4: القيمة المُرجعة (Return Values) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-019-get_min_max.py
"""

def get_min_max(numbers: list[float]) -> tuple[float, float]:
    """تُرجع أصغر وأكبر قيمة في القائمة."""
    return min(numbers), max(numbers)

low, high = get_min_max([5, 2, 9, 1, 7])
print(f"الأصغر: {low}, الأكبر: {high}")  # الأصغر: 1, الأكبر: 9
