#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-015-calculate_sum.py
"""

def calculate_sum(*numbers: float) -> float:
    """تجمع أي عدد من الأرقام الممررة."""
    total = 0
    for num in numbers:
        total += num
    return total

print(calculate_sum(1, 2, 3))           # 6
print(calculate_sum(1, 2, 3, 4, 5))     # 15
print(calculate_sum())                   # 0

# داخل الدالة، numbers تصبح tuple
def show_type(*args):
    print(type(args))  # <class 'tuple'>

show_type(1, 2, 3)
