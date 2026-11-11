#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.7: دوال من الدرجة الأولى (First-Class Functions) - مقدمة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-032-square.py
"""

def square(x: int) -> int:
    return x ** 2

# تخزين الدالة في متغير (بدون أقواس استدعاء!)
my_func = square
print(my_func(5))  # 25

# تمرير دالة كمعامل لدالة أخرى
def apply_function(func, value: int) -> int:
    return func(value)

print(apply_function(square, 6))  # 36
