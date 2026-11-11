#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.11: اختبر فهمك (Quiz)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-040-modify.py
"""

def modify(x, lst):
    x += 1
    lst.append(x)

a = 10
b = [1, 2, 3]
modify(a, b)
print(a, b)
