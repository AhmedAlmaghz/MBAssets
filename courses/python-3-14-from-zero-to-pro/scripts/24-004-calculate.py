#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.3: التنسيق الآلي: `black`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-004-calculate.py
"""

# قبل black (ملف messy.py)
def   calculate(x,y) :
    result=x+y
    if result>10:
        return   'كبير'
    else:
        return "صغير"
