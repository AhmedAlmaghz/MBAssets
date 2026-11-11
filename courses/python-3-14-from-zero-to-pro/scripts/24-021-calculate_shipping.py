#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.9: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-021-calculate_shipping.py
"""

def calculate_shipping(weight):
    if weight > 20:
        return weight * 5.5
    return weight * 3.2
