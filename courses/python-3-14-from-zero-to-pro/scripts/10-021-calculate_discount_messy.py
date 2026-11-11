#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.4: القيمة المُرجعة (Return Values) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-021-calculate_discount_messy.py
"""

def calculate_discount_messy(price: float, is_member: bool) -> float:
    if price > 0:
        if is_member:
            return price * 0.85
        else:
            return price
    else:
        return 0.0
